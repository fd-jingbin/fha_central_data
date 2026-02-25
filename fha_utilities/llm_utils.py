# parallel_openai_asker.py

import importlib

import config.llm_config as llm_cfg

importlib.reload(llm_cfg)

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import json
import re
import time
import random
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

from openai import OpenAI


# =========================
# tokens.txt 读取
# =========================

def load_env_tokens_txt(path: str | Path = llm_cfg.API_TOKEN_DIR, key_prefix: str = "OPENAI_API_KEY_") -> Dict[str, str]:
    """
    读取形如 KEY=VALUE 的 txt，返回 {KEY: VALUE}
    - 支持空行
    - 支持注释 # ...
    - 支持 export KEY=VALUE
    - 只提取以 key_prefix 开头的 key
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Token txt not found: {p}")

    tokens: Dict[str, str] = {}
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k.startswith(key_prefix) and v:
            tokens[k] = v

    if not tokens:
        raise ValueError(f"No tokens found with prefix '{key_prefix}' in {p}")
    return tokens


# =========================
# 文件名安全化
# =========================

def slugify(value: Any, max_len: int = 80) -> str:
    """
    把 task value 变成适合当文件名的一段：
    - 保留中英文、数字、下划线、短横线
    - 其他字符替换成 _
    """
    s = str(value).strip()
    # 常见分隔符替换
    s = s.replace("/", "-").replace("\\", "-").replace(" ", "_")
    # 去掉不可见字符
    s = re.sub(r"[\r\n\t]+", "_", s)
    # 只允许：中文、字母数字、_、-、.
    s = re.sub(r"[^\w\u4e00-\u9fff\.\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "NA"
    return s[:max_len]


# =========================
# OpenAI QA Helper（每次传 api_key，线程安全）
# =========================

@dataclass
class AskResult:
    text: str
    sources: List[Dict[str, Any]]
    response_id: Optional[str] = None


class OpenAIQAHelper:
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)

    def ask(
        self,
        prompts: str,
        gpt_model: str,
        web_use: bool = True,
        thinking: bool = False,
        reasoning_effort: Optional[Dict[str, str]] = None,
    ) -> AskResult:
        payload: Dict[str, Any] = {
            "model": gpt_model,
            "input": prompts,
        }

        if web_use:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
            payload["include"] = ["web_search_call.action.sources"]

        if thinking:
            payload["reasoning"] = reasoning_effort or {"effort": "high"}

        try:
            resp = self.client.responses.create(**payload)
        except Exception as e:
            msg = str(e)
            # 降级：关 web_search 再试一次
            if web_use:
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                payload.pop("include", None)
                resp = self.client.responses.create(**payload)
            else:
                raise RuntimeError(f"OpenAI API call failed: {msg}") from e

        text = getattr(resp, "output_text", None) or ""

        sources: List[Dict[str, Any]] = []
        output_items = getattr(resp, "output", None) or []
        for item in output_items:
            if getattr(item, "type", None) == "web_search_call":
                action = getattr(item, "action", None)
                if action is not None:
                    srcs = getattr(action, "sources", None)
                    if srcs:
                        for s in srcs:
                            if isinstance(s, dict):
                                sources.append(s)
                            else:
                                sources.append(getattr(s, "__dict__", {"value": str(s)}))

        return AskResult(
            text=text,
            sources=sources,
            response_id=getattr(resp, "id", None),
        )


# =========================
# 并发 Runner（新存储机制：suffix + task关键词 + daterun）
# =========================

@dataclass
class TaskResult:
    task: Dict[str, Any]
    status: str                    # "OK" | "SKIP" | "FAIL"
    text: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    response_id: Optional[str] = None
    error: Optional[str] = None
    token_label: Optional[str] = None
    out_path: Optional[str] = None
    cache_hit_from: Optional[str] = None  # 如果是跨日期命中旧文件，会记录旧文件路径


class ParallelOpenAIAsker:
    """
    你只需要：
    - tokens（从 txt 读出来）
    - tasks（for loop 的变量组合，每个 task 是 dict）
    - template（字符串模板）
    就能并发跑。

    存储机制：
      {suffix}_{taskvalue1}_{taskvalue2(ifany)}_{daterun}.json
    断点续跑：
      - 当天文件存在：直接 skip
      - 当天文件不存在但历史日期存在同 prefix：读最新那份，并复制成当天文件（SKIP）
    """

    def __init__(
        self,
        tokens: Dict[str, str],                # {"OPENAI_API_KEY_A": "sk-...", ...}
        out_dir: str | Path,
        *,
        max_workers: int = 16,
        per_token_max_inflight: int = 2,
        retries: int = 4,
        backoff_base: float = 0.7,
        jitter: float = 0.25,
        name_keys: Optional[List[str]] = None, # 用哪些 task key 来拼文件名（按顺序）；默认优先 region, industry
        max_name_values: int = 2,              # taskvalue 取几个（你要求 value1/value2）
    ):
        if not tokens:
            raise ValueError("tokens is empty")

        self.tokens = tokens
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = int(max_workers)
        self.per_token_max_inflight = max(1, int(per_token_max_inflight))
        self.retries = max(0, int(retries))
        self.backoff_base = float(backoff_base)
        self.jitter = float(jitter)

        self.name_keys = name_keys
        self.max_name_values = max(1, int(max_name_values))

        self._token_items: List[Tuple[str, str]] = list(tokens.items())
        self._token_cycle = cycle(self._token_items)
        self._token_pick_lock = threading.Lock()

        self._semaphores: Dict[str, threading.Semaphore] = {
            label: threading.Semaphore(self.per_token_max_inflight)
            for label, _ in self._token_items
        }

    def _pick_token(self) -> Tuple[str, str]:
        with self._token_pick_lock:
            return next(self._token_cycle)

    @staticmethod
    def _render_prompt(template: str, task: Dict[str, Any]) -> str:
        return template.format(**task)

    def _pick_name_values(self, task: Dict[str, Any]) -> List[str]:
        """
        从 task 里按规则取 value1/value2：
        - 如果传了 name_keys，就按 name_keys 顺序取
        - 否则优先 region, industry
        - 不足再按 key 名排序补齐
        """
        picked: List[str] = []
        seen_keys = set()

        if self.name_keys:
            for k in self.name_keys:
                if k in task and k not in seen_keys:
                    picked.append(slugify(task[k]))
                    seen_keys.add(k)
                if len(picked) >= self.max_name_values:
                    return picked

        # 默认偏好
        for k in ["region", "industry"]:
            if k in task and k not in seen_keys:
                picked.append(slugify(task[k]))
                seen_keys.add(k)
            if len(picked) >= self.max_name_values:
                return picked

        # 补齐：按 key 排序
        for k in sorted(task.keys()):
            if k not in seen_keys:
                picked.append(slugify(task[k]))
                seen_keys.add(k)
            if len(picked) >= self.max_name_values:
                return picked

        return picked

    def _prefix(self, task: Dict[str, Any], suffix: str) -> str:
        parts: List[str] = []
        sfx = slugify(suffix) if suffix else ""
        if sfx:
            parts.append(sfx)

        vals = self._pick_name_values(task)
        parts.extend([v for v in vals if v])

        if not parts:
            parts = ["task"]

        return "_".join(parts)

    def _result_path(self, task: Dict[str, Any], suffix: str, run_date: str) -> Path:
        prefix = self._prefix(task, suffix)
        return self.out_dir / f"{prefix}_{run_date}.json"

    def _find_latest_cache(self, task: Dict[str, Any], suffix: str) -> Optional[Path]:
        """
        找历史日期里“同 prefix”的最新文件：{prefix}_YYYY-MM-DD.json
        优先按文件名里的日期排序；解析失败再按 mtime。
        """
        prefix = self._prefix(task, suffix)
        pattern = f"{prefix}_*.json"
        candidates = list(self.out_dir.glob(pattern))
        if not candidates:
            return None

        date_re = re.compile(rf"^{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})\.json$")
        parsed: List[Tuple[str, Path]] = []
        rest: List[Path] = []

        for p in candidates:
            m = date_re.match(p.name)
            if m:
                parsed.append((m.group(1), p))
            else:
                rest.append(p)

        if parsed:
            # 日期字符串按字典序就等价于时间序（YYYY-MM-DD）
            parsed.sort(key=lambda x: x[0], reverse=True)
            return parsed[0][1]

        # fallback：按修改时间
        rest.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return rest[0] if rest else None

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _run_one(
        self,
        task: Dict[str, Any],
        template: str,
        *,
        suffix: str,
        run_date: str,
        gpt_model: str,
        web_use: bool,
        thinking: bool,
        reasoning_effort: Optional[Dict[str, str]],
        skip_if_exists: bool,
        reuse_cache_across_dates: bool,
        write_today_copy_on_cache_hit: bool,
    ) -> TaskResult:
        out_path = self._result_path(task, suffix, run_date)
        print(out_path)

        # 1) 当天存在 => 直接 skip
        if skip_if_exists and out_path.exists():
            try:
                rec = self._read_json(out_path)
                return TaskResult(**rec)
            except Exception:
                # 文件坏了就继续执行，覆盖
                pass

        # 2) 跨日期 cache 命中：找同 prefix 的历史最新文件
        if skip_if_exists and reuse_cache_across_dates:
            cached = self._find_latest_cache(task, suffix)
            if cached and cached.exists():
                try:
                    rec = self._read_json(cached)
                    # 如果不想再写当天文件，就直接返回；否则复制一份当日命名
                    tr = TaskResult(**rec)
                    tr.status = "SKIP"
                    tr.out_path = str(out_path)
                    tr.cache_hit_from = str(cached)

                    if write_today_copy_on_cache_hit:
                        self._write_json(out_path, tr.__dict__)
                    return tr
                except Exception:
                    # 缓存坏了 => 继续正常跑
                    pass

        prompt = self._render_prompt(template, task)

        last_err = None
        for i in range(self.retries + 1):
            token_label, api_key = self._pick_token()
            sem = self._semaphores.get(token_label)
            acquired = False
            try:
                if sem is not None:
                    sem.acquire()
                    acquired = True

                helper = OpenAIQAHelper(api_key=api_key)
                res = helper.ask(
                    prompts=prompt,
                    gpt_model=gpt_model,
                    web_use=web_use,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                )

                tr = TaskResult(
                    task=task,
                    status="OK",
                    text=res.text,
                    sources=res.sources,
                    response_id=res.response_id,
                    token_label=token_label,
                    out_path=str(out_path),
                )
                self._write_json(out_path, tr.__dict__)
                return tr

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                sleep_s = (self.backoff_base * (2 ** i)) + random.random() * self.jitter
                time.sleep(sleep_s)

            finally:
                if acquired and sem is not None:
                    sem.release()

        tr = TaskResult(
            task=task,
            status="FAIL",
            error=last_err,
            out_path=str(out_path),
        )
        self._write_json(out_path, tr.__dict__)
        return tr

    def run(
        self,
        tasks: Sequence[Dict[str, Any]],
        template: str,
        *,
        suffix: str = "",
        run_date: Optional[str] = None,                 # 默认当天 YYYY-MM-DD
        gpt_model: str = "gpt-5.2",
        web_use: bool = True,
        thinking: bool = True,
        reasoning_effort: Optional[Dict[str, str]] = None,
        skip_if_exists: bool = True,
        reuse_cache_across_dates: bool = True,          # 关键：日期变了也能复用历史结果（断点续跑更强）
        write_today_copy_on_cache_hit: bool = True,     # 关键：命中历史缓存时，也写一份当天日期文件
    ) -> List[TaskResult]:
        """
        - suffix：文件名前缀
        - run_date：默认当天；你也可以手动指定
        """
        if run_date is None:
            run_date = datetime.datetime.now().strftime("%Y-%m-%d")

        results: List[TaskResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [
                ex.submit(
                    self._run_one,
                    task,
                    template,
                    suffix=suffix,
                    run_date=run_date,
                    gpt_model=gpt_model,
                    web_use=web_use,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    skip_if_exists=skip_if_exists,
                    reuse_cache_across_dates=reuse_cache_across_dates,
                    write_today_copy_on_cache_hit=write_today_copy_on_cache_hit,
                )
                for task in tasks
            ]
            for fu in as_completed(futs):
                results.append(fu.result())

        summary_path = self.out_dir / f"_summary_{slugify(suffix) if suffix else 'run'}_{run_date}.json"
        summary_path.write_text(
            json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return results


# =========================
# 示例用法
# =========================

if __name__ == "__main__":
    # 1) 从 txt 读取所有 token（不要把 token commit 到 git）
    tokens = load_env_tokens_txt(llm_cfg.API_TOKEN_DIR)  # 改成你的文件路径

    # 2) 生成 tasks（替代你原本的 for loop）
    regions = ["美国", "日本"]
    industries = ["半导体", "汽车", "银行"]
    tasks = [{"region": r, "industry": i} for r in regions for i in industries]

    # 3) 模板：推荐用 {industry}/{region}
    template = "请总结 {region} 的 {industry} 行业趋势，列出要点，并给出来源链接。"

    # 如果你还没改模板，也可以继续用旧占位符：
    # template = "请总结 REGION_PLACEHOLDER 的 INDUSTRY_PLACEHOLDER 行业趋势..."

    # 4) 跑并发
    # 4) 跑并发
    runner = ParallelOpenAIAsker(
        tokens=tokens,
        out_dir=r"C:\Python\data\j2\测试",
        max_workers=24,
        per_token_max_inflight=1,
        retries=4,
        name_keys=["region", "industry"],
        max_name_values=2,
    )

    results = runner.run(
        tasks=tasks,
        template=template,
        suffix="行业趋势",
        gpt_model="gpt-4o",
        web_use=True,
        thinking=False,
        reasoning_effort=None,
        skip_if_exists=True,
        reuse_cache_across_dates=True,
        write_today_copy_on_cache_hit=True,
    )

    ok = sum(r.status == "OK" for r in results)
    fail = sum(r.status == "FAIL" for r in results)
    skip = sum(r.status == "SKIP" for r in results)
    print(f"done. OK={ok}, SKIP={skip}, FAIL={fail}. out_dir=./runs/industry_trends")
