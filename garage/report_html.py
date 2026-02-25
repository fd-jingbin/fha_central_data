# -*- coding: utf-8 -*-
"""
dict[str, DataFrame] -> 交互式 HTML 报表

你的当前列结构（每个 df）：
- 字符串列：子行业 / 子行业多空(Long|Neutral|Short) / 建议交易Ticker(可能多个，用 | 分隔)
- Pipe 字符串列：结论（拆出：置信度、时间、D1..D3、M1..M3）
- 解释段落列：M1..M3、D1..D3（合并成：详细解释，大 dropdown 包小 dropdown）

HTML 效果：
- 不显示 index
- 子行业多空 conditional formatting：Long 绿 / Neutral 黄 / Short 红
- D1..D3、M1..M3 conditional formatting：1 绿 / 0 黄 / -1 红
- 置信度：data bar
- 建议交易Ticker：每个 ticker 变为可点击链接，跳转 Yahoo Finance（sg.finance.yahoo.com）
- ticker_suffix：入口参数控制后缀；None=不加；否则链接用 XXXX.<suffix>（支持 suffix="T" 或 suffix=".T"）
"""

import pandas as pd
import re
import html as ihtml


# ===== 配置区 =====
SIGNAL_COLS = ["D1", "D2", "D3", "M1", "M2", "M3"]
PIPE_FIELDS = ["置信度", "时间"] + SIGNAL_COLS

DEFAULT_EXPLAIN_COLS = ("M1", "M2", "M3", "D1", "D2", "D3")
DEFAULT_PIPE_COL = "结论"
DEFAULT_TICKER_COL = "建议交易Ticker"
DEFAULT_SIDE_COL = "子行业多空"

YAHOO_BASE = "https://sg.finance.yahoo.com/quote/"

# 非信号列宽度权重（剩余宽度会按权重分配）
WIDTH_WEIGHTS = {
    "子行业": 4.0,
    "子行业多空": 0.8,
    "建议交易Ticker": 5.2,
    "置信度": 0.8,
    "时间": 1.0,
    "详细解释": 7.0,
}


# ===== 工具函数 =====
def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _escape_for_html_text(s: str) -> str:
    """转义 + 保留换行"""
    if _is_nan(s):
        return ""
    s = ihtml.escape(str(s))
    return s.replace("\n", "<br/>")


def _normalize_suffix(ticker_suffix):
    """
    None -> None
    "T" -> ".T"
    ".T" -> ".T"
    """
    if ticker_suffix is None:
        return None
    suf = str(ticker_suffix).strip()
    if not suf:
        return None
    return suf if suf.startswith(".") else "." + suf


def build_ticker_links(cell, ticker_suffix=None, base_url=YAHOO_BASE):
    """
    把 "ETN | HUBB | ABB" -> "<a ...>ETN</a> | <a ...>HUBB</a> | ..."
    链接跳转到 https://sg.finance.yahoo.com/quote/<SYMBOL>/
    其中 SYMBOL = ticker + suffix（如果 suffix 非 None）
    """
    if _is_nan(cell):
        return ""

    s = str(cell).strip()
    if not s:
        return ""

    suf = _normalize_suffix(ticker_suffix)

    # 用 | 分隔（同时兼容逗号/中文逗号）
    raw_parts = re.split(r"\s*\|\s*|[,，]\s*", s)
    tickers = [p.strip() for p in raw_parts if p and p.strip()]

    links = []
    for t in tickers:
        # 显示文本用原 ticker，链接用加后缀后的 symbol
        t = t.replace('.SH', '.SS')
        symbol = t + suf if suf else t
        url = f"{base_url}{ihtml.escape(symbol)}/"
        links.append(f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{ihtml.escape(symbol)}</a>")

    return " | ".join(links)


def build_detail_html(row: pd.Series, explain_cols=DEFAULT_EXPLAIN_COLS) -> str:
    """
    单元格结构：
      <details class='detail-outer'>
        <summary>详细解释</summary>
        <ul>
          <li><details class='detail-inner'><summary>M1</summary>...</details></li>
          ...
        </ul>
      </details>
    """
    inner_items = []
    for c in explain_cols:
        txt = row.get(c, "")
        if _is_nan(txt):
            continue
        txt = str(txt).strip()
        if not txt:
            continue

        safe = _escape_for_html_text(txt)
        inner_items.append(
            f"<li><details class='detail-inner'>"
            f"<summary>{ihtml.escape(c)}</summary>"
            f"<div class='detail-text'>{safe}</div>"
            f"</details></li>"
        )

    if not inner_items:
        return ""

    return (
        "<details class='detail-outer'>"
        "<summary>详细解释</summary>"
        "<ul class='detail-list'>"
        + "".join(inner_items) +
        "</ul></details>"
    )


def split_pipe_kv_column(df: pd.DataFrame, source_col: str, drop_source: bool = True, strip_parens: bool = True) -> pd.DataFrame:
    """
    把形如：
      '置信度 62 | 时间 T1+T2 | D1 (0) | ... | M3 (0)'
    拆成列：置信度、时间、D1..M3
    """
    out = df.copy()

    def parse_one(cell):
        result = {k: None for k in PIPE_FIELDS}
        if _is_nan(cell):
            return result

        s = str(cell).strip()
        if not s:
            return result

        parts = [p.strip() for p in s.split("|") if p.strip()]
        for part in parts:
            m = re.match(r"^(\S+)\s+(.*)$", part)
            if not m:
                continue

            key, value = m.group(1).strip(), m.group(2).strip()
            key = re.sub(r"[:：]$", "", key)  # 兼容 D1: / D1：

            if key in result:
                if strip_parens:
                    value = re.sub(r"^\((.*)\)$", r"\1", value.strip())  # (0)->0
                result[key] = value

        return result

    parsed = out[source_col].apply(parse_one).apply(pd.Series).reindex(columns=PIPE_FIELDS)
    out = pd.concat([out, parsed], axis=1)

    if drop_source and source_col in out.columns:
        out = out.drop(columns=[source_col])

    return out


def preprocess_df(
    df: pd.DataFrame,
    pipe_col: str = DEFAULT_PIPE_COL,
    explain_cols=DEFAULT_EXPLAIN_COLS,
    ticker_col: str = DEFAULT_TICKER_COL,
    side_col: str = DEFAULT_SIDE_COL,
    ticker_suffix=None,
    keep_raw_pipe: bool = False,
) -> pd.DataFrame:
    """
    处理顺序（关键）：
    1) 详细解释 = 合并解释列（M1..D3 段落）
    2) drop 原解释列（避免与解析出来的 D1..M3 数值列撞名）
    3) 解析 pipe 列（结论）-> 置信度/时间/D1..M3
    4) 数值化（便于上色/data bar）
    5) Ticker 列转成 hyperlink HTML
    6) 调整列顺序
    """
    cur = df.copy()

    if keep_raw_pipe and pipe_col in cur.columns:
        cur[pipe_col + "原文"] = cur[pipe_col]

    # 1) 合并解释段落
    cur["详细解释"] = cur.apply(lambda r: build_detail_html(r, explain_cols), axis=1)

    # 2) 删原解释列（避免撞名）
    cur = cur.drop(columns=[c for c in explain_cols if c in cur.columns])

    # 3) 拆结论 pipe -> 数值信号列
    if pipe_col in cur.columns:
        cur = split_pipe_kv_column(cur, source_col=pipe_col, drop_source=True, strip_parens=True)

    # 4) 数值化
    if "置信度" in cur.columns:
        cur["置信度"] = pd.to_numeric(cur["置信度"], errors="coerce")
    for c in SIGNAL_COLS:
        if c in cur.columns:
            cur[c] = pd.to_numeric(cur[c], errors="coerce")

    # 5) ticker hyperlink（保留列名不变）
    if ticker_col in cur.columns:
        cur[ticker_col] = cur[ticker_col].apply(lambda x: build_ticker_links(x, ticker_suffix=ticker_suffix))

    # 6) 列顺序（按阅读习惯）
    preferred = []
    for c in ["子行业", side_col, ticker_col, "置信度", "时间"] + SIGNAL_COLS + ["详细解释"]:
        if c in cur.columns:
            preferred.append(c)
    for c in cur.columns:
        if c not in preferred:
            preferred.append(c)

    return cur[preferred]


def _build_column_width_styles(columns, signal_pct: float = 3.0):
    """
    - 信号列（D1..M3）每列固定占 signal_pct%（默认 3%） -> 很窄
    - 剩余宽度按 WIDTH_WEIGHTS 分配给其他列
    """
    cols = list(columns)
    present_signal = [c for c in cols if c in SIGNAL_COLS]

    signal_total = signal_pct * len(present_signal)
    remaining = max(0.0, 100.0 - signal_total)

    other_cols = [c for c in cols if c not in present_signal]
    other_weights = [WIDTH_WEIGHTS.get(c, 1.5) for c in other_cols]
    sw = sum(other_weights) if other_weights else 1.0

    styles = []
    for i, c in enumerate(cols):
        if c in present_signal:
            pct = signal_pct
        else:
            w = WIDTH_WEIGHTS.get(c, 1.5)
            pct = remaining * (w / sw)

        styles.append({
            "selector": f"th.col{i}, td.col{i}",
            "props": [("width", f"{pct:.2f}%")],
        })
    return styles


def style_df_to_html(df: pd.DataFrame, side_col: str = DEFAULT_SIDE_COL, signal_pct: float = 3.0) -> str:
    """
    输出一个 HTML table：
    - 不显示 index
    - 子行业多空 conditional formatting（Long/Neutral/Short）
    - 信号列上色（1/0/-1）
    - 置信度 data bar
    - 详细解释 <details> 可交互
    - ticker 列是 <a> hyperlink
    - 列宽：信号列窄，其他列按权重分配
    """
    styler = df.style

    # 允许 HTML（<details> / <a>）生效
    styler = styler.format(escape=None)

    # 不显示 index（兼容不同 pandas 版本）
    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass

    # 子行业多空上色
    def color_side(v):
        if _is_nan(v):
            return ""
        s = str(v).strip()
        if s == "Long":
            return "background-color:#C8E6C9; font-weight:600;"  # 绿
        if s == "Neutral":
            return "background-color:#FFF59D; font-weight:600;"  # 黄
        if s == "Short":
            return "background-color:#FFCDD2; font-weight:600;"  # 红
        return ""

    if side_col in df.columns:
        styler = styler.applymap(color_side, subset=[side_col])

    # 信号列上色（1/0/-1）
    def color_signal(v):
        if _is_nan(v):
            return ""
        try:
            x = float(v)
        except Exception:
            return ""
        if x == 1:
            return "background-color:#C8E6C9;"
        if x == 0:
            return "background-color:#FFF59D;"
        if x == -1:
            return "background-color:#FFCDD2;"
        return ""

    subset = [c for c in SIGNAL_COLS if c in df.columns]
    if subset:
        styler = styler.applymap(color_signal, subset=subset)

    # 置信度 data bar（默认 0~100）
    if "置信度" in df.columns:
        styler = styler.bar(subset=["置信度"], vmin=0, vmax=100, color="#C4C4C4")
        styler = styler.format({"置信度": "{:.0f}"})

    # 换行 + 顶部对齐
    styler = styler.set_properties(**{
        "white-space": "normal",
        "overflow-wrap": "anywhere",
        "vertical-align": "top",
    })

    # 列宽（信号列窄、其他列分配）
    col_styles = _build_column_width_styles(df.columns, signal_pct=signal_pct)
    styler = styler.set_table_attributes('style="table-layout:fixed;width:100%;"')
    styler = styler.set_table_styles(col_styles, overwrite=False)

    return styler.to_html()


def dict_to_interactive_html(
    df_dict: dict,
    out_html_path: str,
    pipe_col: str = DEFAULT_PIPE_COL,
    explain_cols=DEFAULT_EXPLAIN_COLS,
    ticker_col: str = DEFAULT_TICKER_COL,
    side_col: str = DEFAULT_SIDE_COL,
    ticker_suffix=None,          # None=不加后缀；"T" 或 ".T" 等=加后缀
    keep_raw_pipe: bool = False,
    signal_pct: float = 3.0,     # 每个信号列占比（越小越窄）
    run_date=""
):
    """
    主入口：dict[str, DataFrame] -> 单个 HTML 文件（每个 key 一段）
    """
    page_css = """
    <style>
      body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif; margin: 20px; }
      h2 { margin-top: 28px; }

      table { border-collapse: collapse; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; }
      th { position: sticky; top: 0; background: #f6f6f6; z-index: 1; }

      /* links */
      a { color: #0366d6; text-decoration: none; }
      a:hover { text-decoration: underline; }

      /* dropdown 样式 */
      .detail-outer > summary { cursor:pointer; font-weight:700; }
      .detail-inner > summary { cursor:pointer; font-weight:600; }
      .detail-list { margin: 6px 0 0 0; padding-left: 18px; }
      .detail-text { margin-top: 6px; line-height: 1.35; }
    </style>
    """

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        page_css,
        "</head><body>",
        f"<h1>行业总览 {run_date}</h1>",
    ]

    for key, df in df_dict.items():
        if df is None or df.empty:
            continue

        cur = preprocess_df(
            df,
            pipe_col=pipe_col,
            explain_cols=explain_cols,
            ticker_col=ticker_col,
            side_col=side_col,
            ticker_suffix=ticker_suffix,
            keep_raw_pipe=keep_raw_pipe,
        )

        parts.append(f"<h2>{ihtml.escape(str(key))}</h2>")
        parts.append(style_df_to_html(cur, side_col=side_col, signal_pct=signal_pct))

    parts.append("</body></html>")

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
