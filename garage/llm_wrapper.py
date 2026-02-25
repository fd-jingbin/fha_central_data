import fha_utilities.llm_utils as lm_ut
import fha_utilities.time_utils as tm_ut
import fha_utilities.file_utils as fl_ut
import garage.config as ga_cfg
import garage.report_html as rh_ut

import importlib
for pkg in [lm_ut, ga_cfg, tm_ut, fl_ut, rh_ut]:
    importlib.reload(pkg)

import pandas as pd


class AgentHelper:

    def __init__(self):
        self.tokens = lm_ut.load_env_tokens_txt()
        self.industry_def = ga_cfg.OPTIMIZED_INDUSTRIES
        self.region_def = ga_cfg.REGION_DEF
        self.industry_placeholder = 'INDUSTRY_PLACEHOLDER'
        self.region_placeholder = 'REGION_PLACEHOLDER'
        self.industry_folder = ga_cfg.INDUSTRY_ANALYSIS_FOLDER
        self.industry_pickle_fn = ga_cfg.INDUSTRY_PICKLE_FN
        self.industry_suffix = 'Industry'

    def runner_helper(
            self,
            tasks,
            template,
            suffix,
            name_keys,
            max_name_values,
            out_dir,
            gpt_model="gpt-5.2",
            web_use=True,
            thinking=True,
            reasoning_effort={"effort": "high"},
            skip_if_exists=True,
            reuse_cache_across_dates=False,
            write_today_copy_on_cache_hit=True,
            max_works=36,
            per_token_max_inflight=1,
            retries=4,
            tokens=None):
        if tokens is None:
            tokens = self.tokens
        runner = lm_ut.ParallelOpenAIAsker(
            tokens=tokens,
            out_dir=out_dir,
            max_workers=max_works,
            per_token_max_inflight=per_token_max_inflight,
            retries=retries,
            name_keys=name_keys,
            max_name_values=max_name_values,
        )

        results = runner.run(
            tasks=tasks,
            template=template,
            suffix=suffix,
            gpt_model=gpt_model,
            web_use=web_use,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            skip_if_exists=skip_if_exists,
            reuse_cache_across_dates=reuse_cache_across_dates,
            write_today_copy_on_cache_hit=write_today_copy_on_cache_hit,
        )
        return results


    def generate_industry_analysis(self):
        industries = list(self.industry_def.values())
        regions = self.region_def
        tasks = [
            {
                self.region_placeholder: r,
                self.industry_placeholder: t} for t in industries for r in regions
        ]

        results = self.runner_helper(
            tasks,
            template=ga_cfg.INDUSTRY_ASK,
            suffix=self.industry_suffix,
            name_keys=[self.region_placeholder, self.industry_pickle_fn],
            max_name_values=2,
            out_dir=self.industry_folder,
            gpt_model="gpt-5.2",
            web_use=True,
            thinking=True,
            reasoning_effort={"effort": "high"},
            skip_if_exists=True,
            reuse_cache_across_dates=False,
            write_today_copy_on_cache_hit=True,
            max_works=36,
            per_token_max_inflight=1,
            retries=4,
            tokens=None
        )

        df_list = []

        for res in results:
            print(res.task['REGION_PLACEHOLDER'], res.task['INDUSTRY_PLACEHOLDER'], res.status)
            result_like = res.text
            text = fl_ut.extract_text_only(result_like)
            ticker_df = fl_ut.json_text_to_dataframe(text) \
                .pipe(lambda d: d.assign(子行业=d['子行业'].astype(str).str.partition("（")[0].str.strip(),
                                         Region=res.task['REGION_PLACEHOLDER'],
                                         Industry=res.task['INDUSTRY_PLACEHOLDER'])).drop(['使用的URL链接引用'], axis=1)
            df_list.append(ticker_df)

        res_df = pd.concat(df_list, ignore_index=True)

        res_df.to_pickle(fl_ut.join_path_name(self.industry_folder, self.industry_pickle_fn))

        for reg in regions:
            html_dict = dict()
            reg_df = res_df[res_df.Region == reg]
            for ind in industries:
                html_dict[ind] = reg_df[reg_df.Industry == ind].drop(['Region', 'Industry'], axis=1)

            html_dir = fl_ut.join_path_name(ga_cfg.INDUSTRY_ANALYSIS_FOLDER, f'{reg}行业Flipbook.html')
            rh_ut.dict_to_interactive_html(html_dict, html_dir, pipe_col="结论", ticker_suffix=None, signal_pct=2.0)

    def load_saved_industry_analysis(self):
        return pd.read_pickle(fl_ut.join_path_name(self.industry_folder, self.industry_pickle_fn))


if __name__ == '__main__':
    ag = AgentHelper()
    ag.generate_industry_analysis()
