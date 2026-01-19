# region Import Modules

# External Imports
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, date

# Internal Imports
import importlib as imp
import data_config as da_cfg
import data_utils.utils as da_ut

for pkg in [da_cfg, da_ut]:
    imp.reload(pkg)
# endregion


# region Internal Data

class EnfusionData:

    def __init__(self, cfg=da_cfg):
        self.cfg = cfg
        self.raw_file_path = str(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}.pkl'))
        self.processed_file_path = str(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.ENFUSION_PROCESSED_DATA_PICKLE_NAME}.pkl'))

    def save_raw_data_chunks_20230101_20230131(self):
        xls_path = os.path.join(self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR, self.cfg.ENFUSION_20230101_20250227_FILE_NAME)
        xls_data = pd.ExcelFile(xls_path)
        df_list = []
        for sheet in xls_data.sheet_names:
            try:
                df = pd.read_excel(xls_path, sheet_name=sheet, engine="openpyxl")
                df_list.append(df)
            except Exception as e:
                print(f"Error reading sheet {sheet}: {e}")

        df_20230101_20250131 = pd.concat(df_list, ignore_index=True, copy=False) \
            .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
            .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
            .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date))) \
            .pipe(lambda d: d[pd.to_datetime(d.Date) <= '2025-01-31'])

        df_20230101_20250131.to_pickle(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_20230101_20250131.pkl'))

    def save_raw_data_20250201_20250228(self):
        df_20250201_20250228 = pd.read_excel(str(os.path.join(self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR,
                                             self.cfg.ENFUSION_20250201_20250228_FILE_NAME))) \
            .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
            .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
            .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))

        df_20250201_20250228.to_pickle(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_20250201_20250228.pkl'))

    def save_raw_data_20250301_20250331(self):
        df_20250301_20250331 = pd.read_excel(str(os.path.join(self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR,
                                             self.cfg.ENFUSION_20250301_20250331_FILE_NAME))) \
            .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
            .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
            .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))

        df_20250301_20250331.to_pickle(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_20250301_20250331.pkl'))

    def save_raw_data_20250401_20250430(self):
        df_20250401_20250430 = pd.read_excel(str(os.path.join(self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR,
                                             self.cfg.ENFUSION_20250401_20250430_FILE_NAME))) \
            .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
            .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
            .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))

        df_20250401_20250430.to_pickle(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_20250401_20250430.pkl'))

    def get_data_with_given_date(self, dd):
        dd_file = da_ut.find_earliest_enfusion_daily(
            self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR,
            da_ut.next_day(dd).strftime('%Y-%m-%d'))
        if dd_file is None:
            dd_file = da_ut.find_earliest_enfusion_daily(
                self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR,
                dd.strftime('%Y-%m-%d'), latest=True)
        else:
            month, day, year, hour, minute = re.search(r"(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})", dd_file).groups()
            latest_date = pd.to_datetime(da_ut.get_latest_trading_date(
                da_ut.to_hongkong_datetime(int(year), int(month), int(day), int(hour), int(minute))))
            if da_ut.convert_to_datetime(dd) != da_ut.convert_to_datetime(latest_date):
                dd_file = da_ut.find_earliest_enfusion_daily(
                    self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR,
                    dd.strftime('%Y-%m-%d'), latest=True)
        if dd_file is None:
            print(f'No file found for {dd}')
            return None
        else:
            df = pd.read_excel(os.path.join(self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR, dd_file)) \
                .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
                .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
                .pipe(lambda d: d.assign(Date=dd)) \
                .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))
            return df

    def get_raw_data_between_two_dates(self, start_int, end_int):
        start_int = da_ut.previous_business_day(start_int)
        date_list = list(da_ut.business_days_between(start_int, end_int))

        date_group_list = [date_list[i:i+2] for i in range(len(date_list)-1)]
        df_till_latest_list = []
        for dd_group in date_group_list:
            mrg_cols = ['LE Name', 'Book Name', 'Account', 'BB Yellow Key', 'RIC', 'ISIN', 'SEDOL']
            df1 = self.get_data_with_given_date(dd_group[0])[mrg_cols + ['$ YTD P&L']] \
                .groupby(mrg_cols)[['$ YTD P&L']].sum().reset_index()
            df2 = self.get_data_with_given_date(dd_group[1]) \
                .pipe(lambda d: d.assign(Quantity=d.Quantity.astype(float).fillna(0)))
            df2['Market Price'] = da_ut.clean_numeric_series(df2['Market Price'])
            df2['Market Price'] = np.where(df2['Market Price'] == 0, np.nan, df2['Market Price'])

            df2['Trade/Book FX Rate'] = da_ut.clean_numeric_series(df2['Trade/Book FX Rate'])
            df2['Trade/Book FX Rate'] = np.where(df2['Trade/Book FX Rate'] == 0, np.nan, df2['Trade/Book FX Rate'])

            df2 = df2 \
                .pipe(lambda d: d.assign(
                PnLDay=d['$ Daily P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
                PnLMTD=d['$ MTD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
                PnLYTD=d['$ YTD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0))) \
                .groupby(mrg_cols).agg({'$ Daily P&L': 'sum', '$ MTD P&L': 'sum', '$ YTD P&L': 'sum', 'Quantity': 'sum',
                                        'Trade/Book FX Rate': 'mean', 'Market Price': 'mean'}).reset_index()

            recalc_pnl_day = df2.merge(df1, on=mrg_cols, how='left', suffixes=("", "_old")) \
                .pipe(lambda d: d.assign(PnLDay=d['$ YTD P&L'].fillna(0) - d['$ YTD P&L_old'].fillna(0)))
            recalc_pnl_day['$ Daily P&L'] = recalc_pnl_day['PnLDay']
            recalc_pnl_day = recalc_pnl_day.drop(['PnLDay', '$ YTD P&L_old'], axis=1)
            df_till_latest_list.append(recalc_pnl_day.assign(Date=pd.to_datetime(dd_group[1])))

        if len(df_till_latest_list) > 0:
            df = pd.concat(df_till_latest_list, axis=0)
            for col in self.cfg.ENFUSION_RAW_DATA_COLS:
                if col not in df.columns:
                    df[col] = None
            df = df[self.cfg.ENFUSION_RAW_DATA_COLS]
        else:
            df = pd.DataFrame()

        return df


    def get_raw_data_between_two_dates_backdate(self, start_int, end_int):
        start_int = da_ut.previous_business_day(start_int)
        date_list = list(da_ut.business_days_between(start_int, end_int))

        date_group_list = [date_list[i:i+2] for i in range(len(date_list)-1)]
        df_till_latest_list = []
        for dd_group in date_group_list:
            mrg_cols = ['LE Name', 'Book Name', 'Account', 'BB Yellow Key', 'RIC', 'ISIN', 'SEDOL']
            df1 = self.get_data_with_given_date(dd_group[1])[mrg_cols + ['$ ITD P&L']] \
                .groupby(mrg_cols)[['$ ITD P&L']].sum().reset_index()
            df2 = self.get_data_with_given_date(dd_group[0]) \
                .pipe(lambda d: d.assign(Quantity=d.Quantity.astype(float).fillna(0)))
            df2['Market Price'] = da_ut.clean_numeric_series(df2['Market Price'])
            df2['Market Price'] = np.where(df2['Market Price'] == 0, np.nan, df2['Market Price'])

            df2['Trade/Book FX Rate'] = da_ut.clean_numeric_series(df2['Trade/Book FX Rate'])
            df2['Trade/Book FX Rate'] = np.where(df2['Trade/Book FX Rate'] == 0, np.nan, df2['Trade/Book FX Rate'])

            df2 = df2 \
                .pipe(lambda d: d.assign(
                PnLDay=d['$ Daily P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
                PnLMTD=d['$ MTD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
                PnLYTD=d['$ YTD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
                PnLITD=d['$ ITD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0))) \
                .groupby(mrg_cols).agg({'$ Daily P&L': 'sum', '$ MTD P&L': 'sum', '$ YTD P&L': 'sum', '$ ITD P&L': 'sum', 'Quantity': 'sum',
                                        'Trade/Book FX Rate': 'mean', 'Market Price': 'mean'}).reset_index()

            recalc_pnl_day = df1.merge(df2, on=mrg_cols, how='left', suffixes=("", "_new")) \
                .pipe(lambda d: d.assign(PnLDay=d['$ ITD P&L'].fillna(0) - d['$ ITD P&L_new'].fillna(0)))
            recalc_pnl_day['$ Daily P&L'] = recalc_pnl_day['PnLDay']
            recalc_pnl_day['$ MTD P&L'] = recalc_pnl_day['$ MTD P&L'] + recalc_pnl_day['$ Daily P&L']
            recalc_pnl_day['$ YTD P&L'] = recalc_pnl_day['$ YTD P&L'] + recalc_pnl_day['$ Daily P&L']
            recalc_pnl_day = recalc_pnl_day.drop(['PnLDay', '$ ITD P&L', '$ ITD P&L_new'], axis=1)
            df_till_latest_list.append(recalc_pnl_day.assign(Date=pd.to_datetime(dd_group[1])))

        if len(df_till_latest_list) > 0:
            df = pd.concat(df_till_latest_list, axis=0)
            for col in self.cfg.ENFUSION_RAW_DATA_COLS:
                if col not in df.columns:
                    df[col] = None
            df = df[self.cfg.ENFUSION_RAW_DATA_COLS]
        else:
            df = pd.DataFrame()

        return df

    def save_raw_data_all_weeklies_after_20250430(self):
        avail_weekly_files = da_ut.find_avail_files(
            self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR, self.cfg.ENFUSION_WEEKLY_FILE_KEYWORD, 'csv')
        df_weeklies = pd.concat(
            [pd.read_csv(ww) for ww in avail_weekly_files], axis=0) \
            .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
            .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
            .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date))) \
            .pipe(lambda d: d[pd.to_datetime(d.Date) > '2025-04-30'])

        df_weeklies.to_pickle(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR,
                         f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_weeklies_after_20250430.pkl'))

    def save_raw_data_all_daily_rest(self):
        df_weeklies = pd.read_pickle(os.path.join(self.cfg.EXPOSURE_DATA_DIR,
                         f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_weeklies_after_20250430.pkl'))
        start_date = da_ut.next_business_day(pd.to_datetime(df_weeklies.Date.dropna()).max())
        date_today = da_ut.get_today_date()
        df_till_latest = self.get_raw_data_between_two_dates(start_date, date_today)

        df_till_latest.to_pickle(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR,
                         f'{self.cfg.ENFUSION_RAW_DATA_PICKLE_NAME}_till_latest.pkl'))

    def get_latest_weekly_data(self):
        fl_nm = da_ut.find_latest_file(
            self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR, self.cfg.ENFUSION_WEEKLY_FILE_KEYWORD, 'csv')
        print(f'Using {fl_nm} to update...')
        return pd.read_csv(fl_nm) \
                .rename(columns=self.cfg.ENFUSION_DATE_COL_DICT) \
                .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))

    def get_latest_data(self):
        latest_fn = da_ut.find_latest_file(
            self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR, self.cfg.ENFUSION_DAILY_FILE_KEYWORD, 'xls')
        month, day, year, hour, minute = re.search(r"(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})", latest_fn).groups()
        latest_date = pd.to_datetime(da_ut.get_latest_trading_date(
            da_ut.to_hongkong_datetime(int(year), int(month), int(day), int(hour), int(minute))))
        df = self.get_raw_data_between_two_dates(latest_date, latest_date)
        return df

    def insert_raw_data(self, df):
        if not da_ut.exist_file(self.raw_file_path):
            pd.DataFrame(columns=self.cfg.ENFUSION_RAW_DATA_COLS).to_pickle(self.raw_file_path)

        raw_df = pd.read_pickle(self.raw_file_path).pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))
        df = df.assign(Date=pd.to_datetime(df.Date))
        raw_df = raw_df[~raw_df.Date.isin(df.Date.unique())]
        raw_df = pd.concat([raw_df, df], axis=0).reset_index(drop=True)
        raw_df.to_pickle(self.raw_file_path)

    def insert_processed_data(self, df):
        processed_df = pd.read_pickle(self.processed_file_path).pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date)))
        processed_df = processed_df[~processed_df.Date.isin(df.Date.unique())]
        df = pd.concat([processed_df, df], axis=0).reset_index(drop=True)
        df.to_pickle(self.processed_file_path)

    def save_full_hist_raw_adhoc(self):
        full_hist = pd.concat([pd.read_pickle(os.path.join(self.cfg.EXPOSURE_DATA_DIR, fn)) for fn in
                               ['enfusion_raw_20230101_20250131.pkl', 'enfusion_raw_20250201_20250228.pkl',
                                'enfusion_raw_20250301_20250331.pkl', 'enfusion_raw_20250401_20250430.pkl',
                                'enfusion_raw_weeklies_after_20250430.pkl', 'enfusion_raw_till_latest.pkl']],
                              axis=0).reset_index(drop=True)
        self.insert_raw_data(full_hist)

    def process_data(self, df):
        df = da_ut.clean_ticker(df, self.cfg.EXCHANGE_CODE_REPLACE_DICT, 'BB Yellow Key')
        val_cols = ['NMV', 'Quantity', 'PnLDay', 'PnLMTD', 'PnLYTD']
        out = df \
            .pipe(lambda d: d[~d['BB Yellow Key'].isna()]) \
            .pipe(lambda d: d.assign(
            Quantity=d.Quantity.astype(float).fillna(0),
            LocalPrice=da_ut.clean_numeric_series(d['Market Price']),
            FXRate=da_ut.clean_numeric_series(d['Trade/Book FX Rate']))) \
            .pipe(lambda d: d.assign(Price=d.LocalPrice / d.FXRate)) \
            .pipe(lambda d: d.assign(
            Date=pd.to_datetime(d.Date),
            NMV=d.Quantity * d.Price,
            PnLDay=d['$ Daily P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
            PnLMTD=d['$ MTD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0),
            PnLYTD=d['$ YTD P&L'].fillna('0').astype(str).str.replace(r'[\$ ,()]', '', regex=True).astype(float).fillna(0))) \
            .rename(columns={'Book Name': 'Book', 'LE Name': 'LegalEntity'}) \
            .groupby(['LegalEntity', 'Date', 'Book', 'Ticker'])[val_cols].sum() \
            .reset_index()
        return out

    def save_full_processed_data(self, df):
        if not da_ut.exist_file(self.processed_file_path):
            raw_df = pd.read_pickle(self.raw_file_path)
            processed_df = self.process_data(raw_df)
            processed_df.to_pickle(self.processed_file_path)

        processed_df = self.process_data(df)
        org_processed_df = pd.read_pickle(self.processed_file_path) \
            .pipe(lambda d: d[~d.Date.isin(processed_df.Date.unique())])
        processed_df = pd.concat([org_processed_df, processed_df], axis=0).reset_index(drop=True)
        processed_df.to_pickle(self.processed_file_path)

    def load_processed_data(self):
        return pd.read_pickle(self.processed_file_path)

    def load_raw_data(self):
        return pd.read_pickle(self.raw_file_path)

    def daily_update(self, pd_date_tuple=None):
        if pd_date_tuple is None:
            df_raw = self.get_latest_data()
        else:
            df_raw = self.get_raw_data_between_two_dates(pd_date_tuple[0], pd_date_tuple[0])
        df_processed = self.process_data(df_raw)
        self.insert_raw_data(df_raw)
        self.insert_processed_data(df_processed)

    def adhoc_daily_update(self, start, end):
        df_raw = self.get_raw_data_between_two_dates(start, end)
        df_processed = self.process_data(df_raw)
        self.insert_raw_data(df_raw)
        self.insert_processed_data(df_processed)

    def check_weekly_available(self):
        fl_nm = da_ut.find_latest_file(
            self.cfg.ENFUSION_EXCEL_AUTO_SAVE_DIR, self.cfg.ENFUSION_WEEKLY_FILE_KEYWORD, 'csv')
        print(f'Using {fl_nm} to update...')
        match = re.search(r"(\d{2}_\d{2}_\d{4})", fl_nm)
        if not match:
            return False  # No valid date found
        file_date = datetime.strptime(match.group(1), "%m_%d_%Y").date()
        return file_date == date.today()

    def weekly_update(self):
        df_raw = self.get_latest_weekly_data()
        df_processed = self.process_data(df_raw)
        self.insert_raw_data(df_raw)
        self.insert_processed_data(df_processed)

    def load_processed_full_le(self, le=None):
        le = self.cfg.ENFUSION_LEGAL_ENTITY_FHA if le is None else le
        df = self.load_processed_data() \
            .pipe(lambda d: d[d.LegalEntity == le]) \
            .pipe(lambda d: d[d.Date.dt.weekday < 5]) \
            .pipe(lambda d: d.assign(Ticker=d.Ticker.map(self.cfg.TICKER_OVERRIDES).fillna(d.Ticker))) \
            .pipe(lambda d: d[~d.Ticker.isin(self.cfg.TICKER_EXCLUDES)]) \
            .groupby(['Date', 'Book', 'Ticker'])[['NMV', 'Quantity', 'PnLDay', 'PnLMTD', 'PnLYTD']].sum() \
            .reset_index() \
            .sort_values(['Book', 'Ticker', 'Date']) \
            .pipe(lambda d: d.assign(PrevQty=d.groupby(['Book', 'Ticker']).Quantity.shift())) \
            .pipe(lambda d: d[(d.Quantity != 0) | (d.PrevQty != 0) | (d.PrevQty.isna())]) \
            .drop(columns='PrevQty') \
            .pipe(lambda d: d[~(d[['NMV','Quantity','PnLDay','PnLMTD','PnLYTD']].round(0).eq(0).all(1))]) \
            .pipe(lambda d: d.assign(
            NMVStart=d.sort_values(by=['Book', 'Date', 'Ticker']).groupby(['Book', 'Ticker']).NMV.shift(1).fillna(0),
            QuantityStart=d.sort_values(by=['Book', 'Date', 'Ticker']).groupby(['Book', 'Ticker']).Quantity.shift(1).fillna(0)))
        return df

    def load_final_enfusion_data(self, le=None):
        data_hist = self.load_processed_full_le(le=le)
        qty_check = da_ut.quantity_continuity_check(data_hist, 'QuantityStart', 'Quantity')
        if len(qty_check) > 0:
            print('Check Quantity Continuity Issue')
        df = data_hist \
            .rename(columns={'Quantity': 'QuantityEnd', 'NMV': 'NMVEnd'}) \
            .pipe(lambda d: d.assign(GMVStart=d.NMVStart.abs(),
                                     GMVEnd=d.NMVEnd.abs(),
                                     TotalPnL=d.PnLDay)) \
            .drop(['PnLDay', 'PnLMTD', 'PnLYTD'], axis=1)
        return df

    def quick_position_check(self):

        saved_latest = self.load_final_enfusion_data() \
            .pipe(lambda d: d[d.Date == d.Date.max()])[['Book', 'Ticker', 'NMVEnd', 'QuantityEnd']]
        new_latest = self.process_data(self.get_latest_data()) \
            .pipe(lambda d: d[d.LegalEntity == self.cfg.ENFUSION_LEGAL_ENTITY_FHA])[['Book', 'Ticker', 'NMV', 'Quantity']]
        res = new_latest.merge(saved_latest, on=['Book', 'Ticker'], how='outer') \
            .pipe(lambda d: d[((d.NMV != 0) & (~d.NMV.isna())) | ((d.NMVEnd != 0) & (~d.NMVEnd.isna()))]) \
            .rename(columns={'NMV': 'NMVEnd', 'Quantity': 'QuantityEnd', 'QuantityEnd': 'QuantityStart', 'NMVEnd': 'NMVStart'})

        return res

    def quick_pnl_check(self):
        new_latest = self.get_latest_data() \
            .pipe(lambda d: d[d['LE Name'].isin([self.cfg.ENFUSION_LEGAL_ENTITY_FHA])]) \
            .groupby('Book Name')['$ MTD P&L', '$ YTD P&L'].sum() \
            .pipe(lambda d: (d / 1e6).round(1)) \
            .pipe(lambda d: d[(d['$ MTD P&L'].abs() >= 0.5) | (d['$ YTD P&L'].abs() >= 0.5)]) \
            .sort_values(by='$ MTD P&L', ascending=False).reset_index()
        return new_latest


class MsfsData:

    def __init__(self, cfg=da_cfg):
        self.cfg = cfg
        self.cached_pickle_fn = str(
            os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.MSFS_CACHED_DATA_PICKLE_NAME}.pkl'))

    def process_helper(self, df):
        rename_col_dict = {'G/L Period': 'TotalPnL',
                           'Market Value Net Start': 'NMVStart',
                           'Market Value Net': 'NMVEnd',
                           'Quantity Start': 'QuantityStart',
                           'Quantity': 'QuantityEnd'}
        val_cols = list(rename_col_dict.values())
        df = df \
            .pipe(lambda d: d[~d.Ticker.isna()]) \
            .pipe(lambda d: d.assign(Date=pd.to_datetime(d.Date))) \
            .pipe(lambda d: da_ut.clean_ticker(d, self.cfg.EXCHANGE_CODE_REPLACE_DICT)) \
            .rename(columns=rename_col_dict) \
            .drop('Bloomberg Code', axis=1)
        for col in val_cols:
            df[col] = df[col].astype(str).fillna('0').astype(float)
        df = df.groupby(['Date', 'Ticker'])[val_cols].sum().reset_index()
        return df

    def get_latest_full_data(self):

        df = pd.read_csv(
            da_ut.find_latest_file(
                self.cfg.MSFS_EXCEL_AUTO_SAVE_DIR, self.cfg.MSFS_FULL_HIST_FILE_KEYWORD, 'csv')) \
            .drop(['Day'], axis=1) \
            .pipe(lambda d: d.assign(Ticker=d['Bloomberg Code'] + ' EQUITY'))
        df = self.process_helper(df)

        return df

    def save_full_raw_data(self):
        df = self.get_latest_full_data()
        df.to_pickle(os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.MSFS_RAW_DATA_PICKLE_NAME}.pkl'))

    def load_full_raw_data(self):
        df = pd.read_pickle(os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.MSFS_RAW_DATA_PICKLE_NAME}.pkl')) \
            .pipe(lambda d: d.assign(GMVStart=d.NMVStart.abs(),
                                     GMVEnd=d.NMVEnd.abs(),
                                     Book='FHA_Old'))
        return df

    def save_untouched_hist_data(self):
        df = self.load_full_raw_data() \
            .pipe(lambda d: d[d.Date <= '2022-12-31'])
        df.to_pickle(self.cached_pickle_fn)

    def load_untouched_hist_data(self):
        val_cols = ['NMVEnd', 'QuantityEnd', 'NMVStart', 'QuantityStart', 'GMVStart', 'GMVEnd', 'TotalPnL']
        df = pd.read_pickle(self.cached_pickle_fn)
        for c in val_cols:
            df[c] = df[c].astype(float)
        return df

    def daily_update(self):
        latest_fn = da_ut.find_latest_file(
            self.cfg.MSFS_EXCEL_AUTO_SAVE_DIR, self.cfg.MSFS_DAILY_FILE_KEYWORD, 'csv')
        position_date = pd.to_datetime(re.search(r'(\d{8})', latest_fn).group(1))
        df = pd.read_csv(latest_fn) \
                 .rename(columns={'Market Value Net incl Futures % Equity': 'Market Value Net Start',
                                  'BBG Code Short': 'Bloomberg Code'}) \
                 .drop(['Security Description', 'Ticker'], axis=1) \
                 .pipe(lambda d: d.assign(Ticker=d['Bloomberg Code'] + ' EQUITY')) \
                 .pipe(lambda d: d.assign(Date=position_date))
        df = self.process_helper(df)
        hist_df = self.load_full_raw_data() \
            .pipe(lambda d: d[~d.Date.isin(df.Date.unique())])
        new_df = pd.concat([hist_df, df], axis=0).reset_index(drop=True)
        new_df.to_pickle(os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{self.cfg.MSFS_RAW_DATA_PICKLE_NAME}.pkl'))


class InternalAttributes:

    def __init__(self, cfg=da_cfg):
        self.cfg = cfg
        self.enfusion_api = EnfusionData()
        self.msfs_api = MsfsData()

    def load_position_data(self, df=None, add_hist=False, le=None, refresh=True, hardcode_team=da_cfg.BOOK_RESTRUCTURE_DICT, hardcode_ticker=da_cfg.MULTI_LISTING_DICT):

        def _apply_new_book_structure(_df):
            return _df.assign(Book=lambda d: d.Book.replace(hardcode_team)).groupby(['Date', 'Book', 'Ticker'])[
                ['NMVEnd', 'QuantityEnd', 'NMVStart', 'QuantityStart', 'GMVStart', 'GMVEnd',
                 'TotalPnL']].sum().reset_index()

        def _apply_underlying_clean(_df):
            return _df.assign(Ticker=lambda d: d.Ticker.replace(hardcode_ticker)).groupby(['Date', 'Book', 'Ticker'])[
                ['NMVEnd', 'QuantityEnd', 'NMVStart', 'QuantityStart', 'GMVStart', 'GMVEnd',
                 'TotalPnL']].sum().reset_index()

        if df is not None:
            if hardcode_team is not None:
                df = _apply_new_book_structure(df)
            if hardcode_ticker is not None:
                df = _apply_underlying_clean(df)
            df = da_ut.add_internal_attributes(df)
            return df

        le = self.cfg.ENFUSION_LEGAL_ENTITY_FHA if le is None else le
        fn = f'portfolio_data_with_hist_{le}' if add_hist else f'portfolio_data_no_hist_{le}'
        file_dir = str(os.path.join(self.cfg.EXPOSURE_DATA_DIR, f'{fn}.pkl'))
        if not refresh:
            if da_ut.exist_file(file_dir):
                df = pd.read_pickle(file_dir)
                return df

        df = self.enfusion_api.load_final_enfusion_data(le=le)
        if hardcode_team is not None:
            df = _apply_new_book_structure(df)
        if hardcode_ticker is not None:
            df = _apply_underlying_clean(df)
        if add_hist:
            df_hist = self.msfs_api.load_untouched_hist_data()[df.columns]
            df = pd.concat([df_hist, df], axis=0).reset_index(drop=True)

        df = da_ut.add_internal_attributes(df)
        df.to_pickle(file_dir)

        return df


class InternalDataLoader:

    def __init__(self, cfg=da_cfg):
        self.cfg = cfg
        self.enfu_api = EnfusionData()
        self.inta_api = InternalAttributes()

    def load_enfusion_data_processed(self, le=None):
        return self.enfu_api.load_processed_full_le(le=le)

    def load_enfusion_data_raw(self):
        return self.enfu_api.load_raw_data()

    def load_processed_exposure_data(self, df=None, add_hist=False, le=None, refresh=True):
        return self.inta_api.load_position_data(df=df, add_hist=add_hist, le=le, refresh=refresh)
