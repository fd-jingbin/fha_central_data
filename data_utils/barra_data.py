import sys
sys.path.append(r'/')
sys.path.append(r'C:\Python\fha-research')

import re
import os
import paramiko
import pandas as pd
from datetime import datetime as ddtt
from collections import defaultdict
import zipfile
from pathlib import Path
import numpy as np

import barra_config as ba_cfg
import data_utils.utils as da_ut
import cb_utilities.time_utils as tm_ut
import data_utils.figi_utils as fg_ut

import logging
logger = logging.getLogger(__name__)

class BarraDataLoader:

    def __init__(self, cfg=ba_cfg):
        self.cfg = ba_cfg
        self.daily_raw_dir = str(os.path.join(self.cfg.BARRA_RAW_DATA_DIR, 'daily'))
        self.hist_raw_dir = str(os.path.join(self.cfg.BARRA_RAW_DATA_DIR, 'hist'))
        self.processed_dir = self.cfg.BARRA_PROCESSED_DATA_DIR

    def build_sftp_connection(self):
        # Connect and download
        transport = paramiko.Transport((self.cfg.HOST, self.cfg.PORT))
        transport.connect(username=self.cfg.USERNAME, password=self.cfg.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return sftp

    def generate_hist_file_list(self):
        raise Exception('Not Implemented!')

    def copy_hist_raw_files(self):
        da_ut.exist_create_folder(self.hist_raw_dir)
        sftp = self.build_sftp_connection()

        pattern1 = re.compile(r"GMD_APACEFMTRD_100_D_(20[2-5]\d)(_\d+)?\.zip$")
        pattern2 = re.compile(r"GMD_APACEFMTR_100_D_(20[2-5]\d)(_\d+)?\.zip$")

        # List all files in the remote directory
        for filename in sftp.listdir(self.cfg.SERVER_HIST_PATH_PREFIX):
            print(filename)
            if pattern1.match(filename) or pattern2.match(filename):
                remote_path = os.path.join(self.cfg.SERVER_HIST_PATH_PREFIX, filename)
                local_path = os.path.join(self.hist_raw_dir, filename)
                print(f"Downloading: {filename}")
                sftp.get(remote_path, local_path)

    def copy_daily_raw_files(self, dt_lists=None):
        dt_lists = [da_ut.previous_business_day(ddtt.today()).strftime('%y%m%d')] if dt_lists is None else dt_lists

        local_path_prefix = str(os.path.join(self.cfg.BARRA_RAW_DATA_DIR, 'daily'))
        sftp = self.build_sftp_connection()
        remote_path = ''

        missing_data = []
        for dt in dt_lists:
            for f in self.cfg.DAILY_FILE_LIST:
                try:
                    remote_path = self.cfg.SERVER_DAILY_PATH_PREFIX + f.replace('DATEPLACEHOLDER', dt)
                    local_path = local_path_prefix + rf"\{f}".replace('DATEPLACEHOLDER', dt)
                    print(f'Copying {remote_path}')
                    sftp.get(remote_path, local_path)
                except Exception as e:
                    print(e, remote_path)
                    missing_data.append(remote_path)

        if len(missing_data) > 0:
            raise Exception(f'The following files are missing from copy: {", ".join(missing_data)}')

    @staticmethod
    def read_flat_file(file, columns, delimiter):
        """Reads a structured flat file and converts it to a Pandas DataFrame."""
        try:
            df = pd.read_csv(file, sep=delimiter, comment="!", names=columns, dtype=str)

            # Remove rows containing "[End of File]"
            if df.iloc[-1, 0] == "[End of File]":
                df = df.iloc[:-1]
            return df

        except Exception as e:
            print(f"  Error reading file: {e}")
            return None

    def generate_hist_year_file_dict_by_year(self, pattern):
        regex = re.compile(pattern)
        files_by_year = defaultdict(list)

        for file_name in os.listdir(self.hist_raw_dir):
            match = regex.match(file_name)
            if match:
                year = int(match.group(1))
                files_by_year[year].append(file_name)

        # Convert defaultdict to normal dict
        return dict(files_by_year)

    def process_full_hist_file(self, latest_year=False):

        for data_dict in self.cfg.HIST_DATA_STRUCTURE_DICT.values():
            patterns = data_dict['pattern']
            file_types = data_dict["file_types"]
            columns = data_dict["columns"]
            delimiter = data_dict["delimiter"]

            files_all = self.generate_hist_year_file_dict_by_year(patterns)
            if latest_year:
                files_all = {max(files_all.keys()): files_all[max(files_all.keys())]}
            for file_type, file_patterns in file_types.items():
                print(file_type, file_patterns)
                for year, f_list in files_all.items():
                    data_res_list = []

                    for zip_fn in f_list:
                        print(zip_fn)
                        with zipfile.ZipFile(os.path.join(self.hist_raw_dir, zip_fn), 'r') as zip_ref:
                            for fn in zip_ref.namelist():
                                match = re.match(file_patterns, fn)
                                if match:
                                    print(f"  Extracting {file_type} from {fn}")

                                    with zip_ref.open(fn) as file:
                                        df = self.read_flat_file(file, columns[file_type], delimiter)

                                        if df is not None:
                                            data_res_list.append(df)

                    df = pd.concat(data_res_list, ignore_index=True)
                    df.to_parquet(os.path.join(self.processed_dir, f"{file_type}_{year}.parquet"))
                    print(f"Saved: {file_type}_{year}.parquet")

    def find_latest_zip(self, pattern):
        folder_path = Path(self.daily_raw_dir)
        zip_files = [f for f in folder_path.iterdir() if f.is_file() and re.match(pattern, f.name)]
        if not zip_files:
            return None
        latest_zip = max(zip_files, key=lambda f: f.stat().st_mtime)  # by modification time
        return str(latest_zip)

    @staticmethod
    def extract_distinct_years(dates):
        # Convert each int to string and take the first 4 characters (year)
        years = {str(date)[:4] for date in dates}
        return sorted(years)

    def save_latest_to_pqt(self, data_type, df_new):

        years = self.extract_distinct_years(df_new.DataDate.unique().tolist())
        for yy in years:
            df_yy = df_new[df_new['DataDate'].astype(str).str.startswith(yy)]
            pqt_file = Path(os.path.join(self.processed_dir, f"{data_type}_{yy}.parquet"))

            if pqt_file.exists():
                df_old = pd.read_parquet(pqt_file)
                df_old['DataDate'] = df_old['DataDate'].astype(int)
                new_dates = df_yy.DataDate.astype(int).unique().tolist()
                # Filter out rows in df_yy that already exist in df_old
                df_old = df_old[~df_old.DataDate.isin(new_dates)]
                merged = pd.concat([df_old, df_yy], ignore_index=True)
                added_rows = len(merged) - len(df_old)
            else:
                merged = df_yy.copy()
                added_rows = len(df_yy)
            if added_rows > 0:
                merged.to_parquet(pqt_file)

            print(f"{data_type}_{yy}.parquet updated: {added_rows} new rows added")

    def update_latest_mapping(self):
        print('Start updating latest figi mapping...')
        zip_path = Path(self.find_latest_zip(r"GMD_ASIA_FIGI_ID.*\.zip"))
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            print(f'Processing latest figi mapping...')
            for file_name in zipf.namelist():
                match = re.fullmatch("ASIA_FIGI_Asset_ID\\.(\\d{8})", Path(file_name).name)
                if match:
                    df = self.read_flat_file(zipf.open(file_name), columns=['Barrid', 'AssetIDType', 'AssetID', 'StartDate', 'EndDate'], delimiter='|')
                    df.to_parquet(os.path.join(self.processed_dir, 'figi_mapping.parquet'))
                    print('Updated latest figi mapping')

    def update_daily_volume_data(self):

        data_type = "daily_volume"
        pattern = r"APACEFMTR_Market_Data\.(\d{8})"

        try:
            df_list = []
            for zip_path in Path(self.daily_raw_dir).glob(r'GMD_APACEFMTR_Market_Data*.zip'):
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    print(f'Processing {data_type} {zip_path}')
                    for file_name in zipf.namelist():
                        match = re.fullmatch(pattern, Path(file_name).name)
                        if match:
                            date_str = match.group(1)
                            df = self.read_flat_file(zipf.open(file_name), self.cfg.HIST_ASSET_DATA_STRUCTURE['columns']['daily_volume'], '|')
                            df["DataDate"] = df["DataDate"].astype(int)
                            df_list.append(df)
            print(len(df_list))
            self.save_latest_to_pqt(data_type, pd.concat(df_list, ignore_index=True))

        except Exception as e:
            print(f"Error processing latest {data_type}: {e}")

    def update_latest_specific_return_risk_price_fx_factor_return_data(self):
        for data_type, pattern in self.cfg.LATEST_DATA_DICT["file_types"].items():
            if data_type != 'factor_return':
                try:
                    df_list = []
                    for zip_path in Path(self.daily_raw_dir).glob(r"GMD_APACEFMTRD_100*.zip"):
                        with zipfile.ZipFile(zip_path, 'r') as zipf:
                            logging.info(f'Processing {data_type} {zip_path}')
                            for file_name in zipf.namelist():
                                match = re.fullmatch(pattern, Path(file_name).name)
                                if match:
                                    df = self.read_flat_file(zipf.open(file_name),
                                                             self.cfg.LATEST_DATA_DICT["columns"][data_type],
                                                             self.cfg.LATEST_DATA_DICT["delimiter"])
                                    df["DataDate"] = df["DataDate"].astype(int)
                                    df_list.append(df)
                    logging.info(len(df_list))
                    self.save_latest_to_pqt(data_type, pd.concat(df_list, ignore_index=True))
                except Exception as e:
                    logging.info(f"Error processing latest {data_type}: {e}")
            else:
                zip_path = Path(self.find_latest_zip(r"GMD_APACEFMTRD_100.*\.zip"))
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    logging.info(f'Processing {data_type} {zip_path}')
                    for file_name in zipf.namelist():
                        match = re.fullmatch(pattern, Path(file_name).name)
                        if match:
                            df = self.read_flat_file(zipf.open(file_name),
                                                     self.cfg.LATEST_DATA_DICT["columns"][data_type],
                                                     self.cfg.LATEST_DATA_DICT["delimiter"])
                    df.to_parquet(os.path.join(self.processed_dir, 'factor_return.parquet'))

    def load_figi_mapping(self):
        return pd.read_parquet(os.path.join(self.processed_dir, 'figi_mapping.parquet'))

    def read_country_calendar(self):
        files = list(Path(self.daily_raw_dir).glob(f"APACEFMTR_CountryHolidays*"))
        file_name = max(files, key=lambda f: f.stat().st_mtime)
        cal = self.read_flat_file(file_name, columns=['ISOCountry', 'HolidayDate'], delimiter='|')
        return cal

    def load_factor_return(self):
        return pd.read_parquet(os.path.join(self.processed_dir, 'factor_return.parquet')).pipe(lambda d: d.assign(DlyReturn=d.DlyReturn.astype(float)))

    @staticmethod
    def get_years_required(start, end):
        return [str(x) for x in range(int(start/1e4), int(end / 1e4) + 1)]

    def load_fx_rate_data(self, start_date_id, end_date_id):
        years = self.get_years_required(start_date_id, end_date_id)
        full_data = []
        for year in years:
            try:
                df = pd.read_parquet(os.path.join(self.processed_dir, f'fx_rate_{year}.parquet'))
            except FileNotFoundError:
                print(f'FX data not found for {year}')
                df = pd.DataFrame()
            full_data.append(df)
        full_data = pd.concat(full_data, ignore_index=True) \
            .pipe(lambda d: d.assign(DataDate=d.DataDate.astype(int),
                                     USDxrate=d.USDxrate.astype(float),
                                     RFRate=d['RFRate%'].astype(float))) \
            .drop('RFRate%', axis=1)
        return full_data

    def load_asset_price_data(self, start_date_id, end_date_id, barrid_list=None):
        years = self.get_years_required(start_date_id, end_date_id)
        full_data = []
        kept_cols = ['Barrid', 'Price', 'DlyReturn%', 'Currency', 'DataDate']
        for year in years:
            print(f'Reading price data for {year}')
            try:
                df = pd.read_parquet(os.path.join(self.processed_dir, f'daily_price_{year}.parquet'))[kept_cols]
                if barrid_list is not None:
                    df = df[df.Barrid.isin(barrid_list)]
            except FileNotFoundError:
                print(f'Price data not found for {year}')
                df = pd.DataFrame()
            full_data.append(df)
        full_data = pd.concat(full_data, ignore_index=True) \
             .pipe(lambda d: d.assign(DataDate=d.DataDate.astype(int),
                                      Price=d.Price.astype(float),
                                      DlyReturn=d['DlyReturn%'].astype(float)))
        return full_data

    def load_asset_volume_data(self, start_date_id, end_date_id, barrid_list=None):
        years = self.get_years_required(start_date_id, end_date_id)
        full_data = []
        kept_cols = ['Barrid', 'BidAskSpread', 'DailyVolume', 'ADTV_90', 'IssuerMarketCap', 'DataDate']
        for year in years:
            print(f'Reading volume data for {year}')
            try:
                df = pd.read_parquet(os.path.join(self.processed_dir, f'daily_volume_{year}.parquet'))[kept_cols]
                if barrid_list is not None:
                    df = df[df.Barrid.isin(barrid_list)]
            except FileNotFoundError:
                print(f'Volume data not found for {year}')
                df = pd.DataFrame()
            full_data.append(df)
        full_data = pd.concat(full_data, ignore_index=True) \
              .pipe(lambda d: d.assign(DataDate=d.DataDate.astype(int),
                                       BidAskSpread=d.BidAskSpread.astype(float),
                                       DailyVolume=d.DailyVolume.astype(float),
                                       IssuerMarketCap=d.IssuerMarketCap.astype(float),
                                       ADTV_90=d.ADTV_90.astype(float)))
        return full_data

    def load_asset_specific_return(self, start_date_id, end_date_id, barrid_list=None):
        years = self.get_years_required(start_date_id, end_date_id)
        full_data = []
        for year in years:
            print(f'Reading specific return data for {year}')
            try:
                df = pd.read_parquet(os.path.join(self.processed_dir, f'specific_return_{year}.parquet'))
                if barrid_list is not None:
                    df = df[df.Barrid.isin(barrid_list)]
            except FileNotFoundError:
                print(f'Alpha return data not found for {year}')
                df = pd.DataFrame()
            full_data.append(df)
        full_data = pd.concat(full_data, ignore_index=True) \
               .pipe(lambda d: d.assign(DataDate=d.DataDate.astype(int),
                                        SpecificReturn=d.SpecificReturn.astype(float)))
        return full_data

    def load_asset_risk_data(self, start_date_id, end_date_id, barrid_list=None):
        years = self.get_years_required(start_date_id, end_date_id)
        full_data = []
        for year in years:
            print(f'Read asset risk data for {year}')
            try:
                df = pd.read_parquet(os.path.join(self.processed_dir, f'asset_data_{year}.parquet'))
                if barrid_list is not None:
                    df = df[df.Barrid.isin(barrid_list)]
            except FileNotFoundError:
                print(f'Asset risk data not found for {year}')
                df = pd.DataFrame()
            full_data.append(df)
        full_data = pd.concat(full_data, ignore_index=True) \
              .pipe(lambda d: d.assign(DataDate=d.DataDate.astype(int),
                        Yield=d['Yield%'].astype(float),
                        TotalRisk=d['TotalRisk%'].astype(float),
                        SpecRisk=d['SpecRisk%'].astype(float),
                        HistBeta=d.HistBeta.astype(float),
                        PredBeta=d.PredBeta.astype(float))) \
              .drop(["Yield%", "TotalRisk%", "SpecRisk%"], axis=1)
        return full_data

    def load_loading_data(self, start_date_id, end_date_id, barrid_list=None, factor_list=None):
        years = self.get_years_required(start_date_id, end_date_id)
        full_data = []
        for year in years:
            print(f'Read factor loading data for {year}')
            try:
                df = pd.read_parquet(os.path.join(self.processed_dir, f'asset_exposure_{year}.parquet'))
                if barrid_list is not None:
                    df = df[df.Barrid.isin(barrid_list)]
                if factor_list is not None:
                    df = df[df.Factor.isin(factor_list)]
            except FileNotFoundError:
                print(f'Asset factor loading not found for {year}')
                df = pd.DataFrame()
            full_data.append(df)
        full_data = pd.concat(full_data, ignore_index=True) \
              .pipe(lambda d: d.assign(DataDate=d.DataDate.astype(int),
                        Exposure=d.Exposure.astype(float)))
        return full_data

    def get_max_hist_date(self):
        today = int(da_ut.previous_business_day(ddtt.today()).strftime('%Y%m%d'))
        return self.load_fx_rate_data(today, today).DataDate.max()

    @staticmethod
    def remove_raw_zip_files(file_path):
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

    def daily_update(self):
        logging.info('Start barra data update...')
        start = da_ut.next_business_day(str(self.get_max_hist_date())).strftime('%Y%m%d')
        end = da_ut.previous_business_day(ddtt.today()).strftime('%Y%m%d')
        dt_lists = [dd.strftime('%y%m%d') for dd in da_ut.business_days_between(start, end)]

        logging.info('Copy raw files...')
        self.copy_daily_raw_files(dt_lists)

        logging.info('Update latest mapping...')
        self.update_latest_mapping()

        logging.info('Update daily volume data...')
        self.update_daily_volume_data()

        logging.info('Start specific return, risks, price, fx, factor return...')
        self.update_latest_specific_return_risk_price_fx_factor_return_data()

        logging.info('Remove copied files...')
        for fn in da_ut.find_avail_files(self.daily_raw_dir, 'GMD_APACEFMTR', 'zip'):
            self.remove_raw_zip_files(fn)

    def load_factor_mapping(self):
        out = pd.DataFrame(self.cfg.FACTOR_MAPPING) \
            .pipe(lambda d: d.assign(Region=np.where(d.Factor.str.contains('_JP'), 'Japan', 'exJapan')))
        return out

    def get_style_factors(self):
        return self.load_factor_mapping().pipe(lambda d: d[d.FactorGroup == '1-Risk Indices'])

    def load_ticker_to_barrid_mapping(self):
        return fg_ut.get_figi_mapping() \
            .assign(Ticker=lambda d: d.securityDescription.str.upper() + ' ' + d.exchCode.str.upper() + ' EQUITY',
                    FIGI=lambda d: d.figi)[['FIGI', 'Ticker']].dropna() \
            .merge(self.load_figi_mapping().rename(columns={'AssetID': 'FIGI'})[['Barrid', 'FIGI']], on='FIGI',
                   how='inner') \
            .drop_duplicates(subset='Ticker')[['Ticker', 'Barrid']]

    def convert_all_pkl_to_parquet(self, overwrite=False):
        input_folder = Path(self.processed_dir)
        if not input_folder.is_dir():
            raise NotADirectoryError(f"{input_folder} 不是有效文件夹")

        pkl_files = list(input_folder.glob("*.pkl"))
        if not pkl_files:
            print("没有找到 .pkl 文件")
            return

        for pkl_path in pkl_files:
            parquet_path = pkl_path.with_suffix(".parquet")
            if parquet_path.exists() and not overwrite:
                print(f"跳过已存在: {parquet_path}")
                continue

            try:
                # 读取 gzip 压缩的 pickle
                df = pd.read_parquet(pkl_path)

                # 写 parquet
                df.to_parquet(parquet_path, index=False)
                print(f"转换完成: {pkl_path} -> {parquet_path}")
            except Exception as e:
                print(f"转换失败: {pkl_path} | 错误: {e}")