import data_utils.exposure_data as ex_ut
import data_utils.market_data as mk_ut
import data_config as da_cfg
from logging_utils.email_utils import notify_on_failure

import importlib as imp
for pkg in [da_cfg, ex_ut, mk_ut]:
    imp.reload(pkg)

import faulthandler
faulthandler.enable()
import pandas as pd
from datetime import datetime

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def refresh_exposure_data():
    ex_ut.EnfusionData().daily_update()

    df1 = ex_ut.InternalAttributes().load_position_data(add_hist=True, refresh=True, le=da_cfg.ENFUSION_LEGAL_ENTITY_FHA)
    df2 = ex_ut.InternalAttributes().load_position_data(add_hist=False, refresh=True, le=da_cfg.ENFUSION_LEGAL_ENTITY_FAI)
    df3 = ex_ut.InternalAttributes().load_position_data(add_hist=False, refresh=True, le=da_cfg.ENFUSION_LEGAL_ENTITY_FHA)


def refresh_market_data():
    def _refresh_stock_time_series(ticker_list):
        for bdh_field, pkl_name in da_cfg.BDH_DATA_PKL_NAMES.items():
            logging.info(f'Refreshing Stock {bdh_field} data')
            mk_ut.BbgExcelLoader().save_bdh_to_structured_pickle(
                name_list = ticker_list,
                field_value = bdh_field,
                extra_params = ['BEST_FPERIOD_OVERRIDE', '1GBF'] if pkl_name in ['TP', 'PE'] else None,
                start_date = da_cfg.START_DATE,
                end_date = latest_date,
                temp_excel_path = f'{pkl_name}_temp.xlsx',
                final_excel_path = f'{pkl_name}_final.xlsx',
                pickle_path = f'{pkl_name}.pkl')

    def _refresh_stock_static_information(ticker_list):
        logging.info(f'Refreshing Stock Static data')
        static_df = mk_ut.BbgExcelLoader().save_bdp_to_structured_pickle(
            name_list=ticker_list,
            field_dict=da_cfg.STATIC_FIELDS,
            temp_excel_path=f'{da_cfg.STATIC_FN}_temp.xlsx',
            final_excel_path=f'{da_cfg.STATIC_FN}_final.xlsx',
            pickle_path=f'{da_cfg.STATIC_FN}.pkl',
            return_df=True)
        return (static_df.REL_INDEX + ' INDEX').unique().tolist()

    def _refresh_index_time_series_data(ticker_list):
        logging.info(f'Refreshing Index data')
        for extra_params, pkl_nm in zip([['Currency', 'USD'], None], [da_cfg.IDX_USD_PICKLE_FN, da_cfg.IDX_PICKLE_FN]):
            mk_ut.BbgExcelLoader().save_bdh_to_structured_pickle(
                name_list=list(set(ticker_list).union(set(da_cfg.MKT_INDEX_MAPPING.values())).union(
                    set(da_cfg.MKT_INDEX_MAPPING_HEDGE.values()))),
                field_value='PX_LAST',
                extra_params=extra_params,
                start_date=da_cfg.START_DATE,
                end_date=latest_date,
                temp_excel_path=f'{pkl_nm}_temp.xlsx',
                final_excel_path=f'{pkl_nm}_final.xlsx',
                pickle_path=f'{pkl_nm}.pkl')

    def _refresh_earning_info_data(ticker_list):
        logging.info(f'Refreshing Earnings data')
        today = datetime.today()
        # last business day of this month
        last_bday = (pd.Timestamp(today) + pd.offsets.MonthEnd(0))
        if last_bday.weekday() >= 5:  # Sat(5) or Sun(6)
            last_bday -= pd.offsets.BDay(1)
        if today.day == last_bday.date().day:
            mk_ut.BbgExcelLoader().save_bds_to_structured_pickle(
                name_list=ticker_list,
                field_value=da_cfg.BBG_EARN_FIELD,
                extra_params="headers=t",
                n_space=6,
                temp_excel_path=f'{da_cfg.EARN_DATA_PKL_FN}_temp.xlsx',
                final_excel_path=f'{da_cfg.EARN_DATA_PKL_FN}_final.xlsx',
                pickle_path=f'{da_cfg.EARN_DATA_PKL_FN}.pkl')

    df = ex_ut.InternalAttributes().load_position_data(add_hist=False, refresh=False, le=da_cfg.ENFUSION_LEGAL_ENTITY_FHA)
    tickers = df.Ticker.unique().tolist()
    latest_date = df.Date.max().strftime('%Y-%m-%d')
    _refresh_stock_time_series(tickers)
    idx_list = _refresh_stock_static_information(tickers)
    _refresh_index_time_series_data(idx_list)
    _refresh_earning_info_data(idx_list)



@notify_on_failure(to="jingbin@fengheasia.com",
                   capture_level=logging.INFO,  # or logging.DEBUG for deeper traces
                   include_logs=True,
                   max_log_chars=60_000)
def daily_exposure_market_data_refresh():
    logging.info('Starting data refresh job...')

    logging.info('Refreshing Exposure data...')
    refresh_exposure_data()
    logging.info('Refreshing Market data...')
    refresh_market_data()
    logging.info('Saving Data to S3...')
    save_refreshed_data()

daily_exposure_market_data_refresh()