import data_utils.exposure_data as ex_ut
import data_utils.market_data as mk_ut
import data_config as da_cfg

import importlib as imp
for pkg in [da_cfg, ex_ut, mk_ut]:
    imp.reload(pkg)

import faulthandler
faulthandler.enable()

out1 = mk_ut.BbgExcelLoader().save_bdh_to_structured_pickle(
    name_list=['USGG2YR INDEX', 'USGG10YR INDEX', 'XLP US EQUITY', 'XLY US EQUITY', 'NVDA US EQUITY'],
    field_value='PX_LAST',
    extra_params=['Currency', 'USD'],
    start_date='2020-01-01',
    end_date='2026-01-16',
    temp_excel_path='test_excel_pull_temp.xlsx',
    final_excel_path='test_excel_pull_final.xlsx',
    pickle_path='test_pickle.pkl',
    return_df=True
)

out2 = mk_ut.BbgExcelLoader().save_bds_to_structured_pickle(
    name_list=['AAPL US EQUITY', 'NVDA US EQUITY', 'TSLA US EQUITY'],
    field_value = da_cfg.BBG_EARN_FIELD,
    extra_params = "headers=t",
    n_space=6,
    temp_excel_path='test_excel_pull_temp.xlsx',
    final_excel_path='test_excel_pull_final.xlsx',
    pickle_path='test_pickle.pkl',
    return_df=True)

out3 = mk_ut.BbgExcelLoader().save_bdp_to_structured_pickle(
    name_list=['AAPL US EQUITY', 'NVDA US EQUITY', 'TSLA US EQUITY'],
    field_dict=da_cfg.STATIC_FIELDS,
    temp_excel_path='test_excel_pull_temp.xlsx',
    final_excel_path='test_excel_pull_final.xlsx',
    pickle_path='test_pickle.pkl',
    return_df=True)


out4 = mk_ut.BbgExcelLoader().save_bdh_to_structured_pickle(
    name_list=['AAPL US EQUITY', 'SPX INDEX', '6857 JP EQUITY'],
    field_value='BEST_TARGET_PRICE',
    extra_params=['BEST_FPERIOD_OVERRIDE', '1GBF'],
    start_date='2020-01-01',
    end_date='2026-01-16',
    temp_excel_path='test_excel_pull_temp.xlsx',
    final_excel_path='test_excel_pull_final.xlsx',
    pickle_path='test_pickle.pkl',
    return_df=True)

out5 = mk_ut.BbgExcelLoader().save_bql_consensus_data_pickle(
    name_list=['AAPL US EQUITY', 'SPX INDEX', '6857 JP EQUITY'],
    temp_excel_path='test_excel_pull_temp.xlsx',
    final_excel_path='test_excel_pull_final.xlsx',
    pickle_path='test_pickle.pkl',
    return_df=True)