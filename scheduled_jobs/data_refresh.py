import data_utils.exposure_data as ex_ut
import data_utils.market_data as mk_ut
import data_config as da_cfg

import importlib as imp
for pkg in [da_cfg, ex_ut, mk_ut]:
    imp.reload(pkg)

import time
import faulthandler
faulthandler.enable()


df = ex_ut.EnfusionData().get_latest_data()

df = df[(df.Quantity != 0) & (~df['BB Yellow Key'].isna()) & (df['LE Name'] != 'FAI AI') & (~df['Book Name'].str.startswith('FHA_01_JP'))]

tickers = df['BB Yellow Key'].str.upper().unique()

latest_date = df.Date.max().strftime('%Y-%m-%d')


# Update Price Volume Data

def retry_run(func, max_retries=3, sleep_sec=2, *args, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(sleep_sec)


for bdh_field, pkl_name in da_cfg.BDH_DATA_PKL_NAMES.items():

    retry_run(
        mk_ut.BbgExcelLoader().save_bdh_to_structured_pickle,
        max_retries=3,
        sleep_sec=2,
        name_list=tickers,
        field_value=bdh_field,
        extra_params=['Currency', 'USD'] if pkl_name in ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        else ['BEST_FPERIOD_OVERRIDE', '1GBF'] if pkl_name in ['TP', 'PE'] else None,
        start_date='2015-01-01',
        end_date=latest_date,
        temp_excel_path=f'{pkl_name}_temp.xlsx',
        final_excel_path=f'{pkl_name}_final.xlsx',
        pickle_path=f'{pkl_name}.pkl'
    )



import faulthandler
faulthandler.enable()
pkl_name = 'OPEN'
name_list = tickers
field_value = 'PX_OPEN'
extra_params = ['Currency', 'USD']
start_date = '2015-01-01'
end_date = '2026-01-15'
temp_excel_path=f'{pkl_name}_temp.xlsx'
final_excel_path = f'{pkl_name}_final.xlsx'
pickle_path = f'{pkl_name}.pkl'
return_df=False
batch_size=150
project_path=None
wait_seconds=max(int(len(tickers) * 0.1), 30)
input_filename = temp_excel_path
output_filename = final_excel_path


self = BbgExcelLoader()
if project_path is None:
    project_path = self.cfg.BBG_DEFAULT_PICKLE_DIR

list_of_tickers = chunk(name_list, batch_size)
out_list = []

for idx, tickers in enumerate(list_of_tickers):
    print('insert formulas')