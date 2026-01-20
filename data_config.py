import os


# region Enfusion Config

ENFUSION_EXCEL_AUTO_SAVE_DIR = r'C:\Python\data\excel\enfusion_exposure_report'
ENFUSION_LEGAL_ENTITY_FHA = 'FengHe Asia Fund Ltd'
ENFUSION_LEGAL_ENTITY_FAI = 'FAI AI'
ENFUSION_20230101_20250227_FILE_NAME = 'Daily Position Report for JingBin.xlsx'
ENFUSION_20250201_20250228_FILE_NAME = 'Exposure report by Analyst for Jing Bing - February.xlsx'
ENFUSION_20250301_20250331_FILE_NAME = 'Exposure report by Analyst for Jing Bin- March.xlsx'
ENFUSION_20250401_20250430_FILE_NAME = 'Exposure report by Analyst for Jing Bing - April.xlsx'
ENFUSION_WEEKLY_FILE_KEYWORD = 'Exposure report by Analyst for Jing Bin - Weekly'
ENFUSION_DAILY_FILE_KEYWORD = 'Exposure report by Analyst for Jing Bin v2'
ENFUSION_RAW_DATA_PICKLE_NAME = 'enfusion_raw'
ENFUSION_PROCESSED_DATA_PICKLE_NAME = 'enfusion_processed'
ENFUSION_RAW_DATA_COLS = [
    'Date', 'LE Name', 'Book Name', 'Description', 'Account', 'BB Yellow Key', 'RIC', 'ISIN', 'SEDOL', 'Quantity',
    '$ Net Avg Cost', 'Native Overall Cost Per Lot', '$ Overall Cost', '$ Daily P&L', '$ MTD P&L', '$ YTD P&L',
    '$ ITD P&L', '$ Underlying Market Price', '$ NMV', 'Market Price', 'Native NMV', 'Trade/Book FX Rate',
    'Expiry Date', 'Active']

EXCHANGE_CODE_REPLACE_DICT = {
    ' C1 ': ' CH ',
    ' C2 ': ' CH ',
    ' H1 ': ' HK ',
    ' H2 ': ' HK ',
    ' JT ': ' JP ',
    ' GY ': ' GR '}

ENFUSION_DATE_COL_DICT = {
    'Position Scenario Date Adjusted': 'Date',
    'Position Scenario Date By TimeZone': 'Date'
}

# endregion


# region Book Structure Related

BOOK_RESTRUCTURE_DICT = {'SS': 'SSS', 'FHCIPO': 'FHC', 'QXIPO': 'QX', 'GXIPO': 'GX'}
MULTI_LISTING_DICT = {'BABA US EQUITY': '9988 HK EQUITY',
                      '688331 CH EQUITY': '9995 HK EQUITY',
                      '6031 HK EQUITY': '600031 CH EQUITY',
                      '2050 HK EQUITY': '002050 CH EQUITY',
                      'ASML NA EQUITY': 'ASML US EQUITY',
                      'LOGN SW EQUITY': 'LOGI US EQUITY'}
# endregion


# region MSFS Config

MSFS_EXCEL_AUTO_SAVE_DIR = r'C:\Python\data\excel\msfs_exposure_report'
MSFS_FULL_HIST_FILE_KEYWORD = 'Summary'
MSFS_DAILY_FILE_KEYWORD = 'Daily Position Report'
MSFS_RAW_DATA_PICKLE_NAME = 'msfs_raw'
MSFS_CACHED_DATA_PICKLE_NAME = 'msfs_before_2023'

# endregion


# region Overrides

TICKER_EXCLUDES = ['MSLDLQI LX EQUITY', 'GSUTLRI ID EQUITY']
TICKER_OVERRIDES = {
    'GOLD US EQUITY': 'B US EQUITY',
    '0124501F KS EQUITY': '012450 KS EQUITY'
}
# endregion


# region File Directions
DATA_MASTER_DIR = r'C:\Python\data'
EXPOSURE_DATA_DIR = r'C:\Python\data\exposure_data'
TABLEAU_DATA_DIR = r'C:\Python\data\tableau_data'
# endregion



# region BBG API Config
START_DATE = '2015-01-01'

OPENFIGI_API_KEY = os.environ.get("OPENFIGI_API_KEY", "b1c361a4-9a52-4d7b-ba13-d2c014e43399")
OPENFIGI_BASE_URL = "https://api.openfigi.com/v3/mapping"

BBG_EXCEL_TEMP_FILE_DIR = r'C:\Python\data\market_data\bbg_excel_temp'
BBG_DEFAULT_PICKLE_DIR = r'C:\Python\data\market_data'
FIGI_MAPPING_PKL_FN = 'figi_mapping.pkl'

BDH_DATA_PKL_NAMES = {
    'BETA_RAW_OVERRIDABLE': 'BETA',
    'PX_LAST': 'CLOSE',
    'PX_VOLUME': 'VOLUME',
    'PX_OPEN': 'OPEN',
    # 'PX_HIGH': 'HIGH',
    # 'PX_LOW': 'LOW',
    # 'BEST_TARGET_PRICE': 'TP',
    # 'BEST_PE_RATIO': 'PE',
}

BBG_EARN_FIELD = 'EARN_ANN_DT_TIME_HIST_WITH_EPS'
EARN_DATA_PKL_FN = 'EARN'

STATIC_FN = 'static_mapping'

STATIC_FIELDS = {
    'GicsSector': 'GICS_SECTOR_NAME',
    'GicsIndustryGroup': 'GICS_INDUSTRY_GROUP_NAME',
    'GicsIndustry': 'GICS_INDUSTRY_NAME',
    'BicsSector': 'BICS_LEVEL_1_SECTOR_NAME',
    'BicsIndustryGroup': 'BICS_LEVEL_2_INDUSTRY_GROUP_NAME',
    'BicsIndustry': 'BICS_LEVEL_3_INDUSTRY_NAME',
    'BicsSubIndustry': 'BICS_LEVEL_4_SUB_INDUSTRY_NAME',
    'CountryOfDomicile': 'CNTRY_OF_DOMICILE',
    'CountryOfRisk': 'CNTRY_OF_RISK',
    'SecurityDescription': 'NAME',
    'SecurityType': 'SECURITY_TYP',
    'NextEarnDate': 'EXPECTED_REPORT_DT',
    'FIGI': "COMPOSITE_ID_BB_GLOBAL",
    'ISIN': 'ID_ISIN',
    'REL_INDEX': 'REL_INDEX'
}

STOCK_IMPLIED_CALENDAR_PICKLE_FN = 'CALENDAR'

IDX_USD_PICKLE_FN = 'INDEX_USD'
IDX_PICKLE_FN = 'INDEX'

MKT_INDEX_MAPPING = {
    'CN': 'SHSZ300 INDEX',
    'HK': 'HSI INDEX',
    'GB': 'UKX INDEX',
    'US': 'SPX INDEX',
    'JP': 'NKY INDEX',
    'CL': 'SPX INDEX', # CHILE
    'NL': 'SXXP INDEX',
    'IL': 'SPX INDEX',
    'AU': 'AS51 INDEX',
    'DE': 'DAX INDEX',
    'TW': 'TWSE INDEX',
    'CH': 'SXXP INDEX',
    'KR': 'KOSPI INDEX',
    'FR': 'CAC INDEX',
    'SG': 'ASEAN40 INDEX',
    'CA': 'SPX INDEX',
    'IT': 'SXXP INDEX',
    'ES': 'SXXP INDEX',
    'LU': 'SXXP INDEX',
    'VN': 'VNINDEX INDEX',
    'ID': 'ASEAN40 INDEX',
    'TH': 'ASEAN40 INDEX',
    'PH': 'ASEAN40 INDEX',
    'MO': 'SHSZ300 INDEX',
    'JE': 'SPX INDEX'
}

MKT_INDEX_MAPPING_HEDGE = {
    'CN': 'XIN9I INDEX',
    'HK': 'XIN9I INDEX',
    'GB': 'SX5E INDEX',
    'US': 'SPX INDEX',
    'JP': 'NKY INDEX',
    'CL': 'SPX INDEX', # CHILE
    'NL': 'SX5E INDEX',
    'IL': 'SX5E INDEX',
    'AU': 'SPX INDEX',
    'DE': 'SX5E INDEX',
    'TW': 'SPX INDEX',
    'CH': 'SX5E INDEX',
    'KR': 'KOSPI2 INDEX',
    'FR': 'SX5E INDEX',
    'SG': 'ASEAN40 INDEX',
    'CA': 'SPX INDEX',
    'IT': 'SX5E INDEX',
    'ES': 'SX5E INDEX',
    'LU': 'SX5E INDEX',
    'VN': 'ASEAN40 INDEX',
    'ID': 'ASEAN40 INDEX',
    'TH': 'ASEAN40 INDEX',
    'PH': 'ASEAN40 INDEX',
    'MO': 'XIN9I INDEX',
    'JE': 'SPX INDEX'
}

# endregion