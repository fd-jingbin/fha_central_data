import pandas as pd
import numpy as np
from datetime import datetime, time
import datetime as ddtt
import pytz
import os
import re
from pathlib import Path
import glob


def quantity_continuity_check(df, qty_start_col='QuantityStart', qty_end_col='QuantityEnd'):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])

    # Previous day info
    df['PrevDate'] = df.groupby('Ticker')['Date'].shift()
    df['PrevEnd'] = df.groupby('Ticker')[qty_end_col].shift()

    # Business day gaps
    df['BusGap'] = df.apply(
        lambda r: np.busday_count(r['PrevDate'].date(), r['Date'].date())
        if pd.notna(r['PrevDate']) else 0,
        axis=1
    )

    # Violations:
    # (1) Gap present
    # (2) Either: last day QuantityEnd != 0 OR today's QuantityStart != 0
    violations = df[(df['BusGap'] > 1) & ((df['PrevEnd'] != 0) | (df[qty_start_col] != 0))]
    return violations


def clean_numeric_series(series):
    """
    Convert a Pandas Series to numeric by removing known currency symbols
    and unexpected whitespace/commas.
    Returns a float Series.
    """
    return (series.fillna('0')
                  .astype(str)
                  .str.replace(r'[^\d.\-]', '', regex=True)   # remove everything except digits, dot, minus
                  .replace('', '0').str                       # handle empty after cleaning
                  .replace('-', '').astype(float))


def convert_to_datetime(date):
    if isinstance(date, int):
        return datetime.strptime(str(date), "%Y%m%d")
    elif isinstance(date, str):
        try:
            return datetime.strptime(date, "%Y%m%d")
        except ValueError:
            return datetime.strptime(date, "%Y-%m-%d")
    elif isinstance(date, np.datetime64):
        return pd.Timestamp(date)
    elif isinstance(date, pd.Timestamp):
        return date.to_pydatetime()  # Convert to Python datetime
    elif isinstance(date, datetime):
        return date
    elif isinstance(date, ddtt.date):
        return date
    else:
        raise ValueError("Unsupported date format")


def next_day(date):
    date = convert_to_datetime(date)
    next_bus_day = pd.Timestamp(date) + pd.tseries.offsets.Day(1)

    return next_bus_day.date()


def previous_business_day(date):

    date = convert_to_datetime(date)
    prev_bus_day = pd.Timestamp(date) - pd.tseries.offsets.BDay(1)

    return prev_bus_day.date()


def business_days_between(start_date, end_date):

    start_date = convert_to_datetime(start_date)
    end_date = convert_to_datetime(end_date)
    business_days_count = pd.bdate_range(start=start_date, end=end_date)

    return business_days_count


def next_business_day(date):
    date = convert_to_datetime(date)
    next_bus_day = pd.Timestamp(date) + pd.tseries.offsets.BDay(1)

    return next_bus_day.date()


def get_today_date():
    return datetime.today().date()


def to_hongkong_datetime(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    hk_tz = pytz.timezone("Asia/Hong_Kong")
    naive_dt = datetime(year, month, day, hour, minute)
    hk_dt = hk_tz.localize(naive_dt)
    return hk_dt


def get_latest_trading_date(input_dt: datetime = None):
    hk_tz = pytz.timezone("Asia/Hong_Kong")

    # Handle input datetime
    if input_dt is None:
        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    else:
        if input_dt.tzinfo is None:
            now_utc = input_dt.replace(tzinfo=pytz.utc)
        else:
            now_utc = input_dt.astimezone(pytz.utc)

    # Convert to Hong Kong time
    now_hk = now_utc.astimezone(hk_tz)

    # Apply 7:30 AM cutoff logic
    if now_hk.time() > time(7, 30):
        return now_hk.date()
    else:
        return pd.Timestamp(now_hk.date()) - pd.tseries.offsets.BDay(1)


def find_earliest_enfusion_daily(folder_path, target_date, latest=False):
    pattern = re.compile(r'_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.xls$', re.IGNORECASE)
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')

    matched_files = []

    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            month, day, year, hour, minute, second = map(int, match.groups())
            file_dt = datetime(year, month, day, hour, minute, second)

            if file_dt.date() == target_dt.date():
                matched_files.append((file_dt, filename))

    if not matched_files:
        print(f"No files found for {target_date} in {folder_path}.")
        return None

    matched_files.sort()

    if latest:
        return matched_files[-1][1]
    else:
        return matched_files[0][1]


def exist_file(file_path):
    return Path(file_path).exists()


def exist_create_folder(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def find_avail_files(folder_path, keywords, file_format):
    avail_files = glob.glob(os.path.join(folder_path, f"{keywords}*.{file_format}"))
    if not avail_files:
        raise Exception("No matching files found.")
    return avail_files


def find_latest_file(folder_path, keywords, file_format):
    latest_file = max(find_avail_files(folder_path, keywords, file_format), key=os.path.getmtime)
    return latest_file


def clean_ticker(df, replace_dict, ticker_col='Ticker', drop=False):
    df = df.pipe(lambda d: d.assign(Ticker=d[ticker_col].replace(replace_dict, regex=True).str.upper()))
    if drop:
        df = df.drop_duplicates(subset='Ticker')
    return df


def fill_event_dates_before_and_after(group, event_col, days=20):
    reaction_dates = group[event_col].dropna()

    for reaction_date in reaction_dates.index:
        start = max(reaction_date - days, group.index[0])
        end = min(reaction_date + days, group.index[-1])
        group.loc[start:end, event_col] = group.loc[reaction_date, event_col]

    return group


def shift_df_date(df, N, ticker_col='Ticker', date_col='Date'):
    if N == 0:
        return df
    # Ensure data is sorted by Ticker and Date
    df = df.sort_values(by=[ticker_col, date_col])
    sign = '-' if N > 0 else '+'
    # Compute price N rows back using groupby and shift
    df[f'Date_T{sign}{abs(N)}'] = df.groupby(ticker_col)[date_col].shift(N)

    return df


def age_calculation(df, action_col, pnl_col='TotalPnL', ticker_col='Ticker', date_col='Date'):
    d = df.sort_values([ticker_col, date_col]).reset_index(drop=True).copy()

    out = np.empty(len(d), dtype=float)

    for ticker, sub in d.groupby(ticker_col, sort=False):
        action = 0.0
        new_actions = []

        for a, pnl in zip(sub[action_col].to_numpy(), sub[pnl_col].to_numpy()):
            if pd.notna(a):
                action = float(a)
            elif pnl != 0:
                action += 1.0
            new_actions.append(action)

        out[sub.index] = new_actions

    d[action_col] = out
    return d


def add_age_and_specific_count_id(
        df,
        cycle_start_criteria_col,
        cycle_start_criteria_values,
        cycle_term_structure_col,
        reset_cycle_by,
        cycle_name,
        start_from=0,
        add_id=True):

    df = df \
        .pipe(lambda d: d.assign(cycle_term_structure_col=np.where(d[cycle_start_criteria_col].isin(cycle_start_criteria_values),
                                              start_from, np.nan))) \
        .rename(columns={'cycle_term_structure_col': cycle_term_structure_col}) \
        .pipe(lambda d: age_calculation(d, cycle_term_structure_col))

    if add_id:
        df = df.pipe(lambda d: d.assign(CycleId=np.where(d[cycle_start_criteria_col].isin(cycle_start_criteria_values), 1, np.nan)) \
             .sort_values(by='Date')) \
             .pipe(lambda d: d.assign(CycleId=d.groupby(reset_cycle_by).CycleId.transform(
            lambda x: x.notna().cumsum().sub(0).where(x.notna())))) \
             .pipe(lambda d: d.assign(CycleId=d.groupby(reset_cycle_by).CycleId.ffill().fillna(1))) \
             .pipe(lambda d: d.assign(UniqueGroup=d[reset_cycle_by] + '_' + d.CycleId.astype(int).astype(str))) \
             .rename(columns={'CycleId': f'{cycle_name}Id', 'UniqueGroup': cycle_name})
    return df


def add_internal_attributes(df):
    # 预先过滤数据，减少后续计算量
    df = df[(df.NMVStart != 0) | (df.NMVEnd != 0)].copy()

    # --- 1. 基础列计算 (向量化) ---
    df['QuantityStart'] = df['QuantityStart'].astype(float)
    df['QuantityEnd'] = df['QuantityEnd'].astype(float)

    # Side 计算
    cond_start_0 = df['NMVStart'] == 0
    cond_end_pos = df['NMVEnd'] > 0
    cond_start_pos = df['NMVStart'] > 0

    df['Side'] = np.where(cond_start_0,
                          np.where(cond_end_pos, 'Long', 'Short'),
                          np.where(cond_start_pos, 'Long', 'Short'))

    # TotalReturn 计算
    denom = np.where(df['GMVStart'] == 0, df['GMVEnd'], df['GMVStart'])
    df['TotalReturn'] = np.where(denom != 0, df['TotalPnL'] / denom, 0.0)

    # --- 2. Groupby Transform 优化 ---
    g_date_book = df.groupby(['Book', 'Date'])['GMVEnd']
    g_date_book_side = df.groupby(['Date', 'Book', 'Side'])['GMVEnd']

    sum_gmv_date_book = g_date_book.transform('sum')
    sum_gmv_date_book_side = g_date_book_side.transform('sum')

    # Position Size
    df['PositionSize'] = df['GMVEnd'] / sum_gmv_date_book
    df['PositionSizeSide'] = df['GMVEnd'] / sum_gmv_date_book_side

    # ENP (Effective N) 计算
    def calc_enp_vectorized(x):
        return (x.sum() ** 2) / (np.sum(x ** 2) + 1e-12)

    df['ENP'] = g_date_book.transform(calc_enp_vectorized)
    df['ENPSide'] = g_date_book_side.transform(calc_enp_vectorized)

    # ConvLvl 计算
    inv_enp = 1.0 / df['ENP']
    inv_enp_side = 1.0 / df['ENPSide']

    df['ConvLvl'] = np.select(
        [df['PositionSize'] > inv_enp, df['PositionSize'] < (0.5 * inv_enp)],
        ['Large', 'Small'],
        default='Medium'
    )

    df['ConvLvlSide'] = np.select(
        [df['PositionSizeSide'] > inv_enp_side, df['PositionSizeSide'] < (0.5 * inv_enp_side)],
        ['Large', 'Small'],
        default='Medium'
    )

    # --- 3. 核心修复：向量化 calculate_position_change ---
    q_s = df['QuantityStart'].values
    q_e = df['QuantityEnd'].values

    c_start_0 = (q_s == 0)
    c_start_not_0 = ~c_start_0

    # 【修正处】统一变量名为 cond_flip
    cond_flip = ((q_s < 0) & (q_e > 0)) | ((q_s > 0) & (q_e < 0))

    pos_chg = np.zeros_like(q_s)

    with np.errstate(divide='ignore', invalid='ignore'):
        pos_chg = np.where(c_start_not_0, (q_e / np.where(c_start_not_0, q_s, 1.0)) - 1, 0.0)

    # 这里使用修正后的 cond_flip
    pos_chg = np.where(cond_flip, 1.0, pos_chg)
    pos_chg = np.where(c_start_0 & (q_e != 0), np.nan, pos_chg)

    df['PositionChange'] = pos_chg

    # --- 4. 核心修复：向量化 calculate_idea_type ---
    cond_init = (q_s == 0) & (q_e != 0)
    cond_close = (q_s != 0) & (q_e == 0)
    # cond_flip 已定义
    cond_intraday = (q_s == 0) & (q_e == 0)

    abs_s = np.abs(q_s)
    abs_e = np.abs(q_e)
    cond_shrink = abs_s > abs_e
    cond_grow = abs_s < abs_e

    choices = [
        (cond_init, 'Init'),
        (cond_close, 'Close'),
        (cond_flip, 'Flip'),  # 这里之前报错，现在 cond_flip 已正确定义
        (cond_intraday, 'Intraday'),
        (cond_shrink, 'Shrink'),
        (cond_grow, 'Grow')
    ]

    conds = [c for c, v in choices]
    vals = [v for c, v in choices]

    df['IdeaType'] = np.select(conds, vals, default=None)

    # --- 5. 后续 Age 计算 ---
    outputs = []
    for book, sub in df.groupby('Book'):
        out = add_age_and_specific_count_id(
            sub, 'IdeaType', ['Init'], 'PositionAge', 'Ticker', 'TradeTicker', 0)
        outputs.append(out)

    df = pd.concat(outputs, ignore_index=True)

    outputs = []
    for book, sub in df.groupby('Book'):
        out = add_age_and_specific_count_id(
            sub, 'IdeaType', ['Init', 'Grow', 'Shrink', 'Flip'], 'IdeaAge', 'TradeTicker', 'TradeIdeaTicker', 0)
        outputs.append(out)

    df = pd.concat(outputs, ignore_index=True)

    # --- 6. 最终累计计算 ---
    df = df.sort_values(by='Date')

    g_book_trade = df.groupby(['Book', 'TradeTicker'])
    g_book_idea = df.groupby(['Book', 'TradeIdeaTicker'])

    df['PosCumSumRt'] = g_book_trade['TotalReturn'].cumsum()
    df['IdeaCumSumRt'] = g_book_idea['TotalReturn'].cumsum()
    df['IdeaState'] = g_book_trade['IdeaType'].ffill()

    df['MaxPosCumSumRt'] = g_book_trade['PosCumSumRt'].cummax()
    df['MaxIdeaCumSumRt'] = g_book_idea['IdeaCumSumRt'].cummax()

    df['PosDDRt'] = df['PosCumSumRt'] - df['MaxPosCumSumRt']
    df['IdeaDDRt'] = df['IdeaCumSumRt'] - df['MaxIdeaCumSumRt']

    return df