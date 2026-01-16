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