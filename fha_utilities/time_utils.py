from datetime import datetime, time
import pytz
import numpy as np
import pandas as pd
import datetime as ddtt
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from typing import List, Tuple, Union


# Function to convert different input formats to datetime
def convert_to_datetime(dt):
    if isinstance(dt, int):
        return datetime.strptime(str(dt), "%Y%m%d")
    elif isinstance(dt, str):
        try:
            return datetime.strptime(dt, "%Y%m%d")
        except ValueError:
            return datetime.strptime(dt, "%Y-%m-%d")
    elif isinstance(dt, np.datetime64):
        return pd.Timestamp(dt)
    elif isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()  # Convert to Python datetime
    elif isinstance(dt, datetime):
        return dt
    elif isinstance(dt, ddtt.date):
        return dt
    else:
        raise ValueError("Unsupported date format")


def previous_business_day(dt):

    dt = convert_to_datetime(dt)
    prev_bus_day = pd.Timestamp(dt) - pd.tseries.offsets.BDay(1)

    return prev_bus_day.date()


def previous_n_working_days(dt, n_days):

    date = convert_to_datetime(dt)
    prev_bus_day = pd.Timestamp(dt) - pd.tseries.offsets.BDay(n_days)

    return prev_bus_day.date()


def next_business_day(dt):
    dt = convert_to_datetime(dt)
    next_bus_day = pd.Timestamp(dt) + pd.tseries.offsets.BDay(1)

    return next_bus_day.date()


def next_day(dt):
    dt = convert_to_datetime(dt)
    next_bus_day = pd.Timestamp(dt) + pd.tseries.offsets.Day(1)

    return next_bus_day.date()


def business_days_between(start_date, end_date):

    start_date = convert_to_datetime(start_date)
    end_date = convert_to_datetime(end_date)
    business_days_count = pd.bdate_range(start=start_date, end=end_date)

    return business_days_count


def get_today_date():
    return datetime.today().date()


def to_hongkong_datetime(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    hk_tz = pytz.timezone("Asia/Hong_Kong")
    naive_dt = datetime(year, month, day, hour, minute)
    hk_dt = hk_tz.localize(naive_dt)
    return hk_dt



def get_latest_trading_date(input_dt: datetime = None, cut_off_hour: int = 7, cut_off_min: int = 30):
    hk_tz = pytz.timezone("Asia/Hong_Kong")

    # Handle input datetime (assume naive input is UTC)
    if input_dt is None:
        now_utc = datetime.now(tz=pytz.utc)
    else:
        if input_dt.tzinfo is None:
            now_utc = input_dt.replace(tzinfo=pytz.utc)
        else:
            now_utc = input_dt.astimezone(pytz.utc)

    # Convert to Hong Kong time
    now_hk = now_utc.astimezone(hk_tz)

    # Candidate date based on cutoff
    cutoff = time(cut_off_hour, cut_off_min)
    if now_hk.time() > cutoff:
        candidate = pd.Timestamp(now_hk.date())
    else:
        candidate = pd.Timestamp(now_hk.date()) - pd.tseries.offsets.BDay(1)

    # If candidate is Sat/Sun, roll back to latest business day (Fri)
    candidate = pd.tseries.offsets.BDay().rollback(candidate)

    return candidate.date()

def shift_df_date(df, n, ticker_col='Ticker', date_col='Date'):
    if n == 0:
        return df
    # Ensure data is sorted by Ticker and Date
    df = df.sort_values(by=[ticker_col, date_col])
    sign = '-' if n > 0 else '+'
    # Compute price N rows back using groupby and shift
    df[f'Date_T{sign}{abs(n)}'] = df.groupby(ticker_col)[date_col].shift(n)

    return df


def fill_event_dates_before_and_after(group, event_col, days=20):
    reaction_dates = group[event_col].dropna()

    for reaction_date in reaction_dates.index:
        start = max(reaction_date - days, group.index[0])
        end = min(reaction_date + days, group.index[-1])
        group.loc[start:end, event_col] = group.loc[reaction_date, event_col]

    return group


def adjust_to_working_day(d: date) -> date:
    """
    If d is Saturday or Sunday, return the nearest weekday:
    - Saturday -> Friday
    - Sunday -> Monday
    """
    if d.weekday() < 5:  # Mon-Fri are 0-4
        return d
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    # Sunday
    return d + timedelta(days=1)


def business_days_from_date(date, n):

    date = convert_to_datetime(date)

    # Find the target business day using pandas BDay offset
    target_date = pd.Timestamp(date) + pd.tseries.offsets.BDay(n)

    return target_date.date()


def get_last_day_of_quarter(start, end):
    start = convert_to_datetime(start)
    end = convert_to_datetime(end)


    quarter_ends = pd.date_range(start=start, end=end, freq='Q')

    # For each quarter-end, get the last business day (adjust if not already)
    last_business_days = [
        date if pd.tseries.offsets.BDay().rollback(date) == date else pd.tseries.offsets.BDay().rollback(date)
        for date in quarter_ends
    ]

    return [x.date() for x in last_business_days]


def monthly_last_working_days(start_date, end_date):
    # Ensure timestamps
    start_date = pd.Timestamp(convert_to_datetime(start_date))
    end_date = pd.Timestamp(convert_to_datetime(end_date))

    # Get all month ends between start and end
    month_ends = pd.date_range(start=start_date, end=end_date, freq='M')

    last_working_days = []
    for date in month_ends:
        # If month-end is after end_date, skip
        if date > end_date:
            continue
        # If weekend, roll back
        while date.weekday() >= 5:  # Sat=5, Sun=6
            date -= pd.Timedelta(days=1)
        last_working_days.append(date)

    return pd.Series(last_working_days, name="Last_Working_Day")


def month_start_end(dd):
    dd = pd.Timestamp(convert_to_datetime(dd))
    start = dd.replace(day=1)
    end = (start + pd.offsets.MonthEnd(0))  # last day of the same month
    return start, end


def first_day_last_year_next_month(d):
    # last year's same month
    d = convert_to_datetime(d)
    y = d.year - 1
    m = d.month
    # next month (roll year if December)
    if m == 12:
        y += 1
        m = 1
    else:
        m += 1
    return date(y, m, 1)


def first_day_next_month_n_months_ago(n, ref_date=None):
    if ref_date is None:
        ref_date = date.today()
    # Step 1: go back N months
    past_date = ref_date - relativedelta(months=n)
    # Step 2: move to first day of next month
    next_month_first = (past_date.replace(day=1) + relativedelta(months=1))
    return next_month_first


def check_now_asia_am_pm():
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Singapore")
    except Exception:
        import pytz
        tz = pytz.timezone("Asia/Singapore")

    now = datetime.now(tz)
    return 'AM' if now.hour < 12 else "PM"


def get_today_date_id():
    return int(ddtt.datetime.today().strftime('%Y%m%d'))


DateLike = Union[str, date, datetime]

def monthly_ranges(start: DateLike, end: DateLike) -> List[Tuple[str, str]]:
    """
    Return list of (month_start, month_end) tuples as 'YYYY-MM-DD' strings,
    clipped to the [start, end] interval (inclusive).

    Accepts 'YYYY-MM-DD' string, datetime.date, or datetime.datetime.
    """
    def to_date(x: DateLike) -> date:
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        if isinstance(x, str):
            return date.fromisoformat(x)
        raise TypeError("start/end must be 'YYYY-MM-DD' string, date, or datetime")

    s = to_date(start)
    e = to_date(end)
    if s > e:
        raise ValueError("startdate must be <= enddate")

    # iterate month by month
    y, m = s.year, s.month
    out: List[Tuple[str, str]] = []

    while True:
        # 当月第一天与最后一天
        month_first = date(y, m, 1)
        last_day = calendar.monthrange(y, m)[1]
        month_last = date(y, m, last_day)

        # 与区间 [s, e] 相交并裁剪
        seg_start = max(s, month_first)
        seg_end = min(e, month_last)

        if seg_start <= seg_end:
            out.append((seg_start.isoformat(), seg_end.isoformat()))

        # 跳到下个月
        if month_last >= e:
            break
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1

    return out


def check_expected_date(df, date_col='Date', cut_off_hour=11, cut_off_min=30, shift=0):
    latest_date = convert_to_datetime(df[date_col].max()).date()
    expected_date = get_latest_trading_date(cut_off_hour=cut_off_hour, cut_off_min=cut_off_min)
    if shift != 0:
        expected_date = previous_business_day(expected_date)

    if latest_date >= expected_date:
        return True
    else:
        return False