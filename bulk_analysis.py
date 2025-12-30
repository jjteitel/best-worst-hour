import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import time

CACHE_DIR = Path("cache")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CACHE_DIR.mkdir(exist_ok=True)


@st.cache_data(ttl=86400)
def get_nasdaq_tickers(min_market_cap=1_000_000_000):
    """Fetch top NASDAQ tickers by market cap from NASDAQ website"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    # NASDAQ screener API - returns all NASDAQ stocks
    url = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nasdaq&download=true'

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.error(f"Failed to fetch NASDAQ tickers: {r.status_code}")
        return []

    data = r.json()
    rows = data.get('data', {}).get('rows', [])

    # Filter and parse market cap
    stocks = []
    for row in rows:
        symbol = row.get('symbol', '')
        # Skip symbols with special characters (warrants, units, etc.)
        if symbol and not any(c in symbol for c in ['^', '.', '/', '-']):
            try:
                market_cap = float(row.get('marketCap', '0').replace(',', ''))
            except:
                market_cap = 0
            if market_cap >= min_market_cap: stocks.append((symbol, market_cap))

    # Sort by market cap descending and take top N
    stocks.sort(key=lambda x: x[1], reverse=True)
    tickers = [s[0] for s in stocks]

    return tickers


def get_market_hour(ts):
    hour = ts.hour
    minute = ts.minute
    if hour == 9 and minute >= 30:
        return '9:30-10'
    elif hour == 10 and minute == 0:
        return '10-11'
    elif hour == 10 and minute == 30:
        return '10-11'
    elif hour == 11 and minute == 0:
        return '11-12'
    elif hour == 11 and minute == 30:
        return '11-12'
    elif hour == 12 and minute == 0:
        return '12-13'
    elif hour == 12 and minute == 30:
        return '12-13'
    elif hour == 13 and minute == 0:
        return '13-14'
    elif hour == 13 and minute == 30:
        return '13-14'
    elif hour == 14 and minute == 0:
        return '14-15'
    elif hour == 14 and minute == 30:
        return '14-15'
    elif hour == 15 and minute == 0:
        return '15-16'
    elif hour == 15 and minute == 30:
        return '15-16'
    return None


def calculate_hourly_stats(df, ticker):
    """Calculate mean and median return by market hour for a single ticker"""
    if df.empty:
        return None

    df_copy = df.copy()

    # Flatten multi-level columns if present
    if isinstance(df_copy.columns, pd.MultiIndex):
        df_copy.columns = df_copy.columns.get_level_values(0)

    # Convert to Eastern time
    if df_copy.index.tz is None:
        df_copy.index = df_copy.index.tz_localize('UTC')
    df_copy.index = df_copy.index.tz_convert('US/Eastern')

    df_copy['market_hour'] = df_copy.index.map(get_market_hour)
    df_copy['date'] = df_copy.index.date

    market_data = df_copy[df_copy['market_hour'].notna()].copy()

    if market_data.empty:
        return None

    grouped = market_data.groupby(['date', 'market_hour']).agg({
        'Open': 'first',
        'Close': 'last'
    })
    grouped['pct_return'] = (grouped['Close'] - grouped['Open']) / grouped['Open'] * 100

    # Calculate both mean and median
    stats = grouped.groupby('market_hour')['pct_return'].agg(['mean', 'median'])

    hour_order = ['9:30-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16']
    stats = stats.reindex(hour_order)

    result = {'ticker': ticker}
    for hour in hour_order:
        result[f'{hour} Mean'] = stats.loc[hour, 'mean'] if hour in stats.index else None
        result[f'{hour} Median'] = stats.loc[hour, 'median'] if hour in stats.index else None

    return result


def download_with_retry(tickers, start_date, end_date):
    """Download data with retry logic for transient failures"""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval='30m',
                group_by='ticker',
                progress=False,
                threads=False  # Disable threads to avoid SQLite locking
            )
            return data, None
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    return pd.DataFrame(), last_error


def download_batch(tickers, start_date, end_date, batch_size=50):
    """Download data for multiple tickers in batches"""
    all_stats = []
    failed_tickers = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        status_text.text(f"Downloading {i+1}-{min(i+batch_size, len(tickers))} of {len(tickers)}...")

        data, error = download_with_retry(batch, start_date, end_date)

        if error:
            st.warning(f"Batch failed after {MAX_RETRIES} retries: {error}")
            failed_tickers.extend(batch)
            continue

        # Process each ticker
        for ticker in batch:
            try:
                if len(batch) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()

                if not ticker_data.empty:
                    stats = calculate_hourly_stats(ticker_data, ticker)
                    if stats:
                        all_stats.append(stats)
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)
                continue

        progress_bar.progress(min((i + batch_size) / len(tickers), 1.0))
        time.sleep(0.5)  # Rate limiting

    # Retry failed tickers individually
    if failed_tickers:
        status_text.text(f"Retrying {len(failed_tickers)} failed tickers individually...")
        for ticker in failed_tickers:
            data, error = download_with_retry(ticker, start_date, end_date)
            if not data.empty:
                stats = calculate_hourly_stats(data, ticker)
                if stats:
                    all_stats.append(stats)
            time.sleep(1)

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(all_stats)


# Streamlit UI
st.title("NASDAQ Best/Worst Hour Analysis")
st.write("Analyze average hourly returns for all NASDAQ stocks")

# Date inputs
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=42))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())
with col3:
    view_mode = st.selectbox("View", ["Mean", "Median", "Both"])

# Cache file path
cache_file = CACHE_DIR / f"nasdaq_stats_v2_{start_date}_{end_date}.parquet"

col1, col2 = st.columns([1, 1])
with col1:
    run_button = st.button("Run Analysis")
with col2:
    if st.button("Clear Cache"):
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        st.success("Cache cleared!")
        st.rerun()

if run_button:
    if cache_file.exists():
        st.info("Loading from cache...")
    else:
        st.info("Fetching NASDAQ ticker list...")
        tickers = get_nasdaq_tickers()
        st.success(f"Found {len(tickers)} NASDAQ stocks")

        st.info("Downloading price data (this may take a while)...")
        results_df = download_batch(tickers, start_date, end_date)

        # Save to cache
        if not results_df.empty:
            results_df.to_parquet(cache_file)
            st.success(f"Cached results to {cache_file}")

# Load and display data if cache exists
if cache_file.exists():
    results_df = pd.read_parquet(cache_file)
    if not results_df.empty:
        st.subheader(f"Results: {len(results_df)} stocks analyzed")

        # Filter columns based on view mode
        hour_order = ['9:30-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16']

        if view_mode == "Mean":
            cols = ['ticker'] + [f'{h} Mean' for h in hour_order]
            display_df = results_df[cols].copy()
            # Rename columns to remove "Mean" suffix
            display_df.columns = ['ticker'] + hour_order
        elif view_mode == "Median":
            cols = ['ticker'] + [f'{h} Median' for h in hour_order]
            display_df = results_df[cols].copy()
            # Rename columns to remove "Median" suffix
            display_df.columns = ['ticker'] + hour_order
        else:  # Both
            display_df = results_df.copy()

        # Format numeric columns as percentages
        format_dict = {col: '{:.2f}%' for col in display_df.columns if col != 'ticker'}

        # Sortable dataframe
        st.dataframe(
            display_df.style.format(format_dict),
            use_container_width=True,
            height=600
        )

        # Download button
        csv = results_df.round(2).to_csv(index=False)
        st.download_button(
            "Download Full CSV (Mean + Median)",
            csv,
            "nasdaq_hourly_returns.csv",
            "text/csv"
        )
    else:
        st.error("No data retrieved")

# Show cached files
cached_files = list(CACHE_DIR.glob("*.parquet"))
if cached_files:
    st.subheader("Cached Data")
    for f in cached_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(f.name)
        with col2:
            if st.button("Load", key=f.name):
                results_df = pd.read_parquet(f)
                st.dataframe(results_df.round(2), use_container_width=True, height=600)
