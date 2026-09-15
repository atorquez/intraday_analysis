# ======================================================================
# 📉 PRICE TRACKER — DROP VS PREVIOUS CLOSE
# Simple scanner to show biggest price dips in ascending order
# ======================================================================

import importlib
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(layout="wide", page_title="Price Drop Tracker")

st.caption("Version: V1 — Price Drop Tracker (Prev Close vs Current Price)")

st.title("📉 Price Drop Tracker — Biggest Dips vs Previous Close")

# ============================================================
# LOAD UNIVERSE
# ============================================================

def _load_universe():
    try:
        import utils.data_fetch as data_fetch_module
        data_fetch_module = importlib.reload(data_fetch_module)
        tickers = data_fetch_module.load_universe()
        return sorted(set(str(x).upper() for x in tickers))
    except Exception as e:
        st.error(f"Unable to load universe: {e}")
        return []

# ============================================================
# MARKET DATA FETCH
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_market_batch(tickers_tuple):
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    try:
        raw_daily = yf.download(
            tickers,
            period="3mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        raw_intra = yf.download(
            tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        return raw_daily, raw_intra

    except Exception as e:
        st.error(f"Market data download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ============================================================
# HELPERS
# ============================================================

def _flatten_columns(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        known = {"Open","High","Low","Close","Adj Close","Volume"}
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if all(x in known for x in lvl0):
            df.columns = lvl0
        elif all(x in known for x in lvl1):
            df.columns = lvl1
    return df

def _extract_ticker_slice(batch, ticker):
    if batch is None or batch.empty:
        return pd.DataFrame()
    ticker = str(ticker).upper()
    if not isinstance(batch.columns, pd.MultiIndex):
        return _flatten_columns(batch.copy())
    try:
        lvl0 = set(batch.columns.get_level_values(0))
        lvl1 = set(batch.columns.get_level_values(1))
        if ticker in lvl0:
            return _flatten_columns(batch[ticker].copy())
        if ticker in lvl1:
            return _flatten_columns(batch.xs(ticker, axis=1, level=1, drop_level=True).copy())
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def _to_eastern_index(df):
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    eastern = ZoneInfo("America/New_York")
    if idx.tz is not None:
        idx = idx.tz_convert(eastern)
    else:
        idx = idx.tz_localize(eastern)
    df.index = idx
    return df

# ============================================================
# PRICE TRACK ENGINE
# ============================================================

def price_track_engine(tickers, batch_daily, batch_intra):

    rows = []

    for ticker in tickers:

        daily_df = _flatten_columns(_extract_ticker_slice(batch_daily, ticker))
        intra_df = _flatten_columns(_extract_ticker_slice(batch_intra, ticker))

        if daily_df.empty or intra_df.empty:
            continue

        if "Close" not in daily_df.columns:
            continue
        if "Close" not in intra_df.columns:
            continue

        daily_df = daily_df.dropna(subset=["Close"])
        intra_df = intra_df.dropna(subset=["Close"])

        if len(daily_df) < 2:
            continue

        prev_close = float(daily_df["Close"].iloc[-2])

        # Convert intraday to Eastern and regular session only
        intra_df = _to_eastern_index(intra_df).sort_index()
        intra_df = intra_df.between_time("09:30", "16:00")

        if intra_df.empty:
            continue

        current_price = float(intra_df["Close"].iloc[-1])

        if not np.isfinite(current_price) or not np.isfinite(prev_close):
            continue

        pct_change = (current_price - prev_close) / prev_close * 100

        rows.append({
            "Ticker": ticker,
            "Previous_Close": round(prev_close, 2),
            "Current_Price": round(current_price, 2),
            "Current_vs_Previous_%": round(pct_change, 2)
        })


    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sort ascending: biggest drop first (most negative %)
    df = df.sort_values(by="Current_vs_Previous_%", ascending=True)

    return df

# ============================================================
# RUN PAGE
# ============================================================

run_scan = st.button("Run Price Drop Tracker")

if run_scan:
    st.cache_data.clear()

    universe = _load_universe()
    raw_daily, raw_intra = fetch_market_batch(tuple(universe))

    ranking = price_track_engine(universe, raw_daily, raw_intra)

    if ranking.empty:
        st.info("No tickers found.")
    else:
        st.subheader(f"📉 Price Drop Tracker — {len(ranking)} Tickers")
        st.dataframe(ranking, hide_index=True, use_container_width=True)
