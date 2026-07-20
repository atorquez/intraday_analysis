# utils/data_fetch.py — Restored Stable Version (60‑day window)
import yfinance as yf
import pandas as pd

def get_last_10_closes(symbol: str):
    """
    Fetch last 10 daily closes for a ticker.
    Stable, permissive version used before filters were added.
    - 60-day window to avoid Yahoo gaps
    - Only requires Close
    - Volume optional
    - No strict 10-day requirement
    """

    try:
        df = yf.download(
            symbol,
            period="60d",          # <-- expanded window (fixes universe collapse)
            interval="1d",
            progress=False,
            threads=False
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Normalize MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Require only Close
    if "Close" not in df.columns:
        return None

    # Keep Close (Volume optional)
    cols = ["Close"]
    if "Volume" in df.columns:
        cols.append("Volume")

    df = df[cols].dropna(subset=["Close"])

    # Take last 10 rows (even if fewer exist)
    df = df.tail(10)

    if df.empty:
        return None

    return df

# ---------------------------------------------------------
# LOAD UNIVERSE (SP500 + NASDAQ100 + OTHER)
# ---------------------------------------------------------
from data.sp500_list import sp500
from data.nasdaq1000 import nasdaq1000

def load_universe():
    """
    Returns the combined trading universe for the Intraday Ranker.
    Includes:
    - SP500
    - NASDAQ1000   
    """
    return sorted(list(set(sp500 + nasdaq1000)))

