# utils/data_fetch.py — Integrated with Master US Universe
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
# LOAD UNIVERSE (Unified US Market Basket)
# ---------------------------------------------------------
try:
    from data.us_universe_list import us_universe
except ImportError:
    print("WARNING: data.us_universe_list not found. Falling back to an empty list.")
    us_universe = []

def load_universe():
    """
    Returns the combined trading universe for the Intraday Ranker.
    Includes all deduplicated NYSE and NASDAQ tickers ($40–$110).
    """
    return sorted(list(set(us_universe)))


# ---------------------------------------------------------
# UNIVERSE SOURCE HELPER (Upgraded to Exchange Level)
# ---------------------------------------------------------
def get_universe_source(ticker: str):
    """
    Identifies the underlying exchange for a given ticker symbol.
    Uses standard US market listing structures to classify:
    - NASDAQ: Primarily 4-character symbols (e.g., AAPL, MSFT)
    - NYSE: Primarily 1, 2, or 3-character symbols (e.g., T, KO, JPM)
    """
    clean_ticker = ticker.strip().upper()
    
    if clean_ticker not in us_universe:
        return "UNKNOWN"
        
    # Standard rule of thumb: NASDAQ uses 4+ letters, NYSE uses 1-3 letters
    if len(clean_ticker) >= 4:
        return "NASDAQ"
    
    return "NYSE"
