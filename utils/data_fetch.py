# utils/data_fetch.py — Shared universe for every model (penny, momentum, ...)
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
# LOAD UNIVERSE (Unified, price-agnostic US market basket)
# ---------------------------------------------------------
# CHANGE: us_universe_list.py no longer has a price range baked into it
# (see generate_universe.py). The same file/list is now shared by every
# model — penny (1-10), momentum (40-120), or anything else — and each
# model's own min_price/max_price widget does the actual price filtering,
# live, against current data. This also removes the old failure mode where
# forgetting to regenerate the file after switching models meant a scan
# silently ran against the wrong price range with no error.
try:
    from data.us_universe_list import us_universe, us_universe_exchange
except ImportError:
    print("WARNING: data.us_universe_list not found. Falling back to an empty list.")
    us_universe = []
    us_universe_exchange = {}


def load_universe():
    """
    Returns the full, price-agnostic combined trading universe
    (deduplicated NYSE + NASDAQ common stock). Every model filters this
    down to its own price range live, at scan time.
    """
    return sorted(set(us_universe))


# ---------------------------------------------------------
# UNIVERSE SOURCE HELPER
# ---------------------------------------------------------
def get_universe_source(ticker: str):
    """
    Returns the exchange a ticker was actually sourced from
    (NASDAQ / NYSE), as tagged by generate_universe.py at build time from
    which source CSV the ticker came from.

    CHANGE: previously guessed NASDAQ vs NYSE from ticker symbol length
    (>=4 chars -> NASDAQ), which is unreliable in both directions (many
    NASDAQ tickers are 1-3 letters; NYSE has 4-letter tickers too). Now
    looks up the real tag carried through from the source file.
    """
    clean_ticker = ticker.strip().upper()
    return us_universe_exchange.get(clean_ticker, "UNKNOWN")