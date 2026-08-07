import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from analysis.intraday_screener import IntradayScreener, TradeSetup
from utils.data_fetch import load_universe

st.set_page_config(page_title="Intraday Criteria Engine", layout="wide")

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------
st.title("📈 Page 8 — Intraday Criteria Engine")
st.markdown("Professional multi-timeframe trade setup detector (Daily → 5m → 1m).")

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Screener Settings")

min_daily_volume = st.sidebar.number_input(
    "Minimum Daily Volume", value=300_000, step=100_000
)

min_atr_pct = st.sidebar.number_input(
    "Minimum ATR %", value=0.8, step=0.1
)

min_rr = st.sidebar.number_input(
    "Minimum Risk/Reward", value=1.5, step=0.1
)

max_spread_pct = st.sidebar.number_input(
    "Maximum Spread %", value=0.5, step=0.1
)

limit_tickers = st.sidebar.number_input(
    "Limit Universe (0 = full)", value=20, step=10
)

run_button = st.sidebar.button("Run Intraday Screener")

# ---------------------------------------------------------
# SCREENER INSTANCE
# ---------------------------------------------------------
screener = IntradayScreener(
    min_daily_volume=min_daily_volume,
    min_avg_true_range_pct=min_atr_pct,
    min_risk_reward=min_rr,
    max_spread_pct=max_spread_pct
)

# ---------------------------------------------------------
# DATA FETCH HELPERS (PATCHED)
# ---------------------------------------------------------
def fetch_daily(ticker):
    try:
        df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            progress=False
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # --- FULL FLATTEN OF MULTI-INDEX COLUMNS ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(level) for level in col if level])
            for col in df.columns
        ]

    # --- REMOVE DUPLICATE COLUMNS ---
    df = df.loc[:, ~df.columns.duplicated()]

    # --- FORCE SINGLE OHLCV EXTRACTION ---
    def safe(prefix):
        cols = [c for c in df.columns if prefix in c]
        if not cols:
            return None
        col = df[cols[0]]
        if isinstance(col, pd.DataFrame):
            return col.iloc[:, 0]
        return col

    clean = pd.DataFrame()
    clean["Date"] = df.index
    clean["Open"] = safe("Open")
    clean["High"] = safe("High")
    clean["Low"] = safe("Low")
    clean["Close"] = safe("Close")
    clean["Volume"] = safe("Volume")

    # Reject if missing required columns
    if clean["Close"] is None or clean["High"] is None or clean["Low"] is None:
        return None

    clean = clean.dropna(subset=["Open", "High", "Low", "Close"])

    return clean


def fetch_5min(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        df = df.reset_index().rename(columns={df.columns[0]: "Date"})
        return df
    except Exception:
        return None

def fetch_1min(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        df = df.reset_index().rename(columns={df.columns[0]: "Date"})
        return df
    except Exception:
        return None

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if run_button:
    universe = load_universe()

    if limit_tickers > 0:
        universe = universe[:limit_tickers]

    st.write(f"🔍 Scanning **{len(universe)} tickers** from unified US universe...")

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(universe):
        progress.progress((i + 1) / len(universe))

        daily_df = fetch_daily(ticker)
        df_5min = fetch_5min(ticker)
        df_1min = fetch_1min(ticker)

        setups = screener.screen_ticker(ticker, daily_df, df_5min, df_1min)

        for setup in setups:
            results.append(setup)

    st.success(f"Scan complete — {len(results)} total setups found.")

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------
    if len(results) == 0:
        st.warning("No valid trade setups detected.")
    else:
        df = pd.DataFrame([{
            "Ticker": s.ticker,
            "Direction": s.direction,
            "Setup": s.setup_type,
            "Entry": s.entry_price,
            "Stop": s.stop_loss,
            "Target": s.target_price,
            "Risk/Reward": s.risk_reward,
            "Confidence": s.confidence_score,
            "Daily Trend": s.daily_trend,
            "Volume Confirmed": s.volume_confirmed,
            "Key Level": s.key_level
        } for s in results])

        df = df.sort_values("Confidence", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.subheader("🔥 Top 10 Highest Confidence Setups")
        st.table(df.head(10))
