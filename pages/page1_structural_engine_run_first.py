from fileinput import close
import importlib

from yfinance import ticker
import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)
print(">>> USING FILE:", v3.__file__)

import analysis.intraday_ranker_v3 as v3
rank_universe = v3.rank_universe

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz

from utils.data_fetch import load_universe, get_universe_source

# ---------------------------------------------------------
# CACHE WARM-UP BUTTON
# ---------------------------------------------------------
import yfinance as yf
import streamlit as st

@st.cache_data
def warm_daily_10d(ticker):
    return yf.download(ticker, period="10d", interval="1d", progress=False)

@st.cache_data
def warm_daily_30d(ticker):
    return yf.download(ticker, period="30d", interval="1d", progress=False)

@st.cache_data
def warm_intraday_5m(ticker):
    return yf.download(ticker, period="1d", interval="5m", progress=False)

def warm_up_cache(tickers):
    progress = st.progress(0.0, text="Warming up cache...")
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            warm_daily_10d(ticker)
            warm_daily_30d(ticker)
            warm_intraday_5m(ticker)
        except Exception:
            pass
        progress.progress((i+1)/total, text=f"Warming up cache... {ticker}")
    st.success("Cache warm-up complete! Page1 will run much faster now.")

# UI Button
if st.button("🔥 Warm Up Cache (Preload All Tickers)"):
    warm_up_cache(load_universe())  # your list of 730 tickers



st.set_page_config(layout="wide")
st.caption("Version: 2026-07-21")
st.title("📈 Structural Engine Page1")

# ---------------------------------------------------------
# INDEX TREND FUNCTIONS
# ---------------------------------------------------------
def get_index_trend(ticker):
    """
    Returns: Bullish / Bearish / Choppy
    based on EMA9, EMA20, EMA50 alignment + slope.
    """
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty:
            return "Unknown"

        df["EMA9"] = df["Close"].ewm(span=9).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        ema9 = df["EMA9"].iloc[-1]
        ema20 = df["EMA20"].iloc[-1]
        ema50 = df["EMA50"].iloc[-1]

        slope20 = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]

        if ema9 > ema20 > ema50 and slope20 > 0:
            return "Bullish"

        if ema9 < ema20 < ema50 and slope20 < 0:
            return "Bearish"

        return "Choppy"

    except Exception:
        return "Unknown"

def classify_regime(sp500_trend, nasdaq_trend):
    """
    Returns: Trending / Choppy / Mixed / Bearish
    """
    if sp500_trend == "Bullish" and nasdaq_trend == "Bullish":
        return "Trending"

    if sp500_trend == "Bearish" and nasdaq_trend == "Bearish":
        return "Bearish"

    if sp500_trend != nasdaq_trend:
        return "Mixed"

    return "Choppy"

# ---------------------------------------------------------
# CACHED WRAPPER FOR RANK_UNIVERSE (FIXED: Prevents re-download)
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_rank_universe(tickers_tuple, buy_zone_percentile=0.15):
    return rank_universe(list(tickers_tuple), buy_zone_percentile)

# ---------------------------------------------------------
# COLOR HELPER
# ---------------------------------------------------------
def color_execution_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)

    if "Execution" in df.columns:
        style_df["Execution"] = [
            "background-color: #4CAF50; color: white;" if v == "Watch List"
            else "background-color: #FFC107; color: black;" if v == "Crossing Soon"
            else "background-color: #FF9800; color: white;" if v == "Not Watch List"
            else "background-color: #9E9E9E; color: white;"
            for v in df["Execution"]
        ]
    return style_df

# ---------------------------------------------------------
# RESTORE PREVIOUS FILTERED RESULTS IF AVAILABLE
# ---------------------------------------------------------
filtered_results = st.session_state.get("intraday_visual_results", None)
structural_results = st.session_state.get("intraday_filtered_results", None)

# ---------------------------------------------------------
# PRICE FILTERS
# ---------------------------------------------------------
st.markdown("### 🔍 Price Filter")
min_price = st.number_input("Minimum Price", value=50.0, key="intraday_min_price")
max_price = st.number_input("Maximum Price", value=200.0, key="intraday_max_price")

# ---------------------------------------------------------
# FILTER PANEL
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

buy_signal_filter = st.multiselect(
    "Filter by Buy Signal Zone",
    ["DEEP_VALUE_ZONE", "MID_VALUE_ZONE", "NEAR_VALUE_ZONE", "EXTENDED_ZONE", "UNKNOWN"],
    default=["DEEP_VALUE_ZONE", "MID_VALUE_ZONE"],
    key="intraday_buy_signal_filter"
)

execution_filter = st.multiselect(
    "Filter by Execution Status",
    ["Watch List", "Not Watch List", "Crossing Soon", "Setup Only"],
    default=["Crossing Soon"],
    key="intraday_execution_filter"
)

# ---------------------------------------------------------
# RUN RANKER
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model", key="intraday_run_button")

if run_model:
    import time
    start_time = time.time()

    # --- FIX: Local EDT time ---
    from datetime import datetime
    import pytz
    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now().astimezone(eastern)
    st.markdown(f"⏱️ Start Time: {now_est.strftime('%Y-%m-%d %H:%M:%S')}")

    progress_bar = st.progress(0, text="Loading universe...")

    try:
        base_universe = load_universe()
        if not base_universe:
            st.error("The source stock universe list returned empty.")
            st.stop()

        progress_bar.progress(0.1, text=f"Scanning {len(base_universe)} tickers...")

        ranking = cached_rank_universe(tuple(base_universe))

        # ---------------------------------------------------------
        # ADD SP500 / NASDAQ TREND INDICATORS
        # ---------------------------------------------------------
        sp500_trend = get_index_trend("^GSPC")
        nasdaq_trend = get_index_trend("^IXIC")
        regime = classify_regime(sp500_trend, nasdaq_trend)

        ranking["SP500_Trend"] = sp500_trend
        ranking["NASDAQ_Trend"] = nasdaq_trend
        ranking["Market_Regime"] = regime

        # ---------------------------------------------------------
        # ADD SRC (Structural Recovery Candidate)
        # ---------------------------------------------------------
                # ---------------------------------------------------------
        # ADD SRC (Structural Recovery Candidate) — OPTIMIZED
        # ---------------------------------------------------------
        import yfinance as yf

        # Cache helpers
        @st.cache_data
        def get_daily_10d(ticker):
            return yf.download(ticker, period="10d", interval="1d", progress=False)

        @st.cache_data
        def get_daily_30d(ticker):
            return yf.download(ticker, period="30d", interval="1d", progress=False)

        @st.cache_data
        def get_intraday_5m(ticker):
            return yf.download(ticker, period="1d", interval="5m", progress=False)

        def to_scalar(x):
            try:
                if hasattr(x, "item"):
                    return float(x.item())
                return float(x)
            except Exception:
                try:
                    return float(x.iloc[-1])
                except Exception:
                    return float(x)

        def compute_drop_pct(prev_close, current_price):
            prev_close = to_scalar(prev_close)
            current_price = to_scalar(current_price)
            return (prev_close - current_price) / prev_close if prev_close > 0 else 0

        def compute_recovery_probability(ticker):
            df = get_daily_30d(ticker)
            if df.empty or len(df) < 10:
                return 0.0

            recoveries = 0
            total = len(df)

            for i in range(total):
                low = to_scalar(df["Low"].iloc[i])
                close = to_scalar(df["Close"].iloc[i])
                high = to_scalar(df["High"].iloc[i])

                dip = high - low
                recovered = close - low

                if dip > 0 and recovered >= 0.5 * dip:
                    recoveries += 1

            return recoveries / total

        def ema9_cross_ema20_intraday(ticker):
            df = get_intraday_5m(ticker)
            if df.empty or len(df) < 20:
                return False

            df["EMA9"] = df["Close"].ewm(span=9).mean()
            df["EMA20"] = df["Close"].ewm(span=20).mean()

            ema9 = to_scalar(df["EMA9"].iloc[-1])
            ema20 = to_scalar(df["EMA20"].iloc[-1])

            return ema9 > ema20

        # Prime Time filter (10:00–11:30 EST)
        prime_time = (now_est.hour == 10) or (now_est.hour == 11 and now_est.minute <= 30)

        src_flags = []

        # If not prime time, skip heavy SRC computation and keep table fast
        if not prime_time:
            src_flags = ["" for _ in range(len(ranking))]
        else:
            for idx, row in ranking.iterrows():
                ticker = row["Ticker"]
                current_price = to_scalar(row["Close"])

                df_daily = get_daily_10d(ticker)
                if df_daily.empty or len(df_daily) < 2:
                    src_flags.append("")
                    continue

                prev_close = to_scalar(df_daily["Close"].iloc[-2])
                drop_pct = compute_drop_pct(prev_close, current_price)
                recovery_prob = compute_recovery_probability(ticker)
                ema_cross = ema9_cross_ema20_intraday(ticker)

                SRC = (
                    drop_pct >= 0.03 and
                    recovery_prob >= 0.60 and
                    ema_cross and
                    regime in ["Bearish", "Choppy"]
                )

                src_flags.append("YES" if SRC else "")

        ranking["SRC"] = src_flags
#--
        progress_bar.progress(0.6, text="Applying filters...")

        if ranking is None or ranking.empty:
            st.warning("No stock configurations satisfied the technical requirements.")
        else:
            st.markdown(f"**Structural Universe (Daily):** {len(ranking)} tickers")

            ranking["Universe"] = ranking["Ticker"].apply(get_universe_source)

            filtered = ranking.copy()

            filtered = filtered[
                (filtered["Close"] >= min_price) &
                (filtered["Close"] <= max_price)
            ]

            if execution_filter:
                filtered = filtered[
                    filtered["Execution"].isin(execution_filter)
                ]

            st.session_state["intraday_filtered_results"] = ranking
            st.session_state["intraday_visual_results"] = filtered

            st.markdown(f"**User Visual Filters Applied:** {len(filtered)} tickers shown")
            st.markdown(f"**Tickers Passed to Page 2 (Structural):** {len(ranking)} tickers")

            if filtered.empty:
                st.info("No companies matched the selected filters.")
            else:
                display_df = filtered.copy()

                display_df["Ticker"] = display_df.apply(
                    lambda row: f"{row['Ticker']} ({row['Universe']})",
                    axis=1
                )

                display_df = display_df.drop(columns=["Universe"])

                # Add trend indicators
                display_df["SP500_Trend"] = sp500_trend
                display_df["NASDAQ_Trend"] = nasdaq_trend
                display_df["Market_Regime"] = regime

                # Add SRC flag
                display_df["SRC"] = display_df["SRC"].apply(lambda x: "YES" if x else "")

                styled = display_df.style.apply(color_execution_column, axis=None)

                st.subheader(f"🚀 Primary Universe — {len(display_df)} Filtered Results")
                st.dataframe(
                    styled,
                    hide_index=True,
                    use_container_width=True
                )

        progress_bar.empty()
        end_time = time.time()
        elapsed = end_time - start_time
        st.write(f"⏱️ End Time: {pd.Timestamp.now()}")
        st.write(f"⚡ Total Runtime: {elapsed:.2f} seconds")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Model execution failed: {str(e)}")
        st.exception(e)


# ---------------------------------------------------------
# RENDER STORED RESULTS
# ---------------------------------------------------------
elif filtered_results is not None:
    if structural_results is not None:
        st.markdown(f"**Structural Universe (Daily):** {len(structural_results)} tickers")
        st.markdown(f"**User Visual Filters Applied (Stored):** {len(filtered_results)} tickers shown")
        st.markdown(f"**Tickers Passed to Page 2 (Structural):** {len(structural_results)} tickers")

    st.subheader(f"🚀 Primary Universe — {len(filtered_results)} Stored Results")

    st.dataframe(
        filtered_results.style.apply(color_execution_column, axis=None),
        hide_index=True,
        use_container_width=True
    )

