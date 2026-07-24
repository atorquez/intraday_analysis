import sys
import os

# ---------------------------------------------------------
# ROBUST PATH RESOLUTION (Works locally + Streamlit Cloud)
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = parent folder of /pages
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# utils folder inside project root
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")

# Add utils folder to sys.path
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

print(">>> CURRENT_DIR:", CURRENT_DIR)
print(">>> PROJECT_ROOT:", PROJECT_ROOT)
print(">>> UTILS_PATH:", UTILS_PATH)

# ---------------------------------------------------------
# IMPORT MODULES
# ---------------------------------------------------------
import importlib
import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)
print(">>> USING FILE:", v3.__file__)

rank_universe = v3.rank_universe

import streamlit as st
import pandas as pd
import yfinance as yf

# Import directly from utils folder
from data_fetch import load_universe, get_universe_source

st.set_page_config(layout="wide")
st.caption("Version: 2026-07-21")
st.title("📈 Structural Engine Page1")

# ---------------------------------------------------------
# IMPORT MODULES
# ---------------------------------------------------------
import importlib
import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)
print(">>> USING FILE:", v3.__file__)

import analysis.intraday_ranker_v3 as v3
rank_universe = v3.rank_universe

import streamlit as st
import pandas as pd
import yfinance as yf

# Import directly from utils folder
from data_fetch import load_universe, get_universe_source

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
    st.write(f"⏱️ Start Time: {pd.Timestamp.now()}")

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

                # Add trend indicators to display
                display_df["SP500_Trend"] = sp500_trend
                display_df["NASDAQ_Trend"] = nasdaq_trend
                display_df["Market_Regime"] = regime

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
#------------------------------------------------------------------
# Summary and Definitions
#------------------------------------------------------------------
st.markdown("""
---
# Worked Examples — SPC Intraday Ranker v3

This section provides **realistic examples** of how each major calculation behaves.  
It is designed to help traders understand the engine without needing external resources.
---

# 📘 Example 1 — Trend Stack (EMA9 > EMA20 > EMA50)

Assume daily closing prices produce:
- EMA9 = 152.40  
- EMA20 = 150.10  
- EMA50 = 147.80  

Since:
152.40 > 150.10 > 147.80

The trend stack is:
**UP Trend (Bullish Structural Alignment)**

If instead:
EMA9 = 148.20  
EMA20 = 149.10  
EMA50 = 150.30  

Then:
148.20 < 149.10 < 150.30

Trend stack is:
**DOWN Trend (Bearish Structural Alignment)**

If EMAs are mixed:
EMA9 = 150.0  
EMA20 = 149.8  
EMA50 = 150.2  

Trend stack is:
**FLAT (No Alignment)**
---

# 📘 Example 2 — EMA Slope Alignment

Assume EMA20 values over last 5 days:
- Day 1: 148.00  
- Day 5: 149.20  

Slope = 149.20 − 148.00 = **+1.20**

Interpretation:
- Positive slope → **medium‑term upward velocity**
- Negative slope → **medium‑term downward pressure**
---

# 📘 Example 3 — PCA1 (Daily Momentum Cluster)

Assume PCA1 is computed from:
- RSI = 58  
- Bollinger Width = 0.12  
- ROC = +1.8%  
- StochK = 72  
- EMA curvature = positive  
- VWAP distance = positive  
- Volume delta = positive  

If PCA1 = **+0.43**, then:
**Daily momentum cluster is supportive.**

If PCA1 = **−0.27**, then:
**Daily momentum is weakening.**
---

# 📘 Example 4 — ATR% (Volatility Availability)

Assume:
- ATR (14‑day) = 2.40  
- Current price = 120.00  

ATR% = (2.40 / 120.00) × 100 = **2.0%**

Interpretation:
- ATR% > 2% → **good range availability**
- ATR% < 1% → **low range, friction risk**
---

# 📘 Example 5 — RVOL (Relative Volume)

Assume:
- Current volume = 8.2M  
- 20‑day average volume = 5.1M  

RVOL = 8.2 / 5.1 = **1.61**

Interpretation:
- RVOL > 1.5 → **institutional participation**
- RVOL < 1.0 → **weak participation**
---

# 📘 Example 6 — Intraday EMA Micro‑Trend

Assume 1‑minute EMAs:
- EMA9 = 61.52  
- EMA20 = 61.48  

Since EMA9 > EMA20:
**Micro‑trend is upward.**

If EMA9 < EMA20:
**Micro‑trend is downward.**

If EMA9 ≈ EMA20:
**Compression (Crossing Soon).**
---

# 📘 Example 7 — VWAP Reclaim

Assume:
- Price = 62.10  
- VWAP = 61.85  

Price > VWAP → **institutional support**  
Price < VWAP → **liquidity drag**
---

# 📘 Example 8 — PCA1_slope (Intraday Acceleration)

Assume PCA1_slope = **+0.18**

Interpretation:
- Positive → **intraday acceleration forming**
- Negative → **momentum fading**
---

# 📘 Example 9 — Universe Source (SP500 vs NASDAQ1000)

Examples:
- FCX → **SP500**  
- AMD → **NASDAQ1000**  
- PEP → **SP500**  
- CRWD → **NASDAQ1000**

This determines **which index trend matters**.
---

# 📘 Example 10 — SP500 / NASDAQ Trend Indicators

Assume SP500 EMAs:
- EMA9 = 5520  
- EMA20 = 5512  
- EMA50 = 5498  
- EMA20 slope = +4.2  

SP500 Trend = **Bullish**

Assume NASDAQ EMAs:
- EMA9 = 17920  
- EMA20 = 17940  
- EMA50 = 17980  
- EMA20 slope = −6.1  

NASDAQ Trend = **Bearish**
---

# 📘 Example 11 — Market Regime Classification

Given:
- SP500 Trend = Choppy  
- NASDAQ Trend = Bearish  

Market Regime = **Bearish**

Given:
- SP500 Trend = Bullish  
- NASDAQ Trend = Bullish  

Market Regime = **Trending**

Given:
- SP500 Trend = Bullish  
- NASDAQ Trend = Bearish  

Market Regime = **Mixed**
---

# 📘 Example 12 — Execution Labels

### **Watch List Example**
- Trend stack aligned  
- PCA1 > 0  
- PCA1_slope > 0  
- EMA slopes positive  

### **Crossing Soon Example**
- EMA9 ≈ EMA20  
- Compression forming  
- PCA1 positive  

### **Not Watch List Example**
- Trend aligned  
- PCA1 positive  
- PCA1_slope negative  

### **Setup Only Example**
- Trend not aligned  
- PCA1 weak  
- No intraday acceleration  
---

# 📘 Example 13 — Profit Target Expectations (Based on Regime)

### **Trending Regime**
Expect: **2–4%** expansions

### **Mixed Regime**
Expect: **1–2%** compression → expansion

### **Choppy Regime**
Expect: **1–1.5%** small pops

### **Bearish Regime**
Expect: **0.8–1.2%** defensive scalps
---

# 📘 Example 14 — FCX Example (Real Case)

Assume:
- FCX Universe = SP500  
- SP500 Trend = Choppy  
- NASDAQ Trend = Bearish  
- Market Regime = Mixed/Bearish  
- FCX EMA9/EMA20 compressed  
- PCA1 positive  
- PCA1_slope positive  

Interpretation:
- **Compression → expansion setup**
- **Expect 1–1.5% profit window**
- **Exit on expansion peak**

This matches the FCX trade executed earlier.

---
""")
