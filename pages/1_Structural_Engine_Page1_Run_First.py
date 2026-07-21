# ==============================================================================
# PAGE 1 — INTRADAY ANALYSIS (FULL PERSISTENT VERSION)
# PATCHED VERSION — Structural universe passed to Page 2
# ==============================================================================

import streamlit as st
import pandas as pd
from analysis.intraday_ranker import rank_universe
from utils.data_fetch import load_universe

st.set_page_config(layout="wide")
st.caption("Version: 2026-07-20")
st.title("📈 Strong Tickers Analysis")

# ---------------------------------------------------------
# CACHED WRAPPER FOR RANK_UNIVERSE (FIXED: Prevents re-download)
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_rank_universe(tickers_tuple, buy_zone_percentile=0.15):
    """
    Cached wrapper for rank_universe.
    tickers_tuple must be a tuple (hashable) for Streamlit caching.
    """
    return rank_universe(list(tickers_tuple), buy_zone_percentile)


# ---------------------------------------------------------
# COLOR HELPER (FIXED: Defined once, not duplicated)
# ---------------------------------------------------------
def color_execution_column(df):
    """Apply background colors to the Execution column."""
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    if "Execution" in df.columns:
        style_df["Execution"] = [
            "background-color: #4CAF50; color: white;" if v == "Ready"
            else "background-color: #FFC107; color: black;" if v == "Crossing Soon"
            else "background-color: #FF9800; color: white;" if v == "False Ready"
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
# PRICE FILTERS (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🔍 Price Filter")
min_price = st.number_input("Minimum Price", value=5.0, key="intraday_min_price")
max_price = st.number_input("Maximum Price", value=50.0, key="intraday_max_price")

# ---------------------------------------------------------
# FILTER PANEL (BEFORE RUN)
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
    ["Ready", "False Ready", "Crossing Soon", "Setup Only"],
    default=["Ready"],
    key="intraday_execution_filter"
)

# ---------------------------------------------------------
# RUN RANKER
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model", key="intraday_run_button")

if run_model:
    progress_bar = st.progress(0, text="Loading universe...")

    try:
        base_universe = load_universe()
        if not base_universe:
            st.error("The source stock universe list returned empty.")
            st.stop()

        progress_bar.progress(0.1, text=f"Scanning {len(base_universe)} tickers...")

        ranking = cached_rank_universe(tuple(base_universe))

        progress_bar.progress(0.6, text="Applying filters...")

        if ranking is None or ranking.empty:
            st.warning("No stock configurations satisfied the technical requirements.")
        else:
            # Structural universe summary (before visual filters)
            st.markdown(
                f"**Structural Universe (Daily):** {len(ranking)} tickers"
            )

            # Apply visual filters for display only
            filtered = ranking.copy()

            filtered = filtered[
                (filtered["Close"] >= min_price) &
                (filtered["Close"] <= max_price)
            ]

            if buy_signal_filter:
                filtered = filtered[
                    filtered["BuyZone_Heatmap"].isin(buy_signal_filter)
                ]

            if execution_filter:
                filtered = filtered[
                    filtered["Execution"].isin(execution_filter)
                ]

            # Store structural universe for Page 2
            st.session_state["intraday_filtered_results"] = ranking

            # Store visual subset for Page 1 display
            st.session_state["intraday_visual_results"] = filtered

            # Visual universe summary
            st.markdown(
                f"**User Visual Filters Applied:** {len(filtered)} tickers shown"
            )
            st.markdown(
                f"**Tickers Passed to Page 2 (Structural):** {len(ranking)} tickers"
            )

            if filtered.empty:
                st.info("No companies matched the selected filters.")
            else:
                st.subheader(f"🚀 Primary Universe — {len(filtered)} Filtered Results")

                st.dataframe(
                    filtered.style.apply(color_execution_column, axis=None),
                    hide_index=True,
                    use_container_width=True
                )

        progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        st.error(f"Model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# RENDER STORED FILTERED RESULTS WHEN USER RETURNS TO PAGE
# ---------------------------------------------------------
elif filtered_results is not None:
    if structural_results is not None:
        st.markdown(
            f"**Structural Universe (Daily):** {len(structural_results)} tickers"
        )
        st.markdown(
            f"**User Visual Filters Applied (Stored):** {len(filtered_results)} tickers shown"
        )
        st.markdown(
            f"**Tickers Passed to Page 2 (Structural):** {len(structural_results)} tickers"
        )

    st.subheader(f"🚀 Primary Universe — {len(filtered_results)} Stored Results")

    st.dataframe(
        filtered_results.style.apply(color_execution_column, axis=None),
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------------
# SUMMARY AND DEFINITIONS (PATCHED & SYNCHRONIZED VERSION)
# ---------------------------------------------------------

# ---------------------------------------------------------
# SUMMARY AND DEFINITIONS (PATCHED & SYNCHRONIZED VERSION)
# ---------------------------------------------------------
st.markdown("### 📘 Intraday Ranker — Parameter Definitions")

st.markdown("""
---
# Executive Summary — Intraday Readiness Model

The Model is a multi‑factor, PCA‑enhanced intraday scanning system designed to identify statistically favorable trading locations during the trading session. It integrates daily historical structure with real‑time intraday momentum, volatility, and participation signals to produce a ranked universe of tickers and classify each into two orthogonal dimensions:

1. **BuyZone_Heatmap** — Value-proximity structural alignment  
2. **Execution** — Intraday microstructural actionability  

The model is intentionally **location‑based**, not predictive. It identifies *where* structural advantage exists, while the trader determines *when* to execute through visual confirmation and chart monitoring.

---

## Two‑Stage Scientific Framework

### **Stage 1 — Daily Structure (Value + Trend Foundation)**  
Using 3 months of daily OHLCV data, the model computes macro baseline conditions:
- **Trend Alignment:** Stacked orientation checks (`EMA9 > EMA20 > EMA50`)  
- **Volatility (ATR%):** Daily average percentage movement relative to price  
- **Participation (RVOL):** Immediate volume scaled against a 20-day simple moving average  
- **Opening Pressure (Gap%):** Premium or discount valuation at the opening bell  
- **Value Floors:** Lower rolling price percentiles to map value zone margins  

---

### **Stage 2 — Intraday PCA Engine (Momentum + Volatility + Participation)**  
Using intraday‑sensitive indicators, the model applies `StandardScaler` + `PCA` to compress complex variables into clean, actionable alpha metrics. By converting absolute price strings into stationary spreads (such as `Close - VWAP` and `EMA9 - EMA20`), the engine extracts real-time acceleration shifts without data leakage or structural drift.

---

## Decision Layer — BuyZone Heatmap Matrix

The engine maps price location to structural historical baselines to define the **BuyZone_Heatmap** category:
- **DEEP_VALUE_ZONE:** Asset is trading at or below its 10-period daily rolling value floor. High institutional accumulation profile.
- **MID_VALUE_ZONE:** Price is securely stabilized near intermediate short-term value support bands.
- **NEAR_VALUE_ZONE:** Within a 2% proximity threshold of structural value boundaries.
- **EXTENDED_ZONE:** Price is extended aggressively above historical value. High chase risk.

This dimension answers:  
**"Is the ticker currently in a statistically favorable location?"**

---

## — Intraday Execution Readiness Status

Execution readiness evaluates whether an asset is **actionable right now on the 1-minute chart**, combining moving average vectors, Volume-Weighted Average Price (VWAP) claims, volume tracking, and PCA slopes.

The model classifies entries into:
- **Ready:** Trend vectors are bullish, price has reclaimed its VWAP line, volume expansion is supportive, and the PCA slope is accelerating upward. Fully actionable.
- **Crossing Soon:** Momentum indicators are expanding, and short-term EMA or VWAP reclaims are actively forming.
- **Intraday False Ready:** Visual indicators look bullish, but the entry lacks necessary relative volume (RVOL) or volatility support. High trap risk.
- **Setup Only:** Daily trend structure remains healthy, but 1-minute short-term momentum or institutional participation is absent. Watchlist tracking only.
- **UNKNOWN:** Conflicting indicator signals or insufficient chronological data.

This dimension answers:  
**"Is the ticker structurally actionable right now?"**

---

## Trader Workflow

The trader runs the model periodically throughout the session (e.g., at key structural windows like **09:45**, **10:30**, or **11:30 AM EST**), isolates top-ranked **DEEP_VALUE_ZONE + Ready** candidates, and opens their E*TRADE execution terminal to visually confirm:
1. Clean 1-minute EMA9 reclaims  
2. Intraday higher-low structural chart prints  
3. Ascending multi-bar RVOL bars  
4. Stabilization above the dynamic intraday VWAP line  

This creates a disciplined intraday workflow where **the model isolates structural location**, and **the trader confirms tactical timing**.

---

## 🏛 Technical Parameter Reference Matrix

### **Trend Scoring Metric Formula (0–70 Points)**
Every stock passing basic filters receives a baseline score of **+5 points**. Additional weighting scales up according to institutional trend strength:
- `EMA9 > EMA20` (Short-term velocity) → **+10 Points**
- `EMA20 > EMA50` (Macro structural backing) → **+10 Points**
- `RVOL > 2.0` (Significant institutional participation spike) → **+20 Points**
- `2% <= Gap% <= 5%` (Optimal momentum morning opening pressure) → **+10 Points**
- `ATR% > 5%` (Sufficient daily range availability to clear friction costs) → **+15 Points**

**Total Score = Trend (0–20) + RVOL (0–20) + Gap (0–10) + ATR (0–15) + Baseline (5)**

---

# 📘 Glossary — Core Acronym Definitions

### **OHLCV — Open, High, Low, Close, Volume**
Fundamental raw financial price array data sets used to compute technical indicators.

### **VWAP — Volume‑Weighted Average Price**
Represents the true average cost basis for all market participants throughout the day. It acts as the ultimate line of intraday structural trend validation.

### **RVOL — Relative Volume**
Compares current volume speed against historical averages. High RVOL signals institutional block execution; low RVOL alerts you to erratic, low-liquidity retail noise.

### **PCA — Principal Component Analysis**
An advanced statistical tool that reduces multiple dimensions of data down to a single core index. This index tracks the hidden acceleration of momentum and volatility without indicators lagging behind the current print.

### **ATR — Average True Range**
A reliable volatility index that tracks true structural trading ranges, accounting for overnight gaps and session flushes. Used to establish dynamic trade sizing and stop boundaries.
""")

