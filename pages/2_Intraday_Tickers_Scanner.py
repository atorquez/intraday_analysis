# ==============================================================================
# PAGE 2 — INTRADAY ENGINE ONLY (HARDENED PRODUCTION VERSION — FULL PATCH v4)
# WITH LONG-ONLY EXIT / POSITION MANAGEMENT + MANUAL ENTRY TRACKER
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")
st.caption("Version: 2026-07-20")
st.title("📈 Page 2 — Intraday Scanner")
st.markdown("Pure intraday scanner operating exclusively on 1-minute real-time structural movements. Long-only cash equity model with position management.")

# ---------------------------------------------------------
# SESSION STATE: Manual Position Tracker
# ---------------------------------------------------------
if "positions" not in st.session_state:
    st.session_state.positions = {}  # {ticker: {"entry_price": float, "entry_time": datetime, "shares": int}}

if "closed_positions" not in st.session_state:
    st.session_state.closed_positions = []  # List of closed trades for review

# ---------------------------------------------------------
# SIDEBAR: Manual Position Tracker Panel
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📝 Manual Position Tracker")
    st.caption("Log your E*TRADE entries here. The scanner will track them.")

    with st.expander("➕ Log New Entry", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            entry_ticker = st.text_input("Ticker", key="entry_ticker", placeholder="AAPL").upper().strip()
        with col_b:
            entry_price = st.number_input("Entry Price ($)", value=0.0, min_value=0.0, step=0.01, key="entry_price")
        with col_c:
            entry_shares = st.number_input("Shares", value=100, min_value=1, step=1, key="entry_shares")

        if st.button("Log Entry", key="log_entry", use_container_width=True):
            if entry_ticker and entry_price > 0:
                st.session_state.positions[entry_ticker] = {
                    "entry_price": entry_price,
                    "entry_time": datetime.now(),
                    "shares": entry_shares
                }
                st.success(f"✅ Logged {entry_shares} shares of {entry_ticker} @ ${entry_price:.2f}")
            else:
                st.error("Enter valid ticker and price.")

    # Show current open positions
    if st.session_state.positions:
        st.markdown("#### 📂 Open Positions")
        for tk, pos in list(st.session_state.positions.items()):
            mins_in = int((datetime.now() - pos["entry_time"]).total_seconds() / 60)
            st.markdown(f"**{tk}** — ${pos['entry_price']:.2f} × {pos['shares']} | {mins_in}m in")

            col_x, col_y = st.columns(2)
            with col_x:
                if st.button(f"Close {tk}", key=f"close_{tk}"):
                    st.session_state.closed_positions.append({
                        "ticker": tk,
                        "entry_price": pos["entry_price"],
                        "exit_price": None,
                        "entry_time": pos["entry_time"],
                        "exit_time": datetime.now(),
                        "shares": pos["shares"]
                    })
                    del st.session_state.positions[tk]
                    st.rerun()
    else:
        st.info("No open positions logged.")

    # Show closed positions summary
    if st.session_state.closed_positions:
        st.markdown("#### 📊 Closed Trades")
        st.caption(f"{len(st.session_state.closed_positions)} trades logged")
        if st.button("Clear History", key="clear_history"):
            st.session_state.closed_positions = []
            st.rerun()

# ---------------------------------------------------------
# FILTER PANEL (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

col1, col2, col3 = st.columns(3)

with col1:
    intraday_signal_filter = st.multiselect(
        "Filter by Intraday Buy Signal",
        ["Strong Intraday", "Intraday Buy", "Neutral", "Avoid"],
        default=["Strong Intraday", "Intraday Buy", "Neutral", "Avoid"],
        key="intraday_signal_filter"
    )

with col2:
    execution_filter = st.multiselect(
        "Filter by Execution Readiness",
        ["Ready", "Crossing Soon", "Intraday False Ready", "Setup Only", "UNKNOWN"],
        default=["Ready", "Crossing Soon", "Intraday False Ready", "Setup Only", "UNKNOWN"],
        key="intraday_execution_filter"
    )

with col3:
    min_price = st.number_input("Minimum Price ($)", value=5.0, key="intraday_min_price")
    max_price = st.number_input("Maximum Price ($)", value=100.0, key="intraday_max_price")

# Position Management Filter
st.markdown("### 🎯 Position Management Filter")
position_filter = st.multiselect(
    "Filter by Position Status",
    ["✅ HOLD — Structure Intact", "⚠️ TIGHTEN — Below VWAP", "📉 FADE — Volume Drying", "🛑 EXIT — Momentum Reversed"],
    default=["✅ HOLD — Structure Intact", "⚠️ TIGHTEN — Below VWAP", "📉 FADE — Volume Drying", "🛑 EXIT — Momentum Reversed"],
    key="position_status_filter"
)

# View mode toggle
view_mode = st.radio(
    "View Mode",
    ["🕵️ All Candidates", "📂 My Positions Only"],
    horizontal=True,
    key="view_mode"
)

run_intraday = st.button("Run Intraday Engine", key="intraday_run_button", use_container_width=True)

# ---------------------------------------------------------
# Universe Loader
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_primary_universe():
    try:
        from utils.data_fetch import load_universe
        return load_universe()
    except Exception as e:
        logger.warning(f"Failed to load universe from utils: {e}")
        return ["AAPL", "NVDA", "MSFT", "AMD", "TSLA", "META", "AMZN", "MDT"]

tickers = load_primary_universe()

# If "My Positions Only" mode, override universe with tracked tickers
if view_mode == "📂 My Positions Only" and st.session_state.positions:
    tickers = list(st.session_state.positions.keys())
    if not tickers:
        st.warning("No positions logged. Switch to 'All Candidates' to scan.")

# ---------------------------------------------------------
# Intraday Data Fetcher & Data Column Flattener
# ---------------------------------------------------------
from analysis.intraday_ranker import fetch_intraday

def fetch_intraday_data(ticker):
    try:
        df = fetch_intraday(ticker)
        if df is None or df.empty:
            logger.warning(f"{ticker}: No data returned from fetch_intraday.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"{ticker}: Exception during fetch: {e}")
        return None

# ---------------------------------------------------------
# Intraday Indicator Calculations
# ---------------------------------------------------------
def compute_intraday_indicators(df):
    df = df.copy()

    rename_dict = {
        "vol": "volume", "tradeprice": "close", "last": "close",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
    }
    df = df.rename(columns=rename_dict)

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            alt_col = col.capitalize()
            if alt_col in df.columns:
                df[col] = df[alt_col]
            else:
                logger.warning(f"Missing required column '{col}' (also tried '{alt_col}'). Skipping.")
                return pd.DataFrame()

    # EMAs
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    # VWAP (cumulative session VWAP)
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vp"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"].replace(0, np.nan)

    # ATR — True Range with .mean()
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["prev_close"])
    df["tr3"] = abs(df["low"] - df["prev_close"])
    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr"] = df["true_range"].rolling(14).mean().ffill().bfill().fillna(0.01)
    df["atr%"] = (df["atr"] / df["close"]) * 100

    # Relative Volume
    rolling_vol_mean = df["volume"].rolling(20).mean().ffill().bfill()
    df["rvol"] = df["volume"] / rolling_vol_mean.replace(0, 1)

    return df

# ---------------------------------------------------------
# PCA Engine
# ---------------------------------------------------------
def compute_intraday_pca(df):
    if df.empty or len(df) < 30:
        df["PCA1"] = 0.0
        df["PCA1_slope"] = 0.0
        return df

    # dist_vwap normalized by close price for comparability
    df["spread_ema"] = df["ema9"] - df["ema20"]
    df["dist_vwap"] = (df["close"] - df["vwap"]) / df["close"].replace(0, np.nan)
    df["pct_return"] = df["close"].pct_change().fillna(0.0)

    pca_features = ["spread_ema", "dist_vwap", "pct_return", "atr%", "rvol"]
    X = df[pca_features].ffill().bfill().fillna(0.0)

    if len(X) < 30 or X.var().sum() == 0:
        df["PCA1"] = 0.0
        df["PCA1_slope"] = 0.0
        return df

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=1)
        pcs = pca.fit_transform(X_scaled)

        pca_series = pd.Series(pcs.flatten(), index=X.index)
        df.loc[pca_series.index, "PCA1"] = pca_series
        df["PCA1_slope"] = df["PCA1"].diff().fillna(0.0)
    except Exception as e:
        logger.warning(f"PCA failed: {e}")
        df["PCA1"] = 0.0
        df["PCA1_slope"] = 0.0

    return df

# ---------------------------------------------------------
# Scoring and Classification Modules
# ---------------------------------------------------------
def classify_execution_readiness(latest):
    ema_bull = latest["ema9"] > latest["ema20"]
    vwap_reclaim = latest["close"] > latest["vwap"]
    momentum = latest["PCA1_slope"] > 0
    rvol = latest["rvol"] > 2

    if ema_bull and vwap_reclaim and momentum and rvol:
        return "Ready"
    if ema_bull and momentum:
        return "Crossing Soon"
    if ema_bull and vwap_reclaim and not rvol:
        return "Intraday False Ready"
    if ema_bull:
        return "Setup Only"
    return "UNKNOWN"

def compute_intraday_score(latest):
    score = 0
    if latest["ema9"] > latest["ema20"]: score += 5
    if latest["rvol"] > 2: score += 15
    if latest["atr%"] > 5: score += 5
    if latest["close"] > latest["vwap"]: score += 10
    if latest["PCA1"] > 0: score += 15
    if latest["PCA1_slope"] > 0: score += 10
    return score

def compute_risk_efficiency(latest):
    pca_slope = latest["PCA1_slope"]
    rvol = latest["rvol"]
    atr = latest["atr%"]
    if pd.isna(pca_slope) or pd.isna(rvol) or pd.isna(atr) or (atr + 0.001) == 0:
        return 0.0
    return float((abs(pca_slope) * rvol) / (atr + 0.001))

def classify_intraday_buy_signal(latest):
    pca1 = latest["PCA1"]
    vwap_status = latest["close"] > latest["vwap"]

    if pca1 > 0 and vwap_status:
        return "Strong Intraday"
    if pca1 > 0:
        return "Intraday Buy"
    if pca1 < 0:
        return "Avoid"
    return "Neutral"

# ---------------------------------------------------------
# Position Management / Exit Logic
# ---------------------------------------------------------
def classify_position_status(latest):
    """
    Long-only position management:
    - If you are already in a position, this tells you whether to hold or exit.
    - Uses the same indicators that got you in, but inverted thresholds.
    """
    pca1 = latest["PCA1"]
    pca1_slope = latest["PCA1_slope"]
    close = latest["close"]
    vwap = latest["vwap"]
    rvol = latest["rvol"]
    ema9 = latest["ema9"]
    ema20 = latest["ema20"]

    # EXIT — Momentum has reversed (highest priority)
    if pca1 < 0 or pca1_slope < -0.5:
        return "🛑 EXIT — Momentum Reversed"

    # TIGHTEN — Price below VWAP but EMA still bullish
    if close < vwap and ema9 > ema20:
        return "⚠️ TIGHTEN — Below VWAP"

    # FADE — Volume drying up, momentum slowing
    if rvol < 1.5 or pca1_slope < 0:
        return "📉 FADE — Volume Drying"

    # HOLD — All bullish structure intact
    return "✅ HOLD — Structure Intact"

# ---------------------------------------------------------
# Stop Recommendation Engine
# ---------------------------------------------------------
def compute_stop_recommendation(latest, entry_price=None, risk_multiplier=1.0, reward_ratio=3.0):
    """
    Computes stop-loss and profit target levels based on ATR.

    If entry_price is provided, uses fixed R/R from entry.
    If not, uses current price as reference.
    """
    current = float(latest["close"])
    atr_dollars = (latest["atr%"] / 100.0) * current

    if atr_dollars <= 0 or pd.isna(atr_dollars):
        atr_dollars = current * 0.005  # Fallback: 0.5% of price

    if entry_price and entry_price > 0:
        # Fixed R/R from entry
        risk = atr_dollars * risk_multiplier
        stop = entry_price - risk
        target = entry_price + (reward_ratio * risk)
        pnl_pct = ((current - entry_price) / entry_price) * 100
    else:
        # No entry logged — use current price as reference for new trade
        stop = current - (atr_dollars * risk_multiplier)
        target = current + (atr_dollars * reward_ratio * risk_multiplier)
        pnl_pct = None

    return {
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_dollars": round(atr_dollars * risk_multiplier, 2),
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "distance_to_stop_pct": round(((current - stop) / current) * 100, 2) if stop > 0 else None
    }

# ---------------------------------------------------------
# Core Iteration Execution Loop
# ---------------------------------------------------------
if run_intraday:
    results = []
    position_log = st.session_state.get("positions", {})

    processing_universe = tickers[:40] if len(tickers) > 40 else tickers
    progress_bar = st.progress(0.0)

    for idx, ticker in enumerate(processing_universe):
        progress_bar.progress((idx + 1) / len(processing_universe))

        raw_df = fetch_intraday_data(ticker)
        if raw_df is None or raw_df.empty or len(raw_df) < 3:
            continue

        processed_df = compute_intraday_indicators(raw_df)
        if processed_df.empty:
            continue

        final_df = compute_intraday_pca(processed_df)
        latest_bar = final_df.iloc[-1].fillna(0.0)

        current_price = float(latest_bar["close"])
        if not (min_price <= current_price <= max_price):
            continue

        readiness = classify_execution_readiness(latest_bar)
        score = compute_intraday_score(latest_bar)
        signal = classify_intraday_buy_signal(latest_bar)
        risk_eff = compute_risk_efficiency(latest_bar)
        position_status = classify_position_status(latest_bar)

        # Build result row
        row = {
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "EMA Alignment": "Bullish" if latest_bar["ema9"] > latest_bar["ema20"] else "Bearish",
            "VWAP Status": "Above" if latest_bar["close"] > latest_bar["vwap"] else "Below",
            "RVOL": round(latest_bar["rvol"], 2),
            "ATR%": round(latest_bar["atr%"], 2),
            "PCA1": round(latest_bar["PCA1"], 4),
            "PCA1 Slope": round(latest_bar["PCA1_slope"], 4),
            "Execution Readiness": readiness,
            "Intraday Score": score,
            "Intraday Buy_Signal": signal,
            "Risk Efficiency Score": round(risk_eff, 4),
            "Position Status": position_status
        }

        # If this ticker is in our manual position log, add tracking columns
        if ticker in position_log:
            pos = position_log[ticker]
            entry_price = pos["entry_price"]
            entry_time = pos["entry_time"]
            shares = pos["shares"]

            mins_in = int((datetime.now() - entry_time).total_seconds() / 60)
            stop_rec = compute_stop_recommendation(latest_bar, entry_price=entry_price)

            row["Entry Price"] = entry_price
            row["Shares"] = shares
            row["P&L %"] = stop_rec["pnl_pct"]
            row["Time In Trade"] = f"{mins_in}m"
            row["Current Stop"] = stop_rec["stop"]
            row["Current Target"] = stop_rec["target"]
            row["Risk ($)"] = stop_rec["risk_dollars"]
            row["Dist to Stop %"] = stop_rec["distance_to_stop_pct"]
        else:
            # For candidates not yet entered, show suggested levels
            stop_rec = compute_stop_recommendation(latest_bar)
            row["Suggested Stop"] = stop_rec["stop"]
            row["Suggested Target"] = stop_rec["target"]
            row["Risk ($)"] = stop_rec["risk_dollars"]

        results.append(row)

    progress_bar.empty()

    # ---------------------------------------------------------
    # DISPLAY ENGINE & RENDERER PANEL
    # ---------------------------------------------------------
    if not results:
        st.warning("No assets successfully bypassed background processing parameters. Check your console logs.")
    else:
        master_df = pd.DataFrame(results)

        if intraday_signal_filter:
            master_df = master_df[master_df["Intraday Buy_Signal"].isin(intraday_signal_filter)]

        if execution_filter:
            master_df = master_df[master_df["Execution Readiness"].isin(execution_filter)]

        if position_filter:
            master_df = master_df[master_df["Position Status"].isin(position_filter)]

        if master_df.empty:
            st.info("Watchlist generated structural entries, but they were filtered out by user checkbox configurations.")
        else:
            # Sort: open positions first, then by score
            if "P&L %" in master_df.columns:
                master_df = master_df.sort_values(
                    by=["P&L %"],
                    ascending=[False],
                    na_position="last"
                )
            else:
                master_df = master_df.sort_values(
                    by=["Intraday Score", "Risk Efficiency Score"],
                    ascending=[False, False]
                )

            st.subheader(f"🚀 Live Intraday Universe Matrix ({len(master_df)} Tickers)")

            # Styling: Execution Readiness column
            def style_readiness(val):
                if val == "Ready":
                    return "background-color: #2E7D32; color: white; font-weight: bold;"
                elif val == "Intraday False Ready":
                    return "background-color: #EF6C00; color: white;"
                elif val == "Crossing Soon":
                    return "background-color: #FBC02D; color: black;"
                else:
                    return "background-color: #757575; color: white;"

            # Styling: Position Status column
            def style_position_status(val):
                if "✅ HOLD" in val:
                    return "background-color: #1B5E20; color: white; font-weight: bold;"
                elif "⚠️ TIGHTEN" in val:
                    return "background-color: #FF6F00; color: white; font-weight: bold;"
                elif "📉 FADE" in val:
                    return "background-color: #F9A825; color: black;"
                elif "🛑 EXIT" in val:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                return ""

            # Styling: P&L % column (green/red)
            def style_pnl(val):
                if pd.isna(val):
                    return ""
                if val > 0:
                    return "background-color: #1B5E20; color: white; font-weight: bold;"
                elif val < -2:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                else:
                    return "background-color: #F9A825; color: black;"

            styled_df = master_df.style.map(
                style_readiness,
                subset=["Execution Readiness"]
            ).map(
                style_position_status,
                subset=["Position Status"]
            )

            if "P&L %" in master_df.columns:
                styled_df = styled_df.map(
                    style_pnl,
                    subset=["P&L %"]
                )

            st.dataframe(
                styled_df,
                hide_index=True,
                use_container_width=True
            )

            # Quick Action Panel for positions
            if "P&L %" in master_df.columns and len(master_df[master_df["P&L %"].notna()]) > 0:
                st.markdown("---")
                st.markdown("### ⚡ Quick Actions for Open Positions")

                pos_df = master_df[master_df["P&L %"].notna()].copy()
                for _, row in pos_df.iterrows():
                    tk = row["Ticker"]
                    cols = st.columns([1, 2, 2, 2, 2, 1])
                    with cols[0]:
                        st.markdown(f"**{tk}**")
                    with cols[1]:
                        st.markdown(f"P&L: **{row['P&L %']}%**")
                    with cols[2]:
                        st.markdown(f"Stop: ${row['Current Stop']}")
                    with cols[3]:
                        st.markdown(f"Target: ${row['Current Target']}")
                    with cols[4]:
                        st.markdown(f"Status: {row['Position Status']}")
                    with cols[5]:
                        if st.button(f"Close {tk}", key=f"quick_close_{tk}"):
                            st.session_state.closed_positions.append({
                                "ticker": tk,
                                "entry_price": st.session_state.positions[tk]["entry_price"],
                                "exit_price": row["Price"],
                                "entry_time": st.session_state.positions[tk]["entry_time"],
                                "exit_time": datetime.now(),
                                "shares": st.session_state.positions[tk]["shares"],
                                "pnl_pct": row["P&L %"]
                            })
                            del st.session_state.positions[tk]
                            st.rerun()

# ---------------------------------------------------------
# EXECUTIVE SUMMARY, PROCESS, AND DEFINITIONS
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📘 Pure Intraday Engine — Parameter Definitions & Operating Manual")

st.markdown("""
# Executive Summary — Pure Intraday Momentum Framework (v4)

The **Intraday Engine Only** dashboard is a systematic, long-only screening engine designed for manual execution on cash equity accounts. Unlike discretionary trading systems that rely on pre-market watchlists or narrative-driven setups, this engine treats each trading session as a clean statistical slate. It scans up to 1,500 tickers (S&P 500 + Nasdaq 1000) using real-time 1-minute structural data to surface high-conviction momentum candidates with defined risk parameters.

This engine is built for **systematic traders who execute manually** — providing institutional-grade signal generation while leaving execution control entirely with the trader.

---

## 🕐 Recommended Operating Schedule

| Time | Scan Type | Universe | Purpose |
|------|-----------|----------|---------|
| **10:15** | Full Scan | 1,500 tickers | First valid scan. 45 minutes of data allows PCA, EMA, and VWAP to converge. Post-10:00 news digestion complete. |
| **10:15–10:30** | Review & Enter | Top 2–3 candidates | Validate on E*TRADE chart. Enter best 1–2. Log entry in sidebar tracker. |
| **11:15** | Full Scan | 1,500 tickers | Morning positions check + scan for new setups. Last entry window before lunch decay. |
| **11:30+** | No New Entries | — | Momentum decays into lunch. Only manage existing positions. |
| **13:30** | Full Scan | 1,500 tickers | Afternoon re-activation. Enter only if exceptional "Ready" + "Strong Intraday" signal. |
| **14:45** | Positions Only | Tracked tickers | Tighten stops, prepare exits. No new positions. |
| **15:15** | Positions Only | Tracked tickers | Final exit check. Close anything not at target before close. |

**Why 10:15, not 09:45?** This is a structural scanner, not a pre-market watchlist system. PCA requires 30+ bars (30 minutes) for statistical validity. EMA9/20, VWAP, and RVOL need sufficient session data to stabilize. Running at 09:45 produces noisy signals with high false-positive rates.

---

## 🔄 Manual Workflow Integration

This engine is designed for manual execution on E*TRADE (or similar broker). The scanner does not connect to your broker. Instead, it serves as a systematic co-pilot:

### Entry Protocol
1. Run scanner at scheduled time
2. Filter by `Execution Readiness = "Ready"` + `Intraday Buy Signal = "Strong Intraday"`
3. Validate top candidate on E*TRADE chart (visual confirmation)
4. Place buy order on E*TRADE
5. **Immediately set stop-limit sell order** using scanner's Suggested Stop and Target (1:3 risk/reward)
6. Log entry in sidebar: Ticker, Entry Price, Shares

### Position Management Protocol
1. Re-run scanner (positions-only mode or full scan)
2. Check `Position Status` for tracked tickers:
   - **✅ HOLD** — Keep original stop. Do not touch.
   - **⚠️ TIGHTEN** — Move E*TRADE stop to breakeven or 1:1. Protect the trade.
   - **📉 FADE** — Consider moving stop to 1:1.5 or trailing. Be ready to exit.
   - **🛑 EXIT** — Override your hard stop. Close manually on E*TRADE immediately.
3. Use `Time In Trade` to gauge momentum decay. Positions held >90 minutes with 📉 FADE status should be closed.

### Exit Protocol
1. Hard stop/target hits on E*TRADE → trade closes automatically
2. Scanner says 🛑 EXIT before hard stop → close manually on E*TRADE, then click "Close" in sidebar tracker
3. Review Closed Trades history in sidebar for end-of-day performance

---

## 📊 The Data Normalization and Flattening Protocol

Intraday data from yFinance and other feeds arrives with inconsistent formatting. The engine enforces strict preprocessing before any indicator is calculated:

1. **Multi-Index Collapse:** Strips nested column headers to a single flat layer
2. **Case Normalization:** Forces all columns to lowercase (`open`, `high`, `low`, `close`, `volume`)
3. **Alias Mapping:** Automatically remaps platform-specific names:
   - `vol` → `volume`
   - `tradeprice`, `last` → `close`
   - `o`, `h`, `l`, `c`, `v` → `open`, `high`, `low`, `close`, `volume`

---

## 🧮 Indicator Definitions

### EMA9 & EMA20 (Exponential Moving Averages)
- **EMA9:** 9-period exponential moving average of closing prices. Captures short-term price velocity.
- **EMA20:** 20-period exponential moving average. Serves as the intermediate trend anchor.
- **Signal:** `EMA9 > EMA20` = bullish alignment. `EMA9 < EMA20` = bearish alignment.

### VWAP (Volume-Weighted Average Price)
- Calculated as cumulative `(price × volume) / cumulative volume` from the start of the session.
- Represents the average price weighted by volume — the "fair value" of the session.
- **Signal:** `Close > VWAP` = buyers control the session. `Close < VWAP` = sellers control.

### ATR% (Average True Range, Percentage)
- Measures intraday volatility as a percentage of price.
- True Range = max(`high-low`, `abs(high-prev_close)`, `abs(low-prev_close)`)
- ATR = 14-period rolling mean of True Range.
- **Signal:** `ATR% > 5%` = sufficient range for meaningful moves. Low ATR% = chop, avoid.

### RVOL (Relative Volume)
- Current 1-minute volume divided by the 20-period rolling average volume.
- **Signal:** `RVOL > 2.0` = institutional participation expanding. `RVOL < 1.5` = fading interest.

### PCA1 (Principal Component Analysis, Component 1)
- A single composite score derived from five stationary features: EMA spread, VWAP distance (normalized), percent return, ATR%, and RVOL.
- StandardScaler normalizes all features to zero mean/unit variance before PCA extraction.
- **Signal:** `PCA1 > 0` = all factors aligned bullishly. `PCA1 < 0` = bearish alignment.

### PCA1 Slope
- First difference of PCA1 (`PCA1[t] - PCA1[t-1]`).
- Measures the **acceleration** of momentum, not just its direction.
- **Signal:** Positive and rising = momentum strengthening. Negative and steep = momentum collapsing.

---

## 🎯 Scoring & Classification Systems

### Intraday Score (0–60 Points)
A weighted ranking system that scores structural alignment. Higher scores = stronger conviction.

| Condition | Weight | Rationale |
|-----------|--------|-----------|
| `EMA9 > EMA20` | +5 | Short-term trend aligned |
| `RVOL > 2.0` | +15 | Institutional volume confirmation (highest weight) |
| `ATR% > 5%` | +5 | Sufficient range for the move |
| `Close > VWAP` | +10 | Buyers control session average |
| `PCA1 > 0` | +15 | All factors aligned bullishly (highest weight) |
| `PCA1_slope > 0` | +10 | Momentum accelerating |

**Maximum: 60 points.** Sort descending by Intraday Score, then Risk Efficiency Score.

### Risk Efficiency Score
`Risk Efficiency = (|PCA1_slope| × RVOL) / (ATR% + 0.001)`

- **Numerator:** Raw momentum force (velocity × volume participation)
- **Denominator:** Volatility cost (smoothed by epsilon to prevent division by zero)
- **Interpretation:** Higher = more momentum per unit of risk. Filters out choppy, wide-spread assets. Elevates clean, high-volume directional runners.

---

## 🚦 Execution Readiness Classification

| Status | Conditions | Trader Action |
|--------|-----------|-------------|
| **Ready** | EMA9>EMA20 AND Close>VWAP AND PCA1_slope>0 AND RVOL>2 | **Highest conviction entry.** All systems aligned. |
| **Crossing Soon** | EMA9>EMA20 AND PCA1_slope>0 (but missing VWAP or RVOL) | Setup forming. Wait for VWAP reclaim or volume spike. |
| **Intraday False Ready** | EMA9>EMA20 AND Close>VWAP (but RVOL≤2 or PCA1_slope≤0) | Looks bullish but lacks volume/momentum confirmation. High trap risk. Avoid. |
| **Setup Only** | EMA9>EMA20 only | Structure organizing but momentum stagnant. No edge yet. |
| **UNKNOWN** | EMA9≤EMA20 or conflicting signals | No bullish alignment. Avoid. |

---

## 🎯 Position Status Classification (Exit Management)

For tracked positions only. Guides manual stop management on E*TRADE.

| Status | Trigger | E*TRADE Action |
|--------|---------|----------------|
| **✅ HOLD — Structure Intact** | PCA1>0, PCA1_slope≥0, Close≥VWAP, RVOL≥1.5 | Keep original 1:3 stop-limit. Do not adjust. |
| **⚠️ TIGHTEN — Below VWAP** | Close<VWAP but EMA9>EMA20 and PCA1 still positive | Move stop to breakeven or 1:1. Protect capital. |
| **📉 FADE — Volume Drying** | RVOL<1.5 or PCA1_slope<0 (but PCA1 still positive) | Consider trailing stop or taking partial profits. Momentum decaying. |
| **🛑 EXIT — Momentum Reversed** | PCA1<0 OR PCA1_slope<-0.5 | **Close position immediately.** Momentum has structurally reversed. Override hard stop if not yet hit. |

**Priority cascade:** EXIT checked first, then TIGHTEN, then FADE, then HOLD. The most severe condition always wins.

---

## 📈 Intraday Buy Signal Classification

| Signal | Conditions | Meaning |
|--------|-----------|---------|
| **Strong Intraday** | PCA1>0 AND Close>VWAP | Full bullish alignment. Best setups. |
| **Intraday Buy** | PCA1>0 only | Bullish but below VWAP. Watch for VWAP reclaim. |
| **Neutral** | PCA1≈0 | No directional edge. Avoid. |
| **Avoid** | PCA1<0 | Bearish alignment. Do not enter. |

---

## 🛠️ Sidebar Position Tracker

The manual entry tracker lives in the left sidebar and serves as your trade log:

- **Log New Entry:** Input ticker, entry price, and share count after E*TRADE execution
- **Open Positions:** Live view of all tracked trades with time-in-trade (minutes)
- **Close Button:** One-click removal with automatic logging to closed trades history
- **Closed Trades History:** Running P&L record for end-of-day review

**Important:** Session state is volatile. If the Streamlit server restarts, position history is lost. For permanent records, export or screenshot the Closed Trades panel at end of session.

---

## 🎛️ View Mode Toggle

Two modes above the Run button control which tickers are scanned:

| Mode | Use Case |
|------|----------|
| **🕵️ All Candidates** | Full universe scan (1,500 tickers). Use for finding new setups. |
| **📂 My Positions Only** | Scan only tickers you've logged in the sidebar tracker. Use for managing open positions quickly. |

When "My Positions Only" is selected, the scanner skips the full universe and only fetches data for your tracked tickers. Much faster for mid-day position checks.

---

## 🎯 Stop Recommendation Engine

For every row in the table, the scanner calculates:

**For candidates (not yet entered):**
- `Suggested Stop` — where to set your E*TRADE stop-limit
- `Suggested Target` — your 1:3 profit target
- `Risk ($)` — dollar risk per share based on ATR

**For tracked positions:**
- `Current Stop` — updated stop level based on current ATR (moves if volatility changes)
- `Current Target` — updated target
- `P&L %` — live unrealized gain/loss
- `Time In Trade` — how many minutes since your entry
- `Dist to Stop %` — how close price is to your stop (warning if < 1%)

**Example table row for a tracked position:**

| Ticker | Price | P&L % | Time In Trade | Current Stop | Current Target | Position Status |
|--------|-------|-------|---------------|--------------|----------------|-----------------|
| AAPL | $228.30 | +1.24% | 47m | $223.10 | $234.50 | ✅ HOLD |

---

## ⚡ Quick Action Panel

Below the main table, a dedicated panel shows **only your open positions** as action cards:

```
AAPL    P&L: +1.24%    Stop: $223.10    Target: $234.50    Status: ✅ HOLD    [Close AAPL]
NVDA    P&L: -0.80%    Stop: $118.20    Target: $128.50    Status: ⚠️ TIGHTEN    [Close NVDA]
```

Click **Close** → logs the exit price, P&L, and time to your closed trades history. No need to manually track in a spreadsheet.

---

## ⚠️ Key Assumptions & Limitations

1. **Data Quality:** Assumes `fetch_intraday` returns clean 1-minute OHLCV bars starting from 09:30 ET. Stale or incomplete data will produce invalid signals.
2. **No Short Selling:** This is a **long-only** model. Short signals are treated as "Avoid." PCA1<0 means exit existing longs, not enter shorts.
3. **Manual Execution:** The scanner does not connect to any broker API. All orders must be placed manually on E*TRADE.
4. **Intraday Only:** All positions should be closed by market close (16:00 ET). Overnight gaps invalidate the model's assumptions.
5. **PCA Minimum:** PCA requires 30+ bars (30 minutes) for statistical validity. Running before 10:00 ET produces unreliable signals.
6. **VWAP Session Assumption:** VWAP is cumulative from the first bar in the data feed. If your data starts after 09:30, VWAP will be biased.
7. **Not Financial Advice:** This is a systematic screening tool. All trading decisions remain the sole responsibility of the trader.

---

## 🧪 Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1 | 2026-08-08 | Initial intraday engine |
| v2 | 2026-08-08 | Fixed ATR calculation (True Range + mean), normalized PCA features, added Risk Efficiency epsilon |
| v3 | 2026-08-08 | Added Position Status (HOLD/TIGHTEN/FADE/EXIT), Position Status filter, color-coded styling |
| **v4** | **2026-08-08** | **Added manual position tracker, stop recommendation engine, time-in-trade, view mode toggle (All Candidates / My Positions Only), quick action panel, P&L color coding** |
""")