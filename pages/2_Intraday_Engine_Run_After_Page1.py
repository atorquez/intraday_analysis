# ==============================================================================
# PAGE 2 — INTRADAY ENGINE + FUNDAMENTAL WARNING
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf

logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")
st.caption("Version: 2026-07-20 (Exit Trigger + Price Extension + Fundamental Warning — Patched Universe)")
st.title("📈 Page 2 — Intraday Scanner + Exit Triggers + Price Extension + Fundamentals")
st.markdown("Pure intraday scanner with automated exit triggers, price extension warnings, and fundamental risk sizing. Long-only cash equity model with position management.")

# Import shared fundamental scoring
from utils.fundamental_scoring import get_fundamental_scores_cached

# ---------------------------------------------------------
# SESSION STATE: Manual Position Tracker
# ---------------------------------------------------------
if "positions" not in st.session_state:
    st.session_state.positions = {}

if "closed_positions" not in st.session_state:
    st.session_state.closed_positions = []

if "page2_last_results" not in st.session_state:
    st.session_state.page2_last_results = None

# ---------------------------------------------------------
# SIDEBAR: Manual Position Tracker + FUNDAMENTAL SETTINGS
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

    st.markdown("---")
    st.markdown("### ⚙️ Fundamental Warning Settings")

    st.markdown("**Warning Thresholds**")

    weak_fund_threshold = st.slider(
        "Weak Fundamentals Warning (< X)",
        min_value=30,
        max_value=70,
        value=50,
        step=5,
        key="weak_fund_threshold",
        help="Stocks with Fund Score below this get a ⚠️ warning"
    )

    strong_fund_threshold = st.slider(
        "Strong Fundamentals Highlight (≥ X)",
        min_value=60,
        max_value=95,
        value=80,
        step=5,
        key="strong_fund_threshold",
        help="Stocks with Fund Score above this get a ✅ highlight"
    )

    show_fund_details = st.checkbox("Show Fundamental Sub-scores", value=True, key="show_fund_details")

    st.markdown("---")
    st.markdown("**Position Sizing Guide**")
    st.caption("Based on Fund Score:")
    st.markdown(f"""
    - ✅ **≥ {strong_fund_threshold}** — Normal size
    - ⚠️ **{weak_fund_threshold}-{strong_fund_threshold-1}** — Reduce 25%
    - 🚨 **< {weak_fund_threshold}** — Reduce 50% or avoid
    """)

    st.markdown("---")
    st.markdown("### 🔢 Scan Size Settings")

    max_scan = st.slider(
        "Max tickers to scan (Page 2)",
        min_value=40,
        max_value=1500,
        value=1500,
        step=20,
        help="Controls how many tickers Page 2 will process. Uses Page 1 filtered universe when available."
    )

# ---------------------------------------------------------
# FILTER PANEL (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

col1, col2, col3 = st.columns(3)

with col1:
    intraday_signal_filter = st.multiselect(
        "Filter by Intraday Buy Signal",
        ["Strong Intraday", "Intraday Buy", "Neutral", "Avoid"],
        default=["Strong Intraday", "Intraday Buy", "Neutral"],
        key="intraday_signal_filter"
    )

with col2:
    execution_filter = st.multiselect(
        "Filter by Execution Readiness",
        ["Ready", "Crossing Soon", "Intraday False Ready", "Setup Only", "UNKNOWN"],
        default=["Ready", "Crossing Soon"],
        key="intraday_execution_filter"
    )

with col3:
    min_price = st.number_input("Minimum Price ($)", value=5.0, key="intraday_min_price")
    max_price = st.number_input("Maximum Price ($)", value=100.0, key="intraday_max_price")

st.markdown("### 🎯 Position Management Filter")
position_filter = st.multiselect(
    "Filter by Position Status",
    ["✅ HOLD — Structure Intact", "⚠️ TIGHTEN — Below VWAP", "📉 FADE — Volume Drying", "🛑 EXIT — Momentum Reversed"],
    default=["✅ HOLD — Structure Intact", "⚠️ TIGHTEN — Below VWAP", "📉 FADE — Volume Drying"],
    key="position_status_filter"
)

view_mode = st.radio(
    "View Mode",
    ["🕵️ All Candidates", "📂 My Positions Only"],
    horizontal=True,
    key="view_mode"
)

run_intraday = st.button("Run Intraday Engine", key="intraday_run_button", use_container_width=True)

# ---------------------------------------------------------
# Universe Loader (Patched: Prefer Page 1 filtered universe)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_primary_universe():
    try:
        from utils.data_fetch import load_universe
        return load_universe()
    except Exception as e:
        logger.warning(f"Failed to load universe from utils: {e}")
        return ["AAPL", "NVDA", "MSFT", "AMD", "TSLA", "META", "AMZN", "MDT"]

# Prefer Page 1 filtered results if available
page1_results = st.session_state.get("intraday_filtered_results", None)

if page1_results is not None and isinstance(page1_results, pd.DataFrame) and not page1_results.empty:
    tickers = page1_results["Ticker"].tolist()
    st.caption(f"Using {len(tickers)} tickers from Page 1 filtered universe.")
else:
    tickers = load_primary_universe()
    st.caption(f"Using raw primary universe ({len(tickers)} tickers). Consider running Page 1 first for structural filtering.")

# =========================================================
# PAGE 2 — Structural Universe Validation Banner
# =========================================================

# Load structural universe from Page 1
tickers = st.session_state.get("intraday_filtered_results", None)

if tickers is None or len(tickers) == 0:
    st.error("No structural universe found. Please run Page 1 first.")
    st.stop()

structural_count = len(tickers)
scan_limit = st.session_state.get("intraday_scan_limit", 1500)  # or your slider variable
tickers_scanned = min(structural_count, scan_limit)

# Display validation banner
st.markdown("""
### 🔎 Universe Validation Summary
""")

st.markdown(
    f"""
**Structural Universe from Page 1:** {structural_count} tickers  
**Intraday Scan Limit:** {scan_limit}  
**Tickers Scanned:** {tickers_scanned}  
"""
)

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

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vp"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"].replace(0, np.nan)

    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["prev_close"])
    df["tr3"] = abs(df["low"] - df["prev_close"])
    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr"] = df["true_range"].rolling(14).mean().ffill().bfill().fillna(0.01)
    df["atr%"] = (df["atr"] / df["close"]) * 100

    df["session_open"] = df["open"].iloc[0] if len(df) > 0 else None

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

def classify_position_status(latest):
    pca1 = latest["PCA1"]
    pca1_slope = latest["PCA1_slope"]
    close = latest["close"]
    vwap = latest["vwap"]
    rvol = latest["rvol"]
    ema9 = latest["ema9"]
    ema20 = latest["ema20"]

    if pca1 < 0 or pca1_slope < -0.5:
        return "🛑 EXIT — Momentum Reversed"
    if close < vwap and ema9 > ema20:
        return "⚠️ TIGHTEN — Below VWAP"
    if rvol < 1.5 or pca1_slope < 0:
        return "📉 FADE — Volume Drying"
    return "✅ HOLD — Structure Intact"

# ---------------------------------------------------------
# EXIT TRIGGER ENGINE
# ---------------------------------------------------------
def classify_exit_trigger(latest, entry_price=None, session_open=None):
    current = float(latest["close"])
    ema9 = latest["ema9"]
    ema20 = latest["ema20"]
    pca1 = latest["PCA1"]
    pca1_slope = latest["PCA1_slope"]
    rvol = latest["rvol"]

    if entry_price and entry_price > 0:
        pnl_pct = ((current - entry_price) / entry_price) * 100

        if pnl_pct > 10:
            if current < ema9:
                return "🎯 TAKE PROFIT — +10% gain, price broke EMA9. Capture immediately."
            elif pca1_slope < -0.3 and rvol < 1.5:
                return "⚠️ TIGHTEN STOP — +10% gain, momentum fading. Move stop to breakeven."

        elif pnl_pct > 5:
            if current < ema9:
                return "🎯 TAKE PROFIT — +5% gain, price broke EMA9. Capture 50%."
            elif pca1 < 0:
                return "⚠️ TIGHTEN STOP — +5% gain, PCA turning negative. Trail at EMA9."

    if session_open and session_open > 0:
        extension = ((current - session_open) / session_open) * 100

        if extension > 15:
            if current < ema9 and pca1_slope < 0:
                return "🚨 EXTREME EXIT — +15% extended, EMA9 broken, momentum reversing. Full exit."
        elif extension > 10:
            if current < ema9 and pca1_slope < 0:
                return "🚨 REVERSAL EXIT — +10% extended, EMA9 broken, PCA negative. Exit 75%."
            elif current < ema9:
                return "⚠️ EMA9 BREACH — Extended move, price below EMA9. Exit 50%, watch EMA20."
            elif pca1_slope < -0.5 and rvol < 1.5:
                return "📉 MOMENTUM FADE — Extended move, volume drying, PCA collapsing. Tighten stop."

    if current < ema20 and pca1 < 0:
        return "🛑 TREND FAILURE — Price below EMA20, PCA negative. Full exit, structure broken."

    return None

def get_exit_sizing_guidance(exit_signal):
    if exit_signal is None:
        return None, "HOLD", "Maintain current position."

    if "TAKE PROFIT" in exit_signal:
        return "EXIT 50-100%", "GREEN", "Lock in gains. Scale out in two tranches."
    elif "EXTREME EXIT" in exit_signal:
        return "EXIT 100%", "RED", "Parabolic exhaustion. Full exit immediately."
    elif "REVERSAL EXIT" in exit_signal:
        return "EXIT 75%", "RED", "Extended move reversing. Keep 25% only if EMA20 holds."
    elif "EMA9 BREACH" in exit_signal:
        return "EXIT 50%", "ORANGE", "First warning. Exit half, watch EMA20 for remainder."
    elif "TREND FAILURE" in exit_signal:
        return "EXIT 100%", "RED", "Structural break. No mercy, full exit."
    elif "TIGHTEN STOP" in exit_signal:
        return "REDUCE RISK", "YELLOW", "Move stop to breakeven or 1:1. Protect capital."
    elif "MOMENTUM FADE" in exit_signal:
        return "TRAIL STOP", "YELLOW", "Tighten trailing stop to EMA9. Prepare for exit."

    return None, "HOLD", "No action required."

# ---------------------------------------------------------
# PRICE EXTENSION CLASSIFIER
# ---------------------------------------------------------
def classify_price_extension(current_price, prev_close):
    if prev_close is None or prev_close <= 0:
        return "❓ Unknown", 0.0

    extension = ((current_price - prev_close) / prev_close) * 100

    if extension >= 10:
        return "🚨 Extended", extension
    elif extension >= 5:
        return "⚠️ Chasing", extension
    elif extension >= 2:
        return "✅ Good Entry", extension
    else:
        return "💤 Baseline", extension

def get_extension_sizing_guidance(extension_label):
    guidance = {
        "💤 Baseline": "Normal size — early momentum, favorable entry.",
        "✅ Good Entry": "Normal size — momentum confirmed, manageable risk.",
        "⚠️ Chasing": "Reduce 50% — extended move, pullback likely. Wait for EMA9 touch or VWAP reclaim.",
        "🚨 Extended": "Avoid new entry or reduce 75% — parabolic exhaustion. Only add if strong pullback to EMA20.",
        "❓ Unknown": "Use technical stops only — no extension data available."
    }
    return guidance.get(extension_label, "Use technical stops only.")

# ---------------------------------------------------------
# Stop Recommendation Engine
# ---------------------------------------------------------
def compute_stop_recommendation(latest, entry_price=None, risk_multiplier=1.0, reward_ratio=3.0):
    current = float(latest["close"])
    atr_dollars = (latest["atr%"] / 100.0) * current

    if atr_dollars <= 0 or pd.isna(atr_dollars):
        atr_dollars = current * 0.005

    if entry_price and entry_price > 0:
        risk = atr_dollars * risk_multiplier
        stop = entry_price - risk
        target = entry_price + (reward_ratio * risk)
        pnl_pct = ((current - entry_price) / entry_price) * 100
    else:
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
# FUNDAMENTAL WARNING CLASSIFIER
# ---------------------------------------------------------
def classify_fundamental_warning(fund_score, weak_thresh, strong_thresh):
    if fund_score is None:
        return "❓ No Data", "gray", "Fundamental data unavailable. Use technical stops only."

    if fund_score >= strong_thresh:
        return "✅ Strong", "green", f"Fund Score {fund_score}: Normal position size acceptable."
    elif fund_score >= weak_thresh:
        return "⚠️ Moderate", "orange", f"Fund Score {fund_score}: Consider reducing position by 25%."
    else:
        return "🚨 Weak", "red", f"Fund Score {fund_score}: High risk. Reduce 50% or avoid overnight hold."

# ---------------------------------------------------------
# Core Iteration Execution Loop (WITH FUNDAMENTAL WARNINGS)
# ---------------------------------------------------------
if run_intraday:
    results = []
    position_log = st.session_state.get("positions", {})

    weak_thresh = st.session_state.get("weak_fund_threshold", 50)
    strong_thresh = st.session_state.get("strong_fund_threshold", 80)
    show_fund = st.session_state.get("show_fund_details", True)

    if len(tickers) > max_scan:
        processing_universe = tickers[:max_scan]
        st.caption(f"Scanning {len(processing_universe)} tickers (capped by slider).")
    else:
        processing_universe = tickers
        st.caption(f"Scanning full universe subset ({len(processing_universe)} tickers).")

    progress_bar = st.progress(0.0, text="Initializing scan...")

    for idx, ticker in enumerate(processing_universe):
        progress_bar.progress((idx + 1) / len(processing_universe), text=f"Scanning {ticker}... ({idx+1}/{len(processing_universe)})")

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
        tech_score = compute_intraday_score(latest_bar)
        signal = classify_intraday_buy_signal(latest_bar)
        risk_eff = compute_risk_efficiency(latest_bar)
        position_status = classify_position_status(latest_bar)

        session_open = latest_bar.get("session_open", current_price)
        ext_label, ext_pct = classify_price_extension(current_price, session_open)
        ext_guidance = get_extension_sizing_guidance(ext_label)

        fund_data = get_fundamental_scores_cached(ticker)

        if fund_data:
            fund_score = fund_data["Fund_Score"]
            warn_label, warn_color, warn_msg = classify_fundamental_warning(
                fund_score, weak_thresh, strong_thresh
            )
        else:
            fund_score = None
            warn_label = "❓ No Data"
            warn_color = "gray"
            warn_msg = "Fundamental data unavailable."

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
            "Intraday Score": tech_score,
            "Intraday Buy_Signal": signal,
            "Risk Efficiency Score": round(risk_eff, 4),
            "Position Status": position_status,
            "Price Extension": ext_label,
            "Extension %": round(ext_pct, 2),
            "Extension Guidance": ext_guidance,
            "Fund Score": fund_score,
            "Fund Warning": warn_label,
            "Fund Message": warn_msg,
        }

        if show_fund and fund_data:
            row["Fund Valuation"] = fund_data["Fund_Valuation"]
            row["Fund Growth"] = fund_data["Fund_Growth"]
            row["Fund Profit"] = fund_data["Fund_Profit"]
            row["Fund Risk"] = fund_data["Fund_Risk"]
            row["Sector"] = fund_data["Sector"]

        if ticker in position_log:
            pos = position_log[ticker]
            entry_price = pos["entry_price"]
            entry_time = pos["entry_time"]
            shares = pos["shares"]

            mins_in = int((datetime.now() - entry_time).total_seconds() / 60)
            stop_rec = compute_stop_recommendation(latest_bar, entry_price=entry_price)

            exit_signal = classify_exit_trigger(latest_bar, entry_price=entry_price, session_open=session_open)
            exit_action, exit_color, exit_msg = get_exit_sizing_guidance(exit_signal)

            row["Entry Price"] = entry_price
            row["Shares"] = shares
            row["P&L %"] = stop_rec["pnl_pct"]
            row["Time In Trade"] = f"{mins_in}m"
            row["Current Stop"] = stop_rec["stop"]
            row["Current Target"] = stop_rec["target"]
            row["Risk ($)"] = stop_rec["risk_dollars"]
            row["Dist to Stop %"] = stop_rec["distance_to_stop_pct"]
            row["Exit Signal"] = exit_signal
            row["Exit Action"] = exit_action
            row["Exit Message"] = exit_msg
        else:
            stop_rec = compute_stop_recommendation(latest_bar)
            row["Suggested Stop"] = stop_rec["stop"]
            row["Suggested Target"] = stop_rec["target"]
            row["Risk ($)"] = stop_rec["risk_dollars"]

        results.append(row)

    progress_bar.empty()

    if not results:
        st.warning("No assets successfully bypassed background processing parameters. Check your console logs.")
    else:
        master_df = pd.DataFrame(results)

        st.session_state["page2_last_results"] = master_df

        if intraday_signal_filter:
            master_df = master_df[master_df["Intraday Buy_Signal"].isin(intraday_signal_filter)]

        if execution_filter:
            master_df = master_df[master_df["Execution Readiness"].isin(execution_filter)]

        if position_filter:
            master_df = master_df[master_df["Position Status"].isin(position_filter)]

        if master_df.empty:
            st.info("Watchlist generated structural entries, but they were filtered out by user checkbox configurations.")
        else:
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

            strong_count = (master_df["Fund Warning"] == "✅ Strong").sum()
            moderate_count = (master_df["Fund Warning"] == "⚠️ Moderate").sum()
            weak_count = (master_df["Fund Warning"] == "🚨 Weak").sum()
            no_data_count = (master_df["Fund Warning"] == "❓ No Data").sum()

            ext_baseline = (master_df["Price Extension"] == "💤 Baseline").sum()
            ext_good = (master_df["Price Extension"] == "✅ Good Entry").sum()
            ext_chasing = (master_df["Price Extension"] == "⚠️ Chasing").sum()
            ext_extended = (master_df["Price Extension"] == "🚨 Extended").sum()

            st.caption(
                f"Fundamental Profile: ✅ Strong {strong_count} | ⚠️ Moderate {moderate_count} | 🚨 Weak {weak_count} | ❓ No Data {no_data_count}"
            )
            st.caption(
                f"Price Extension Profile: 💤 Baseline {ext_baseline} | ✅ Good {ext_good} | ⚠️ Chasing {ext_chasing} | 🚨 Extended {ext_extended}"
            )

            st.dataframe(master_df, hide_index=True, use_container_width=True)

elif st.session_state.get("page2_last_results") is not None:
    st.subheader("📂 Last Intraday Scan Results (Stored)")
    st.dataframe(st.session_state["page2_last_results"], hide_index=True, use_container_width=True)