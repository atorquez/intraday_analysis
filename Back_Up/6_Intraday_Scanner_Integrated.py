# ==============================================================================
# PAGE 2 — INTRADAY ENGINE + FUNDAMENTAL WARNING (v5.1)
# ==============================================================================
# Replaces the Fundamental Filter/Gate with a Warning System.
# All candidates pass through. Fundamentals influence risk sizing, not entry.

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
st.caption("Version: 2026-07-20 (Exit Trigger + Price Extension + Fundamental Warning)")
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

    # REMOVED: Minimum Fundamental Score slider (no longer filters)
    # ADDED: Warning thresholds
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

# Position Management Filter
st.markdown("### 🎯 Position Management Filter")
position_filter = st.multiselect(
    "Filter by Position Status",
    ["✅ HOLD — Structure Intact", "⚠️ TIGHTEN — Below VWAP", "📉 FADE — Volume Drying", "🛑 EXIT — Momentum Reversed"],
    default=["✅ HOLD — Structure Intact", "⚠️ TIGHTEN — Below VWAP", "📉 FADE — Volume Drying"],
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

    # ATR + Previous Close for extension calculation
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["prev_close"])
    df["tr3"] = abs(df["low"] - df["prev_close"])
    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr"] = df["true_range"].rolling(14).mean().ffill().bfill().fillna(0.01)
    df["atr%"] = (df["atr"] / df["close"]) * 100

    # Store session open price for extension calculation (first bar of the day)
    df["session_open"] = df["open"].iloc[0] if len(df) > 0 else None

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
# EXIT TRIGGER ENGINE (For Extended Moves & Profit Capture)
# ---------------------------------------------------------
def classify_exit_trigger(latest, entry_price=None, session_open=None):
    """
    Generates exit triggers for positions in extended or momentum stocks.

    Rules:
      1. Profit Capture: If up > 5% from entry, tighten stop to EMA9 breach
      2. Extended Reversal: If > 10% from open and EMA9 breaks with negative PCA
      3. EMA20 Failure: If price breaks EMA20 after extended move, full exit

    Returns exit signal string or None if no trigger.
    """
    current = float(latest["close"])
    ema9 = latest["ema9"]
    ema20 = latest["ema20"]
    pca1 = latest["PCA1"]
    pca1_slope = latest["PCA1_slope"]
    rvol = latest["rvol"]

    # Rule 1: Profit Capture — if up > 5% from entry, watch EMA9
    if entry_price and entry_price > 0:
        pnl_pct = ((current - entry_price) / entry_price) * 100

        if pnl_pct > 10:
            # Big winner — very tight risk management
            if current < ema9:
                return "🎯 TAKE PROFIT — +10% gain, price broke EMA9. Capture immediately."
            elif pca1_slope < -0.3 and rvol < 1.5:
                return "⚠️ TIGHTEN STOP — +10% gain, momentum fading. Move stop to breakeven."

        elif pnl_pct > 5:
            # Moderate winner — standard profit capture
            if current < ema9:
                return "🎯 TAKE PROFIT — +5% gain, price broke EMA9. Capture 50%."
            elif pca1 < 0:
                return "⚠️ TIGHTEN STOP — +5% gain, PCA turning negative. Trail at EMA9."

    # Rule 2: Extended Reversal — parabolic move exhaustion
    if session_open and session_open > 0:
        extension = ((current - session_open) / session_open) * 100

        if extension > 15:
            # Extreme extension — any EMA9 break is exit
            if current < ema9 and pca1_slope < 0:
                return "🚨 EXTREME EXIT — +15% extended, EMA9 broken, momentum reversing. Full exit."

        elif extension > 10:
            # Standard extended move — watch for structure break
            if current < ema9 and pca1_slope < 0:
                return "🚨 REVERSAL EXIT — +10% extended, EMA9 broken, PCA negative. Exit 75%."
            elif current < ema9:
                return "⚠️ EMA9 BREACH — Extended move, price below EMA9. Exit 50%, watch EMA20."
            elif pca1_slope < -0.5 and rvol < 1.5:
                return "📉 MOMENTUM FADE — Extended move, volume drying, PCA collapsing. Tighten stop."

    # Rule 3: EMA20 Failure — trend structure broken
    if current < ema20 and pca1 < 0:
        return "🛑 TREND FAILURE — Price below EMA20, PCA negative. Full exit, structure broken."

    return None


def get_exit_sizing_guidance(exit_signal):
    """Returns position sizing action for exit triggers."""
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
# PRICE EXTENSION CLASSIFIER (Momentum Exhaustion Warning)
# ---------------------------------------------------------
def classify_price_extension(current_price, prev_close):
    """
    Flags how extended the price is from the previous close.
    Used for entry timing and risk sizing, not filtering.

    Zones:
      💤 Baseline   (< 2%)  — Normal range, good entry opportunity
      ✅ Good Entry  (2-5%)  — Momentum building, still favorable
      ⚠️ Chasing     (5-10%) — Extended, reduce size or wait for pullback
      🚨 Extended    (> 10%) — Parabolic, high reversal risk, avoid new entries
    """
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
    """Returns position sizing guidance based on extension zone."""
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
    """
    Returns a warning label and position sizing guidance.
    Does NOT filter — all candidates pass through.
    """
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

    # Get user settings
    weak_thresh = st.session_state.get("weak_fund_threshold", 50)
    strong_thresh = st.session_state.get("strong_fund_threshold", 80)
    show_fund = st.session_state.get("show_fund_details", True)

    processing_universe = tickers[:40] if len(tickers) > 40 else tickers

    progress_bar = st.progress(0.0, text="Initializing scan...")

    # Phase 1: Technical scan
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

        # --- PRICE EXTENSION CALCULATION ---
        # Use session open as proxy for previous close (intraday context)
        # For true previous close, we'd need daily data fetch
        session_open = latest_bar.get("session_open", current_price)
        ext_label, ext_pct = classify_price_extension(current_price, session_open)
        ext_guidance = get_extension_sizing_guidance(ext_label)

        # --- FUNDAMENTAL OVERLAY (WARNING SYSTEM) ---
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
            "Intraday Score": tech_score,
            "Intraday Buy_Signal": signal,
            "Risk Efficiency Score": round(risk_eff, 4),
            "Position Status": position_status,
            # Price Extension — momentum exhaustion warning
            "Price Extension": ext_label,
            "Extension %": round(ext_pct, 2),
            # Fundamental warning columns
            "Fund Score": fund_score,
            "Fund Warning": warn_label,
            "Fund Message": warn_msg,
        }

        # Add fundamental sub-scores if enabled
        if show_fund and fund_data:
            row["Fund Valuation"] = fund_data["Fund_Valuation"]
            row["Fund Growth"] = fund_data["Fund_Growth"]
            row["Fund Profit"] = fund_data["Fund_Profit"]
            row["Fund Risk"] = fund_data["Fund_Risk"]
            row["Sector"] = fund_data["Sector"]

        # Position management columns
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

        # Store in session state for persistence
        st.session_state["page2_last_results"] = master_df

        # Apply filters (technical only — fundamentals don't filter)
        if intraday_signal_filter:
            master_df = master_df[master_df["Intraday Buy_Signal"].isin(intraday_signal_filter)]

        if execution_filter:
            master_df = master_df[master_df["Execution Readiness"].isin(execution_filter)]

        if position_filter:
            master_df = master_df[master_df["Position Status"].isin(position_filter)]

        if master_df.empty:
            st.info("Watchlist generated structural entries, but they were filtered out by user checkbox configurations.")
        else:
            # Sort: open positions first, then by technical score
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

            # Summary stats
            strong_count = (master_df["Fund Warning"] == "✅ Strong").sum()
            moderate_count = (master_df["Fund Warning"] == "⚠️ Moderate").sum()
            weak_count = (master_df["Fund Warning"] == "🚨 Weak").sum()
            no_data_count = (master_df["Fund Warning"] == "❓ No Data").sum()

            ext_baseline = (master_df["Price Extension"] == "💤 Baseline").sum()
            ext_good = (master_df["Price Extension"] == "✅ Good Entry").sum()
            ext_chasing = (master_df["Price Extension"] == "⚠️ Chasing").sum()
            ext_extended = (master_df["Price Extension"] == "🚨 Extended").sum()

            st.caption(f"Fundamental Profile: ✅ {strong_count} Strong | ⚠️ {moderate_count} Moderate | 🚨 {weak_count} Weak | ❓ {no_data_count} No Data")
            st.caption(f"Price Extension: 💤 {ext_baseline} Baseline | ✅ {ext_good} Good Entry | ⚠️ {ext_chasing} Chasing | 🚨 {ext_extended} Extended")

            # Styling functions
            def style_readiness(val):
                if val == "Ready":
                    return "background-color: #2E7D32; color: white; font-weight: bold;"
                elif val == "Intraday False Ready":
                    return "background-color: #EF6C00; color: white;"
                elif val == "Crossing Soon":
                    return "background-color: #FBC02D; color: black;"
                else:
                    return "background-color: #757575; color: white;"

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

            def style_pnl(val):
                if pd.isna(val):
                    return ""
                if val > 0:
                    return "background-color: #1B5E20; color: white; font-weight: bold;"
                elif val < -2:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                else:
                    return "background-color: #F9A825; color: black;"

            def style_fund_warning(val):
                if "✅" in val:
                    return "background-color: #1B5E20; color: white; font-weight: bold;"
                elif "⚠️" in val:
                    return "background-color: #FF6F00; color: white;"
                elif "🚨" in val:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                else:
                    return "background-color: #757575; color: white;"

            def style_extension(val):
                if "🚨" in val:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                elif "⚠️" in val:
                    return "background-color: #FF6F00; color: white; font-weight: bold;"
                elif "✅" in val:
                    return "background-color: #2E7D32; color: white; font-weight: bold;"
                else:
                    return "background-color: #757575; color: white;"

            def style_exit_trigger(val):
                if val == "—" or pd.isna(val):
                    return ""
                if "🚨 EXTREME" in val or "🛑 TREND" in val:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                elif "🚨 REVERSAL" in val:
                    return "background-color: #D32F2F; color: white; font-weight: bold;"
                elif "🎯 TAKE PROFIT" in val:
                    return "background-color: #2E7D32; color: white; font-weight: bold;"
                elif "⚠️ EMA9" in val or "📉 MOMENTUM" in val:
                    return "background-color: #FF6F00; color: white; font-weight: bold;"
                elif "⚠️ TIGHTEN" in val:
                    return "background-color: #F9A825; color: black; font-weight: bold;"
                return ""

            def style_exit_action(val):
                if val == "HOLD" or pd.isna(val):
                    return ""
                if "EXIT 100%" in val:
                    return "background-color: #B71C1C; color: white; font-weight: bold;"
                elif "EXIT 75%" in val:
                    return "background-color: #D32F2F; color: white; font-weight: bold;"
                elif "EXIT 50%" in val:
                    return "background-color: #FF6F00; color: white; font-weight: bold;"
                elif "REDUCE RISK" in val or "TRAIL STOP" in val:
                    return "background-color: #F9A825; color: black; font-weight: bold;"
                return ""

            styled_df = master_df.style.map(
                style_readiness,
                subset=["Execution Readiness"]
            ).map(
                style_position_status,
                subset=["Position Status"]
            ).map(
                style_fund_warning,
                subset=["Fund Warning"]
            ).map(
                style_extension,
                subset=["Price Extension"]
            )

            # Add exit trigger styling if columns exist
            if "Exit Trigger" in master_df.columns:
                styled_df = styled_df.map(
                    style_exit_trigger,
                    subset=["Exit Trigger"]
                )
            if "Exit Action" in master_df.columns:
                styled_df = styled_df.map(
                    style_exit_action,
                    subset=["Exit Action"]
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

            # --- Deep Dive Section ---
            st.markdown("---")
            st.markdown("### 🔍 Deep Dive Research")
            st.caption("Select a ticker to view full fundamental analysis (opens Page 4)")

            selected_ticker = st.selectbox(
                "Select ticker for deep research:",
                options=[""] + master_df["Ticker"].tolist(),
                key="deep_dive_select"
            )

            if selected_ticker:
                st.session_state["deep_dive_ticker"] = selected_ticker
                st.session_state["research_ticker"] = selected_ticker

                col_research, col_info = st.columns([1, 3])
                with col_research:
                    if st.button(f"🔬 Research {selected_ticker}", key=f"research_{selected_ticker}", use_container_width=True):
                        try:
                            st.switch_page("pages/4_Ticker_Summary.py")
                        except Exception as e:
                            st.warning(f"Navigation to Page 4 failed: {e}")
                            st.info(f"Manual navigation: Go to Page 4 and enter ticker '{selected_ticker}'")
                with col_info:
                    fund_row = master_df[master_df["Ticker"] == selected_ticker]
                    if not fund_row.empty:
                        fs = fund_row.iloc[0].get("Fund Score")
                        fw = fund_row.iloc[0].get("Fund Warning", "")
                        if pd.notna(fs):
                            st.markdown(f"**Fund Score:** {fs:.0f}/100 | **Warning:** {fw}")

            # Quick Action Panel for positions
            if "P&L %" in master_df.columns and len(master_df[master_df["P&L %"].notna()]) > 0:
                st.markdown("---")
                st.markdown("### ⚡ Quick Actions for Open Positions")

                # Show exit trigger summary if any exist
                exit_df = master_df[master_df["P&L %"].notna()]
                active_exits = exit_df[exit_df["Exit Trigger"] != "—"] if "Exit Trigger" in exit_df.columns else pd.DataFrame()
                if len(active_exits) > 0:
                    st.warning(f"🚨 {len(active_exits)} position(s) with active exit triggers! Review immediately.")

                pos_df = master_df[master_df["P&L %"].notna()].copy()
                for _, row in pos_df.iterrows():
                    tk = row["Ticker"]

                    # Show exit trigger prominently if active
                    exit_trigger = row.get("Exit Trigger", "—")
                    exit_action = row.get("Exit Action", "HOLD")
                    exit_note = row.get("Exit Note", "")

                    if exit_trigger != "—":
                        st.markdown(f"""
                        <div style="padding: 8px; border-left: 4px solid {'#B71C1C' if 'EXIT 100%' in exit_action else '#FF6F00' if 'EXIT' in exit_action else '#F9A825'}; background-color: #1a1a1a; margin-bottom: 4px;">
                            <strong>{tk}</strong> — {exit_trigger}<br/>
                            <small>{exit_action}: {exit_note}</small>
                        </div>
                        """, unsafe_allow_html=True)

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
                                "pnl_pct": row["P&L %"],
                                "exit_trigger": exit_trigger if exit_trigger != "—" else "Manual Close"
                            })
                            del st.session_state.positions[tk]
                            st.rerun()

# ---------------------------------------------------------
# RENDER STORED RESULTS WHEN USER RETURNS TO PAGE
# ---------------------------------------------------------
elif st.session_state.get("page2_last_results") is not None:
    stored_df = st.session_state["page2_last_results"]
    st.subheader(f"🚀 Stored Intraday Matrix ({len(stored_df)} Tickers)")
    st.caption("Results from last scan. Click 'Run Intraday Engine' to refresh.")
    st.dataframe(stored_df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------
# EXECUTIVE SUMMARY, PROCESS, AND DEFINITIONS
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📘 Pure Intraday Engine + Fundamental Warning System — Parameter Definitions")

st.markdown("""
# Executive Summary — Integrated Intraday Momentum Framework (v5.1)

The **Intraday Engine + Fundamental Warning System** is a systematic, long-only screening engine that combines **real-time 1-minute technical signals** with **fundamental risk warnings** to surface high-conviction momentum candidates while respecting capital preservation.

Unlike the previous version (v5) which used a **Fundamental Quality Gate** (filtering out weak stocks), this version uses a **Fundamental Warning System** that lets all candidates through but flags risk levels for position sizing.

---

## 🆕 What's New in v5.3 — Exit Trigger Engine + Price Extension + Fundamental Warning System

### Exit Trigger Engine (For Tracked Positions)

A new **automated exit signal system** for open positions that identifies when to capture profits or cut losses based on real-time structural breaks.

**Three Core Rules:**

| Rule | Trigger | Action | Color |
|------|---------|--------|-------|
| **Profit Capture** | P&L > 5% + price breaks EMA9 | Exit 50% | 🎯 Green |
| **Profit Capture** | P&L > 10% + price breaks EMA9 | Exit 100% | 🎯 Green |
| **Extended Reversal** | Price > 10% from open + EMA9 break + PCA negative | Exit 75% | 🚨 Red |
| **Extreme Reversal** | Price > 15% from open + EMA9 break | Exit 100% | 🚨 Red |
| **Trend Failure** | Price below EMA20 + PCA negative | Exit 100% | 🛑 Red |
| **Momentum Fade** | Extended move + volume drying + PCA collapsing | Trail stop | 📉 Yellow |

**How it works:**
1. Scanner detects structural break (EMA9 breach, EMA20 failure, PCA collapse)
2. Cross-references with your entry price and session extension
3. Generates specific exit signal with sizing guidance
4. Displays prominently in Quick Action Panel with color-coded alerts

**Example (AEVA at $17.64):**
- Entry: $17.17, Current: $17.64 (+2.7%)
- Price Extension: 🚨 Extended (+10.7% from open)
- If price breaks EMA9: → 🚨 REVERSAL EXIT — Exit 75%
- If price reaches EMA20: → 🛑 TREND FAILURE — Exit 100%

---

### Price Extension — Momentum Exhaustion Detector

A new column **"Price Extension"** flags how far the current price has moved from the session open:

| Zone | Extension | Color | Action |
|------|-----------|-------|--------|
| 💤 Baseline | < 2% | Gray | Normal size — early momentum, favorable entry |
| ✅ Good Entry | 2–5% | Green | Normal size — momentum confirmed, manageable risk |
| ⚠️ Chasing | 5–10% | Orange | **Reduce 50%** — extended move, pullback likely. Wait for EMA9 touch or VWAP reclaim |
| 🚨 Extended | > 10% | Red | **Avoid new entry or reduce 75%** — parabolic exhaustion. Only add if strong pullback to EMA20 |

**Why this matters:** When a stock is already up 10%+ from the open, the easy money is made. The risk/reward shifts dramatically — you're buying into supply, not demand. This warning prevents chasing parabolic moves and protects capital.

**Example:** AEVA at $17.60 (up 10.69% from open) triggers 🚨 Extended. The scanner still shows the technical setup, but the warning tells you to either wait for a pullback to EMA9/VWAP or size down significantly.

---

### Why Warnings Instead of Filters?

**The Gate (v5):** Fund Score < 50 → Candidate discarded.  
**The Warning (v5.1):** Fund Score < 50 → Candidate flagged with 🚨, position size reduced 50%.

**Why the change?**

| Scenario | Gate (v5) | Warning (v5.1) |
|----------|-----------|----------------|
| GME short squeeze | ❌ Filtered out | ✅ Traded at 50% size |
| AMC gamma ramp | ❌ Filtered out | ✅ Traded at 50% size |
| Junk stock with perfect technicals | ❌ Missed | ✅ Captured with caution |
| MSFT clean breakout | ✅ Passed | ✅ Normal size |

**Institutional desks don't filter momentum — they size it.**

---

## 🚦 The Three Warning Levels

| Warning | Fund Score | Color | Position Sizing | Rationale |
|---------|-----------|-------|-----------------|-----------|
| **✅ Strong** | ≥ 80 | Green | 100% (normal size) | Quality company. Sleep well. |
| **⚠️ Moderate** | 50–79 | Orange | 75% (reduce 25%) | Decent but not exceptional. Tighten stop. |
| **🚨 Weak** | < 50 | Red | 50% (reduce 50%) or avoid | High risk. Only trade if technicals are exceptional. Never hold overnight. |
| **❓ No Data** | N/A | Gray | Technical stops only | Unknown quality. Assume worst case. |

---

## 🎯 How the Warning System Works

### Step 1: Technical Scan (Unchanged)
The engine runs its standard 1-minute analysis:
- EMA9/20 alignment
- VWAP position
- RVOL (relative volume)
- ATR% (volatility)
- PCA1 + PCA1_slope (momentum composite)
- Intraday Score (0-60)

### Step 2: Fundamental Fetch (Cached)
For each candidate, the engine fetches:
- Forward P/E, Trailing P/E
- Revenue Growth, Earnings Growth
- Profit Margin
- Beta, Debt-to-Equity

All via `yf.Ticker(ticker).info`, cached 1 hour.

### Step 3: Warning Classification
Based on Fund Score and user-defined thresholds:
- **Weak Threshold** (default 50): Below this = 🚨 Weak
- **Strong Threshold** (default 80): Above this = ✅ Strong
- Between = ⚠️ Moderate

### Step 4: Results Display
All candidates appear in the table. The **Fund Warning** column shows the risk level. Traders adjust size accordingly.

---

## 📊 Position Sizing Rules

### Default Rules (Adjustable in Sidebar)

| Fund Warning | Position Size | Stop Behavior | Overnight Hold? |
|-------------|---------------|---------------|-----------------|
| ✅ Strong | 100% | Normal 1:3 R/R | Acceptable |
| ⚠️ Moderate | 75% | Tighten to 1:2 | Avoid if possible |
| 🚨 Weak | 50% or skip | Tighten to 1:1.5 | **Never** |
| ❓ No Data | 50% | Technical only | **Never** |

### Example

You have a $20,000 account, 2% risk per trade = $400 risk.

| Ticker | Technical Score | Fund Warning | Adjusted Size | Risk |
|--------|----------------|--------------|---------------|------|
| MSFT | 55 | ✅ Strong | $400 | Normal |
| GME | 58 | 🚨 Weak | $200 | Reduced 50% |
| AMC | 60 | 🚨 Weak | Skip or $200 | Reduced 50% |
| AMD | 52 | ⚠️ Moderate | $300 | Reduced 25% |

---

## 🔄 Recommended Operating Schedule

Same as v4/v5:

| Time | Scan Type | Focus |
|------|-----------|-------|
| **10:15** | Full Scan | First valid scan with 45min of data |
| **10:15–10:30** | Review & Enter | Top candidates by Technical Score. Check Fund Warning for sizing. |
| **11:15** | Full Scan | Morning check + new setups |
| **11:30+** | No New Entries | Manage existing positions only |
| **13:30** | Full Scan | Afternoon re-activation (selective) |
| **14:45** | Positions Only | Tighten stops, prepare exits |
| **15:15** | Positions Only | Final exit check |

---

## 🎛️ Sidebar Settings

### Warning Thresholds
- **Weak Fundamentals Warning (< X)**: Default 50. Candidates below this get 🚨.
- **Strong Fundamentals Highlight (≥ X)**: Default 80. Candidates above this get ✅.

### Show Fundamental Sub-scores
When enabled, the results table displays:
- `Fund Score` (0-100)
- `Fund Warning` (✅/⚠️/🚨/❓)
- `Fund Valuation`, `Fund Growth`, `Fund Profit`, `Fund Risk`
- `Sector`

---

## 🔍 Deep Dive Research

Below the results table, select any ticker and click **"Research [TICKER]"** to navigate to Page 4 (`Ticker_summary`) for full narrative, all metrics, and historical context.

---

## ⚡ Quick Action Panel

Below the main table, a dedicated panel shows **only your open positions** as action cards with:
- Live P&L
- Current stop/target
- Position Status
- One-click close button

---

## ⚠️ Key Assumptions & Limitations

1. **Fundamental Data Lag:** `yf.Ticker().info` may be 1-24 hours delayed. For intraday decisions, this is acceptable.
2. **Not All Tickers Have Fundamentals:** Small-cap or international stocks may return no data. These get ❓ No Data warning.
3. **Warning System is Guidance, Not Law:** A 🚨 Weak stock with exceptional technicals (Ready + Strong Intraday + RVOL > 5) may still be worth a reduced-size trade. Use judgment.
4. **Manual Execution:** The scanner does not connect to any broker API. All orders must be placed manually on E*TRADE.
5. **Intraday Only:** All positions should be closed by market close (16:00 ET).
6. **PCA Minimum:** Requires 30+ bars (30 minutes) for statistical validity.

---

## 🧪 Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1 | 2026-08-08 | Initial intraday engine |
| v2 | 2026-08-08 | Fixed ATR, normalized PCA |
| v3 | 2026-08-08 | Position Status, color-coded styling |
| v4 | 2026-08-08 | Manual position tracker, stop engine |
| v5 | 2026-08-08 | Fundamental overlay, Combined Score, quality gate |
| **v5.1** | **2026-08-08** | **Replaced gate with warning system. All candidates pass. Fundamentals guide sizing, not entry.** |
""")