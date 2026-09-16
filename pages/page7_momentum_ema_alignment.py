# ======================================================================
# 🚀 PAGE 7B — MOMENTUM + EARLY DEVELOPMENT + FULL ALIGNMENT
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib

st.set_page_config(layout="wide", page_title="Momentum + EMA Structure")

st.caption(
    "Version: V2 — Momentum + Early Development + Full Alignment "
    "+ Structure_Type"
)
st.title("🚀 Momentum + EMA Structure — Early + Full Alignment")

EST = ZoneInfo("America/New_York")

# ============================================================
# LOAD UNIVERSE
# ============================================================

def _load_universe():
    try:
        import utils.data_fetch as data_fetch_module
        data_fetch_module = importlib.reload(data_fetch_module)
        tickers = data_fetch_module.load_universe()
        return sorted(
            set(
                str(x).strip().upper()
                for x in tickers
                if x
            )
        )
    except Exception as e:
        st.error(f"Unable to load universe: {e}")
        return []

# ============================================================
# MARKET DATA FETCH
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_market_batch(tickers_tuple):

    tickers = list(tickers_tuple)

    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    try:
        raw_daily = yf.download(
            tickers,
            period="3mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        raw_intra = yf.download(
            tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        return raw_daily, raw_intra

    except Exception as e:
        st.error(f"Market data download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ============================================================
# HELPERS
# ============================================================

def _flatten_columns(df):

    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):

        known = {
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        }

        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if all(x in known for x in lvl0):
            df.columns = lvl0
        elif all(x in known for x in lvl1):
            df.columns = lvl1

    return df


def _extract_ticker_slice(batch, ticker):

    if batch is None or batch.empty:
        return pd.DataFrame()

    ticker = str(ticker).strip().upper()

    if not isinstance(batch.columns, pd.MultiIndex):
        return _flatten_columns(batch.copy())

    try:
        lvl0 = set(
            str(x).strip().upper()
            for x in batch.columns.get_level_values(0)
        )
        lvl1 = set(
            str(x).strip().upper()
            for x in batch.columns.get_level_values(1)
        )

        if ticker in lvl0:
            result = batch[ticker].copy()
        elif ticker in lvl1:
            result = batch.xs(
                ticker,
                axis=1,
                level=1,
                drop_level=True
            ).copy()
        else:
            return pd.DataFrame()

        return _flatten_columns(result)

    except Exception:
        return pd.DataFrame()


def _to_eastern_index(df):

    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    eastern = ZoneInfo("America/New_York")

    if idx.tz is not None:
        idx = idx.tz_convert(eastern)
    else:
        idx = idx.tz_localize(eastern)

    df.index = idx
    return df

# ============================================================
# PAGE 7B ENGINE — MOMENTUM + EMA STRUCTURE
# ============================================================

def page7b_engine(
    tickers,
    batch_daily,
    batch_intra,
    min_price,
    max_price,
    min_momentum_score
):

    rows = []

    for ticker in tickers:

        daily_df = _flatten_columns(
            _extract_ticker_slice(batch_daily, ticker)
        )
        intra_df = _flatten_columns(
            _extract_ticker_slice(batch_intra, ticker)
        )

        if daily_df.empty or intra_df.empty:
            continue

        if "Close" not in daily_df.columns:
            continue

        if "Close" not in intra_df.columns:
            continue

        daily_df = daily_df.dropna(subset=["Close"])
        intra_df = intra_df.dropna(subset=["Close"])

        if len(daily_df) < 40:
            continue

        # ----------------------------------------------------
        # INTRADAY TO EASTERN + REGULAR SESSION
        # ----------------------------------------------------
        intra_df = _to_eastern_index(intra_df).sort_index()
        intra_df = intra_df.between_time("09:30", "16:00")

        if intra_df.empty:
            continue

        close_raw = (
            pd.to_numeric(
                intra_df["Close"],
                errors="coerce"
            )
            .dropna()
            .values
            .astype(float)
        )

        if len(close_raw) < 5:
            continue

        current_price = float(close_raw[-1])

        if (
            current_price < min_price
            or current_price > max_price
        ):
            continue

        # ====================================================
        # MOMENTUM COMPONENTS (ESSENTIAL)
        # ====================================================

        last5 = close_raw[-5:]

        price_change_5bar_pct = (
            (
                last5[-1] - last5[0]
            )
            / last5[0]
            * 100
            if last5[0] > 0
            else 0.0
        )

        ema9_series = (
            pd.Series(close_raw)
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
            .values
        )

        if ema9_series[-5] > 0:
            ema9_slope_10_pct = (
                (
                    ema9_series[-1]
                    - ema9_series[-5]
                )
                / ema9_series[-5]
                * 100
            )
        else:
            ema9_slope_10_pct = 0.0

        # Velocity score (same thresholds as momentum page)
        if ema9_slope_10_pct > 0.60:
            velocity_score = 4
        elif ema9_slope_10_pct > 0.30:
            velocity_score = 3
        elif ema9_slope_10_pct > 0.15:
            velocity_score = 2
        elif ema9_slope_10_pct > 0.00:
            velocity_score = 1
        else:
            velocity_score = 0

        # RVOL
        vol_i = (
            pd.to_numeric(
                intra_df["Volume"],
                errors="coerce"
            )
            .fillna(0)
            .values
        )

        intraday_total_volume = float(
            np.sum(vol_i)
        )

        vol_d = (
            pd.to_numeric(
                daily_df["Volume"],
                errors="coerce"
            )
            .dropna()
            .values
            .astype(float)
        )

        if len(vol_d) < 20:
            continue

        avg_volume_20d = float(
            np.mean(vol_d[-20:])
        )

        latest_bar_time = intra_df.index[-1]

        market_open = latest_bar_time.replace(
            hour=9,
            minute=30,
            second=0,
            microsecond=0
        )

        elapsed_minutes = (
            latest_bar_time - market_open
        ).total_seconds() / 60.0

        elapsed_minutes = max(
            1.0,
            min(
                elapsed_minutes,
                390.0
            )
        )

        expected_volume_by_now = (
            avg_volume_20d
            * (elapsed_minutes / 390.0)
        )

        if expected_volume_by_now > 0:
            rvol = (
                intraday_total_volume
                / expected_volume_by_now
            )
        else:
            rvol = 1.0

        if rvol > 5.0:
            rvol_score = 4
        elif rvol > 3.0:
            rvol_score = 3
        elif rvol > 2.0:
            rvol_score = 2
        elif rvol > 1.2:
            rvol_score = 1
        else:
            rvol_score = 0

        momentum_score = (
            velocity_score
            + rvol_score
        )

        if momentum_score < min_momentum_score:
            continue

        # ====================================================
        # EMA STRUCTURE — EARLY DEVELOPMENT + FULL ALIGNMENT
        # ====================================================

        ema9_i = (
            intra_df["Close"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
            .values
            .astype(float)
        )

        ema20_i = (
            intra_df["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
            .values
            .astype(float)
        )

        ema9_now = float(ema9_i[-1])
        ema20_now = float(ema20_i[-1])

        ema9_slope = float(
            ema9_i[-1] - ema9_i[-5]
        )
        ema20_slope = float(
            ema20_i[-1] - ema20_i[-5]
        )

        last5_ema9 = ema9_i[-5:]

        # --- Early Development (Pattern B) ---
        bar1_to_3_below = bool(
            np.all(
                last5[:3] <= last5_ema9[:3]
            )
        )
        bar4_above = last5[3] > last5_ema9[3]
        bar5_above = last5[4] > last5_ema9[4]

        patternB = (
            bar1_to_3_below
            and bar4_above
            and bar5_above
            and (ema9_now > ema20_now)
            and (ema9_slope > 0)
            and (ema20_slope > 0)
            and (current_price > ema20_now)
        )

        # --- Full Alignment ---
        full_alignment = (
            np.all(last5 > last5_ema9)
            and (ema9_now > ema20_now)
            and (ema9_slope > 0)
            and (ema20_slope > 0)
            and (current_price > ema20_now)
        )

        # --- Structure Type ---
        if patternB:
            structure_type = "Early Development"
        elif full_alignment:
            structure_type = "Full Alignment"
        else:
            structure_type = None

        if structure_type is None:
            continue

        # Combined score: momentum + structural bonus
        if structure_type == "Early Development":
            structure_bonus = 5
        else:
            structure_bonus = 3

        combined_score = (
            momentum_score
            + structure_bonus
        )

        rows.append({
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "Momentum_Score": round(momentum_score, 2),
            "EMA9_Slope": round(ema9_slope, 4),
            "EMA20_Slope": round(ema20_slope, 4),
            "RVOL": round(rvol, 2),
            "Price_Change_5B_Pct": round(price_change_5bar_pct, 2),
            "Structure_Type": structure_type,
            "Combined_Score": combined_score,
            "Timestamp": latest_bar_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        by=["Combined_Score", "Momentum_Score"],
        ascending=False
    )

    return df

# ============================================================
# PAGE UI
# ============================================================

st.markdown("### 🔍 Price Boundaries Filter")

min_price = st.number_input(
    "Minimum Price ($)",
    value=40.0,
    min_value=0.0,
    max_value=500.0
)

max_price = st.number_input(
    "Maximum Price ($)",
    value=120.0,
    min_value=0.0,
    max_value=500.0
)

st.markdown("### 🎛️ Momentum Score Filter")

min_momentum_score = st.number_input(
    "Minimum Momentum Score",
    value=4.0,
    min_value=0.0,
    max_value=8.0,
    step=0.5
)

run_scan = st.button(
    "Run Momentum + EMA Structure Scan"
)

if run_scan:

    st.cache_data.clear()

    now_est = datetime.now(EST)

    st.markdown(
        "⏱️ Scan Time: "
        f"**{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**"
    )

    universe = _load_universe()

    if not universe:
        st.error("Universe is empty.")
    else:
        raw_daily, raw_intra = fetch_market_batch(
            tuple(universe)
        )

        ranking = page7b_engine(
            universe,
            raw_daily,
            raw_intra,
            min_price,
            max_price,
            min_momentum_score
        )

        if ranking.empty:
            st.info(
                "No tickers with momentum + EMA structure "
                "(early development or full alignment) "
                "under current filters."
            )
        else:
            st.subheader(
                f"🚀 Momentum + EMA Structure — "
                f"{len(ranking)} Tickers"
            )

            st.caption(
                "Structure_Type = Early Development or Full Alignment. "
                "Combined_Score = Momentum_Score + structural bonus."
            )

            st.dataframe(
                ranking,
                hide_index=True,
                use_container_width=True
            )
