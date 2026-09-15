# ==============================================================================
# 📈 INSTITUTIONAL EMA ALIGNMENT MODEL
# PURPOSE:
# Identify institutional-quality tickers showing early EMA alignment.
# Opportunity categorization describes the price relationship to the
# previous-day close without changing the EMA qualification logic.

# ==============================================================================
import importlib
import streamlit as st
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(layout="wide", page_title="Institutional EMA Alignment")

st.caption(
    "Version: V6 2026-09-15 — Institutional EMA Alignment + "
    "Opportunity Categorization + Price Increase Score + "
    "Development v4 + Rejection Diagnostics"
)

st.title("📈 Institutional EMA Alignment Model")

# ============================================================
# MODEL PARAMETERS
# ============================================================

MIN_DAILY_HISTORY = 40
MIN_INTRADAY_BARS = 5
MIN_REAL_DAY_BARS = 10
MIN_AVG_VOLUME_20D = 250000

# ============================================================
# SESSION STATE
# ============================================================

if "ema_alignment_raw_ranking" not in st.session_state:
    st.session_state["ema_alignment_raw_ranking"] = pd.DataFrame()

if "ema_alignment_rejections" not in st.session_state:
    st.session_state["ema_alignment_rejections"] = pd.DataFrame()

# ============================================================
# DATAFRAME HELPERS
# ============================================================

def _flatten_columns(df):

    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):

        if df.columns.nlevels == 2:

            known_fields = {
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume"
            }

            level_0 = df.columns.get_level_values(0)
            level_1 = df.columns.get_level_values(1)

            if all(x in known_fields for x in level_0):
                df.columns = level_0

            elif all(x in known_fields for x in level_1):
                df.columns = level_1

    return df

def _extract_ticker_slice(batch, ticker):

    if batch is None or batch.empty:
        if str(ticker).upper() == "TEM":
            print("TEM DIAGNOSTIC: batch is None or empty")
        return pd.DataFrame()

    ticker = str(ticker).strip().upper()

    if not isinstance(batch.columns, pd.MultiIndex):
        result = _flatten_columns(batch.copy())
        if ticker == "TEM":
            print("TEM DIAGNOSTIC: non-MultiIndex batch")
            print("TEM extracted shape:", result.shape)
            print("TEM columns:", list(result.columns))
            print(result.tail(10))
        return result

    try:
        level0 = set(str(x).strip().upper()
                     for x in batch.columns.get_level_values(0))
        level1 = set(str(x).strip().upper()
                     for x in batch.columns.get_level_values(1))

        if ticker in level0:
            result = batch[ticker].copy()
        elif ticker in level1:
            result = batch.xs(
                ticker, axis=1, level=1, drop_level=True
            ).copy()
        else:
            if ticker == "TEM":
                print("TEM DIAGNOSTIC: ticker not found in either MultiIndex level")
                print("Level 0 sample:", list(level0)[:20])
                print("Level 1 sample:", list(level1)[:20])
            return pd.DataFrame()

        result = _flatten_columns(result)

        if ticker == "TEM":
            print("TEM DIAGNOSTIC: MultiIndex extraction successful")
            print("TEM extracted shape:", result.shape)
            print("TEM columns:", list(result.columns))
            if not result.empty:
                print("TEM first index:", result.index.min())
                print("TEM last index:", result.index.max())
                print("TEM tail:")
                print(result.tail(10))

        return result

    except Exception as e:
        if ticker == "TEM":
            print("TEM DIAGNOSTIC: extraction exception:", repr(e))
            print("Batch columns:", batch.columns)
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
# UNIVERSE
# ============================================================

def _load_universe():

    try:
        import inspect
        import sys
        import importlib
        import utils.data_fetch as data_fetch_module
        import data.us_universe_list as universe_module

        # ----------------------------------------------------
        # TEMPORARY FORCE RELOAD
        # ----------------------------------------------------

        universe_module = importlib.reload(universe_module)
        data_fetch_module = importlib.reload(data_fetch_module)

        # Get load_universe AFTER reloading data_fetch
        load_universe = data_fetch_module.load_universe

        # ----------------------------------------------------
        # UNIVERSE SOURCE DIAGNOSTIC
        # ----------------------------------------------------

        st.write("### 🔎 Universe Source Diagnostic")

        st.write("### 🐍 PYTHON ENVIRONMENT")

        st.write(
            f"Python executable: `{sys.executable}`"
        )

        st.write(
            f"Python version: `{sys.version}`"
        )

        st.write(
            f"Page 5 file: **{__file__}**"
        )

        st.write(
            f"data_fetch file: **{data_fetch_module.__file__}**"
        )

        st.write(
            f"us_universe file: **{universe_module.__file__}**"
        )

        direct_count = len(universe_module.us_universe)

        st.write(
            f"Direct us_universe length: **{direct_count:,}**"
        )

        # ----------------------------------------------------
        # LOAD UNIVERSE
        # ----------------------------------------------------

        tickers = load_universe()

        if tickers is None:
            st.error("load_universe() returned None.")
            return []

        st.write(
            f"load_universe() length: **{len(tickers):,}**"
        )

        # ----------------------------------------------------
        # SHOW FUNCTION SOURCE
        # ----------------------------------------------------

        st.write(
            "load_universe source:"
        )

        st.code(
            inspect.getsource(load_universe),
            language="python"
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        normalized_tickers = sorted(
            set(
                str(x).strip().upper()
                for x in tickers
                if x
            )
        )

        st.write(
            f"Normalized universe length: "
            f"**{len(normalized_tickers):,}**"
        )

        return normalized_tickers

    except Exception as e:

        st.error(
            "Unable to load the master US universe "
            "from utils/data_fetch.py: "
            f"{e}"
        )

        return []

# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_clean_market_batch(tickers_tuple):

    ticker_list = list(tickers_tuple)

    if not ticker_list:
        return pd.DataFrame(), pd.DataFrame()

    try:

        raw_daily = yf.download(
            ticker_list,
            period="3mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        raw_intra = yf.download(
            ticker_list,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        return raw_daily, raw_intra

    except Exception as e:

        st.error(
            f"Market data download failed: {e}"
        )

        return pd.DataFrame(), pd.DataFrame()

# ============================================================
# PRICE INCREASE SCORE
# ============================================================

def price_increase_score(price_increase_pct):

    try:
        x = float(price_increase_pct)

    except (TypeError, ValueError):
        return 0

    if not np.isfinite(x):
        return 0

    if x >= 1.00:
        return 3

    if x >= 0.50:
        return 2

    if x >= 0.25:
        return 1

    return 0

def price_increase_label(score):

    return {
        3: "Strong",
        2: "Good",
        1: "Developing",
        0: "Weak"
    }.get(
        int(score),
        "Weak"
    )

# ============================================================
# OPPORTUNITY CATEGORY
# ============================================================

def opportunity_category(gap_vs_prev_close):
    """
    Classify a QUALIFIED EMA ticker by its relationship
    to the previous trading day's close.

    Informational only. This does NOT reject or qualify a ticker.
    """
    try:
        x = float(gap_vs_prev_close)
    except (TypeError, ValueError):
        return "Unknown"

    if not np.isfinite(x):
        return "Unknown"

    if x <= -5.0:
        return "Deep Recovery"
    if x < -2.0:
        return "Moderate Recovery"
    if x <= 2.0:
        return "Near Previous Close"
    if x <= 5.0:
        return "Continuation"
    return "Extended"


def session_phase(timestamp):
    """Informational market-session classification."""
    try:
        total_minutes = timestamp.hour * 60 + timestamp.minute
    except Exception:
        return "Unknown"

    if total_minutes < 9 * 60 + 30:
        return "Pre-Market"
    if total_minutes < 10 * 60 + 10:
        return "Early Session (09:30-10:10)"
    if total_minutes < 10 * 60 + 30:
        return "Morning (10:10-10:30)"
    if total_minutes < 12 * 60:
        return "Late Morning (10:30-12:00)"
    if total_minutes < 14 * 60:
        return "Midday (12:00-14:00)"
    return "Afternoon (14:00-16:00)"


# ============================================================
# DISPLAY COLORING
# ============================================================

def color_score_columns(df):

    style = pd.DataFrame(
        "",
        index=df.index,
        columns=df.columns
    )

    # --------------------------------------------------------
    # PRICE SCORE
    # --------------------------------------------------------

    if "Price_Increase_Score" in df.columns:

        for i, v in enumerate(
            df["Price_Increase_Score"]
        ):

            try:
                v = float(v)

            except Exception:
                continue

            if v >= 3:

                css = (
                    "background-color:#006400;"
                    "color:white;"
                    "font-weight:bold;"
                )

            elif v >= 2:

                css = (
                    "background-color:#32CD32;"
                    "color:black;"
                    "font-weight:bold;"
                )

            elif v >= 1:

                css = (
                    "background-color:#FFD700;"
                    "color:black;"
                    "font-weight:bold;"
                )

            else:

                css = (
                    "background-color:#FF9800;"
                    "color:white;"
                )

            style.iloc[
                i,
                df.columns.get_loc(
                    "Price_Increase_Score"
                )
            ] = css

    # --------------------------------------------------------
    # PRICE LABEL
    # --------------------------------------------------------

    if "Price_Increase_Label" in df.columns:

        for i, v in enumerate(
            df["Price_Increase_Label"]
        ):

            if v == "Strong":

                css = (
                    "background-color:#006400;"
                    "color:white;"
                    "font-weight:bold;"
                )

            elif v == "Good":

                css = (
                    "background-color:#32CD32;"
                    "color:black;"
                    "font-weight:bold;"
                )

            elif v == "Developing":

                css = (
                    "background-color:#FFD700;"
                    "color:black;"
                    "font-weight:bold;"
                )

            else:

                css = (
                    "background-color:#FF9800;"
                    "color:white;"
                )

            style.iloc[
                i,
                df.columns.get_loc(
                    "Price_Increase_Label"
                )
            ] = css

    return style

# ============================================================
# REJECTION ROW TEMPLATE
# ============================================================

def _rejection_row(
    ticker,
    status="REJECTED",
    reason=""
):

    return {

        "Ticker": ticker,

        "Status": status,

        "Reason": reason,

        "Price": np.nan,

        "Opportunity_Category": "N/A",

        "Session_Phase": "N/A",

        "Avg_Volume_20d": np.nan,

        "EMA_Score": np.nan,

        "Price_Increase_%_5Bars": np.nan,

        "Price_Increase_Score": np.nan,

        "Price_Above_EMA9": "N/A",

        "Price_Above_EMA20": "N/A",

        "EMA9_Above_EMA20": "N/A",

        "EMA9_Slope_Pos": "N/A",

        "EMA20_Slope_Pos": "N/A",

        "Last5_Above_EMA9": "N/A",

        "LastBar_Higher_3": "N/A",

        "LastBar_Higher_4": "N/A",

        "Development_Signal": "N/A",

        "Development_Reason": "N/A",

        "Development_Current_Above_EMA9": "N/A",

        "Last5_Rising_Trend": "N/A",

        "Daily_Data": "N/A",

        "Intraday_Data": "N/A",

        "Latest_Real_Day": "N/A"
    }

# ============================================================
# EMA ALIGNMENT ENGINE
# ============================================================

def ema_alignment_engine(
    tickers,
    batch_daily,
    batch_intra,
    min_price,
    max_price
):

    rows = []

    rejection_rows = []

    # ========================================================
    # NO DAILY DATA
    # ========================================================

    if batch_daily is None or batch_daily.empty:

        for ticker in tickers:

            rejection_rows.append(
                _rejection_row(
                    ticker,
                    reason="No daily data returned"
                )
            )

        return (
            pd.DataFrame(),
            pd.DataFrame(rejection_rows)
        )

    # ========================================================
    # NO INTRADAY DATA
    # ========================================================

    if batch_intra is None or batch_intra.empty:

        for ticker in tickers:

            rejection_rows.append(
                _rejection_row(
                    ticker,
                    reason="No intraday data returned"
                )
            )

        return (
            pd.DataFrame(),
            pd.DataFrame(rejection_rows)
        )

    # ========================================================
    # DETERMINE AVAILABLE TICKERS
    # ========================================================

    try:

        available_daily = set(
            batch_daily.columns.get_level_values(0)
        )

        available_intra = set(
            batch_intra.columns.get_level_values(0)
        )

        active_pool = sorted(
            set(tickers)
            .intersection(available_daily)
            .intersection(available_intra)
        )

    except Exception:

        active_pool = sorted(
            set(tickers)
        )

    # ========================================================
    # TICKERS MISSING FROM YAHOO DATA
    # ========================================================

    active_set = set(active_pool)

    for ticker in sorted(
        set(tickers) - active_set
    ):

        r = _rejection_row(ticker)

        r["Reason"] = (
            "Ticker missing from daily or "
            "intraday Yahoo data"
        )

        rejection_rows.append(r)

    # ========================================================
    # PROCESS EACH TICKER
    # ========================================================

    for ticker in active_pool:

        r = _rejection_row(ticker)

        # ----------------------------------------------------
        # EXTRACT DATA
        # ----------------------------------------------------

        daily_df = _flatten_columns(
            _extract_ticker_slice(
                batch_daily,
                ticker
            )
        )

        intraday_df = _flatten_columns(
            _extract_ticker_slice(
                batch_intra,
                ticker
            )
        )

        if str(ticker).upper() == "TEM":
            print("\n" + "=" * 80)
            print("TEM DATA DIAGNOSTIC")
            print("=" * 80)
            print("Daily rows after extraction:", len(daily_df))
            print("Intraday rows after extraction:", len(intraday_df))
            print("Daily columns:", list(daily_df.columns))
            print("Intraday columns:", list(intraday_df.columns))
            if not daily_df.empty:
                print("Daily first index:", daily_df.index.min())
                print("Daily last index:", daily_df.index.max())
                print(daily_df.tail(5))
            if not intraday_df.empty:
                print("Intraday first index:", intraday_df.index.min())
                print("Intraday last index:", intraday_df.index.max())
                print(intraday_df.tail(10))
            print("=" * 80)



        r["Daily_Data"] = (
            "OK"
            if not daily_df.empty
            else "Missing"
        )

        r["Intraday_Data"] = (
            "OK"
            if not intraday_df.empty
            else "Missing"
        )

        # ----------------------------------------------------
        # EMPTY DATA
        # ----------------------------------------------------

        if daily_df.empty:

            r["Reason"] = "Daily data empty"

            rejection_rows.append(r)

            continue

        if intraday_df.empty:

            r["Reason"] = "Intraday data empty"

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        if "Close" not in daily_df.columns:

            r["Reason"] = (
                "Daily Close column missing"
            )

            rejection_rows.append(r)

            continue

        if "Close" not in intraday_df.columns:

            r["Reason"] = (
                "Intraday Close column missing"
            )

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # CLEAN CLOSE DATA
        # ----------------------------------------------------

        daily_df = daily_df.dropna(
            subset=["Close"]
        )

        intraday_df = intraday_df.dropna(
            subset=["Close"]
        )


        r["Daily_Data"] = (
            f"OK ({len(daily_df)} rows)"
        )

        r["Intraday_Data"] = (
            f"OK ({len(intraday_df)} rows)"
        )

        # ----------------------------------------------------
        # DAILY HISTORY
        # ----------------------------------------------------

        if len(daily_df) < MIN_DAILY_HISTORY:

            r["Reason"] = (
                f"Daily history below "
                f"{MIN_DAILY_HISTORY} rows"
            )

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # CONVERT INTRADAY TO EASTERN
        # ----------------------------------------------------

        intraday_df = _to_eastern_index(
            intraday_df
        ).sort_index()

        # ----------------------------------------------------
        # REGULAR SESSION ONLY
        # ----------------------------------------------------

        intraday_df = intraday_df.between_time(
            "09:30",
            "16:00"
        )

        if intraday_df.empty:

            r["Reason"] = (
                "No regular-session intraday bars"
            )

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # FIND REAL TRADING DAY
        # ----------------------------------------------------

        day_counts = pd.Series(
            intraday_df.index.date
        ).value_counts()


        eligible_days = sorted(
            day_counts[
                day_counts >= MIN_REAL_DAY_BARS
            ].index
        )

        if not eligible_days:

            r["Reason"] = (
                "No trading day with at least "
                f"{MIN_REAL_DAY_BARS} intraday bars"
            )

            rejection_rows.append(r)

            continue

        latest_real_day = eligible_days[-1]

        intraday_df = intraday_df[
            intraday_df.index.date
            == latest_real_day
        ].copy()


        r["Latest_Real_Day"] = str(
            latest_real_day
        )

        if len(intraday_df) < MIN_REAL_DAY_BARS:

            r["Reason"] = (
                f"Latest real day has fewer than "
                f"{MIN_REAL_DAY_BARS} bars"
            )

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # INTRADAY CLOSE ARRAY
        # ----------------------------------------------------

        close_raw = (
            pd.to_numeric(
                intraday_df["Close"],
                errors="coerce"
            )
            .dropna()
            .values
            .astype(float)
        )

        if len(close_raw) < MIN_INTRADAY_BARS:

            r["Reason"] = (
                f"Intraday bars below "
                f"{MIN_INTRADAY_BARS}"
            )

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # CURRENT INTRADAY PRICE
        # ----------------------------------------------------

        current_price = float(
            close_raw[-1]
        )

        r["Price"] = round(
            current_price,
            2
        )

        if not np.isfinite(
            current_price
        ):

            r["Reason"] = (
                "Current intraday price invalid"
            )

            rejection_rows.append(r)

            continue

        # ====================================================
        # PRICE FILTER
        # ====================================================

        if (
            current_price < min_price
            or current_price > max_price
        ):

            r["Reason"] = (
                f"Price outside range "
                f"${min_price:.2f}–${max_price:.2f}"
            )

            rejection_rows.append(r)

            continue

        # ====================================================
        # VOLUME FILTER
        # ====================================================

        if "Volume" not in daily_df.columns:

            r["Reason"] = (
                "Daily Volume column missing"
            )

            rejection_rows.append(r)

            continue

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

            r["Reason"] = (
                "Fewer than 20 valid "
                "daily volume observations"
            )

            rejection_rows.append(r)

            continue

        avg_volume_20d = float(
            np.mean(
                vol_d[-20:]
            )
        )

        r["Avg_Volume_20d"] = round(
            avg_volume_20d,
            0
        )

        if (
            not np.isfinite(avg_volume_20d)
            or avg_volume_20d < MIN_AVG_VOLUME_20D
        ):

            r["Reason"] = (
                f"20-day average volume below "
                f"{MIN_AVG_VOLUME_20D:,}"
            )

            rejection_rows.append(r)

            continue

        # ====================================================
        # EMA CALCULATIONS
        # ====================================================

        ema9_i = (
            intraday_df["Close"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
            .values
            .astype(float)
        )

        ema20_i = (
            intraday_df["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
            .values
            .astype(float)
        )

        if len(close_raw) < 5:

            r["Reason"] = (
                "Fewer than 5 valid "
                "intraday closes"
            )

            rejection_rows.append(r)

            continue

        # ----------------------------------------------------
        # CURRENT EMA VALUES
        # ----------------------------------------------------

        ema9_now = float(
            ema9_i[-1]
        )

        ema20_now = float(
            ema20_i[-1]
        )

        # ----------------------------------------------------
        # EMA SLOPES
        # ----------------------------------------------------

        ema9_slope = float(
            ema9_i[-1]
            - ema9_i[-5]
        )

        ema20_slope = float(
            ema20_i[-1]
            - ema20_i[-5]
        )

        # ====================================================
        # LAST 5 BARS
        # ====================================================

        last5 = close_raw[-5:]

        last5_ema9 = ema9_i[-5:]

        # ----------------------------------------------------
        # PRICE INCREASE
        # ----------------------------------------------------

        price_increase_pct = (

            (
                last5[-1]
                - last5[0]
            )
            / last5[0]
            * 100

            if last5[0] > 0

            else 0.0
        )

        p_score = price_increase_score(
            price_increase_pct
        )

        # ====================================================
        # EMA CONDITIONS
        # ====================================================

        cond_price_above_ema9 = (
            current_price > ema9_now
        )

        cond_price_above_ema20 = (
            current_price > ema20_now
        )

        cond_ema9_above_ema20 = (
            ema9_now > ema20_now
        )

        cond_ema9_slope_pos = (
            ema9_slope > 0
        )

        cond_ema20_slope_pos = (
            ema20_slope > 0
        )

        cond_last5_above_ema9 = bool(
            np.all(
                last5 > last5_ema9
            )
        )

        #cond_lastbar_higher_3 = bool(
        #    last5[-1] > last5[-3]
        #)

        #cond_lastbar_higher_4 = bool(
        #    last5[-1] > last5[-4]
        #)

        # ====================================================
        # DEVELOPMENT SIGNAL — DIAGNOSTIC ONLY
        # ====================================================
        # DEVELOPMENT v4 — flexible early-transition hypothesis.
        #
        # Allow either of these recent patterns:
        #   A) only the current (5th) bar is above EMA9, while the
        #      preceding 4 bars are at/below EMA9; OR
        #   B) the 4th and 5th bars are above EMA9, while the
        #      preceding 3 bars are at/below EMA9.
        #
        # This captures an early transition that has already held
        # above EMA9 for two consecutive bars without requiring
        # a single-bar crossover pattern.
        #
        # This is DIAGNOSTIC ONLY and does not replace Strong.

        # DEVELOPMENT v4 — flexible early-transition hypothesis.
        #
        # Allow either of these recent patterns:
        #   A) only the current (5th) bar is above EMA9, while the
        #      preceding 4 bars are at/below EMA9; OR
        #   B) the 4th and 5th bars are above EMA9, while the
        #      preceding 3 bars are at/below EMA9.
        #
        # This captures an early transition that has already held
        # above EMA9 for two consecutive bars without requiring a
        # single-bar crossover pattern.
        recent2_above_ema9 = bool(
            np.all(last5[-2:] > last5_ema9[-2:])
        )

        preceding3_below_ema9 = bool(
            np.all(last5[:3] <= last5_ema9[:3])
        )

        previous4_below_ema9 = bool(
            np.all(last5[:-1] <= last5_ema9[:-1])
        )

        development_one_bar_transition = bool(
            current_price > ema9_now
            and previous4_below_ema9
        )

        development_two_bar_transition = bool(
            recent2_above_ema9
            and preceding3_below_ema9
        )

        development_recent_ema9_cross = bool(
            development_one_bar_transition
            or development_two_bar_transition
        )

        development_base_conditions = (
            cond_price_above_ema20
            and cond_ema9_above_ema20
            and cond_ema9_slope_pos
            and cond_ema20_slope_pos
        )

        development_signal = bool(
            development_base_conditions
            and development_recent_ema9_cross
        )

        if development_two_bar_transition:
            development_transition_type = "2-Bar Transition"
        elif development_one_bar_transition:
            development_transition_type = "1-Bar Transition"
        else:
            development_transition_type = "None"

        development_reasons = []

        if current_price <= ema9_now:
            development_reasons.append(
                "current price <= EMA9"
            )

        if not development_recent_ema9_cross:
            development_reasons.append(
                "recent EMA9 transition pattern not met: "
                "requires either current bar above EMA9 with previous 4 at/below, "
                "or bars 4-5 above EMA9 with bars 1-3 at/below"
            )

        if not cond_price_above_ema20:
            development_reasons.append("price <= EMA20")

        if not cond_ema9_above_ema20:
            development_reasons.append("EMA9 <= EMA20")

        if not cond_ema9_slope_pos:
            development_reasons.append("EMA9 slope not positive")

        if not cond_ema20_slope_pos:
            development_reasons.append("EMA20 slope not positive")

        if development_signal:
            if development_two_bar_transition:
                development_reason = (
                    "Bars 4-5 above EMA9 after bars 1-3 at/below EMA9, "
                    "with positive EMA9/EMA20 structure"
                )
            else:
                development_reason = (
                    "Current bar above EMA9 after previous 4 bars "
                    "at/below EMA9, with positive EMA9/EMA20 structure"
                )
        else:
            development_reason = "; ".join(
                development_reasons
            )

# EMA ALIGNMENT SCORE
        # ====================================================

        if str(ticker).upper() == "TEM":
            print("\n" + "=" * 80)
            print("TEM EMA DIAGNOSTIC")
            print("=" * 80)
            print("Latest intraday timestamp:", intraday_df.index[-1])
            print("Current price:", current_price)
            print("EMA9:", ema9_now)
            print("EMA20:", ema20_now)
            print("EMA9 slope:", ema9_slope)
            print("EMA20 slope:", ema20_slope)
            print("Last 5 closes:", last5)
            print("Last 5 EMA9:", last5_ema9)
            print("Price > EMA9:", cond_price_above_ema9)
            print("Price > EMA20:", cond_price_above_ema20)
            print("EMA9 > EMA20:", cond_ema9_above_ema20)
            print("EMA9 slope positive:", cond_ema9_slope_pos)
            print("EMA20 slope positive:", cond_ema20_slope_pos)
            print("All last 5 closes > EMA9:", cond_last5_above_ema9)
            print("=" * 80)

        ema_score = 0

        if cond_price_above_ema9:
            ema_score += 2

        if cond_price_above_ema20:
            ema_score += 2

        if cond_ema9_above_ema20:
            ema_score += 2

        if cond_ema9_slope_pos:
            ema_score += 1

        if cond_ema20_slope_pos:
            ema_score += 1

        # ====================================================
        # STORE DIAGNOSTIC CONDITIONS
        # ====================================================

        r.update({

            "EMA_Score":
                ema_score,

            "Price_Increase_%_5Bars":
                round(
                    price_increase_pct,
                    3
                ),

            "Price_Increase_Score":
                p_score,

            "Price_Above_EMA9":
                (
                    "PASS"
                    if cond_price_above_ema9
                    else "FAIL"
                ),

            "Price_Above_EMA20":
                (
                    "PASS"
                    if cond_price_above_ema20
                    else "FAIL"
                ),

            "EMA9_Above_EMA20":
                (
                    "PASS"
                    if cond_ema9_above_ema20
                    else "FAIL"
                ),

            "EMA9_Slope_Pos":
                (
                    "PASS"
                    if cond_ema9_slope_pos
                    else "FAIL"
                ),

            "EMA20_Slope_Pos":
                (
                    "PASS"
                    if cond_ema20_slope_pos
                    else "FAIL"
                ),

            "Last5_Above_EMA9":
                (
                    "PASS"
                    if cond_last5_above_ema9
                    else "FAIL"
                ),

            #"LastBar_Higher_3":
            #    (
            #        "PASS"
            #        if cond_lastbar_higher_3
            #        else "FAIL"
            #    ),

            #"LastBar_Higher_4":
            #    (
            #        "PASS"
            #        if cond_lastbar_higher_4
            #        else "FAIL"
            #    ),

            "Development_Signal":
                (
                    "DEVELOPMENT"
                    if development_signal
                    else "NO"
                ),

            "Development_Reason":
                development_reason,

            "Development_Current_Above_EMA9":
                (
                    "PASS"
                    if current_price > ema9_now
                    else "FAIL"
                ),

            "Development_Previous4_Below_EMA9":
                (
                    "PASS"
                    if previous4_below_ema9
                    else "FAIL"
                ),

            "Development_Recent2_Above_EMA9":
                (
                    "PASS"
                    if recent2_above_ema9
                    else "FAIL"
                ),

            "Development_Preceding3_Below_EMA9":
                (
                    "PASS"
                    if preceding3_below_ema9
                    else "FAIL"
                ),

            "Development_Transition_Type":
                development_transition_type
        })

        # ====================================================
        # DETERMINE FAILED HARD CONDITIONS
        # ====================================================

        failed = []

        if ema_score < 4:

            failed.append(
                "EMA score < 4/8"
            )

        if not cond_last5_above_ema9:

            failed.append(
                "not all last 5 closes > EMA9"
            )

        # ====================================================
        # REJECTED
        # ====================================================

        if failed:

            r["Reason"] = "; ".join(
                failed
            )

            rejection_rows.append(r)

            continue

        # ====================================================
        # QUALIFIED
        # ====================================================

        close_daily = (
            pd.to_numeric(
                daily_df["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if len(close_daily) >= 2:

            previous_close = float(
                close_daily.iloc[-2]
            )

        else:

            previous_close = current_price


        if previous_close > 0:

            gap_vs_prev_close = (
                (
                    current_price
                    - previous_close
                )
                / previous_close
                * 100
            )

        else:

            gap_vs_prev_close = 0.0

        opportunity_cat = opportunity_category(
            gap_vs_prev_close
        )

        session_cat = session_phase(
            intraday_df.index[-1]
        )

        r["Opportunity_Category"] = opportunity_cat
        r["Session_Phase"] = session_cat

        rows.append({

            "Ticker":
                ticker,

            "Opportunity_Category":
                opportunity_cat,

            "Session_Phase":
                session_cat,

            "Close":
                round(
                    current_price,
                    2
                ),

            "EMA_Alignment_Score":
                ema_score,

            "Price_Increase_%_5Bars":
                round(
                    price_increase_pct,
                    3
                ),

            "Price_Increase_Score":
                p_score,

            "Price_Increase_Label":
                price_increase_label(
                    p_score
                ),

            "EMA9":
                round(
                    ema9_now,
                    4
                ),

            "EMA20":
                round(
                    ema20_now,
                    4
                ),

            "EMA9_Above_EMA20":
                cond_ema9_above_ema20,

            "EMA9_Slope":
                round(
                    ema9_slope,
                    4
                ),

            "EMA20_Slope":
                round(
                    ema20_slope,
                    4
                ),

            "Development_Signal":
                (
                    "DEVELOPMENT"
                    if development_signal
                    else "NO"
                ),

            "Development_Reason":
                development_reason,

            "Development_Current_Above_EMA9":
                (
                    "PASS"
                    if current_price > ema9_now
                    else "FAIL"
                ),

            "Development_Previous4_Below_EMA9":
                (
                    "PASS"
                    if previous4_below_ema9
                    else "FAIL"
                ),

            "Development_Recent2_Above_EMA9":
                (
                    "PASS"
                    if recent2_above_ema9
                    else "FAIL"
                ),

            "Development_Preceding3_Below_EMA9":
                (
                    "PASS"
                    if preceding3_below_ema9
                    else "FAIL"
                ),

            "Development_Transition_Type":
                development_transition_type,

            "Avg_Volume_20d":
                round(
                    avg_volume_20d,
                    0
                ),

            "Previous_Close":
                round(
                    previous_close,
                    2
                ),

            "Gap_vs_Prev_Close_%":
                round(
                    gap_vs_prev_close,
                    3
                ),

            "Data_As_Of":
                intraday_df.index[-1].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "_Latest_Real_Day":
                str(
                    latest_real_day
                )
        })

        # Also put qualified ticker in diagnostics
        r["Status"] = "QUALIFIED"

        r["Reason"] = (
            "v6 hard EMA conditions passed: EMA score >= 4/8 "
            "and all last 5 closes > EMA9"
        )

        rejection_rows.append(r)

    # ========================================================
    # RANKING DATAFRAME
    # ========================================================

    ranking = pd.DataFrame(
        rows
    )

    if not ranking.empty:

        ranking = ranking.sort_values(

            [
                "EMA_Alignment_Score",
                "Price_Increase_Score",
                "Price_Increase_%_5Bars"
            ],

            ascending=[
                False,
                False,
                False
            ]

        ).reset_index(
            drop=True
        )

    # ========================================================
    # REJECTION DATAFRAME
    # ========================================================

    rejections = pd.DataFrame(
        rejection_rows
    )

    if not rejections.empty:

        rejections = rejections.sort_values(

            [
                "Status",
                "Ticker"
            ],

            ascending=[
                True,
                True
            ]

        ).reset_index(
            drop=True
        )

    return (
        ranking,
        rejections
    )

# ============================================================
# USER FILTERS
# ============================================================

st.markdown(
    "### 🔍 Price Boundaries Filter"
)

min_price = st.number_input(

    "Minimum Price ($)",

    value=40.0,

    min_value=40.0,

    max_value=120.0,

    key="ema_alignment_min_price"
)

max_price = st.number_input(

    "Maximum Price ($)",

    value=120.0,

    min_value=40.0,

    max_value=120.0,

    key="ema_alignment_max_price"
)

# ============================================================
# PRICE SCORE INFORMATION
# ============================================================

st.markdown(
    "### 🎯 Price Increase Reference"
)

st.caption(

    "Price Increase Score is informational/ranking only. "

    "0 = <0.25%, "

    "1 = 0.25–<0.50%, "

    "2 = 0.50–<1.00%, "

    "3 = ≥1.00% "

    "over the last 5 one-minute bars."
)

# ============================================================
# RUN BUTTON
# ============================================================

run_model = st.button(

    "Run EMA Alignment Model Scan",

    key="ema_alignment_run_button"
)

# ============================================================
# RUN MODEL
# ============================================================

if run_model:

    try:

        st.cache_data.clear()

        start_time = time.time()

        eastern = ZoneInfo("America/New_York")

        now_est = datetime.now(eastern)

        # ----------------------------------------------------
        # WEEKEND PROTECTION
        # ----------------------------------------------------

        if now_est.weekday() >= 5:

            st.warning(
                "⚠️ U.S. stock market is closed today."
            )

            st.info(

                "The model requires current-day "
                "1-minute data for a live scan. "

                "No current-day scan was performed."
            )

            st.session_state[
                "ema_alignment_raw_ranking"
            ] = pd.DataFrame()

            st.session_state[
                "ema_alignment_rejections"
            ] = pd.DataFrame()

            st.stop()

        # ----------------------------------------------------
        # EXECUTION TIME
        # ----------------------------------------------------

        st.write(

            "⏱️ Scan Execution Time: **"
            + now_est.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " EST**"
        )

        # ----------------------------------------------------
        # LOAD UNIVERSE
        # ----------------------------------------------------

        universe_list = _load_universe()

        if not universe_list:

            st.error(
                "The stock universe could not "
                "be loaded. "
                "Check data/us_universe_list.py "
                "and utils/data_fetch.py."
            )

            st.stop()

        # ----------------------------------------------------
        # TEMPORARY UNIVERSE TRACE
        # ----------------------------------------------------

        st.write("### 🔎 UNIVERSE TRACE")

        st.write(
            f"Page 5 executing file: `{__file__}`"
        )

        st.write(
            f"Universe variable type: `{type(universe_list)}`"
        )

        st.write(
            f"Universe variable length: "
            f"**{len(universe_list):,}**"
        )

        st.write(
            f"First 10 tickers: `{universe_list[:10]}`"
        )

        st.write(
            f"Last 10 tickers: `{universe_list[-10:]}`"
        )

        # ----------------------------------------------------
        # MASTER UNIVERSE COUNT
        # ----------------------------------------------------

        st.write(
            f"Master Universe loaded: "
            f"**{len(universe_list):,} tickers**"
        )

        # ----------------------------------------------------
        # DOWNLOAD DATA
        # ----------------------------------------------------

        progress = st.progress(

            0,

            text=(
                "Downloading daily and "
                "1-minute market data..."
            )
        )

        raw_daily, raw_intra = (
            fetch_clean_market_batch(
                tuple(universe_list)
            )
        )

        if (
            raw_daily is None
            or raw_daily.empty
            or raw_intra is None
            or raw_intra.empty
        ):

            progress.empty()

            st.warning(
                "No market data was returned."
            )

            st.stop()

        # ----------------------------------------------------
        # RUN ENGINE
        # ----------------------------------------------------

        progress.progress(

            0.6,

            text=(
                "Calculating EMA alignment, "
                "price scores, and rejection "
                "diagnostics..."
            )
        )

        ranking, rejections = (
            ema_alignment_engine(

                universe_list,

                raw_daily,

                raw_intra,

                min_price,

                max_price
            )
        )

        progress.progress(

            0.9,

            text=(
                "Preparing ranked results..."
            )
        )

        progress.empty()

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        st.session_state[
            "ema_alignment_raw_ranking"
        ] = ranking

        st.session_state[
            "ema_alignment_rejections"
        ] = rejections

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        st.write(

            f"⚡ Total Model Runtime: "
            f"**{time.time() - start_time:.2f} seconds**"
        )

    except Exception as e:

        st.error(
            f"Model execution failed: {e}"
        )

        st.exception(e)

# ============================================================
# GET RESULTS FROM SESSION STATE
# ============================================================

ranking = st.session_state.get(

    "ema_alignment_raw_ranking",

    pd.DataFrame()
)

rejections = st.session_state.get(

    "ema_alignment_rejections",

    pd.DataFrame()
)

# ============================================================
# V6 NOTE — STRONG SIGNAL SIMPLIFIED
# ============================================================

st.caption(
    "V6 Strong logic: the latest bar does not need to be higher "
    "than bars -3 or -4. Those comparisons remain diagnostic "
    "information only. The 5-bar price increase % and score "
    "indicate recent price strength."
)

# ============================================================
# DEVELOPMENT SIGNALS — DIAGNOSTIC ONLY
# ============================================================

st.markdown(
    "### 🟡 Development Signals — Diagnostic Only"
)

st.caption(
    "Development v4 is an earlier transition hypothesis. "
    "It does NOT replace the existing Strong/qualified logic. "
    "It allows either: (A) only the current 5th bar above EMA9 "
    "with the previous 4 at/below EMA9, or (B) the 4th and 5th "
    "bars above EMA9 with the first 3 at/below EMA9. "
    "Price must also be above EMA20, EMA9 above EMA20, and both "
    "EMA9/EMA20 slopes positive. No strictly rising 5-bar sequence "
    "is required."
)

development_df = pd.DataFrame()

if (
    rejections is not None
    and not rejections.empty
    and "Development_Signal" in rejections.columns
):
    development_df = rejections[
        rejections["Development_Signal"] == "DEVELOPMENT"
    ].copy()

if (
    ranking is not None
    and not ranking.empty
    and "Development_Signal" in ranking.columns
):
    ranking_development = ranking[
        ranking["Development_Signal"] == "DEVELOPMENT"
    ].copy()

    if not ranking_development.empty:
        development_df = pd.concat(
            [development_df, ranking_development],
            ignore_index=True
        )

if not development_df.empty:

    development_df = development_df.drop_duplicates(
        subset=["Ticker"]
    )

    development_display_cols = [
        "Ticker",
        "Development_Signal",
        "Price",
        "EMA_Score",
        "Price_Increase_%_5Bars",
        "Price_Increase_Score",
        "Development_Current_Above_EMA9",
        "Development_Previous4_Below_EMA9",
        "Development_Recent2_Above_EMA9",
        "Development_Preceding3_Below_EMA9",
        "Development_Transition_Type",
        "Last5_Rising_Trend",
        "Price_Above_EMA20",
        "EMA9_Above_EMA20",
        "EMA9_Slope_Pos",
        "EMA20_Slope_Pos",
        "Development_Reason"
    ]

    development_display_cols = [
        c for c in development_display_cols
        if c in development_df.columns
    ]

    st.dataframe(
        development_df[development_display_cols],
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "🟡 DEVELOPMENT is an observation signal only. "
        "The purpose is to measure whether the earlier EMA9 "
        "transition produces better entry opportunities than "
        "waiting for the existing Strong confirmation."
    )

else:
    st.write("No Development candidates detected in this scan.")

# ============================================================
# OPPORTUNITY CATEGORY SUMMARY
# ============================================================

if (
    ranking is not None
    and not ranking.empty
    and "Opportunity_Category" in ranking.columns
):
    st.markdown("### 🧭 Opportunity Category Summary")

    category_summary = (
        ranking["Opportunity_Category"]
        .value_counts()
        .rename_axis("Opportunity_Category")
        .reset_index(name="Qualified_Tickers")
    )

    st.dataframe(
        category_summary,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "This is a summary of categories among the currently qualified EMA tickers. "
        "It is informational only and does not filter or change qualification."
    )


# ============================================================
# QUALIFIED RESULTS
# ============================================================

if (
    ranking is not None
    and not ranking.empty
):

    display_df = ranking.drop(

        columns=[
            "_Latest_Real_Day"
        ],

        errors="ignore"

    ).copy()


    st.subheader(

        f"📊 EMA Alignment Results — "
        f"{len(display_df)} Tickers"
    )

    st.dataframe(

        display_df.style.apply(
            color_score_columns,
            axis=None
        ),

        hide_index=True,

        use_container_width=True
    )

    # --------------------------------------------------------
    # PRICE SCORE EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        "### 🧭 How to read Price Increase Score"
    )

    st.write(

        "The EMA Alignment Score identifies "
        "the structural setup. "

        "The Price Increase Score tells you "
        "whether the stock is actually moving "
        "enough over the last 5 one-minute bars. "

        "It is **NOT a rejection filter**."
    )

    st.markdown(
        "### 🧭 Opportunity Category"
    )

    st.caption(
        "Opportunity Category is informational only. It does NOT "
        "change the EMA qualification rules. Deep Recovery = current "
        "price is at least 5% below the previous trading day's close; "
        "Moderate Recovery = more than 2% and less than 5% below; "
        "Near Previous Close = within ±2%; Continuation = more than 2% "
        "and up to 5% above; Extended = more than 5% above. Session "
        "Phase identifies whether the signal occurs during the first "
        "40 minutes or later."
    )


# ============================================================
# REJECTION DIAGNOSTICS
# ============================================================

st.markdown(
    "### 🧪 Rejection Diagnostics"
)

st.caption(

    "This table records where each ticker "
    "failed. QUALIFIED rows are also included "
    "so we can compare a ticker such as PENG "
    "against the exact conditions used by "
    "the model."
)

if (
    rejections is not None
    and not rejections.empty
):

    diag_cols = [

        "Ticker",

        "Status",

        "Reason",

        "Price",

        "Opportunity_Category",

        "Session_Phase",

        "Avg_Volume_20d",

        "EMA_Score",

        "Price_Increase_%_5Bars",

        "Price_Increase_Score",

        "Price_Above_EMA9",

        "Price_Above_EMA20",

        "EMA9_Above_EMA20",

        "EMA9_Slope_Pos",

        "EMA20_Slope_Pos",

        "Last5_Above_EMA9",

        "LastBar_Higher_3",

        "LastBar_Higher_4",

        "Development_Signal",

        "Development_Current_Above_EMA9",

        "Development_Previous4_Below_EMA9",

        "Development_Recent2_Above_EMA9",

        "Development_Preceding3_Below_EMA9",

        "Development_Transition_Type",

        "Last5_Rising_Trend",

        "Development_Reason",

        "Daily_Data",

        "Intraday_Data",

        "Latest_Real_Day"
    ]

    diag_cols = [

        c

        for c in diag_cols

        if c in rejections.columns
    ]

    st.dataframe(

        rejections[diag_cols],

        hide_index=True,

        use_container_width=True
    )

    # ========================================================
    # INDIVIDUAL TICKER DIAGNOSTIC
    # ========================================================

    diagnostic_ticker = st.selectbox(

        "Select ticker for rejection diagnostics",

        sorted(
            rejections[
                "Ticker"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),

        key="ema_alignment_rejection_ticker"
    )

    if diagnostic_ticker:

        row = rejections[

            rejections[
                "Ticker"
            ].astype(str)
            == str(diagnostic_ticker)

        ].iloc[0]


        st.markdown(

            f"#### 🔎 {diagnostic_ticker}"
        )

        st.write(

            f"**Status:** "
            f"{row['Status']}"
        )


        st.write(

            f"**Reason:** "
            f"{row['Reason']}"
        )

        st.write(

            f"**Latest real trading day:** "
            f"{row['Latest_Real_Day']}"
        )

        st.write(

            f"**Price:** "
            f"{row['Price']}"
        )

        st.write(

            f"**EMA Alignment Score:** "
            f"{row['EMA_Score']} / 8"
        )

        st.write(

            f"**Price Increase:** "
            f"{row['Price_Increase_%_5Bars']}%"
        )

        st.write(

            f"**Price Increase Score:** "
            f"{row['Price_Increase_Score']}"
        )

        if "Development_Signal" in row.index:
            st.write(
                f"**Development Signal:** "
                f"{row['Development_Signal']}"
            )

        if "Development_Reason" in row.index:
            st.write(
                f"**Development Reason:** "
                f"{row['Development_Reason']}"
            )

        if "Development_Current_Above_EMA9" in row.index:
            st.write(
                f"**Recent EMA9 Cross:** "
                f"{row['Development_Current_Above_EMA9']}"
            )

        if "Development_Previous4_Below_EMA9" in row.index:
            st.write(
                f"**Previous 4 closes at/below EMA9:** "
                f"{row['Development_Previous4_Below_EMA9']}"
            )

        if "Development_Transition_Type" in row.index:
            st.write(
                f"**Development Transition Type:** "
                f"{row['Development_Transition_Type']}"
            )

        if "Last5_Rising_Trend" in row.index:
            st.write(
                f"**Last 5 Rising Trend:** "
                f"{row['Last5_Rising_Trend']}"
            )

        st.write(
            "**Condition results:**"
        )

        conditions = {

            "Price > EMA9":
                row[
                    "Price_Above_EMA9"
                ],

            "Price > EMA20":
                row[
                    "Price_Above_EMA20"
                ],

            "EMA9 > EMA20":
                row[
                    "EMA9_Above_EMA20"
                ],

            "EMA9 slope positive":
                row[
                    "EMA9_Slope_Pos"
                ],

            "EMA20 slope positive":
                row[
                    "EMA20_Slope_Pos"
                ],

            "All last 5 closes > EMA9":
                row[
                    "Last5_Above_EMA9"
                ],

            "Latest bar > bar -3":
                row[
                    "LastBar_Higher_3"
                ],

            "Latest bar > bar -4":
                row[
                    "LastBar_Higher_4"
                ],

            "Development: recent EMA9 upside cross":
                row.get(
                    "Development_Current_Above_EMA9",
                    "N/A"
                ),

            "Development: previous 4 closes at/below EMA9":
                row.get(
                    "Development_Previous4_Below_EMA9",
                    "N/A"
                )
        }

        for name, result in conditions.items():

            st.write(

                f"- **{name}:** "
                f"{result}"
            )

# ============================================================
# NO RESULTS
# ============================================================

else:

    if (

        ranking is not None
        and ranking.empty

        and rejections is not None
        and not rejections.empty

    ):

        st.info(

            "No tickers qualified. "
            "Use the Rejection Diagnostics "
            "table to see exactly why."
        )