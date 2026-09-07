# ==============================================================================
# 📈 INSTITUTIONAL EMA ALIGNMENT MODEL
# PURPOSE:
# Identify institutional-quality tickers showing early EMA alignment
# WITHOUT requiring momentum, continuation, VWAP, proximity, or
# price > previous-day close.
#
# VERSION: 2026-09-07
#
# IMPORTANT DESIGN:
# - Uses current intraday price for the price filter.
# - Previous-day close is informational only.
# - Does NOT reject a ticker simply because it opened below yesterday's close.
# - Uses EMA9 / EMA20 structure and short-term price behavior.
# - Designed as an early structural candidate detector.
# ==============================================================================
import streamlit as st
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

import data.us_universe_list as universe_module

file_path = universe_module.__file__

with open(file_path, "r", encoding="utf-8") as f:
    file_lines = f.readlines()

st.write(f"Imported file: **{file_path}**")
st.write(f"Physical lines in file: **{len(file_lines):,}**")
st.write(f"Python list length: **{len(universe_module.us_universe):,}**")

st.set_page_config(layout="wide", page_title="Institutional EMA Alignment")

st.caption(
    "Version: 2026-09-07 — Institutional EMA Alignment + "
    "Price Increase Score + Rejection Diagnostics"
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
        return pd.DataFrame()

    if not isinstance(batch.columns, pd.MultiIndex):
        return _flatten_columns(batch.copy())

    try:
        return batch[ticker].copy()

    except Exception:
        return pd.DataFrame()


def _to_eastern_index(df):

    df = df.copy()

    idx = pd.DatetimeIndex(df.index)

    if idx.tz is not None:
        idx = idx.tz_convert("US/Eastern")

    else:
        idx = idx.tz_localize("US/Eastern")

    df.index = idx

    return df

# ============================================================
# UNIVERSE
# ============================================================

def _load_universe():

    try:
        # ----------------------------------------------------
        # Load the project's master universe
        # ----------------------------------------------------
        from utils.data_fetch import load_universe

        tickers = load_universe()

        if tickers is None:
            st.error(
                "load_universe() returned None."
            )
            return []

        # ----------------------------------------------------
        # Normalize and deduplicate
        # ----------------------------------------------------
        normalized_tickers = sorted(
            set(
                str(x).strip().upper()
                for x in tickers
                if x
            )
        )

        # ----------------------------------------------------
        # Universe diagnostics
        # ----------------------------------------------------
        st.write(
            f"**Master Universe loaded:** "
            f"{len(normalized_tickers):,} tickers"
        )

        # ----------------------------------------------------
        # Verify the specific ticker we are investigating
        # ----------------------------------------------------
        if "PENG" in normalized_tickers:

            st.success(
                "✅ PENG is present in the master universe."
            )

        else:

            st.warning(
                "⚠️ PENG is NOT present in the master universe."
            )

        # ----------------------------------------------------
        # Compare directly with the source module
        # ----------------------------------------------------
        try:

            from data.us_universe_list import us_universe

            raw_count = len(us_universe)

            unique_count = len(
                set(
                    str(x).strip().upper()
                    for x in us_universe
                    if x
                )
            )

            st.write(
                f"Source `us_universe` count: "
                f"**{raw_count:,}**"
            )

            st.write(
                f"Unique normalized `us_universe` count: "
                f"**{unique_count:,}**"
            )

            if raw_count != unique_count:

                st.info(
                    f"ℹ️ The source contains "
                    f"{raw_count - unique_count:,} "
                    f"duplicate/normalization entries."
                )

            # ------------------------------------------------
            # Direct PENG check
            # ------------------------------------------------
            normalized_source = set(
                str(x).strip().upper()
                for x in us_universe
                if x
            )

            st.write(
                "PENG in source `us_universe`: "
                f"**{'YES' if 'PENG' in normalized_source else 'NO'}**"
            )

            st.write(
                "PENG in loaded universe: "
                f"**{'YES' if 'PENG' in normalized_tickers else 'NO'}**"
            )

        except Exception as source_error:

            st.warning(
                "Could not directly inspect "
                f"data.us_universe_list: {source_error}"
            )

        # ----------------------------------------------------
        # Return the normalized master universe
        # ----------------------------------------------------
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


        cond_lastbar_higher_3 = bool(
            last5[-1] > last5[-3]
        )


        cond_lastbar_higher_4 = bool(
            last5[-1] > last5[-4]
        )


        # ====================================================
        # EMA ALIGNMENT SCORE
        # ====================================================

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

            "LastBar_Higher_3":
                (
                    "PASS"
                    if cond_lastbar_higher_3
                    else "FAIL"
                ),

            "LastBar_Higher_4":
                (
                    "PASS"
                    if cond_lastbar_higher_4
                    else "FAIL"
                )
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


        if not cond_lastbar_higher_3:

            failed.append(
                "latest bar <= bar -3"
            )


        if not cond_lastbar_higher_4:

            failed.append(
                "latest bar <= bar -4"
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


        rows.append({

            "Ticker":
                ticker,

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
            "All hard EMA alignment "
            "conditions passed"
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

        eastern = pytz.timezone(
            "US/Eastern"
        )

        now_est = datetime.now(
            eastern
        )


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

                "Check "
                "data.sp500_list.py / "
                "data.nasdaq100_list.py."
            )

            st.stop()


        st.write(

            f"Universe loaded: "
            f"**{len(universe_list)} tickers**"
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
                ]
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