# ==============================================================================
# 📈 WARRIOR MODEL — Clean & Patched Version (Standalone Page 4)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(layout="wide", page_title="Penny Model")
st.caption("Version: W1 — Warrior model, adapted from Penny Model V6. Fixed: slope threshold "
           "was silently read from a bare global instead of being passed in; prior-close "
           "now robust to whether Yahoo's daily bar for 'today' updates live during market hours.")
st.title("📈 Price Acceleration Model")

# ==============================================================================
# MODEL PARAMETERS
# ==============================================================================
MIN_DAILY_HISTORY = 40
MIN_INTRADAY_BARS = 5
MIN_REAL_DAY_BARS = 10
# DEPRECATED as a filter: no longer used to reject tickers. Left here only
# because Avg_Volume_20d is still computed and shown as informational
# context in the output tables.
MIN_AVG_VOLUME_20D = 80000

# DEPRECATED: previously used to give the daily-close price pre-filter
# some slack. Currently unused — daily_prefilter() does an exact daily
# close comparison with no buffer.
PREFILTER_PRICE_BUFFER_PCT = 0.15  # currently unused

# --------------------------------------------------------------------------
# PRICE-ACCELERATION ENGINE THRESHOLDS
#
# MIN_REG_SLOPE_PCT is the DEFAULT used if no override is passed in. The
# Streamlit UI below lets you override this live via a number_input
# (min_reg_slope_pct) — FIX: previously the engine read that UI value as a
# bare module-level global instead of accepting it as a parameter, which
# silently made this constant dead code and was fragile (would break with
# a confusing NameError if this function were ever called from anywhere
# else, e.g. a test or another page). Now it's an explicit parameter with
# this constant as its default.
MIN_REG_SLOPE_PCT = 0.005   # 0.5% of price per bar — starting default only
MIN_CONSISTENCY = 0.25      # loosened from the old EMA version's 0.60 —
                            # quite permissive; worth watching whether this
                            # lets through mostly-flat/choppy tickers.

# Max concurrent Yahoo requests for the intraday fetch stage.
INTRADAY_MAX_WORKERS = 20

# --------------------------------------------------------------------------
# CACHE TTLs — decoupled on purpose.
# --------------------------------------------------------------------------
DAILY_CACHE_TTL_SECONDS = 6 * 60 * 60   # 6 hours — effectively "once per session"
INTRADAY_CACHE_TTL_SECONDS = 120        # 2 minutes — must stay fresh for live scans

# Where the top-5-per-run log persists across separate script runs.
import os
TOP5_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top5_log.csv")
TOP5_LOG_COLUMNS = [
    "Run_Timestamp_ET", "Ticker", "Close",
    "Volume_Day_vs_50d", "Price_Open_PrevClose_%",
    "Price_Increase_%_5Bars", "Avg_Volume_Last5Bars",
    "Segment_Signal", "Status", "Development_Signal",
    "Latest_Real_Day", "Stale_Bars_Last5",
]

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "ema_alignment_raw_ranking" not in st.session_state:
    st.session_state["ema_alignment_raw_ranking"] = pd.DataFrame()

if "ema_alignment_rejections" not in st.session_state:
    st.session_state["ema_alignment_rejections"] = pd.DataFrame()

# ==============================================================================
# HELPERS
# ==============================================================================
def _flatten_columns(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if all(x in known for x in level0):
            df.columns = level0
        elif all(x in known for x in level1):
            df.columns = level1
    return df

def _extract_ticker_slice(batch, ticker):
    if batch is None or batch.empty:
        return pd.DataFrame()

    ticker = str(ticker).upper()

    if not isinstance(batch.columns, pd.MultiIndex):
        return _flatten_columns(batch.copy())

    try:
        if ticker in batch.columns.get_level_values(0):
            return _flatten_columns(batch[ticker].copy())
    except Exception:
        pass

    try:
        if ticker in batch.columns.get_level_values(1):
            return _flatten_columns(
                batch.xs(ticker, axis=1, level=1, drop_level=True).copy()
            )
    except Exception:
        pass

    return pd.DataFrame()

def _to_eastern_index(df):
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    eastern = ZoneInfo("America/New_York")
    if idx.tz is not None:
        df.index = idx.tz_convert(eastern)
    else:
        df.index = idx.tz_localize(eastern)
    return df

def _prior_close_row(daily_df):
    """Return (row, was_today_row: bool) for the daily bar that represents
    the most recently COMPLETED session's close — i.e. genuinely
    "yesterday's close" — not an in-progress, still-updating "today" bar.

    WHY THIS EXISTS: it's untested/unconfirmed whether Yahoo's daily
    download includes today's date as its last row DURING market hours
    with a live-updating Close, or only adds today's row after the close.
    If it's the former and this code just blindly used `.iloc[-1]`
    everywhere, every "vs previous close" comparison (the daily
    pre-filter's price check, and this model's core
    Price_Open_PrevClose_% gap metric) could silently be comparing today
    against ITSELF instead of against yesterday — a meaningful correctness
    bug specifically for the Warrior gap-% criterion.

    This function is robust to BOTH possibilities: if the last row's date
    is today (Eastern time), it steps back one row and returns that as the
    true prior close; otherwise the last row already IS the settled prior
    close and is returned as-is. `was_today_row` is returned so callers
    can aggregate a diagnostic count — see the "Daily-bar freshness check"
    caption in the page execution section, which reports this directly
    from your own data on each run rather than assuming an answer.
    """
    if daily_df is None or daily_df.empty:
        return None, False
    try:
        idx_dates = pd.DatetimeIndex(daily_df.index).date
    except Exception:
        return daily_df.iloc[-1], False

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    last_is_today = len(idx_dates) > 0 and idx_dates[-1] == today_et

    if last_is_today and len(daily_df) >= 2:
        return daily_df.iloc[-2], True
    return daily_df.iloc[-1], last_is_today

def _load_universe():
    try:
        import utils.data_fetch as data_fetch_module
        import data.us_universe_list as universe_module
        universe_module = importlib.reload(universe_module)
        data_fetch_module = importlib.reload(data_fetch_module)
        load_universe = data_fetch_module.load_universe
        tickers = load_universe()

        st.write("Universe size:", len(tickers))

        if tickers is None:
            st.error("load_universe() returned None.")
            return []
        normalized = sorted(set(str(x).strip().upper() for x in tickers if x))
        return normalized
    except Exception as e:
        st.error(f"Unable to load universe: {e}")
        return []

def _load_movers():
    """Load the trader's curated watchlist of known pre-market/after-hours
    movers from data/movers_list.py. Reloaded fresh every run so editing
    the movers list takes effect immediately. Missing module or empty
    list is NOT an error — mover tagging is an optional overlay.

    Expected file: data/movers_list.py
        def load_movers():
            return ["GLND", "GRML", "ABCD"]  # your curated list for today
    """
    try:
        import data.movers_list as movers_module
        movers_module = importlib.reload(movers_module)
        load_movers = movers_module.load_movers
        movers = load_movers()
        if movers is None:
            return []
        return sorted(set(str(x).strip().upper() for x in movers if x))
    except ModuleNotFoundError:
        return []
    except Exception as e:
        st.warning(f"Movers list found but couldn't be loaded: {e}")
        return []

def _rejection_row(ticker, reason=""):
    return {
        "Ticker": ticker,
        "Status": "REJECTED",
        "Reason": reason,
        "Price": np.nan,
        "Opportunity_Category": "N/A",
        "Session_Phase": "N/A",
        "Avg_Volume_20d": np.nan,
        "EMA_Score": np.nan,
        "Price_Increase_%_5Bars": np.nan,
        "Price_Increase_Score": np.nan,
        "Price_Increase_Label": "N/A",
        "Development_Signal": "N/A",
        "Development_Reason": "N/A",
        "Latest_Real_Day": "N/A",
        "Stale_Bars_Last5": np.nan,
    }

def _ensure_log_schema(log_path=TOP5_LOG_PATH, expected_columns=TOP5_LOG_COLUMNS):
    """Self-heals the log file if its on-disk header doesn't match the
    current expected columns. See prior version's comments — this
    prevents a schema change from silently corrupting the whole log."""
    if not os.path.exists(log_path):
        return

    try:
        existing = pd.read_csv(log_path)
        if list(existing.columns) == list(expected_columns):
            return
    except Exception:
        try:
            existing = pd.read_csv(log_path, on_bad_lines="skip")
        except TypeError:
            existing = pd.read_csv(log_path, error_bad_lines=False, warn_bad_lines=False)
        except Exception:
            return

    for col in expected_columns:
        if col not in existing.columns:
            existing[col] = np.nan
    existing = existing[expected_columns]
    existing.to_csv(log_path, index=False)


def log_top5(top_df, log_path=TOP5_LOG_PATH, top_n=5):
    """Append the current run's top-N rows from top_df to a persistent CSV
    log, tagged with the run's Eastern-time timestamp."""
    if top_df is None or top_df.empty:
        return 0

    _ensure_log_schema(log_path)

    run_ts = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")

    top = top_df.head(top_n).copy()
    entry = pd.DataFrame({
        "Run_Timestamp_ET": run_ts,
        "Ticker": top["Ticker"].values,
        "Close": top["Close"].values,
        "Volume_Day_vs_50d": top.get("Volume_Day_vs_50d", pd.Series([np.nan] * len(top))).values,
        "Price_Open_PrevClose_%": top.get("Price_Open_PrevClose_%", pd.Series([np.nan] * len(top))).values,
        "Price_Increase_%_5Bars": top.get("Price_Increase_%_5Bars", pd.Series([np.nan] * len(top))).values,
        "Avg_Volume_Last5Bars": top.get("Avg_Volume_Last5Bars", pd.Series([np.nan] * len(top))).values,
        "Segment_Signal": top.get("Segment_Signal", pd.Series(["N/A"] * len(top))).values,
        "Status": top.get("Status", pd.Series(["N/A"] * len(top))).values,
        "Development_Signal": top.get("EMA_Aligned", pd.Series([np.nan] * len(top))).values,
        "Latest_Real_Day": top.get("Latest_Real_Day", pd.Series(["N/A"] * len(top))).values,
        "Stale_Bars_Last5": top.get("Stale_Bars_Last5", pd.Series([np.nan] * len(top))).values,
    })

    file_exists = os.path.exists(log_path)
    entry.to_csv(log_path, mode="a", header=not file_exists, index=False)
    return len(entry)

def reset_top5_log(log_path=TOP5_LOG_PATH):
    pd.DataFrame(columns=TOP5_LOG_COLUMNS).to_csv(log_path, index=False)

def load_top5_log(log_path=TOP5_LOG_PATH):
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=TOP5_LOG_COLUMNS)
    _ensure_log_schema(log_path)
    try:
        return pd.read_csv(log_path)
    except Exception:
        return pd.DataFrame(columns=TOP5_LOG_COLUMNS)

# ==============================================================================
# MARKET DATA
# ==============================================================================
@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_daily_batch(tickers_tuple):
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()
    try:
        daily = yf.download(
            tickers, period="3mo", interval="1d",
            group_by="ticker", auto_adjust=False,
            progress=False, threads=True
        )
        return daily
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_intraday_batch(tickers_tuple, max_workers=INTRADAY_MAX_WORKERS):
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()

    def _fetch_one(t):
        try:
            df = yf.download(t, period="1d", interval="1m", progress=False)
        except Exception:
            df = pd.DataFrame()
        return t, df

    intra_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            t, df = future.result()

            if df is None or df.empty:
                continue

            try:
                if df.index.tz is None:
                    df = df.tz_localize("UTC")
            except (TypeError, AttributeError):
                continue

            intra_dict[t] = df

    if not intra_dict:
        return pd.DataFrame()

    return pd.concat(intra_dict, axis=1)


def daily_prefilter(tickers, daily_batch, min_price, max_price):
    """Shrink the universe by daily history depth AND a price check.

    FIX: the price check now uses _prior_close_row() instead of a blind
    `.iloc[-1]` — see that function's docstring for why. Also now counts
    how many tickers' daily batch has today's date as the last row, so you
    can see directly (via the caption in page execution) whether Yahoo's
    daily data updates live during market hours in your environment,
    rather than that being an unconfirmed assumption.

    Returns (candidates: list[str], prefilter_rejects: list[dict],
    n_today_rows: int, n_checked: int).
    """
    candidates = []
    prefilter_rejects = []
    n_today_rows = 0
    n_checked = 0

    if daily_batch.empty:
        for t in tickers:
            prefilter_rejects.append(_rejection_row(t, "No daily data"))
        return candidates, prefilter_rejects, n_today_rows, n_checked

    for ticker in tickers:
        daily = _flatten_columns(_extract_ticker_slice(daily_batch, ticker))

        if daily.empty:
            prefilter_rejects.append(_rejection_row(ticker, "No daily data"))
            continue

        daily = daily.dropna(subset=["Close"])

        if len(daily) < MIN_DAILY_HISTORY:
            prefilter_rejects.append(_rejection_row(ticker, "Insufficient daily history"))
            continue

        try:
            prior_row, was_today = _prior_close_row(daily)
            n_checked += 1
            if was_today:
                n_today_rows += 1
            last_close = float(prior_row["Close"])
            if last_close < min_price or last_close > max_price:
                prefilter_rejects.append(_rejection_row(ticker, "Daily price outside range"))
                continue
        except Exception:
            prefilter_rejects.append(_rejection_row(ticker, "Invalid daily price"))
            continue

        candidates.append(ticker)

    return candidates, prefilter_rejects, n_today_rows, n_checked


# ==============================================================================
# SCORING
# ==============================================================================
def price_increase_score(pct):
    try:
        x = float(pct)
    except Exception:
        return 0
    if x >= 1.0: return 3
    if x >= 0.5: return 2
    if x >= 0.25: return 1
    return 0

def opportunity_category(gap):
    try:
        x = float(gap)
    except Exception:
        return "Unknown"
    if x <= -5: return "Deep Recovery"
    if x < -2: return "Moderate Recovery"
    if x <= 2: return "Near Previous Close"
    if x <= 5: return "Continuation"
    return "Extended"

def session_phase(ts):
    m = ts.hour * 60 + ts.minute
    if m < 570: return "Pre-Market"
    if m < 610: return "Early Session"
    if m < 630: return "Morning"
    if m < 720: return "Late Morning"
    if m < 840: return "Midday"
    return "Afternoon"

# ==============================================================================
# PRICE ACCELERATION ENGINE
# ==============================================================================
def price_acceleration_engine(
    tickers, daily_batch, intra_batch, min_price, max_price,
    max_stale_bars_last5=2,
    min_gap_pct=None,
    min_volume_ratio=None,
    min_volume_day=None,
):

    rows = []
    rejects = []
    MAX_STALE_BARS_LAST5 = max_stale_bars_last5

    if daily_batch.empty:
        for t in tickers:
            rejects.append(_rejection_row(t, "No daily data"))
        return pd.DataFrame(), pd.DataFrame(rejects)

    if intra_batch.empty:
        for t in tickers:
            rejects.append(_rejection_row(t, "No intraday data"))
        return pd.DataFrame(), pd.DataFrame(rejects)

    try:
        available_daily = set(daily_batch.columns.get_level_values(0))
        available_intra = set(intra_batch.columns.get_level_values(0))
        active = sorted(set(tickers).intersection(available_daily).intersection(available_intra))
    except Exception:
        active = sorted(tickers)

    for t in sorted(set(tickers) - set(active)):
        rejects.append(_rejection_row(t, "Ticker missing from Yahoo data"))

    for ticker in active:
        r = _rejection_row(ticker)

        daily = _flatten_columns(_extract_ticker_slice(daily_batch, ticker))
        intra = _flatten_columns(_extract_ticker_slice(intra_batch, ticker))

        if daily.empty:
            r["Reason"] = "Daily empty"
            rejects.append(r)
            continue

        if intra.empty:
            r["Reason"] = "Intraday empty"
            rejects.append(r)
            continue

        daily = daily.dropna(subset=["Close"])
        intra = intra[intra["Close"].notna() | intra["Open"].notna()]

        if len(daily) < MIN_DAILY_HISTORY:
            r["Reason"] = "Insufficient daily history"
            rejects.append(r)
            continue

        intra = _to_eastern_index(intra).sort_index()

        intra_regular = intra.between_time("09:30", "16:00")
        intra_regular = intra_regular.loc[~intra_regular.index.duplicated(keep='last')]

        if len(intra_regular) < MIN_REAL_DAY_BARS:
            r["Reason"] = f"Insufficient regular-hours bars ({len(intra_regular)})"
            rejects.append(r)
            continue

        intra_day = intra_regular

        day_counts = pd.Series(intra_day.index.date).value_counts()
        eligible = sorted(day_counts[day_counts >= MIN_REAL_DAY_BARS].index)

        if not eligible:
            r["Reason"] = "No eligible regular-hours trading day"
            rejects.append(r)
            continue

        latest_day = eligible[-1]
        intra = intra_day[intra_day.index.date == latest_day]
        intra = intra.loc[~intra.index.duplicated(keep='last')]

        close_series_clean = pd.to_numeric(intra["Close"], errors="coerce").dropna()
        close_raw = close_series_clean.values.astype(float)

        if len(close_raw) < MIN_INTRADAY_BARS:
            r["Reason"] = "Too few intraday bars"
            rejects.append(r)
            continue

        # Use prior settled daily close (correct Warrior behavior)
        prior_row, _ = _prior_close_row(daily)
        price = float(prior_row["Close"])
        r["Price"] = round(price, 2)

        if price < min_price or price > max_price:
            r["Reason"] = "Price outside range"
            rejects.append(r)
            continue

        # --- Informational 20-day volume ---------------------------------
        vol = pd.to_numeric(daily["Volume"], errors="coerce").dropna().values.astype(float)
        if len(vol) > 0:
            avg_vol = float(np.mean(vol[-20:])) if len(vol) >= 20 else float(np.mean(vol))
            r["Avg_Volume_20d"] = round(avg_vol, 0)

        # --- Stale bar check ------------------------------------------------
        last5_index = close_series_clean.tail(5).index
        bar_vol_last5 = pd.to_numeric(intra["Volume"], errors="coerce").reindex(last5_index).fillna(0.0)
        stale_bars_last5 = int((bar_vol_last5 <= 0).sum())

        if stale_bars_last5 > MAX_STALE_BARS_LAST5:
            r["Reason"] = f"Too many no-trade bars in last 5 ({stale_bars_last5})"
            rejects.append(r)
            continue

        # --- PRICE ACCELERATION --------------------------------------------
        last5 = close_raw[-5:]
        pct = (last5[-1] - last5[0]) / last5[0] * 100 if last5[0] > 0 else 0

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        reg_slope = np.polyfit(x, last5, 1)[0]
        reg_slope_pct = reg_slope / price if price else 0

        diffs = np.diff(last5)
        up_moves = np.sum(diffs > 0)
        down_moves = np.sum(diffs < 0)
        consistency_score = up_moves / 4.0 if up_moves >= down_moves else down_moves / 4.0

        if consistency_score < MIN_CONSISTENCY:
            r["Reason"] = f"Low consistency ({consistency_score:.2f}, need {MIN_CONSISTENCY:.2f})"
            rejects.append(r)
            continue

        # --- EMA9/EMA20 (informational only) -------------------------------
        ema9_series = intra["Close"].ewm(span=9, adjust=False).mean()
        ema20_series = intra["Close"].ewm(span=20, adjust=False).mean()
        ema9_last5 = ema9_series.tail(5).values.astype(float)
        ema20_last5 = ema20_series.tail(5).values.astype(float)
        ema9_now, ema9_prev = ema9_last5[-1], ema9_last5[-2]
        ema20_now, ema20_prev = ema20_last5[-1], ema20_last5[-2]
        ema9_slope_pct = (ema9_now - ema9_prev) / price if price else 0
        ema20_slope_pct = (ema20_now - ema20_prev) / price if price else 0
        ema_aligned = bool(
            price > ema9_now and price > ema20_now and
            ema9_now > ema20_now and
            ema9_slope_pct > 0 and ema20_slope_pct > 0
        )

        bar_vol_values = bar_vol_last5.values.astype(float)

        # --- SEGMENT SIGNAL -------------------------------------------------
        is_peak_bar5 = bool(bar_vol_values[-1] >= np.max(bar_vol_values))
        is_building = bool(bar_vol_values[2] < bar_vol_values[3] < bar_vol_values[4])
        if is_peak_bar5:
            segment_signal = "🔴 Peak Bar"
        elif is_building:
            segment_signal = "🟢 Ignition"
        else:
            segment_signal = "🟡 Mixed"

        # --- WARRIOR METRICS -----------------------------------------------
        vol_series = pd.to_numeric(daily["Volume"], errors="coerce").dropna()
        if len(vol_series) >= 50:
            volume_avg_50d = float(np.mean(vol_series[-50:]))
        else:
            volume_avg_50d = float(np.mean(vol_series)) if len(vol_series) > 0 else np.nan

        volume_day = float(np.sum(pd.to_numeric(intra["Volume"], errors="coerce").fillna(0)))
        volume_day_vs_50d = volume_day / volume_avg_50d if volume_avg_50d and volume_avg_50d > 0 else np.nan

        try:
            prior_row, _ = _prior_close_row(daily)
            prev_close = float(prior_row["Close"])
            intraday_open = float(intra["Open"].iloc[0])
            price_open_prevclose = ((intraday_open - prev_close) / prev_close) * 100 if prev_close > 0 else np.nan
        except Exception:
            price_open_prevclose = np.nan

        # --- WARRIOR FILTERS ------------------------------------------------
        if min_gap_pct is not None and price_open_prevclose < min_gap_pct:
            r["Reason"] = f"Gap too weak ({price_open_prevclose:.2f}%, need {min_gap_pct}%)"
            rejects.append(r)
            continue

        if min_volume_ratio is not None and volume_day_vs_50d < min_volume_ratio:
            r["Reason"] = f"Volume ratio too weak ({volume_day_vs_50d:.2f}, need {min_volume_ratio})"
            rejects.append(r)
            continue

        if min_volume_day is not None and volume_day < min_volume_day:
            r["Reason"] = f"Intraday volume too low ({volume_day:.0f}, need {min_volume_day})"
            rejects.append(r)
            continue

        # --- FINAL ROW ------------------------------------------------------
        rows.append({
            "Ticker": ticker,
            "Close": round(price, 2),
            "Open": round(float(intra["Open"].iloc[0]), 2),
            "Bar5_Close": round(last5[4], 4),
            "Open_vs_PrevClose_%": round(price_open_prevclose, 3),
            "Bar5_Close_vs_Close_%": round(((last5[-1] - price) / price) * 100, 3),
            "Volume_Day": round(volume_day, 1),
            "Avg_Volume_20d": r["Avg_Volume_20d"],
            "Volume_Avg_50d": round(volume_avg_50d, 1),
            "Volume_Day_vs_50d": round(volume_day_vs_50d, 3),
            "Price_Increase_%_5Bars": round(pct, 3),
            "Avg_Vol_Last5Bars": round(float(np.mean(bar_vol_values)), 1),
            "Segment_Signal": segment_signal,
            "Regression_Slope_Pct": round(reg_slope_pct, 5),
            "Consistency": round(consistency_score, 2),
            "EMA_Aligned": ema_aligned,
            "Bar5_EMA9": round(ema9_now, 4),
            "Bar5_EMA20": round(ema20_now, 4),
            "Status": "PRICE",
            "Latest_Real_Day": str(latest_day),
            "Stale_Bars_Last5": stale_bars_last5,
            "Bar1_Close": round(last5[0], 4),
            "Bar2_Close": round(last5[1], 4),
            "Bar3_Close": round(last5[2], 4),
            "Bar4_Close": round(last5[3], 4),
            "Bar5_Close": round(last5[4], 4),
            "Bar1_Volume": int(bar_vol_values[0]),
            "Bar2_Volume": int(bar_vol_values[1]),
            "Bar3_Volume": int(bar_vol_values[2]),
            "Bar4_Volume": int(bar_vol_values[3])
            #"Bar5_Volume": int(bar_vol_values[4]),
        })

    ranking = pd.DataFrame(rows)

    if not ranking.empty:
        ranking = ranking.sort_values(
            ["Volume_Day_vs_50d", "Open_vs_PrevClose_%"],
            ascending=[False, False]
        ).reset_index(drop=True)

    rejects_df = pd.DataFrame(rejects)
    return ranking, rejects_df

# ==============================================================================
# PAGE EXECUTION
# ==============================================================================
_eastern = ZoneInfo("America/New_York")
_now_et = datetime.now(_eastern)
_is_weekday = _now_et.weekday() < 5  # Mon=0 ... Sun=6
_minutes_now = _now_et.hour * 60 + _now_et.minute
_is_market_hours = _is_weekday and (570 <= _minutes_now < 960)  # 09:30-16:00 ET

if _is_market_hours:
    st.success(f"🟢 Market is open — {_now_et.strftime('%A %Y-%m-%d %H:%M %Z')}. Results below reflect live/current-session bars.")
else:
    st.warning(
        f"🟠 Market is CLOSED right now — {_now_et.strftime('%A %Y-%m-%d %H:%M %Z')}. "
        "Any tickers that qualify below were scored on the most recent completed "
        "session (check the Latest_Real_Day column), not live data. Treat this as a "
        "historical/backtest-style run, not a trading signal."
    )

tickers = _load_universe()

movers = _load_movers()
if movers:
    st.caption(f"🔥 Movers list loaded: {len(movers)} ticker(s) — {', '.join(movers)}")
else:
    st.caption("No movers list loaded (optional). Add `data/movers_list.py` with a `load_movers()` function to enable mover tagging.")

st.write("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Price ($)", min_value=0.0, value=2.00, step=0.25)
max_price = st.number_input("Maximum Price ($)", min_value=0.0, value=20.00, step=0.25)

st.write("### 🔍 Warrior Filters")

min_gap_pct = st.number_input(
    "Minimum Price_Open_PrevClose_% (Gap %)",
    min_value=-50.0,
    value=2.0,
    step=0.5
)

min_volume_ratio = st.number_input(
    "Minimum Volume_Day_vs_50d (Relative Volume)",
    min_value=0.0,
    value=0.0,
    step=0.25
)

min_volume_day = st.number_input(
    "Minimum Volume_Day (Raw Intraday Volume)",
    min_value=0.0,
    value=0.0,
    step=5000.0
)

max_stale_bars_last5 = st.slider(
    "Max no-trade bars allowed in last 5",
    min_value=0, max_value=5, value=2,
    help=(
        "Thinly-traded penny stocks can have 1-minute bars with zero volume, "
        "where Yahoo just repeats the last traded price. Tickers with more "
        "no-trade bars than this in their last 5 are rejected. 0 = require "
        "every one of the last 5 bars to have a real trade; 5 = disable."
    ),
)

import time

# ==============================================================================
# RUN BUTTON
# ==============================================================================
force_refresh = st.checkbox(
    "Force fresh intraday data (bypass 2-min cache)",
    value=False,
    help=(
        "Normally intraday data is cached for 2 minutes to avoid hammering "
        "Yahoo on every click. If you're watching a fast-moving ticker and "
        "clicking Run Model faster than every 2 minutes, check this to force "
        "a real, fresh fetch every time instead of possibly getting the same "
        "cached snapshot back."
    ),
)

run_clicked = st.button("🚀 Run Model", type="primary")

if run_clicked:
    t0 = time.time()
    daily_batch = fetch_daily_batch(tuple(tickers))
    t1 = time.time()

    candidates, prefilter_rejects, n_today_rows, n_checked = daily_prefilter(
        tickers, daily_batch, min_price, max_price
    )
    t2 = time.time()

    if force_refresh:
        fetch_intraday_batch.clear()
    intra_batch = fetch_intraday_batch(tuple(candidates))
    t3 = time.time()

    #ranking, engine_rejects = price_acceleration_engine(
    #    candidates, daily_batch, intra_batch, min_price, max_price,
    #    max_stale_bars_last5=max_stale_bars_last5,
    #    )

    ranking, rejects_df = price_acceleration_engine(
        candidates,
        daily_batch,
        intra_batch,
        min_price,
        max_price,
        max_stale_bars_last5,
        min_gap_pct=min_gap_pct,
        min_volume_ratio=min_volume_ratio,
        min_volume_day=min_volume_day,
    )

    t4 = time.time()

    rejects = pd.concat([pd.DataFrame(prefilter_rejects), rejects_df], ignore_index=True)

    st.session_state["model_results"] = {
        "ranking": ranking,
        "rejects": rejects,
        "tickers": tickers,
        "n_candidates": len(candidates),
        "n_universe": len(tickers),
        "timings": (t0, t1, t2, t3, t4),
        "run_at_et": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "n_today_rows": n_today_rows,
        "n_checked": n_checked,
    }

results = st.session_state.get("model_results")

if results is None:
    st.info("Click **🚀 Run Model** above to fetch data and run the scan.")
    ranking = pd.DataFrame()
    rejects = pd.DataFrame()
else:
    ranking = results["ranking"]
    rejects = results["rejects"]
    run_universe = results["tickers"]
    t0, t1, t2, t3, t4 = results["timings"]
    st.caption(f"Last run: {results['run_at_et']}")
    st.write(
        f"### ⏱️ Runtime: {t4 - t0:.2f}s total "
        f"(daily fetch {t1 - t0:.2f}s · pre-filter {t2 - t1:.2f}s · "
        f"intraday fetch {t3 - t2:.2f}s [{results['n_candidates']}/{results['n_universe']} tickers] · "
        f"engine {t4 - t3:.2f}s)"
    )
    if (t1 - t0) < 2.0:
        st.caption("✅ Daily fetch served from cache (fast) — daily history is cached for 6h, not re-downloaded every run.")
    if (t3 - t2) < 1.0 and results["n_candidates"] > 20:
        st.caption(
            "⚠️ Intraday fetch was near-instant — this run likely served CACHED "
            "data (up to 2 min old), not a fresh pull. If you need up-to-the-"
            "second prices, check 'Force fresh intraday data' above and re-run."
        )

    # --- Daily-bar freshness diagnostic (answers open question #4 directly
    # from YOUR data, rather than assuming an answer) ----------------------
    if _is_market_hours and results.get("n_checked", 0) > 0:
        n_today_rows = results["n_today_rows"]
        n_checked = results["n_checked"]
        pct_today = 100 * n_today_rows / n_checked if n_checked else 0
        if n_today_rows > 0:
            st.caption(
                f"ℹ️ Daily-bar freshness check: {n_today_rows}/{n_checked} tickers "
                f"({pct_today:.0f}%) have TODAY's date as the last row in their daily "
                f"batch — meaning Yahoo's daily data appears to update live during "
                f"market hours in your environment. The code already accounts for "
                f"this (Price_Open_PrevClose_% and the daily price pre-filter use the "
                f"prior settled close, not this live row)."
            )
        else:
            st.caption(
                f"ℹ️ Daily-bar freshness check: 0/{n_checked} tickers have today's date "
                f"as the last daily row — Yahoo's daily data appears to only settle "
                f"after the close in your environment, so `.iloc[-1]` alone would "
                f"have been correct anyway. Good to have confirmed either way."
            )

    movers_set = set(movers)
    if not ranking.empty:
        ranking = ranking.copy()
        ranking["Watchlist"] = ranking["Ticker"].apply(lambda t: "Movers List" if t in movers_set else "Not Movers List")
    if not rejects.empty:
        rejects = rejects.copy()
        rejects["Watchlist"] = rejects["Ticker"].apply(lambda t: "Movers List" if t in movers_set else "Not Movers List")

    if movers:
        st.write("### 🔥 Movers List Scorecard")
        scorecard_rows = []
        for m in movers:
            if m not in run_universe:
                scorecard_rows.append({"Ticker": m, "Status": "⚠️ Not in scanned universe (foreign listing, wrong exchange, or delisted — check the ticker)"})
                continue
            if not ranking.empty and m in ranking["Ticker"].values:
                row = ranking[ranking["Ticker"] == m].iloc[0]
                scorecard_rows.append({
                    "Ticker": m,
                    "Status": f"✅ Qualified — Close ${row['Close']}, {row['Price_Increase_%_5Bars']}% (5 bars), EMA_Aligned: {row.get('EMA_Aligned', 'N/A')}, Stale bars: {row.get('Stale_Bars_Last5', 'N/A')}",
                })
                continue
            if not rejects.empty and m in rejects["Ticker"].values:
                reasons = rejects[rejects["Ticker"] == m]["Reason"].dropna().unique().tolist()
                scorecard_rows.append({"Ticker": m, "Status": f"❌ Rejected — {'; '.join(reasons) if reasons else 'reason not recorded'}"})
                continue
            scorecard_rows.append({"Ticker": m, "Status": "❔ Not evaluated (in universe but no result recorded this run)"})
        st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# DISPLAY RESULTS (PRICE ACCELERATION — no EMA gate)
# ==============================================================================
if results is not None:
    if ranking.empty:
        st.warning("No tickers qualified.")
    else:
        st.write("### 📊 Price Acceleration (EMA shown as info only, not a gate)")
        qualified = ranking[ranking["Status"] == "PRICE"]
        if qualified.empty:
            st.info("No tickers qualified this run.")
        else:
            qualified_display = qualified.reset_index(drop=True)
            st.dataframe(qualified_display, use_container_width=True)
            st.caption(
                "Table sorted by Volume_Day_vs_50d then Price_Open_PrevClose_% (descending). "
                "Warrior gap-% and volume-ratio criteria are NOT yet hard filters — every "
                "PRICE-qualified ticker is shown regardless of how well it meets them; sort "
                "order surfaces the strongest matches at the top for manual review."
            )

            n_logged = log_top5(qualified_display)
            st.caption(f"📝 Logged top {n_logged} (by Volume_Day_vs_50d, matching the table above) to `{os.path.basename(TOP5_LOG_PATH)}` at this run's timestamp.")

# ==============================================================================
# TOP-5 RUNNING LOG
# ==============================================================================
st.write("### 📝 Top-5 Log (across runs)")

log_col1, log_col2 = st.columns([3, 1])
with log_col2:
    if "confirm_reset_log" not in st.session_state:
        st.session_state["confirm_reset_log"] = False

    if not st.session_state["confirm_reset_log"]:
        if st.button("Reset Log"):
            st.session_state["confirm_reset_log"] = True
    else:
        st.warning("This clears the entire log. Confirm?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, clear it"):
                reset_top5_log()
                st.session_state["confirm_reset_log"] = False
                st.success("Log cleared.")
        with c2:
            if st.button("Cancel"):
                st.session_state["confirm_reset_log"] = False

top5_log_df = load_top5_log()
with log_col1:
    if top5_log_df.empty:
        st.info("No entries logged yet. Run the model at least once with qualifying tickers to start building history.")
    else:
        st.dataframe(top5_log_df.sort_values("Run_Timestamp_ET", ascending=False), use_container_width=True)
        st.caption(f"{len(top5_log_df)} total logged entries · file: `{TOP5_LOG_PATH}`")

# ==============================================================================
# REJECTION DIAGNOSTICS
# ==============================================================================
if results is not None:
    st.write("### 🧪 Rejection Diagnostics")
    st.write(f"Total Rejects: {len(rejects)}")

    if rejects.empty:
        st.info("No rejections recorded.")
    else:
        st.write("### Rejection Summary")
        st.write(rejects["Reason"].value_counts())

        ticker_list = sorted(rejects["Ticker"].unique())
        selected = st.selectbox("Select ticker for rejection diagnostics", ticker_list)

        diag = rejects[rejects["Ticker"] == selected]
        st.dataframe(diag, use_container_width=True)