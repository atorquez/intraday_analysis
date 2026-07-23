# ==============================================================================
# MASTER REGIME AND MULTI-MODE BULK SCANNER ENGINE (PATCHED & PRODUCTION READY)
# FIXED VERSION — Addresses all critical bugs and performance issues
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# MARKET REGIME DETECTOR
# ---------------------------------------------------------
def automatic_market_regime_detector(market_daily_df, market_5min_df):
    if market_daily_df is None or market_daily_df.empty or market_5min_df is None or market_5min_df.empty:
        return "CHOPPY_MARKET"

    # Flatten MultiIndex columns safely
    if isinstance(market_daily_df.columns, pd.MultiIndex):
        market_daily_df.columns = market_daily_df.columns.get_level_values(0)

    if isinstance(market_5min_df.columns, pd.MultiIndex):
        market_5min_df.columns = market_5min_df.columns.get_level_values(0)

    # Normalize column names cleanly
    market_daily_df.columns = market_daily_df.columns.str.capitalize()
    market_5min_df.columns = market_5min_df.columns.str.capitalize()

    try:
        # Extract guaranteed scalar floating points
        yesterday_close = float(market_daily_df['Close'].iloc[-1])
        today_open = float(market_5min_df['Open'].iloc[0])
        current_price = float(market_5min_df['Close'].iloc[-1])

        # Compute gap and intraday return
        gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100
        intraday_return = ((current_price - today_open) / today_open) * 100

        # Regime logic thresholds
        if gap_percent < -0.4 or intraday_return < -0.3:
            return "BEARISH_MARKET"
        elif gap_percent > 0.4 or intraday_return > 0.3:
            return "BULLISH_MARKET"
        else:
            return "CHOPPY_MARKET"

    except Exception as e:
        logger.warning(f"Regime calculation bypass due to structural index error: {e}")
        return "CHOPPY_MARKET"


# ---------------------------------------------------------
# PART 1 — BULLISH SCANNER
# ---------------------------------------------------------
def part1_bullish_scanner(universe_df):
    if universe_df is None or universe_df.empty:
        return []

    strong = universe_df[
        (universe_df['Trend'] == 'UP') &
        (universe_df['Execution'] == 'Ready')
    ]
    return strong['Ticker'].head(50).tolist()


# ---------------------------------------------------------
# PART 1 — BEARISH SCANNER
# ---------------------------------------------------------
def part1_bearish_scanner(universe_df, mode="SHORT"):
    if universe_df is None or universe_df.empty:
        return []

    if mode == "SHORT":
        weak = universe_df[
            (universe_df['Trend'] == 'DOWN') &
            (universe_df['Execution'].isin(['Ready', 'Crossing Soon']))
        ]
        return weak['Ticker'].head(50).tolist()

    elif mode == "LONG":
        strong_rel = universe_df[
            (universe_df['Gap%'] > 0) & 
            (universe_df['Trend'] == 'UP')
        ]
        return strong_rel['Ticker'].head(50).tolist()

    return []


# ---------------------------------------------------------
# PART 2 — REGIME-AWARE INTRADAY TRIGGER (SINGLE HISTORICAL RUNNER)
# ---------------------------------------------------------
def process_regime_intraday_trigger(stock_df, market_df, mode="LONG"):
    if stock_df is None or stock_df.empty or market_df is None or market_df.empty:
        logger.debug("Empty input dataframes")
        return False

    # Validate mode
    mode = str(mode).upper()
    if mode not in ("LONG", "SHORT"):
        logger.warning(f"Invalid mode '{mode}', defaulting to LONG")
        mode = "LONG"

    # Force uniform naming conventions prior to merge operation
    stock_df = stock_df.copy()
    market_df = market_df.copy()

    if isinstance(stock_df.columns, pd.MultiIndex): 
        stock_df.columns = stock_df.columns.get_level_values(0)
    if isinstance(market_df.columns, pd.MultiIndex): 
        market_df.columns = market_df.columns.get_level_values(0)

    stock_df.columns = stock_df.columns.str.capitalize()
    market_df.columns = market_df.columns.str.capitalize()

    # Align chronological merge keys
    aligned = pd.merge(
        stock_df[['Close', 'Volume']],
        market_df[['Close']],
        left_index=True, right_index=True,
        suffixes=('_stock', '_market')
    ).dropna()

    if len(aligned) < 50:
        logger.debug(f"Insufficient aligned data: {len(aligned)} rows")
        return False

    # Structural Technical Framework Integration
    aligned['ema9'] = aligned['Close_stock'].ewm(span=9, adjust=False).mean()
    aligned['ema20'] = aligned['Close_stock'].ewm(span=20, adjust=False).mean()

    aligned['ema_spread'] = aligned['ema9'] - aligned['ema20']
    aligned['stock_ret'] = aligned['Close_stock'].pct_change()
    aligned['market_ret'] = aligned['Close_market'].pct_change()

    # FIX: Guard against division by zero in beta calculation
    cov = aligned['stock_ret'].rolling(30).cov(aligned['market_ret'])
    var_m = aligned['market_ret'].rolling(30).var()

    # FIX: Replace zero variance with NaN to prevent inf
    var_m_safe = var_m.replace(0, np.nan)
    beta = cov / var_m_safe
    aligned['residual_ret'] = aligned['stock_ret'] - (beta * aligned['market_ret'])

    features = aligned[['ema_spread', 'residual_ret']].dropna()
    if len(features) < 10:
        logger.debug(f"Insufficient features after dropna: {len(features)} rows")
        return False

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=1)
    pca_values = pca.fit_transform(scaled)

    # FIX: Proper index alignment for PCA results
    pca_series = pd.Series(pca_values.flatten(), index=features.index)
    aligned['PCA_Alpha'] = pca_series.reindex(aligned.index)

    current = aligned.iloc[-1]
    prior = aligned.iloc[-2]

    # Binary Condition Execution Engine
    if mode == "LONG":
        ema_cross = (prior['Close_stock'] <= prior['ema9']) and (current['Close_stock'] > current['ema9'])
        ema_stack = current['ema9'] > current['ema20']
        pca_up = current['PCA_Alpha'] > prior['PCA_Alpha']
        return bool(ema_cross and ema_stack and pca_up)

    elif mode == "SHORT":
        ema_cross = (prior['Close_stock'] >= prior['ema9']) and (current['Close_stock'] < current['ema9'])
        ema_stack = current['ema9'] < current['ema20']
        pca_down = current['PCA_Alpha'] < prior['PCA_Alpha']
        return bool(ema_cross and ema_stack and pca_down)

    return False


# ---------------------------------------------------------
# NEW (BULK) PART 2 — REAL-TIME HIGH SPEED EXECUTION PIPELINE
# ---------------------------------------------------------
def process_bulk_regime_triggers(watchlist, market_1min_df, mode="LONG"):
    """
    Downloads and evaluates signals for an entire group of stocks simultaneously.
    Bypasses the iterative loop layout to avoid API rate limits.
    """
    # FIX: Input validation
    if not watchlist:
        logger.info("Empty watchlist provided")
        return []

    if isinstance(watchlist, str):
        watchlist = [watchlist]

    watchlist = [t.strip().upper() for t in watchlist if t and isinstance(t, str)]

    if not watchlist:
        logger.info("Watchlist empty after validation")
        return []

    mode = str(mode).upper()
    if mode not in ("LONG", "SHORT"):
        logger.warning(f"Invalid mode '{mode}', defaulting to LONG")
        mode = "LONG"

    if market_1min_df is None or market_1min_df.empty:
        logger.warning("Empty market dataframe")
        return []

    triggered_tickers = []

    # Flatten and normalize the market index dataframe
    market_1min_df = market_1min_df.copy()
    if isinstance(market_1min_df.columns, pd.MultiIndex):
        market_1min_df.columns = market_1min_df.columns.get_level_values(0)
    market_1min_df.columns = market_1min_df.columns.str.capitalize()

    # Execute a unified 1-minute batch download across the internet
    try:
        bulk_intraday = yf.download(
            tickers=watchlist,
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker"
        )
    except Exception as e:
        logger.error(f"Bulk intraday yfinance extraction failure: {e}")
        return []

    # FIX: Handle both single-ticker (flat) and multi-ticker (MultiIndex) responses
    is_multi = isinstance(bulk_intraday.columns, pd.MultiIndex)

    # Parse and compute individual vectors out of memory
    for ticker in watchlist:
        try:
            # FIX: Extract ticker sub-frame with single-ticker handling
            if is_multi:
                if ticker in bulk_intraday.columns.get_level_values(0):
                    stock_df = bulk_intraday[ticker].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                else:
                    continue
            else:
                if len(watchlist) == 1 and ticker == watchlist[0]:
                    stock_df = bulk_intraday.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                else:
                    continue

            if stock_df.empty or len(stock_df) < 35:
                logger.debug(f"{ticker}: insufficient data ({len(stock_df)} rows)")
                continue

            # Standardize naming configurations before calculating
            stock_df.columns = stock_df.columns.str.capitalize()

            # Pass inputs straight into our core mathematical trigger function
            is_triggered = process_regime_intraday_trigger(stock_df, market_1min_df, mode=mode)

            if is_triggered:
                triggered_tickers.append(ticker)
                logger.info(f"TRIGGER: {ticker} ({mode})")

        except Exception as e:
            logger.warning(f"Bypassing internal calculations on {ticker}: {e}")
            continue

    return triggered_tickers