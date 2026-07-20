import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# MARKET REGIME DETECTOR
# ---------------------------------------------------------
def automatic_market_regime_detector(market_daily_df, market_5min_df):
    # --- FIX 1: Flatten MultiIndex columns ---
    if isinstance(market_daily_df.columns, pd.MultiIndex):
        market_daily_df.columns = market_daily_df.columns.get_level_values(0)

    if isinstance(market_5min_df.columns, pd.MultiIndex):
        market_5min_df.columns = market_5min_df.columns.get_level_values(0)

    # --- FIX 2: Normalize column names ---
    market_daily_df.columns = market_daily_df.columns.str.capitalize()
    market_5min_df.columns = market_5min_df.columns.str.capitalize()

    # --- FIX 3: Extract scalar values ---
    yesterday_close = float(market_daily_df['Close'].iloc[-1])
    today_open = float(market_5min_df['Open'].iloc[0])
    current_price = float(market_5min_df['Close'].iloc[-1])

    # --- Compute gap and intraday return ---
    gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100
    intraday_return = ((current_price - today_open) / today_open) * 100

    # --- Regime logic ---
    if gap_percent < -0.4 or intraday_return < -0.3:
        return "BEARISH_MARKET"
    elif gap_percent > 0.4 or intraday_return > 0.3:
        return "BULLISH_MARKET"
    else:
        return "CHOPPY_MARKET"

# ---------------------------------------------------------
# PART 1 — BULLISH SCANNER
# ---------------------------------------------------------
def part1_bullish_scanner(universe_df):
    strong = universe_df[
        (universe_df['Trend'] == 'UP') &
        (universe_df['Execution_Status'] == 'Ready')
    ]
    return strong['Ticker'].head(50).tolist()


# ---------------------------------------------------------
# PART 1 — BEARISH SCANNER
# ---------------------------------------------------------
def part1_bearish_scanner(universe_df, mode="SHORT"):
    if mode == "SHORT":
        weak = universe_df[
            (universe_df['Trend'] == 'DOWN') &
            (universe_df['Execution_Status'].isin(['Ready', 'Crossing Soon']))
        ]
        return weak['Ticker'].head(50).tolist()

    elif mode == "LONG":
        strong_rel = universe_df[
            (universe_df['Today_Return'] > 0) &
            (universe_df['Trend'] == 'UP')
        ]
        return strong_rel['Ticker'].head(50).tolist()


# ---------------------------------------------------------
# PART 2 — REGIME-AWARE INTRADAY TRIGGER
# ---------------------------------------------------------
def process_regime_intraday_trigger(stock_df, market_df, mode="LONG"):
    aligned = pd.merge(
        stock_df[['Close', 'Volume']],
        market_df[['Close']],
        left_index=True, right_index=True,
        suffixes=('_stock', '_market')
    ).dropna()

    if len(aligned) < 50:
        return False

    aligned['ema9'] = aligned['Close_stock'].ewm(span=9, adjust=False).mean()
    aligned['ema20'] = aligned['Close_stock'].ewm(span=20, adjust=False).mean()

    aligned['ema_spread'] = aligned['ema9'] - aligned['ema20']
    aligned['stock_ret'] = aligned['Close_stock'].pct_change()
    aligned['market_ret'] = aligned['Close_market'].pct_change()

    cov = aligned['stock_ret'].rolling(30).cov(aligned['market_ret'])
    var_m = aligned['market_ret'].rolling(30).var()
    beta = cov / var_m
    aligned['residual_ret'] = aligned['stock_ret'] - (beta * aligned['market_ret'])

    features = aligned[['ema_spread', 'residual_ret']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=1)
    aligned.loc[features.index, 'PCA_Alpha'] = pca.fit_transform(scaled)

    current = aligned.iloc[-1]
    prior = aligned.iloc[-2]

    if mode == "LONG":
        ema_cross = (prior['Close_stock'] <= prior['ema9']) and (current['Close_stock'] > current['ema9'])
        ema_stack = current['ema9'] > current['ema20']
        pca_up = current['PCA_Alpha'] > prior['PCA_Alpha']
        return ema_cross and ema_stack and pca_up

    elif mode == "SHORT":
        ema_cross = (prior['Close_stock'] >= prior['ema9']) and (current['Close_stock'] < current['ema9'])
        ema_stack = current['ema9'] < current['ema20']
        pca_down = current['PCA_Alpha'] < prior['PCA_Alpha']
        return ema_cross and ema_stack and pca_down

    return False

