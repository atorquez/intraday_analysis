import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def calculate_ema_compression(df, window_9=9, window_20=20, window_50=50, threshold_pct=0.015):
    df['EMA9'] = df['Close'].ewm(span=window_9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=window_20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=window_50, adjust=False).mean()

    max_ema = df[['EMA9', 'EMA20', 'EMA50']].max(axis=1)
    min_ema = df[['EMA9', 'EMA20', 'EMA50']].min(axis=1)

    df['EMA_Spread'] = (max_ema - min_ema) / df['EMA50']
    df['Structural_Ready'] = (df['EMA_Spread'] <= threshold_pct) & (df['Close'] > df['EMA50'])
    return df


def calculate_residual_pca(ticker_features, spy_features):
    pca = PCA(n_components=1)
    ticker_pca1 = pca.fit_transform(ticker_features)
    spy_pca1 = pca.fit_transform(spy_features)

    model = LinearRegression()
    model.fit(spy_pca1, ticker_pca1)

    predicted_ticker_pca = model.predict(spy_pca1)
    residual_pca1 = ticker_pca1 - predicted_ticker_pca
    return residual_pca1


def check_intraday_actionability(current_price, previous_close, intraday_df):
    opening_gap_pct = (intraday_df['Open'].iloc[0] - previous_close) / previous_close
    if opening_gap_pct > 0.015:
        return "NO_TRADE: Excess Opening Gap"

    vwap_reclaim = (current_price > intraday_df['VWAP'].iloc[-1])
    rvol_spike = (intraday_df['Volume'].iloc[-1] > (intraday_df['Volume_3Month_Mean'].iloc[-1] * 2.0))
    intraday_acceleration = (intraday_df['PCA1_Slope'].iloc[-1] > 0)

    if vwap_reclaim and rvol_spike and intraday_acceleration:
        return "TRIGGER_LONG_ENTRY"

    return "HOLD: Inside Noise Parameters"
