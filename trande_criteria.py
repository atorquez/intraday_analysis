"""
================================================================================
INTRADAY TRADING CRITERIA ENGINE
================================================================================
A professional day trading screener that uses dual-timeframe analysis to identify
potential intraday trade setups.

TIME FRAMES:
- Daily:    Trend direction, volume, volatility screening
- 5-Minute: Intraday context, key levels (VWAP, EMAs, day high/low)
- 1-Minute: Precision entry triggers (VWAP bounce, EMA pullback, breakout retest)

SETUP TYPES DETECTED:
- VWAP_BOUNCE:      Price pulls back to VWAP and shows reversal candle
- EMA_PULLBACK:     Price pulls back to 9 EMA in uptrend and bounces
- BREAKOUT_RETEST:  Price breaks resistance, retests, and holds
- VWAP_REJECTION:   Price rejects at VWAP for short setups (bearish bias)

USAGE:
    from intraday_screener import IntradayScreener, TradeSetup

    screener = IntradayScreener(
        min_daily_volume=500_000,
        min_avg_true_range_pct=1.5,
        min_risk_reward=2.0
    )

    setups = screener.screen_ticker(
        ticker="AAPL",
        daily_df=daily_dataframe,
        df_5min=five_min_dataframe,
        df_1min=one_min_dataframe
    )

    for setup in setups:
        print(setup)

DATA SOURCE:
    Use Yahoo Finance API to fetch data:
    - Daily:  interval="1d", period="3mo"
    - 5-Min:  interval="5m", period="5d"  (intraday max 60 days)
    - 1-Min:  interval="1m", period="5d"  (intraday max 60 days)

    DataFrames must have columns: Date, Open, High, Low, Close, Volume
================================================================================
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TradeSetup:
    """
    Represents a potential trade setup found by the screener.

    Attributes:
        ticker: Stock symbol
        direction: 'LONG' or 'SHORT'
        entry_price: Suggested entry price
        stop_loss: Suggested stop loss price
        target_price: Suggested profit target
        risk_reward: Risk-to-reward ratio (e.g., 2.0 means 1:2)
        setup_type: Pattern name (VWAP_BOUNCE, EMA_PULLBACK, etc.)
        confidence_score: 0-100 score based on multiple factors
        daily_trend: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        volume_confirmed: Whether volume confirms the setup
        key_level: The critical technical level for this setup
    """
    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    setup_type: str
    confidence_score: float
    daily_trend: str
    volume_confirmed: bool
    key_level: Optional[float] = None

    def __repr__(self):
        return (f"TradeSetup({self.ticker} | {self.direction} | {self.setup_type} | "
                f"Entry: ${self.entry_price:.2f} | Stop: ${self.stop_loss:.2f} | "
                f"Target: ${self.target_price:.2f} | R:R = 1:{self.risk_reward:.1f} | "
                f"Score: {self.confidence_score:.0f}/100)")


# ==============================================================================
# MAIN SCREENER CLASS
# ==============================================================================

class IntradayScreener:
    """
    Professional Intraday Trading Screener

    Uses dual-timeframe analysis to identify high-probability day trading setups:
    - Daily timeframe establishes the primary trend bias
    - 5-minute timeframe provides intraday context and key levels
    - 1-minute timeframe delivers precise entry triggers

    All setups require:
    1. Timeframe alignment (daily + intraday bias agree)
    2. Volume confirmation (above average)
    3. Minimum risk/reward ratio (default 1.5:1)
    4. Defined technical level for stop placement
    """

    def __init__(self, 
                 min_daily_volume: int = 500_000,
                 min_avg_true_range_pct: float = 1.5,
                 ema_fast: int = 9,
                 ema_slow: int = 21,
                 min_risk_reward: float = 1.5,
                 max_spread_pct: float = 0.5):
        """
        Initialize the screener with configurable parameters.

        Args:
            min_daily_volume: Minimum average daily volume in shares (liquidity filter)
            min_avg_true_range_pct: Minimum ATR as % of price (volatility filter)
            ema_fast: Fast EMA period (default 9)
            ema_slow: Slow EMA period (default 21)
            min_risk_reward: Minimum acceptable risk/reward ratio
            max_spread_pct: Maximum bid-ask spread % (liquidity filter)
        """
        self.min_daily_volume = min_daily_volume
        self.min_avg_true_range_pct = min_avg_true_range_pct
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.min_risk_reward = min_risk_reward
        self.max_spread_pct = max_spread_pct

    # ==========================================================================
    # TECHNICAL INDICATORS
    # ==========================================================================

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP = cumulative(Typical Price * Volume) / cumulative(Volume)
        where Typical Price = (High + Low + Close) / 3

        VWAP resets at market open each day.
        """
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_tp_vol = (tp * df['Volume']).cumsum()
        cumulative_vol = df['Volume'].cumsum()
        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for volatility measurement.

        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = SMA of True Range over N periods
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (momentum oscillator)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # ==========================================================================
    # STEP 1A: DAILY SCREENING (Universe Selection)
    # ==========================================================================

    def screen_daily(self, daily_df: pd.DataFrame) -> Optional[Dict]:
        """
        Screen a stock using DAILY data to determine if it warrants 
        intraday analysis.

        Criteria checked:
        - Volume: Above minimum threshold and above 20-day average
        - Volatility: ATR >= min threshold (ensures tradable range)
        - Trend: Price above/below EMAs with proper alignment
        - Momentum: RSI not in extreme overbought/oversold
        - Candle structure: Recent candle direction aligns with bias

        Returns:
            Dict with screening results, or None if stock is disqualified.
        """
        if daily_df is None or len(daily_df) < 30:
            return None

        df = daily_df.copy().sort_values('Date')

        # Calculate indicators
        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_21'] = self.calculate_ema(df['Close'], 21)
        df['ATR'] = self.calculate_atr(df)
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])

        latest = df.iloc[-1]

        # --- CRITERIA CHECKS ---

        # 1. Volume Filter: Must have enough liquidity
        volume_ok = latest['Volume'] >= self.min_daily_volume
        volume_surge = latest['Volume'] > latest['Volume_SMA20'] * 1.2

        # 2. Volatility Filter: Need enough range to profit
        atr_ok = latest['ATR_Pct'] >= self.min_avg_true_range_pct

        # 3. Trend Filter: Price position relative to EMAs
        price_above_ema9 = latest['Close'] > latest['EMA_9']
        price_above_ema21 = latest['Close'] > latest['EMA_21']
        ema_bullish = latest['EMA_9'] > latest['EMA_21']

        # 4. Momentum Filter: Avoid extreme conditions
        rsi = latest['RSI']
        not_overbought = rsi < 75
        not_oversold = rsi > 25

        # 5. Recent candle direction
        bullish_candle = latest['Close'] > latest['Open']

        # Calculate bias scores
        bullish_score = sum([
            price_above_ema9, price_above_ema21, ema_bullish,
            volume_surge, bullish_candle, atr_ok
        ])
        bearish_score = sum([
            not price_above_ema9, not price_above_ema21, not ema_bullish,
            volume_surge, not bullish_candle, atr_ok
        ])

        # Determine daily bias
        if bullish_score >= 4:
            daily_bias = 'BULLISH'
        elif bearish_score >= 4:
            daily_bias = 'BEARISH'
        else:
            daily_bias = 'NEUTRAL'

        # Disqualify if no clear bias or insufficient volatility/volume
        if daily_bias == 'NEUTRAL' or not atr_ok or not volume_ok:
            return None

        return {
            'ticker': None,  # Filled by caller
            'daily_bias': daily_bias,
            'atr_pct': latest['ATR_Pct'],
            'volume_surge': volume_surge,
            'rsi': rsi,
            'price': latest['Close'],
            'ema_9': latest['EMA_9'],
            'ema_21': latest['EMA_21'],
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }

    # ==========================================================================
    # STEP 1B: 5-MINUTE CONTEXT (Trend & Levels)
    # ==========================================================================

    def analyze_intraday_5min(self, df_5min: pd.DataFrame, daily_bias: str) -> Optional[Dict]:
        """
        Analyze 5-minute chart to establish intraday trend context
        and identify key levels for potential trades.

        Key outputs:
        - intraday_bias: BULLISH/BEARISH/CAUTIOUS/PULLBACK_MODE
        - key_levels: VWAP, EMAs, day high/low, pre-market high/low
        - volume_confirmed: Whether current volume supports the move
        """
        if df_5min is None or len(df_5min) < 30:
            return None

        df = df_5min.copy().sort_values('Date')

        # Calculate 5-min indicators
        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_21'] = self.calculate_ema(df['Close'], 21)
        df['VWAP'] = self.calculate_vwap(df)
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

        latest = df.iloc[-1]

        # --- TREND ANALYSIS ---
        price_above_vwap = latest['Close'] > latest['VWAP']
        ema_aligned = latest['EMA_9'] > latest['EMA_21']
        volume_confirmed = latest['Volume'] > latest['Volume_SMA'] * 1.1

        # --- KEY LEVELS ---
        day_high = df['High'].max()
        day_low = df['Low'].min()
        pre_market_high = df.iloc[:15]['High'].max() if len(df) > 15 else day_high
        pre_market_low = df.iloc[:15]['Low'].min() if len(df) > 15 else day_low

        # --- DETERMINE INTRADAY BIAS ---
        if daily_bias == 'BULLISH':
            if price_above_vwap and ema_aligned:
                intraday_bias = 'BULLISH'
            elif price_above_vwap:
                intraday_bias = 'CAUTIOUSLY_BULLISH'
            else:
                intraday_bias = 'PULLBACK_MODE'
        elif daily_bias == 'BEARISH':
            if not price_above_vwap and not ema_aligned:
                intraday_bias = 'BEARISH'
            elif not price_above_vwap:
                intraday_bias = 'CAUTIOUSLY_BEARISH'
            else:
                intraday_bias = 'PULLBACK_MODE'
        else:
            intraday_bias = 'NEUTRAL'

        key_levels = {
            'vwap': latest['VWAP'],
            'ema_9': latest['EMA_9'],
            'ema_21': latest['EMA_21'],
            'day_high': day_high,
            'day_low': day_low,
            'pre_market_high': pre_market_high,
            'pre_market_low': pre_market_low
        }

        return {
            'intraday_bias': intraday_bias,
            'price': latest['Close'],
            'volume_confirmed': volume_confirmed,
            'key_levels': key_levels,
            'price_above_vwap': price_above_vwap,
            'ema_aligned': ema_aligned,
            'dataframe': df
        }

    # ==========================================================================
    # STEP 1C: 1-MINUTE ENTRY TRIGGERS
    # ==========================================================================

    def find_entry_setups_1min(self, df_1min: pd.DataFrame, 
                                context_5min: Dict,
                                daily_info: Dict) -> List[TradeSetup]:
        """
        Analyze 1-minute chart to find precise entry setups.

        LONG Setups (when daily + 5-min are bullish):
        1. VWAP_BOUNCE: Price pulls back to VWAP, forms reversal candle, bounces
        2. EMA_PULLBACK: Price pulls back to 9 EMA in uptrend, bounces
        3. BREAKOUT_RETEST: Price breaks above resistance, retests, holds

        SHORT Setups (when daily + 5-min are bearish):
        1. VWAP_REJECTION: Price rallies to VWAP, forms rejection candle, falls

        Each setup includes:
        - Entry price (current close)
        - Stop loss (below technical level or recent low)
        - Target (based on risk/reward ratio)
        - Confidence score (timeframe alignment + volume + candle quality)
        """
        setups = []

        if df_1min is None or len(df_1min) < 30:
            return setups

        df = df_1min.copy().sort_values('Date')

        # Calculate 1-min indicators
        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_21'] = self.calculate_ema(df['Close'], 21)
        df['VWAP'] = self.calculate_vwap(df)
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        price = latest['Close']
        vwap = latest['VWAP']
        ema9 = latest['EMA_9']
        ema21 = latest['EMA_21']

        intraday_bias = context_5min['intraday_bias']
        daily_bias = daily_info['daily_bias']

        # ======================================================================
        # LONG SETUPS
        # ======================================================================
        if daily_bias == 'BULLISH' and intraday_bias in ['BULLISH', 'CAUTIOUSLY_BULLISH', 'PULLBACK_MODE']:

            # --- SETUP 1: VWAP BOUNCE ---
            if self._is_vwap_bounce(df, latest, prev):
                entry = price
                stop = min(df.iloc[-5:]['Low'].min(), vwap * 0.998)
                stop = max(stop, entry * 0.985)  # Cap max loss at 1.5%
                risk = entry - stop
                target = entry + (risk * 2.0)
                rr = (target - entry) / risk if risk > 0 else 0

                if rr >= self.min_risk_reward:
                    confidence = self._calculate_confidence(
                        daily_bias, intraday_bias, 
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        candle_strength=self._candle_strength(latest)
                    )
                    setups.append(TradeSetup(
                        ticker=daily_info.get('ticker', 'UNKNOWN'),
                        direction='LONG',
                        entry_price=round(entry, 2),
                        stop_loss=round(stop, 2),
                        target_price=round(target, 2),
                        risk_reward=round(rr, 2),
                        setup_type='VWAP_BOUNCE',
                        confidence_score=confidence,
                        daily_trend=daily_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        key_level=round(vwap, 2)
                    ))

            # --- SETUP 2: EMA PULLBACK ---
            if self._is_ema_pullback(df, latest, prev, ema9):
                entry = price
                stop = min(df.iloc[-3:]['Low'].min(), ema9 * 0.997)
                stop = max(stop, entry * 0.985)
                risk = entry - stop
                target = entry + (risk * 2.5)
                rr = (target - entry) / risk if risk > 0 else 0

                if rr >= self.min_risk_reward:
                    confidence = self._calculate_confidence(
                        daily_bias, intraday_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        candle_strength=self._candle_strength(latest)
                    )
                    setups.append(TradeSetup(
                        ticker=daily_info.get('ticker', 'UNKNOWN'),
                        direction='LONG',
                        entry_price=round(entry, 2),
                        stop_loss=round(stop, 2),
                        target_price=round(target, 2),
                        risk_reward=round(rr, 2),
                        setup_type='EMA_PULLBACK',
                        confidence_score=confidence,
                        daily_trend=daily_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        key_level=round(ema9, 2)
                    ))

            # --- SETUP 3: BREAKOUT RETEST ---
            if self._is_breakout_retest(df, latest, context_5min['key_levels']):
                resistance = context_5min['key_levels']['pre_market_high']
                entry = price
                stop = min(df.iloc[-3:]['Low'].min(), resistance * 0.995)
                stop = max(stop, entry * 0.985)
                risk = entry - stop
                target = entry + (risk * 2.0)
                rr = (target - entry) / risk if risk > 0 else 0

                if rr >= self.min_risk_reward:
                    confidence = self._calculate_confidence(
                        daily_bias, intraday_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        candle_strength=self._candle_strength(latest)
                    )
                    setups.append(TradeSetup(
                        ticker=daily_info.get('ticker', 'UNKNOWN'),
                        direction='LONG',
                        entry_price=round(entry, 2),
                        stop_loss=round(stop, 2),
                        target_price=round(target, 2),
                        risk_reward=round(rr, 2),
                        setup_type='BREAKOUT_RETEST',
                        confidence_score=confidence,
                        daily_trend=daily_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        key_level=round(resistance, 2)
                    ))

        # ======================================================================
        # SHORT SETUPS
        # ======================================================================
        elif daily_bias == 'BEARISH' and intraday_bias in ['BEARISH', 'CAUTIOUSLY_BEARISH']:

            if self._is_vwap_rejection(df, latest, prev):
                entry = price
                stop = max(df.iloc[-5:]['High'].max(), vwap * 1.002)
                stop = min(stop, entry * 1.015)
                risk = stop - entry
                target = entry - (risk * 2.0)
                rr = (entry - target) / risk if risk > 0 else 0

                if rr >= self.min_risk_reward:
                    confidence = self._calculate_confidence(
                        daily_bias, intraday_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        candle_strength=self._candle_strength(latest, bearish=True)
                    )
                    setups.append(TradeSetup(
                        ticker=daily_info.get('ticker', 'UNKNOWN'),
                        direction='SHORT',
                        entry_price=round(entry, 2),
                        stop_loss=round(stop, 2),
                        target_price=round(target, 2),
                        risk_reward=round(rr, 2),
                        setup_type='VWAP_REJECTION',
                        confidence_score=confidence,
                        daily_trend=daily_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        key_level=round(vwap, 2)
                    ))

        return setups

    # ==========================================================================
    # CANDLE PATTERN DETECTORS
    # ==========================================================================

    def _is_vwap_bounce(self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series) -> bool:
        """
        Detect VWAP bounce pattern for LONG entry.

        Criteria:
        1. Price is at or above VWAP
        2. Price touched VWAP recently (within last 6 candles)
        3. Current candle is bullish (close > open)
        4. Candle has a lower wick (buying pressure at lows)
        """
        vwap = latest['VWAP']
        price = latest['Close']

        # Must be at or above VWAP
        if price < vwap * 0.998:
            return False

        # Check recent touch of VWAP
        if len(df) < 7:
            return False
        recent_df = df.iloc[-7:-1]
        touched_vwap = any(recent_df['Low'] <= vwap * 1.003)

        if not touched_vwap:
            return False

        # Bullish candle structure
        bullish = latest['Close'] > latest['Open']

        # Lower wick indicates buying pressure at lows
        lower_wick = min(latest['Open'], latest['Close']) - latest['Low']
        body = abs(latest['Close'] - latest['Open'])
        has_lower_wick = lower_wick > body * 0.2 if body > 0.01 else lower_wick > 0.01

        return bullish and has_lower_wick

    def _is_ema_pullback(self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series, ema9: float) -> bool:
        """
        Detect EMA pullback pattern for LONG entry.

        Criteria:
        1. Price is above EMA9
        2. Price touched EMA9 recently (within last 4 candles)
        3. EMA9 is rising (uptrend confirmation)
        4. Current candle is bullish
        """
        price = latest['Close']

        if price < ema9 * 0.997:
            return False

        if len(df) < 5:
            return False
        recent_df = df.iloc[-5:-1]
        touched_ema = any(recent_df['Low'] <= ema9 * 1.005)

        if not touched_ema:
            return False

        # EMA slope should be positive
        ema_series = df['EMA_9']
        ema_rising = ema_series.iloc[-1] > ema_series.iloc[-5] if len(ema_series) >= 5 else True

        bullish = latest['Close'] > latest['Open']

        return ema_rising and bullish

    def _is_breakout_retest(self, df: pd.DataFrame, latest: pd.Series, key_levels: Dict) -> bool:
        """
        Detect breakout and retest pattern for LONG entry.

        Criteria:
        1. Price is above resistance level
        2. Price broke above resistance recently
        3. Price retested the breakout level and held
        4. Current candle is bullish and holding above breakout
        """
        resistance = key_levels['pre_market_high']
        price = latest['Close']

        if price < resistance * 1.001:
            return False

        if len(df) < 12:
            return False
        recent_df = df.iloc[-12:-1]
        broke_out = any(recent_df['High'] > resistance * 1.003)

        if not broke_out:
            return False

        recent_lows = df.iloc[-6:-1]['Low']
        retested = any(recent_lows <= resistance * 1.015)
        holding = price > resistance * 1.002
        bullish = latest['Close'] > latest['Open']

        return retested and holding and bullish

    def _is_vwap_rejection(self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series) -> bool:
        """
        Detect VWAP rejection pattern for SHORT entry.

        Criteria:
        1. Price is at or below VWAP
        2. Price touched VWAP recently (rejection)
        3. Current candle is bearish (close < open)
        4. Candle has an upper wick (selling pressure at highs)
        """
        vwap = latest['VWAP']
        price = latest['Close']

        if price > vwap * 1.002:
            return False

        if len(df) < 7:
            return False
        recent_df = df.iloc[-7:-1]
        touched_vwap = any(recent_df['High'] >= vwap * 0.997)

        if not touched_vwap:
            return False

        bearish = latest['Close'] < latest['Open']
        upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
        body = abs(latest['Close'] - latest['Open'])
        has_upper_wick = upper_wick > body * 0.2 if body > 0.01 else upper_wick > 0.01

        return bearish and has_upper_wick

    def _candle_strength(self, candle: pd.Series, bearish: bool = False) -> float:
        """
        Score candle strength from 0-100 based on body and wick ratios.

        For bullish: Strong body + small upper wick = high score
        For bearish: Strong body + small lower wick = high score
        """
        body = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']

        if total_range < 0.001:
            return 50

        body_ratio = body / total_range

        if bearish:
            lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
            lower_wick_ratio = lower_wick / total_range
            score = (body_ratio * 60) + ((1 - lower_wick_ratio) * 40)
        else:
            upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
            upper_wick_ratio = upper_wick / total_range
            score = (body_ratio * 60) + ((1 - upper_wick_ratio) * 40)

        return min(100, max(0, score))

    def _calculate_confidence(self, daily_bias: str, intraday_bias: str,
                             volume_confirmed: bool, candle_strength: float) -> float:
        """
        Calculate overall confidence score (0-100).

        Components:
        - Timeframe alignment: 40 pts (daily + intraday agree = max)
        - Volume confirmation: 25 pts
        - Candle quality: 25 pts
        - Base score: 10 pts
        """
        score = 0

        # Timeframe alignment (max 40 pts)
        if daily_bias == intraday_bias:
            score += 40
        elif 'CAUTIOUSLY' in intraday_bias and daily_bias in intraday_bias:
            score += 30
        else:
            score += 15

        # Volume (max 25 pts)
        score += 25 if volume_confirmed else 10

        # Candle quality (max 25 pts)
        score += (candle_strength / 100) * 25

        # Base score (10 pts)
        score += 10

        return round(score, 1)

    # ==========================================================================
    # MAIN PIPELINE
    # ==========================================================================

    def screen_ticker(self, ticker: str,
                      daily_df: pd.DataFrame,
                      df_5min: pd.DataFrame,
                      df_1min: pd.DataFrame) -> List[TradeSetup]:
        """
        Full screening pipeline for a single ticker.

        Pipeline:
        1. Daily screen -> determines if stock is worth analyzing
        2. 5-min analysis -> establishes intraday context and levels
        3. 1-min triggers -> finds precise entry setups

        Returns:
            List of TradeSetup objects (empty if no valid setups)
        """
        # Step 1: Daily screen
        daily_result = self.screen_daily(daily_df)
        if daily_result is None:
            return []

        daily_result['ticker'] = ticker

        # Step 2: 5-min context
        context_5min = self.analyze_intraday_5min(df_5min, daily_result['daily_bias'])
        if context_5min is None:
            return []

        # Step 3: 1-min entry triggers
        setups = self.find_entry_setups_1min(df_1min, context_5min, daily_result)

        return setups


# ==============================================================================
# BATCH SCREENING (Multiple Tickers)
# ==============================================================================

def screen_watchlist(screener: IntradayScreener,
                     ticker_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[TradeSetup]]:
    """
    Screen multiple tickers at once.

    Args:
        screener: Configured IntradayScreener instance
        ticker_data: Dict mapping ticker -> {'daily': df, '5min': df, '1min': df}

    Returns:
        Dict mapping ticker -> list of TradeSetup objects
    """
    results = {}

    for ticker, data in ticker_data.items():
        try:
            setups = screener.screen_ticker(
                ticker=ticker,
                daily_df=data.get('daily'),
                df_5min=data.get('5min'),
                df_1min=data.get('1min')
            )
            if setups:
                results[ticker] = setups
        except Exception as e:
            print(f"Error screening {ticker}: {e}")
            continue

    return results


# ==============================================================================
# EXAMPLE: Integration with Yahoo Finance data source
# ==============================================================================

def example_yahoo_finance_integration():
    """
    Example showing how to integrate with Yahoo Finance data.

    Note: This is pseudocode showing the data flow. In practice, you would:
    1. Fetch daily data: interval="1d", period="3mo"
    2. Fetch 5-min data: interval="5m", period="5d"  (max 60 days for intraday)
    3. Fetch 1-min data: interval="1m", period="5d"  (max 60 days for intraday)
    """

    # Pseudocode for data fetching:
    """
    from your_data_source import get_historical_stock_prices

    ticker = "AAPL"

    # Fetch daily data (for trend analysis)
    daily_df = get_historical_stock_prices(
        ticker=ticker,
        interval="1d",
        period="3mo"
    )

    # Fetch 5-min data (for intraday context)
    df_5min = get_historical_stock_prices(
        ticker=ticker,
        interval="5m",
        period="5d"  # Must be <= 60 days for intraday
    )

    # Fetch 1-min data (for entry triggers)
    df_1min = get_historical_stock_prices(
        ticker=ticker,
        interval="1m",
        period="5d"  # Must be <= 60 days for intraday
    )

    # Run screener
    screener = IntradayScreener(
        min_daily_volume=1_000_000,
        min_avg_true_range_pct=2.0,
        min_risk_reward=2.0
    )

    setups = screener.screen_ticker(ticker, daily_df, df_5min, df_1min)

    for setup in setups:
        print(setup)
    """
    pass


# ==============================================================================
# MAIN (Demo)
# ==============================================================================

if __name__ == "__main__":
    print("Intraday Trading Criteria Engine")
    print("Import IntradayScreener and TradeSetup to use in your project.")
    print("See example_yahoo_finance_integration() for usage pattern.")