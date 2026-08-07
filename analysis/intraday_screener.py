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

    def __init__(self,
                 min_daily_volume: int = 300_000,
                 min_avg_true_range_pct: float = 0.8,
                 ema_fast: int = 9,
                 ema_slow: int = 21,
                 min_risk_reward: float = 1.5,
                 max_spread_pct: float = 0.5):

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
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_tp_vol = (tp * df['Volume']).cumsum()
        cumulative_vol = df['Volume'].cumsum()
        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # ==========================================================================  
    # STEP 1A: DAILY SCREENING  
    # ==========================================================================  

    def screen_daily(self, daily_df: pd.DataFrame) -> Optional[Dict]:

        if daily_df is None or len(daily_df) < 30:
            return None

        # --- SANITIZE DAILY DATA ---
        df = daily_df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(level) for level in col if level]) for col in df.columns]

        df = df.loc[:, ~df.columns.duplicated()]

        def safe_extract(prefix):
            cols = [c for c in df.columns if prefix in c]
            if not cols:
                return None
            col = df[cols[0]]
            if isinstance(col, pd.DataFrame):
                return col.iloc[:, 0]
            return col

        clean = pd.DataFrame()
        clean["Date"] = df["Date"] if "Date" in df.columns else df.index
        clean["Open"] = safe_extract("Open")
        clean["High"] = safe_extract("High")
        clean["Low"] = safe_extract("Low")
        clean["Close"] = safe_extract("Close")
        clean["Volume"] = safe_extract("Volume")

        if clean["Close"] is None:
            return None

        clean = clean.dropna(subset=["Open", "High", "Low", "Close"])

        df = clean.copy()

        # --- INDICATORS ---
        df["EMA_9"] = self.calculate_ema(df["Close"], 9)
        df["EMA_21"] = self.calculate_ema(df["Close"], 21)
        df["ATR"] = self.calculate_atr(df)
        df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100
        df["Volume_SMA20"] = df["Volume"].rolling(window=20).mean()
        df["RSI"] = self.calculate_rsi(df["Close"])

        latest = df.iloc[-1]

        # --- DAILY FILTERS ---
        volume_ok = latest['Volume'] >= self.min_daily_volume
        atr_ok = latest['ATR_Pct'] >= self.min_avg_true_range_pct

        price_above_ema9 = latest['Close'] > latest['EMA_9']
        price_above_ema21 = latest['Close'] > latest['EMA_21']
        ema_bullish = latest['EMA_9'] > latest['EMA_21']
        bullish_candle = latest['Close'] > latest['Open']
        volume_surge = latest['Volume'] > latest['Volume_SMA20'] * 1.2

        rsi = latest['RSI']
        not_overbought = rsi < 75
        not_oversold = rsi > 25

        bullish_score = sum([
            price_above_ema9,
            price_above_ema21,
            ema_bullish,
            bullish_candle,
            atr_ok,
            volume_surge
        ])

        bearish_score = sum([
            not price_above_ema9,
            not price_above_ema21,
            not ema_bullish,
            not bullish_candle,
            atr_ok,
            volume_surge
        ])

        # --- PATCH: LOWER THRESHOLD ---
        if bullish_score >= 3:
            daily_bias = 'BULLISH'
        elif bearish_score >= 3:
            daily_bias = 'BEARISH'
        else:
            daily_bias = 'NEUTRAL'

        # Do NOT reject tickers at daily stage.
        # Daily bias is used only for weighting intraday setups.


        return {
            'ticker': None,
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
    # 5-MINUTE CONTEXT  
    # ==========================================================================  

    def analyze_intraday_5min(self, df_5min: pd.DataFrame, daily_bias: str) -> Optional[Dict]:

        if df_5min is None or len(df_5min) < 30:
            return None

        if isinstance(df_5min.columns, pd.MultiIndex):
            df_5min.columns = [c[0] for c in df_5min.columns]

        df = df_5min.copy().sort_values("Date")

        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_21'] = self.calculate_ema(df['Close'], 21)
        df['VWAP'] = self.calculate_vwap(df)
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

        latest = df.iloc[-1]

        price_above_vwap = latest['Close'] > latest['VWAP']
        ema_aligned = latest['EMA_9'] > latest['EMA_21']
        volume_confirmed = latest['Volume'] > latest['Volume_SMA'] * 1.1

        day_high = df['High'].max()
        day_low = df['Low'].min()
        pre_market_high = df.iloc[:15]['High'].max() if len(df) > 15 else day_high
        pre_market_low = df.iloc[:15]['Low'].min() if len(df) > 15 else day_low

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
    # 1-MINUTE ENTRY TRIGGERS  
    # ==========================================================================  

    def find_entry_setups_1min(self, df_1min: pd.DataFrame,
                               context_5min: Dict,
                               daily_info: Dict) -> List[TradeSetup]:

        setups = []

        if df_1min is None or len(df_1min) < 30:
            return setups

        df = df_1min.copy().sort_values("Date")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

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

        if daily_bias in ['BULLISH', 'NEUTRAL'] and intraday_bias in [
            'BULLISH', 'CAUTIOUSLY_BULLISH', 'PULLBACK_MODE'
        ]:

            # VWAP Bounce
            if self._is_vwap_bounce(df, latest, prev):
                entry = price
                stop = min(df.iloc[-5:]['Low'].min(), vwap * 0.998)
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
                        setup_type='VWAP_BOUNCE',
                        confidence_score=confidence,
                        daily_trend=daily_bias,
                        volume_confirmed=latest['Volume'] > latest['Volume_SMA'],
                        key_level=round(vwap, 2)
                    ))

            # EMA Pullback
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

            # Breakout Retest
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

        if daily_bias in ['BEARISH', 'NEUTRAL'] and intraday_bias in [
            'BEARISH', 'CAUTIOUSLY_BEARISH'
        ]:

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
    # PATTERN DETECTORS  
    # ==========================================================================  

    def _is_vwap_bounce(self, df, latest, prev):
        vwap = latest['VWAP']
        price = latest['Close']

        if price < vwap * 0.998:
            return False

        if len(df) < 7:
            return False

        recent_df = df.iloc[-7:-1]
        touched_vwap = any(recent_df['Low'] <= vwap * 1.003)

        if not touched_vwap:
            return False

        bullish = latest['Close'] > latest['Open']
        lower_wick = min(latest['Open'], latest['Close']) - latest['Low']
        body = abs(latest['Close'] - latest['Open'])
        has_lower_wick = lower_wick > body * 0.2 if body > 0.01 else lower_wick > 0.01

        return bullish and has_lower_wick

    def _is_ema_pullback(self, df: pd.DataFrame, latest: pd.Series,
                         prev: pd.Series, ema9: float) -> bool:
        """
        Detect EMA pullback pattern for LONG entry.
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

        ema_series = df['EMA_9']
        ema_rising = ema_series.iloc[-1] > ema_series.iloc[-5] if len(ema_series) >= 5 else True

        bullish = latest['Close'] > latest['Open']

        return ema_rising and bullish

    def _is_breakout_retest(self, df: pd.DataFrame, latest: pd.Series,
                            key_levels: Dict) -> bool:
        """
        Detect breakout and retest pattern for LONG entry.
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

    def _is_vwap_rejection(self, df: pd.DataFrame, latest: pd.Series,
                           prev: pd.Series) -> bool:
        """
        Detect VWAP rejection pattern for SHORT entry.
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
                              volume_confirmed: bool,
                              candle_strength: float) -> float:
        """
        Calculate overall confidence score (0-100).
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

    def screen_ticker(self,
                      ticker: str,
                      daily_df: pd.DataFrame,
                      df_5min: pd.DataFrame,
                      df_1min: pd.DataFrame) -> List[TradeSetup]:
        """
        Full pipeline:
        1. Daily screen
        2. 5-min context
        3. 1-min entry setups
        """
        setups: List[TradeSetup] = []

        # Step 1: Daily
        daily_info = self.screen_daily(daily_df)
        if daily_info is None:
            return setups
        daily_info['ticker'] = ticker

        # Step 2: 5-min
        context_5min = self.analyze_intraday_5min(df_5min, daily_info['daily_bias'])
        if context_5min is None:
            return setups

        # Step 3: 1-min
        setups = self.find_entry_setups_1min(df_1min, context_5min, daily_info)

        return setups
