import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Atl4s-TechnicalLibrary")

class TechnicalLibrary:
    """
    The Codex of Technical Analysis.
    Contains mathematical implementations of all major technical indicators.
    """
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> float:
        try:
            # Optimization: Use only recent data. SMA RSI doesn't need infinite history.
            if len(series) > 500:
                series = series.iloc[-500:]
            
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except: return 50.0

    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
        try:
            if len(series) > 500:
                series = series.iloc[-500:]
                
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
        except: return 0, 0, 0

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        try:
            # EMA converges after ~3.45 * span. 500 is safe for slow=26.
            if len(series) > 500:
                series = series.iloc[-500:]
                
            exp1 = series.ewm(span=fast, adjust=False).mean()
            exp2 = series.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        except: return 0, 0, 0

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) > 500:
                df = df.iloc[-500:]
                
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.iloc[-1]
        except: return 0.0

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> float:
        try:
            # VWAP is cumulative from session start. Backtest assumes continuous?
            # Safe to assume simplified VWAP on recent window for local trend context
            # Real VWAP resets daily. Here we just take last 1000 candles as proxy if optimized?
            # Or keep full if strictly necessary? Let's limit to 1000 for speed.
            if len(df) > 1000:
                df = df.iloc[-1000:]
                
            # Typical Price * Volume
            v = df['volume']
            tp = (df['high'] + df['low'] + df['close']) / 3
            vwap = (tp * v).cumsum() / v.cumsum()
            return vwap.iloc[-1]
        except: return 0.0

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
        try:
            low_min = df['low'].rolling(window=k).min()
            high_max = df['high'].rolling(window=k).max()
            k_line = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
            d_line = k_line.rolling(window=d).mean()
            return k_line.iloc[-1], d_line.iloc[-1]
        except: return 50.0, 50.0

    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame):
        # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
        try:
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            tenkan_sen = (high_9 + low_9) / 2

            # Base Line (Kijun-sen): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            kijun_sen = (high_26 + low_26) / 2

            # Lead 1 (Senkou A): (Conversion + Base)/2
            senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            # Lead 2 (Senkou B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            senkou_b = ((high_52 + low_52) / 2).shift(26)
            
            return tenkan_sen.iloc[-1], kijun_sen.iloc[-1], senkou_a.iloc[-1], senkou_b.iloc[-1]
        except: return 0,0,0,0

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14):
        try:
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            plus_di = 100 * (plus_dm.ewm(alpha = 1/period).mean() / atr)
            minus_di = abs(100 * (minus_dm.ewm(alpha = 1/period).mean() / atr))
            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = dx.ewm(alpha = 1/period).mean()
            return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
        except: return 0, 0, 0

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20):
        try:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: pd.Series(x).mad())
            cci = (tp - sma) / (0.015 * mad)
            return cci.iloc[-1]
        except: return 0.0

    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14):
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            wr = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-9)
            return wr.iloc[-1]
        except: return -50.0

    @staticmethod
    def calculate_obv(df: pd.DataFrame):
        try:
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            return obv.iloc[-1]
        except: return 0.0

    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period=10, multiplier=3):
        # Simplified one-step calculation
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # ATR
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            hl2 = (high + low) / 2
            basic_upper = hl2 + (multiplier * atr)
            basic_lower = hl2 - (multiplier * atr)
            
            # Trend determination requires iteration, simplified here for "instant" check
            # Just check if price is above/below the 'basic' band relative to typical price?
            # A full iterative SuperTrend is expensive in Python loops.
            # We return Basic Upper/Lower and Price relation.
            
            return basic_upper.iloc[-1], basic_lower.iloc[-1], atr.iloc[-1]
        except: return 0,0,0
