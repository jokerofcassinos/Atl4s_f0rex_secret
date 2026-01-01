import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """
    Standard Technical Indicators.
    """

    @staticmethod
    def rsi(series: pd.Series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(series: pd.Series, window=20, num_std=2):
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = mean + (std * num_std)
        lower = mean - (std * num_std)
        return upper, mean, lower

    @staticmethod
    def atr(high, low, close, window=14):
        h_l = high - low
        h_cp = (high - close.shift(1)).abs()
        l_cp = (low - close.shift(1)).abs()
        
        tr = pd.concat([h_l, h_cp, l_cp], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, span=20):
        return series.ewm(span=span, adjust=False).mean()
