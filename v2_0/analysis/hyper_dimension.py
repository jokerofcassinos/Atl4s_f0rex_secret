import pandas as pd
import numpy as np

class HyperDimension:
    def __init__(self):
        pass
        
    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def calculate_rsi(self, df, window=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def analyze_reality(self, df):
        """
        The 'Third Eye': Sees what is hidden.
        Combines standard tools in a complex way.
        """
        if df is None or len(df) < 50: return 0, "BLIND"
        
        upper, lower = self.calculate_bollinger_bands(df)
        rsi = self.calculate_rsi(df)
        
        current_price = df.iloc[-1]['close']
        current_rsi = rsi.iloc[-1]
        
        # Dimensions Check
        # 1. Expansion: Is price outside bands?
        outside_upper = current_price > upper.iloc[-1]
        outside_lower = current_price < lower.iloc[-1]
        
        # 2. Momentum: Is RSI extreme?
        overbought = current_rsi > 70
        oversold = current_rsi < 30
        
        # 3. Manipulation Detection (Wicks)
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        wick_upper = last_candle['high'] - max(last_candle['close'], last_candle['open'])
        wick_lower = min(last_candle['close'], last_candle['open']) - last_candle['low']
        
        manipulation = False
        if wick_upper > 2 * body: manipulation = True # Shooting starish
        if wick_lower > 2 * body: manipulation = True # Hammerish
        
        score = 0
        state = "NEUTRAL"
        
        # Complex Logic
        if outside_upper and overbought:
            if manipulation:
                score = -90 # Strong Reversal Sell
                state = "DIMENSIONAL_SELL_REVERSAL"
            else:
                score = 10 # Strong Trend Buy (Walking the band)
                state = "DIMENSIONAL_TREND_BUY"
        elif outside_lower and oversold:
            if manipulation:
                score = 90
                state = "DIMENSIONAL_BUY_REVERSAL"
            else:
                score = -10
                state = "DIMENSIONAL_TREND_SELL"
                
        return score, state
