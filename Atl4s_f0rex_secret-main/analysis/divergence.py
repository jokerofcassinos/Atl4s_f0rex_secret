import pandas as pd
import logging
import numpy as np

logger = logging.getLogger("Atl4s-Divergence")

class DivergenceHunter:
    def __init__(self):
        self.rsi_period = 14

    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, df):
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def calculate_stoch(self, df, period=14):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        return k

    def analyze(self, df):
        """
        Detects Triple Divergence (RSI, MACD, Stoch).
        """
        if df is None or len(df) < 50:
            return 0, 0, "None"
            
        df = df.copy()
        
        # Calculate Indicators
        if 'RSI_14' not in df.columns:
            df['RSI_14'] = self.calculate_rsi(df['close'], self.rsi_period)
            
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df)
        df['Stoch_K'] = self.calculate_stoch(df)
        
        # Lookback for peaks/troughs
        lookback = 5 
        
        # Find Swing Points (Last 2)
        lows = []
        highs = []
        
        for i in range(len(df)-lookback-1, len(df)-30, -1):
            # Swing Low
            if df['low'].iloc[i] < df['low'].iloc[i-1] and \
               df['low'].iloc[i] < df['low'].iloc[i+1] and \
               df['low'].iloc[i] < df['low'].iloc[i-2] and \
               df['low'].iloc[i] < df['low'].iloc[i+2]:
                lows.append(i)
                if len(lows) >= 2: break
                
            # Swing High
            if df['high'].iloc[i] > df['high'].iloc[i-1] and \
               df['high'].iloc[i] > df['high'].iloc[i+1] and \
               df['high'].iloc[i] > df['high'].iloc[i-2] and \
               df['high'].iloc[i] > df['high'].iloc[i+2]:
                highs.append(i)
                if len(highs) >= 2: break
                
        score = 0
        direction = 0
        div_type = "None"
        
        # --- Bullish Divergence ---
        if len(lows) == 2:
            curr, prev = lows[0], lows[1]
            price_lower = df['low'].iloc[curr] < df['low'].iloc[prev]
            
            # RSI Bullish
            rsi_div = df['RSI_14'].iloc[curr] > df['RSI_14'].iloc[prev]
            
            # MACD Bullish (Histogram or Line)
            macd_curr = df['MACD'].iloc[curr]
            macd_prev = df['MACD'].iloc[prev]
            macd_div = macd_curr > macd_prev
            
            # Stoch Bullish
            stoch_curr = df['Stoch_K'].iloc[curr]
            stoch_prev = df['Stoch_K'].iloc[prev]
            stoch_div = stoch_curr > stoch_prev
            
            if price_lower:
                div_count = sum([rsi_div, macd_div, stoch_div])
                
                if div_count >= 1:
                    score = 60 + (div_count * 10) # 70, 80, 90
                    direction = 1
                    div_type = f"BULL_DIV_x{div_count}"
                    if div_count == 3:
                        logger.info("TRIPLE BULLISH DIVERGENCE DETECTED!")
                    else:
                        logger.info(f"Bullish Divergence Detected (Confluence: {div_count})")

        # --- Bearish Divergence ---
        if len(highs) == 2:
            curr, prev = highs[0], highs[1]
            price_higher = df['high'].iloc[curr] > df['high'].iloc[prev]
            
            # RSI Bearish
            rsi_div = df['RSI_14'].iloc[curr] < df['RSI_14'].iloc[prev]
            
            # MACD Bearish
            macd_curr = df['MACD'].iloc[curr]
            macd_prev = df['MACD'].iloc[prev]
            macd_div = macd_curr < macd_prev
            
            # Stoch Bearish
            stoch_curr = df['Stoch_K'].iloc[curr]
            stoch_prev = df['Stoch_K'].iloc[prev]
            stoch_div = stoch_curr < stoch_prev
            
            if price_higher:
                div_count = sum([rsi_div, macd_div, stoch_div])
                
                if div_count >= 1:
                    score = 60 + (div_count * 10)
                    direction = -1
                    div_type = f"BEAR_DIV_x{div_count}"
                    if div_count == 3:
                        logger.info("TRIPLE BEARISH DIVERGENCE DETECTED!")
                    else:
                        logger.info(f"Bearish Divergence Detected (Confluence: {div_count})")
                
        return score, direction, div_type
