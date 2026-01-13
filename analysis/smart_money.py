import pandas as pd
import numpy as np
from datetime import datetime, time

class SmartMoneyEngine:
    def __init__(self):
        self.fvgs = []
        self.order_blocks = [] # List of {price, type, strength, time}

    def detect_fvg(self, df):
        """
        Detects Fair Value Gaps with strict displacement criteria.
        Bullish FVG: High of candle i-2 < Low of candle i.
        Bearish FVG: Low of candle i-2 > High of candle i.
        """
        fvgs = []
        if len(df) < 5: return []

        # Analyze last 20 candles for efficiency
        lookback = min(len(df), 50)
        start_idx = len(df) - lookback
        
        for i in range(start_idx + 2, len(df)):
            # ATR check for 'displacement' (simplified as large range)
            range_i = abs(df.iloc[i]['high'] - df.iloc[i]['low'])
            avg_range = abs(df.iloc[i-5:i]['high'] - df.iloc[i-5:i]['low']).mean() if i > 5 else range_i
            
            is_impulsive = range_i > (avg_range * 1.2) # 20% larger than average

            # Bullish FVG
            if df.iloc[i-2]['high'] < df.iloc[i]['low']:
                gap_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
                if gap_size > 0: # and is_impulsive:
                    fvgs.append({
                        'type': 'BULLISH',
                        'top': df.iloc[i]['low'],
                        'bottom': df.iloc[i-2]['high'],
                        'index': df.index[i],
                        'strength': gap_size
                    })
            # Bearish FVG
            elif df.iloc[i-2]['low'] > df.iloc[i]['high']:
                gap_size = df.iloc[i-2]['low'] - df.iloc[i]['high']
                if gap_size > 0: # and is_impulsive:
                    fvgs.append({
                        'type': 'BEARISH',
                        'top': df.iloc[i-2]['low'],
                        'bottom': df.iloc[i]['high'],
                        'index': df.index[i],
                        'strength': gap_size
                    })
        self.fvgs = fvgs
        return fvgs

    def detect_order_blocks(self, df):
        """
        Refined Order Block Detection.
        Bullish OB: The last bearish candle before a bullish Imbalance (FVG) and Break of Structure (BOS).
        Bearish OB: The last bullish candle before a bearish Imbalance.
        """
        obs = []
        if len(df) < 10: return []
        
        # We need recent structure
        # Simplified logic: Look for pivots followed by strong moves
        
        for i in range(len(df) - 20, len(df) - 3):
            # Bullish OB Check:
            # Candle i is Bearish (Close < Open)
            is_bearish_candle = df.iloc[i]['close'] < df.iloc[i]['open']
            
            if is_bearish_candle:
                # Next candles must be bullish and impulsive
                next_close = df.iloc[i+1]['close']
                curr_high = df.iloc[i]['high']
                
                # Did price break the high of the bearish candle aggressively?
                if next_close > curr_high:
                     # Calculate body size to filter dojis
                    body = abs(df.iloc[i]['open'] - df.iloc[i]['close'])
                    if body > (df.iloc[i]['high'] - df.iloc[i]['low']) * 0.3: # Solid body
                        obs.append({
                            'type': 'BULLISH',
                            'top': df.iloc[i]['high'],
                            'bottom': df.iloc[i]['low'], # Wicks included for OB zone
                            'index': df.index[i]
                        })

            # Bearish OB Check:
            # Candle i is Bullish
            is_bullish_candle = df.iloc[i]['close'] > df.iloc[i]['open']
            
            if is_bullish_candle:
                # Next candles must be bearish
                next_close = df.iloc[i+1]['close']
                curr_low = df.iloc[i]['low']
                
                if next_close < curr_low:
                    body = abs(df.iloc[i]['open'] - df.iloc[i]['close'])
                    if body > (df.iloc[i]['high'] - df.iloc[i]['low']) * 0.3:
                        obs.append({
                            'type': 'BEARISH',
                            'top': df.iloc[i]['high'],
                            'bottom': df.iloc[i]['low'],
                            'index': df.index[i]
                        })

        self.order_blocks = obs
        return obs

    def analyze(self, df):
        if df is None or df.empty: return 0

        self.detect_fvg(df)
        self.detect_order_blocks(df)
        
        current_price = df.iloc[-1]['close']
        score = 0
        
        # 1. Evaluate FVGs (Magnets or Support)
        # Weight recent FVGs higher
        for fvg in self.fvgs[-3:]:
            midpoint = (fvg['top'] + fvg['bottom']) / 2
            
            # Distance factor
            dist = abs(current_price - midpoint)
            # Normalizing distance roughly...
            
            if fvg['bottom'] <= current_price <= fvg['top']:
                # Price is INSIDE the gap. It is balancing efficiently.
                # If we are effectively reacting?
                if fvg['type'] == 'BULLISH':
                    score += 15 # Support Zone
                else:
                    score -= 15 # Resistance Zone
            elif current_price > fvg['top'] and fvg['type'] == 'BULLISH':
                 # Above a bullish FVG = Strong Support below
                 score += 5

        # 2. Evaluate Order Blocks
        for ob in self.order_blocks[-3:]:
            if ob['bottom'] <= current_price <= ob['top']:
                # Mitigating OB
                if ob['type'] == 'BULLISH':
                    score += 25 # Strong Buy Zone
                else:
                    score -= 25 # Strong Sell Zone
                    
        return max(min(score, 100), -100) # Clamp

class Liquidator:
    """
    The 'Liquidator' Strategy Engine.
    Detects Session Sweeps (Asia/London) and SMT Divergences.
    """
    def __init__(self):
        # Define Session Times (Broker Server Time assumed UTC+2/3 approx)
        # Adjust these based on actual data feed if necessary
        self.asia_start = time(0, 0)
        self.asia_end = time(8, 0) # Frankfurt Open / London Pre-market
        self.london_start = time(8, 0)
        self.london_end = time(16, 0) # NY overlap end
        
        self.asia_high = None
        self.asia_low = None
        self.london_high = None
        self.london_low = None
        self.last_day = None

    def update_session_levels(self, df_h1):
        """
        Scans H1 data to find today's session High/Low.
        """
        if df_h1 is None or df_h1.empty: return

        # Get today's candles
        today = df_h1.index[-1].date()
        df_today = df_h1[df_h1.index.date == today]
        
        if df_today.empty: return

        # Filter for Asia (00:00 - 08:00)
        asia_candles = df_today.between_time(self.asia_start, self.asia_end)
        if not asia_candles.empty:
            self.asia_high = asia_candles['high'].max()
            self.asia_low = asia_candles['low'].min()
            
        # Filter for London (08:00 - 16:00)
        london_candles = df_today.between_time(self.london_start, self.london_end)
        if not london_candles.empty:
            self.london_high = london_candles['high'].max()
            self.london_low = london_candles['low'].min()

    def check_sweep(self, df_m5, current_price):
        """
        Checks if we have just swept a key level and are reversing.
        Returns: 'BUY_ASIA_SWEEP', 'SELL_ASIA_SWEEP', 'BUY_LONDON_SWEEP', etc.
        """
        if not self.asia_high: return None # No levels yet
        
        signal = None
        
        # 1. Asia Low Sweep (Bullish)
        # Price went below Asia Low, but current Close is above it?
        # Or look at last few candles
        
        # Strict definition: Wick below, Close above (Hammer/Pinbar logic on M5)
        last_candle = df_m5.iloc[-1]
        prev_candle = df_m5.iloc[-2]
        
        # Check current candle sweep
        if last_candle['low'] < self.asia_low and last_candle['close'] > self.asia_low:
             signal = "BUY_ASIA_SWEEP"
             
        # Check prev candle sweep (confirmation candle)
        elif prev_candle['low'] < self.asia_low and prev_candle['close'] > self.asia_low and last_candle['close'] > prev_candle['high']:
             signal = "BUY_ASIA_SWEEP_CONFIRMED"
             
        # 2. Asia High Sweep (Bearish)
        elif last_candle['high'] > self.asia_high and last_candle['close'] < self.asia_high:
             signal = "SELL_ASIA_SWEEP"
             
        # 3. London Low Sweep (Bullish)
        if self.london_low:
             if last_candle['low'] < self.london_low and last_candle['close'] > self.london_low:
                  signal = "BUY_LONDON_SWEEP"
        
        # 4. London High Sweep (Bearish)
        if self.london_high:
             if last_candle['high'] > self.london_high and last_candle['close'] < self.london_high:
                  signal = "SELL_LONDON_SWEEP"
                  
        return signal
