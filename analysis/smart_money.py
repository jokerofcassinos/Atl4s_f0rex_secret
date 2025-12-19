import pandas as pd
import numpy as np

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
