import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Atl4s-Sniper")

class Sniper:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Analyzes Market Structure (FVG, Liquidity).
        Returns:
            score (int): Confidence score (0-100)
            direction (int): 1 (Buy), -1 (Sell), 0 (Neutral)
        """
        if df is None or len(df) < 5:
            return 0, 0

        score = 0
        direction = 0

        # Identify Fair Value Gaps (FVG)
        # Bullish FVG: High[i-2] < Low[i] (Gap between candle i-2 and i)
        # Bearish FVG: Low[i-2] > High[i]
        
        # We look at the last completed 3 candles (excluding current forming candle if live)
        # Assuming df includes current candle at -1, we look at -2, -3, -4
        
        # Scan last 20 candles for FVG zones
        lookback = 20
        start_idx = max(0, len(df) - lookback)
        
        fvgs = []
        
        for i in range(start_idx + 2, len(df)):
             # Bullish FVG: Low[i] > High[i-2]
             if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                 gap = df.iloc[i]['low'] - df.iloc[i-2]['high']
                 if gap > 0:
                     fvgs.append({'type': 1, 'top': df.iloc[i]['low'], 'bottom': df.iloc[i-2]['high'], 'idx': i})
                     
             # Bearish FVG: High[i] < Low[i-2]
             elif df.iloc[i]['high'] < df.iloc[i-2]['low']:
                 gap = df.iloc[i-2]['low'] - df.iloc[i]['high']
                 if gap > 0:
                     fvgs.append({'type': -1, 'top': df.iloc[i-2]['low'], 'bottom': df.iloc[i]['high'], 'idx': i})

        current_price = df.iloc[-1]['close']
        
        # Evaluate Active FVGs
        for fvg in fvgs:
            # Check if price is arguably testing this FVG (or inside it)
            # Bullish FVG (Support)
            if fvg['type'] == 1:
                # If price is near/in the gap
                if fvg['bottom'] <= current_price <= fvg['top'] * 1.05:
                    score += 40
                    direction = 1
                    logger.info(f"Sniper: Price in Bullish FVG Zone ({fvg['bottom']:.2f}-{fvg['top']:.2f})")
            
            # Bearish FVG (Resistance)
            if fvg['type'] == -1:
                if fvg['bottom'] * 0.95 <= current_price <= fvg['top']:
                    score += 40
                    direction = -1
                    logger.info(f"Sniper: Price in Bearish FVG Zone ({fvg['bottom']:.2f}-{fvg['top']:.2f})")
                    
        # If score > 0, we found something.
        # Cap score
        if score > 0:
             if score > 100: score = 100
             return score, direction

        # --- Liquidity Sweep (Fallback) ---
        # Did we just grab liquidity (Tortue Soup)?
        # Look back 10 candles for a high/low
        subset = df.iloc[-15:-2] 
        recent_low = subset['low'].min()
        recent_high = subset['high'].max()
        
        last_low = df.iloc[-1]['low']
        last_high = df.iloc[-1]['high']
        last_close = df.iloc[-1]['close']
        
        # Bullish Sweep: Price went below recent low but closed above it
        if last_low < recent_low and last_close > recent_low:
             return 30, 1
             
        # Bearish Sweep: Price went above recent high but closed below it
        if last_high > recent_high and last_close < recent_high:
             return 30, -1

        return 0, 0
