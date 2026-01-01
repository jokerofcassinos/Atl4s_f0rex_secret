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

        # Identify Fair Value Gaps (FVG)
        # Bullish FVG: High[i-2] < Low[i] (Gap between candle i-2 and i)
        # Bearish FVG: Low[i-2] > High[i]
        
        # We look at the last completed 3 candles (excluding current forming candle if live)
        # Assuming df includes current candle at -1, we look at -2, -3, -4
        
        c1 = df.iloc[-4]
        c2 = df.iloc[-3]
        c3 = df.iloc[-2] # Most recent completed candle
        
        score = 0
        direction = 0
        
        # Helper to get float
        def get_val(row, col):
            val = row[col]
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            return float(val)

        c1_high = get_val(c1, 'high')
        c1_low = get_val(c1, 'low')
        c3_high = get_val(c3, 'high')
        c3_low = get_val(c3, 'low')
        
        # Bullish FVG
        if c1_high < c3_low:
            # Gap exists
            fvg_size = c3_low - c1_high
            if fvg_size > 0:
                score += 60 # Increased to 60 for stronger signal
                direction = 1
                logger.info(f"Bullish FVG detected. Size: {fvg_size}")

        # Bearish FVG
        elif c1_low > c3_high:
            # Gap exists
            fvg_size = c1_low - c3_high
            if fvg_size > 0:
                score += 60 # Increased to 60
                direction = -1
                logger.info(f"Bearish FVG detected. Size: {fvg_size}")
        
        # Swing Points (Liquidity)
        # Check if we just swept a low/high
        # Simplified Fractal: Low[-3] was a local low, and Low[-2] broke it?
        # Or Price is reacting to a previous swing.
        
        # For now, let's stick to FVG as the primary "Sniper" signal for entry
        
        return score, direction
