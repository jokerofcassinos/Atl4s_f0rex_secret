import pandas as pd
import numpy as np

class PriceAction:
    """
    Smart Money Concepts & Price Action Analysis.
    """

    @staticmethod
    def detect_fvg(df: pd.DataFrame):
        """
        Identifies Fair Value Gaps (Imbalances).
        Returns Series with 'BullishFVG', 'BearishFVG' or NaN.
        """
        fvg = pd.Series(index=df.index, data=np.nan)
        
        # Bullish FVG: Low[-1] > High[-3] (Indices: current=i, previous=i-1, 2-back=i-2)
        # But in DataFrame, we use shift.
        # i is current candle being analyzed. The FVG is formed by the gap between i's High and i+2's Low?
        # NO. FVG is a 3-candle pattern.
        # Candle 1: Big move start.
        # Candle 2: Big move body.
        # Candle 3: Move continues or pulls back.
        # Gap is between Candle 1 High and Candle 3 Low (Bullish)?? 
        # Actually:
        # Bullish FVG: Gap between High of Candle (i-2) and Low of Candle (i).
        # Bearish FVG: Gap between Low of Candle (i-2) and High of Candle (i).
        
        high_shifted = df['high'].shift(2)
        low_shifted = df['low'].shift(2)
        
        # Bullish FVG condition: Low of current > High of 2 candles ago
        bull_cond = df['low'] > high_shifted
        # Bearish FVG condition: High of current < Low of 2 candles ago
        bear_cond = df['high'] < low_shifted
        
        fvg[bull_cond] = "Bullish"
        fvg[bear_cond] = "Bearish"
        
        return fvg

    @staticmethod
    def identify_structure(df: pd.DataFrame, window=5):
        """
        Identifies Local Highs and Lows (Support/Resistance).
        """
        # Fractal/Pivot High/Low
        # High is highest in window centered at i
        # We need to look ahead, so this is lagging for real-time, 
        # but valid for historic structure.
        
        # For real-time, we just look back.
        roll_max = df['high'].rolling(window=window).max()
        roll_min = df['low'].rolling(window=window).min()
        
        is_high = df['high'] == roll_max
        is_low = df['low'] == roll_min
        
        return is_high, is_low
