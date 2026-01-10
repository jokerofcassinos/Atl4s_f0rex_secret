
import pandas as pd
import numpy as np

class FVGTracker:
    """
    Fair Value Gaps (FVG) / Imbalance Tracker.
    
    Concept:
    Areas of inefficient price delivery where price moved too fast in one direction,
    leaving 'gaps' of unfilled liquidity. Market often returns to fill these.
    
    Structure:
    3-candle pattern.
    Bullish FVG: Gap between Candle 1 High and Candle 3 Low.
    Bearish FVG: Gap between Candle 1 Low and Candle 3 High.
    """

    def detect_fvg(self, df: pd.DataFrame) -> list:
        """
        Scans dataframe for active FVGs.
        Returns list of FVG objects.
        """
        gaps = []
        if df is None or len(df) < 3:
            return gaps
            
        # Scan last 50 candles
        start_idx = max(2, len(df) - 50)
        
        for i in range(start_idx, len(df)):
            candle_1 = df.iloc[i - 2]
            candle_2 = df.iloc[i - 1] # The Imbalance Candle
            candle_3 = df.iloc[i]
            
            # Bullish FVG
            # Candle 2 is green (usually).
            # Candle 3 Low > Candle 1 High
            if candle_3['low'] > candle_1['high']:
                gap_size = candle_3['low'] - candle_1['high']
                
                # Filter tiny gaps (noise) -> e.g. < 0.2 spread? 
                # For now keep raw.
                
                gaps.append({
                    'type': 'BULLISH',
                    'top': candle_3['low'],
                    'bottom': candle_1['high'],
                    'size': gap_size,
                    'time': df.index[i],
                    'filled': False
                })
            
            # Bearish FVG
            # Candle 2 is red.
            # Candle 3 High < Candle 1 Low
            elif candle_3['high'] < candle_1['low']:
                gap_size = candle_1['low'] - candle_3['high']
                
                gaps.append({
                    'type': 'BEARISH',
                    'top': candle_1['low'],
                    'bottom': candle_3['high'],
                    'size': gap_size,
                    'time': df.index[i],
                    'filled': False
                })
        
        return gaps
    
    def price_entering_fvg(self, fvgs: list, price: float, direction: str) -> bool:
        """
        Check if current price is 'filling' a FVG in the direction we want to trade.
        
        If direction='BUY', we want price to drop INTO a BULLISH FVG (Discount).
        If direction='SELL', we want price to rally INTO a BEARISH FVG (Premium).
        """
        for fvg in fvgs:
            if direction == 'BUY' and fvg['type'] == 'BULLISH':
                # Price is inside the gap range
                if fvg['bottom'] <= price <= fvg['top']:
                    return True
                    
            elif direction == 'SELL' and fvg['type'] == 'BEARISH':
                # Price is inside the gap range
                if fvg['bottom'] <= price <= fvg['top']:
                    return True
                    
        return False
