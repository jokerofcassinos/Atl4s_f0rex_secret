
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("RangeScanner")

class RangeScanner:
    """
    Big Beluga Phase 17: Range Scanner (The Box).
    Identifies Lateral Markets (Consolidation) vs Trending Markets.
    Enables 'Ping Pong' Strategy: Buy Low, Sell High.
    """
    def __init__(self, period: int = 50):
        self.period = period
        self.last_state = "UNKNOWN"
        self.range_high = 0.0
        self.range_low = 0.0
        self.mid_point = 0.0
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Scans for box formation.
        Args:
            df: DataFrame with 'high', 'low', 'close' columns.
        Returns:
            Dict: {
                'status': 'RANGING' | 'TRENDING' | 'BREAKOUT',
                'range_high': float,
                'range_low': float,
                'strength': float (0-1),
                'proximity': 'HIGH' | 'LOW' | 'MID' | 'NONE'
            }
        """
        if df is None or len(df) < self.period:
            return {'status': 'UNKNOWN', 'range_high': 0, 'range_low': 0, 'strength': 0.0}

        # 1. Define Analysis Window
        window = df.tail(self.period)
        
        # 2. Find Swing Points (Max/Min)
        recent_high = window['high'].max()
        recent_low = window['low'].min()
        current_close = window['close'].iloc[-1]
        
        # 3. Calculate Volatility (Standard Deviation)
        std_dev = window['close'].std()
        avg_price = window['close'].mean()
        
        # Normalized Volatility (CV)
        vol_ratio = (std_dev / avg_price) * 10000 
        
        # 4. Determine Range Validity
        # If High - Low is huge, it might be a trend, not a range.
        height = recent_high - recent_low
        
        # Heuristic: Range is valid if price spends time inside the box without breaking.
        # Check touch counts
        buffer = height * 0.05 # 5% buffer
        touches_high = len(window[window['high'] > (recent_high - buffer)])
        touches_low = len(window[window['low'] < (recent_low + buffer)])
        
        # Range Strength Score 
        # More touches = Stronger Range. 
        # Low Volatility = Better Range.
        range_score = 0.5
        
        if touches_high >= 2 and touches_low >= 2:
            range_score += 0.3
            
        if vol_ratio < 10.0: # Very tight
            range_score += 0.2
            
        # 5. Determine State
        status = "TRENDING"
        if range_score >= 0.7:
             status = "RANGING"
             
        # Detect Breakout
        # If current close is OUTSIDE the box?
        # Actually, recent_high IS the max, so it can't be outside unless we look at *previous* box.
        # For simple ping pong, we assume current window IS the box.
        
        self.range_high = recent_high
        self.range_low = recent_low
        self.mid_point = (recent_high + recent_low) / 2
        
        # 6. Proximity (Where are we in the box?)
        proximity = "MID"
        dist_to_high = abs(recent_high - current_close)
        dist_to_low = abs(current_close - recent_low)
        
        # If logic says we are RANGING:
        if status == "RANGING":
            if dist_to_high < (height * 0.20): # Top 20%
                proximity = "RANGE_HIGH" # SELL ZONE
            elif dist_to_low < (height * 0.20): # Bottom 20%
                proximity = "RANGE_LOW" # BUY ZONE
        
        return {
            'status': status,
            'range_high': recent_high,
            'range_low': recent_low,
            'strength': range_score,
            'proximity': proximity,
            'height': height
        }
