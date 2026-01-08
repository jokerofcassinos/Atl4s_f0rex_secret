import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger("Atl4s-SMCEngine")

class SmartMoneyEngine:
    """
    The Institutional Algo Reverse Engineer.
    Detects:
    1. Fair Value Gaps (FVG) / Imbalances (Inversion FVG).
    2. Liquidity Grabs (Sweeps of recent highs/lows).
    3. Session Levels (Asian Range, London Open).
    """
    
    def detect_fvgs(self, df: pd.DataFrame, window=30) -> List[Dict]:
        """
        Scans for FVGs in the last N candles.
        Bullish FVG: Low of candle i-2 > High of candle i
        Bearish FVG: High of candle i-2 < Low of candle i
        """
        if len(df) < 5: return []
        
        fvgs = []
        df_slice = df.iloc[-window:]
        
        # Dynamic gap threshold (0.02% of price = ~3 pips for Forex)
        current_price = df_slice['close'].iloc[-1]
        min_gap = current_price * 0.0002  # 0.02% = ~3 pips for 1.34, ~0.03 for 156.x JPY
        
        for i in range(2, len(df_slice)):
            idx = df_slice.index[i]
            prev_idx = df_slice.index[i-2]
            
            # Candle 0 (Current frame of ref i), Candle 1 (i-1), Candle 2 (i-2)
            # Actually indices are i (current), i-1, i-2
            
            c0_high = df_slice['high'].iloc[i]
            c0_low = df_slice['low'].iloc[i]
            
            c2_high = df_slice['high'].iloc[i-2]
            c2_low = df_slice['low'].iloc[i-2]
            
            # BEARISH FVG (Gap Down)
            # Low of candle i-2 is unmatched by High of candle i
            if c2_low > c0_high:
                gap_size = c2_low - c0_high
                if gap_size > min_gap:
                    fvgs.append({
                        'type': 'BEAR_FVG',
                        'top': c2_low,
                        'bottom': c0_high,
                        'time': int(df_slice.index[i-2].timestamp()), # Start time
                        'end_time': int(df_slice.index[i].timestamp())
                    })
                    
            # BULLISH FVG (Gap Up)
            # High of candle i-2 is unmatched by Low of candle i
            elif c2_high < c0_low:
                gap_size = c0_low - c2_high
                if gap_size > min_gap:
                    fvgs.append({
                        'type': 'BULL_FVG',
                        'top': c0_low,
                        'bottom': c2_high,
                        'time': int(df_slice.index[i-2].timestamp()),
                        'end_time': int(df_slice.index[i].timestamp())
                    })
                    
        # Filter for only ACTIVE FVGs - return last 3 only
        return fvgs[-3:]

    def detect_liquidity_grabs(self, df: pd.DataFrame, lookback=20) -> List[Dict]:
        """
        Detects sweeps: Price breaks a recent Swing High/Low but closes back inside range (Wick).
        """
        if len(df) < lookback+2: return []
        
        sweeps = []
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Find recent Swing Highs/Lows
        window = df.iloc[-lookback:-2] # Exclude current and prev
        recent_high = window['high'].max()
        recent_low = window['low'].min()
        
        # Check for Sweep of High (Bearish Sweep)
        # High > Recent High, but Close < Recent High
        if current['high'] > recent_high and current['close'] < recent_high:
             sweeps.append({
                 'type': 'SWEEP_HIGH',
                 'level': recent_high,
                 'time': int(df.index[-1].timestamp())
             })
             
        # Check for Sweep of Low (Bullish Sweep)
        # Low < Recent Low, but Close > Recent Low
        if current['low'] < recent_low and current['close'] > recent_low:
             sweeps.append({
                 'type': 'SWEEP_LOW',
                 'level': recent_low,
                 'time': int(df.index[-1].timestamp())
             })
             
        return sweeps

    def get_session_levels(self, df: pd.DataFrame) -> Dict:
        """
        Returns key levels based on time.
        For now simple High/Low of the last N candles as a proxy for 'Session'.
        """
        day_high = df['high'].iloc[-200:].max()
        day_low = df['low'].iloc[-200:].min()
        return {'day_high': day_high, 'day_low': day_low}
