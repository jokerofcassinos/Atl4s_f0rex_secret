
import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal
from scipy.signal import argrelextrema

logger = logging.getLogger("LiquidityMapSwarm")

class LiquidityMapSwarm(SubconsciousUnit):
    """
    The Hunter.
    Maps Stop Loss Clusters and trades the 'Stop Hunt' (SFP).
    """
    def __init__(self):
        super().__init__("Liquidity_Map_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 100: return None
        
        # 1. Identify Swing Points (Local Min/Max)
        n = 5 # 5 candles left/right
        df_m5['min'] = df_m5['low'].iloc[argrelextrema(df_m5['low'].values, np.less_equal, order=n)[0]]
        df_m5['max'] = df_m5['high'].iloc[argrelextrema(df_m5['high'].values, np.greater_equal, order=n)[0]]
        
        # Recent Swings
        recent_lows = df_m5['min'].dropna().tail(5).values
        recent_highs = df_m5['max'].dropna().tail(5).values
        
        last = df_m5.iloc[-1]
        current_price = last['close']
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        # 2. Detect "Swing Failure Pattern" (SFP)
        # Price pierces a swing low, then closes ABOVE it.
        # This indicates a Stop Hunt (Liquidity Grab) + Rejection.
        
        for low in recent_lows:
            if last['low'] < low: # Pierced
                if last['close'] > low: # Reclaimed
                    # Validation: How deep was the pierce?
                    pierce_dist = low - last['low']
                    if pierce_dist > 0:
                         signal = "BUY"
                         confidence = 90
                         reason = f"Liquidity Grab (SFP) at {low}"
                         
        for high in recent_highs:
            if last['high'] > high: # Pierced
                if last['close'] < high: # Reclaimed
                     signal = "SELL"
                     confidence = 90
                     reason = f"Liquidity Grab (SFP) at {high}"
                     
        # 3. Detect "Pools" (Equal Highs/Lows) - Inducement
        # If we see double bottoms, market is ATTRACTED to them aka "Magnet"
        # Ideally, we predict the run TO the pool.
        
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'level_swept': 0}
            )
            
        return None
