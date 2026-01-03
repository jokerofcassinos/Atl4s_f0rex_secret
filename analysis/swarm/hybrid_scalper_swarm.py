
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("HybridScalperSwarm")

class HybridScalperSwarm(SubconsciousUnit):
    """
    The Unified Field (New + Old).
    Integrates the 'Field Theory' of the legacy ScalperSwarm into the asynchronous Swarm logic.
    Calculates:
    - Particle Velocity (v)
    - Field Pressure (P)
    - Strange Attractor (A)
    """
    def __init__(self):
        super().__init__("HybridScalperSwarm")
        self.last_candle_time = None
        self.threshold = 0.4 # Slightly more sensitive than old 0.5

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        df_m5 = context.get('df_m5')
        tick_data = context.get('tick')

        if df_m1 is None or len(df_m1) < 20: return None

        # --- EXTRACTING THE UNIFIED FIELD COMPONENTS ---
        
        # 1. Particle Velocity (v) - Slope of M1 Close (Recent)
        # Using simple regression slope of last 5 candles
        last_5 = df_m1['close'].tail(5).values
        if len(last_5) == 5:
            x = np.arange(5)
            slope, _ = np.polyfit(x, last_5, 1)
            # Normalize slope relative to price (roughly)
            # A slope of 1.0 on BTC (3000) is tiny. A slope of 1.0 on EURUSD (1.05) is huge.
            # Percent slope is better.
            pct_slope = slope / last_5[-1] * 10000 
            # Sigmoid normalization (-1 to 1)
            v = 2 / (1 + np.exp(-pct_slope)) - 1 
        else:
            v = 0.0

        # 2. Field Pressure (P) - Order Flow Imbalance (Simulated via Volume/Candle Delta)
        # Green Candle Volume vs Red Candle Volume pressure
        last_candle = df_m1.iloc[-1]
        delta = last_candle['close'] - last_candle['open']
        rng = last_candle['high'] - last_candle['low']
        
        p_raw = 0
        if rng > 0:
            # Volume Delta Proxy: Volume * (Close-Open)/Range
            # If Close=High, Delta = Volume. If Close=Open, Delta=0.
            p_raw = last_candle['tick_volume'] * (delta / rng)
            
        # Normalize P (assuming 1000 volume is 'high' for M1)
        P = np.clip(p_raw / 500.0, -1.0, 1.0)

        # 3. Strange Attractor (A) - Mean Reversion to M5 MA
        # Does the "Rubber Band" pull back?
        if not df_m5.empty:
            ma20 = df_m5['close'].rolling(20).mean().iloc[-1]
            price = tick_data['last']
            dist = (price - ma20) / ma20 * 100 # Percents
            
            # If dist is > 0.5%, strong pull back (Attractor)
            # If dist is positive (price > MA), Attractor pulls DOWN (-1)
            A_raw = -dist 
            A = np.clip(A_raw * 5.0, -1.0, 1.0) # Sensitivity
        else:
            A = 0.0

        # --- UNIFIED VECTOR S ---
        # Weights: v (Momentum) 40%, P (Pressure) 30%, A (Reversion) 30%
        # If market is trending (v matches P), we ignore A.
        # If market is extended (A is huge), we ignore v.
        
        w_v, w_p, w_a = 0.4, 0.3, 0.3
        
        # Unified Vector
        S = (w_v * v) + (w_p * P) + (w_a * A)
        
        # Signal Generation
        signal = "WAIT"
        confidence = abs(S) * 100
        
        if S > self.threshold: signal = "BUY"
        elif S < -self.threshold: signal = "SELL"
        
        reason = f"Hybrid Field: v={v:.2f} P={P:.2f} A={A:.2f} | S={S:.2f}"
        
        if signal != "WAIT":
            # logger.info(f"Hybrid Signal: {signal} | {reason}") # Too noisy?
            pass

        return SwarmSignal(
            source="HybridScalper",
            signal_type=signal,
            confidence=confidence,
            timestamp=0,
            meta_data={"reason": reason, "vector":S}
        )
