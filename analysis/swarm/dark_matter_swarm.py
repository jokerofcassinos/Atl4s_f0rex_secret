
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("DarkMatterSwarm")

class DarkMatterSwarm(SubconsciousUnit):
    """
    Phase 74: The Dark Matter Swarm (Invisible Mass).
    
    analyzes the relationship between Price Movement (Gravity) and Volume (Visible Matter).
    Calculates the Mass-to-Light Ratio (Upsilon).
    
    Physics:
    - In clusters, if Mass (Gravity) >> Visible Light, Dark Matter exists.
    - In markets, if Price Move >> Volume, "Dark Liquidity" or "Vacuum" exists.
    
    Logic:
    - High Upsilon (Price moves fast on zero volume) = Unstable/Fake Move -> FADE IT.
    - Low Upsilon (High volume, no price move) = Black Hole (Absorption) -> BREAKOUT IMMINENT.
    """
    def __init__(self):
        super().__init__("Dark_Matter_Swarm")
        self.upsilon_history = []

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 30: return None
        
        # 1. Physics Parameters
        # Gravity (Delta P): Absolute Price Change
        close = df_m5['close'].values
        open_p = df_m5['open'].values
        price_change = np.abs(close - open_p)
        
        # Light (Volume)
        # Handle missing volume
        if 'tick_volume' in df_m5.columns:
            volume = df_m5['tick_volume'].replace(0, 1).values
        else:
            return None # Can't detect Dark Matter without Volume
            
        # 2. Compute Mass-to-Light Ratio (Upsilon)
        # Upsilon = Effect / Cause
        upsilon = price_change / volume 
        
        # Normalize Upsilon relative to recent history (Local Anomaly)
        # We look at the last candle vs the rolling mean of Upsilon
        current_upsilon = upsilon[-1]
        
        # Rolling stats (last 20)
        rolling_mean = np.mean(upsilon[-21:-1])
        rolling_std = np.std(upsilon[-21:-1])
        
        if rolling_std == 0: return None
        
        # Anomaly Score (Z-Score)
        z_score = (current_upsilon - rolling_mean) / rolling_std
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 3. Scenario A: Ghost Move (High Upsilon)
        # Price moved a lot, but Volume was missing.
        # This implies a Liquidity Void or Skewed Order Book. The move is fragile.
        # Expect Reversion.
        
        if z_score > 3.0: # 3 Sigma Anomaly
            direction = "UP" if close[-1] > open_p[-1] else "DOWN"
            
            # If moved UP on low volume -> Unstable -> SHORT
            if direction == "UP":
                signal = "SELL"
                confidence = 85.0
                reason = f"DARK MATTER: High Upsilon (Z={z_score:.1f}). Price moved without Volume (Ghost Pump)."
            else:
                signal = "BUY"
                confidence = 85.0
                reason = f"DARK MATTER: High Upsilon (Z={z_score:.1f}). Price moved without Volume (Ghost Dump)."
                
        # 4. Scenario B: Black Hole (Low Upsilon -> Approaches 0)
        # Massive Volume, Zero Price Move.
        # Absorption / Accumulation / Distribution.
        # Energy is condensing. Expect Explosion.
        
        elif z_score < -1.5 and volume[-1] > (np.mean(volume[-21:-1]) * 1.5):
            # High Volume, Low Displacement. "Black Hole".
            # Direction? Hard to tell. Absorption usually favors the trend *reversal* if at extremes.
            
            # Simple heuristic: If at Local Low + Black Hole -> Accumulation (Buy)
            # If at Local High + Black Hole -> Distribution (Sell)
            
            recent_high = np.max(df_m5['high'].values[-20:])
            recent_low = np.min(df_m5['low'].values[-20:])
            current_p = close[-1]
            
            # Position in Range
            range_pos = (current_p - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            
            if range_pos > 0.8: # Absorb at Top
                signal = "SELL"
                confidence = 75.0
                reason = "DARK MATTER: Black Hole at Resistance (Absorption)."
            elif range_pos < 0.2: # Absorb at Low
                signal = "BUY"
                confidence = 75.0
                reason = "DARK MATTER: Black Hole at Support (Absorption)."

        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'upsilon_z': z_score, 'reason': reason}
            )
            
        return None
