
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("HiggsSwarm")

class HiggsSwarm(SubconsciousUnit):
    """
    Phase 81: The Higgs Swarm (The God Particle).
    
    Models the 'Mass' of Price Action using the Higgs Mechanism.
    
    Physics:
    - The 'Higgs Field' is the Liquidity Pool (Limit Orders).
    - Price 'acquires mass' by interacting with this field (Filling Orders).
    - Coupling Constant (g): How much Volume is required to move Price.
      - g = Volume / |Delta Price|
      
    Logic:
    - Massless State (g -> 0): Symmetrical Phase.
      - Price moves with ZERO friction. A 'Superfluid' market.
      - PREDICTION: Violent, Fast Trend (Gamma Squeeze / Flash Crash).
      - Action: BUY BREAKOUT / SELL BREAKDOWN.
      
    - Massive State (g -> Infinity): Broken Symmetry.
      - Price requires infinite energy to move an inch.
      - PREDICTION: Hitting a Wall. Absorption.
      - Action: REVERSAL / TAKE PROFIT.
    """
    def __init__(self):
        super().__init__("Higgs_Swarm")
        self.lookback = 10

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 20: return None
        
        # 1. Fetch Data
        closes = df_m5['close'].values
        
        # Safe Volume Access (Phase 78 Fix)
        if 'tick_volume' in df_m5.columns:
             volumes = df_m5['tick_volume'].replace(0, 1).values
        elif 'Volume' in df_m5.columns:
             volumes = df_m5['Volume'].replace(0, 1).values
        else:
             return None
        
        # 2. Calculate Einstein-Higgs Coupling (g)
        # We calculate 'g' for the last candle or last N candles.
        
        # Delta Price must be non-zero
        delta_p = abs(closes[-1] - closes[-2])
        if delta_p < 0.00001: delta_p = 0.00001
        
        vol = volumes[-1]
        
        # g = "Density of Resistance"
        g = vol / delta_p
        
        # 3. Normalize 'g'
        # We need to know if 'g' is Low (Vacuum) or High (Wall) relative to history.
        
        # Rolling 'g' for last 20 candles
        hist_deltas = np.abs(np.diff(closes[-21:]))
        hist_deltas = np.where(hist_deltas < 0.00001, 0.00001, hist_deltas)
        hist_vols = volumes[-20:]
        
        hist_g = hist_vols / hist_deltas
        
        avg_g = np.mean(hist_g)
        std_g = np.std(hist_g)
        if std_g == 0: std_g = 1
        
        # Z-Score of current Mass Coupling
        z_score = (g - avg_g) / std_g
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 4. Classification
        
        # A. Massless Phase (The God Particle Moment)
        # Z-Score < -1.5 (Extremely Low Volume for Data Moved, or Huge Move for Normal Volume)
        # Wait, if g is Low, it means Vol is Low OR DeltaP is Huge.
        # Low g = Low Resistance.
        
        if z_score < -1.5:
            # Symmetrical Phase (Superfluidity)
            # Price is slipping through the field.
            direction = "UP" if closes[-1] > closes[-2] else "DOWN"
            
            if direction == "UP":
                signal = "BUY"
                confidence = 90.0 # High Conviction
                reason = f"HIGGS: Massless Phase (Z={z_score:.2f}). Superfluid Breakout UP."
            else:
                signal = "SELL"
                confidence = 90.0
                reason = f"HIGGS: Massless Phase (Z={z_score:.2f}). Superfluid Crash DOWN."
                
        # B. Massive Phase (The Wall)
        # Z-Score > 2.0 (Huge Volume, Small Move)
        # Absorption.
        
        elif z_score > 2.5:
             # Infinite Mass.
             # Trend is dying.
             direction = "UP" if closes[-1] > closes[-2] else "DOWN"
             
             # If moving UP but hitting MASS -> REVERSE DOWN
             if direction == "UP":
                 signal = "SELL"
                 confidence = 85.0
                 reason = f"HIGGS: Massive Phase (Z={z_score:.2f}). Hitting Resistance Wall."
             else:
                 signal = "BUY"
                 confidence = 85.0
                 reason = f"HIGGS: Massive Phase (Z={z_score:.2f}). Hitting Support Wall."

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'g_z_score': z_score, 'reason': reason}
            )
            
        return None
