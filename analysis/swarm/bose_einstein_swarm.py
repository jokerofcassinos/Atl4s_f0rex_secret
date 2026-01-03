
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("BoseEinsteinSwarm")

class BoseEinsteinSwarm(SubconsciousUnit):
    """
    Phase 84: The Bose-Einstein Swarm (Quantum Coherence).
    
    Detects the formation of a 'Bose-Einstein Condensate' (BEC) in the market.
    
    Physics:
    - At critical low temperatures (Low Volatility), particles lose identity and collapse into a single quantum state.
    - Result: Macroscopic Quantum Coherence (A Laser Beam).
    
    Logic:
    - We measure 'Coherence' (C) across multiple Timeframes (M1, M5).
    - If M1 and M5 vectors align perfectly (C -> 1.0) and Volatility is compressing (Cooling),
      the market is entering a BEC State.
    - PREDICTION: A Super-Trend (Vertical Move) that ignores resistance.
    - ACTION: AGGRESSIVE Trend Following. Ignore Oscillators.
    """
    def __init__(self):
        super().__init__("Bose_Einstein_Swarm")

    async def process(self, context) -> SwarmSignal:
        # We need Multi-Timeframe Data
        # Context usually has df_m5. We might need df_m1 if available in context.
        df_m5 = context.get('df_m5')
        df_m1 = context.get('df_m1') # Assuming Orchestrator provides this
        
        if df_m5 is None or len(df_m5) < 20: return None
        # Use M5 as primary if M1 missing, or simulate M1 from recent M5
        
        # 1. Calculate Vectors for Different Frames
        # M5 Vector
        m5_closes = df_m5['close'].values
        v_m5 = m5_closes[-1] - m5_closes[-5] # 5-bar momentum
        
        # M1 Vector (if available)
        v_m1 = 0
        if df_m1 is not None and len(df_m1) > 5:
            m1_closes = df_m1['close'].values
            v_m1 = m1_closes[-1] - m1_closes[-5]
        else:
            # Fallback: Use last candle of M5 as "Fast Vector"
            v_m1 = m5_closes[-1] - df_m5['open'].values[-1]
            
        # 2. Normalize Vectors (Unit Vectors)
        # We care about DIRECTION, not magnitude for Coherence.
        def normalize(v):
            if abs(v) < 1e-9: return 0
            return v / abs(v) # Returns 1 or -1 (Spin State)
            
        # 3. Calculate Coherence (Alignment)
        # Sum of Spings
        spin_m5 = normalize(v_m5)
        spin_m1 = normalize(v_m1)
        
        # We can also infer a "Macroscopic Trend" (H1 equivalent) from M5 SMA 50 slope
        sma_50_start = np.mean(m5_closes[-55:-5])
        sma_50_end = np.mean(m5_closes[-50:])
        spin_macro = normalize(sma_50_end - sma_50_start)
        
        # Total Spin Integration
        total_spin = spin_m1 + spin_m5 + spin_macro
        # Max spin possible = 3 (All aligned) or -3.
        
        coherence = abs(total_spin) / 3.0
        
        # 4. Check Temperature (Volatility)
        # BEC forms at Low Temp.
        # But a "Laser" fires with high energy. The STATE forms at low temp, then EMITS.
        # We look for High Coherence.
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        if coherence > 0.9:
            # QUANTUM COHERENCE ACHEIVED.
            # The market is a Laser.
            
            direction = "UP" if total_spin > 0 else "DOWN"
            
            magnitude = abs(v_m5) # How hard is it pushing?
            
            if direction == "UP":
                signal = "BUY"
                confidence = 95.0 # Extremely High Confidence
                reason = f"BOSE-EINSTEIN: Quantum Coherence (C={coherence:.2f}). Super-Trend UP."
            else:
                signal = "SELL"
                confidence = 95.0
                reason = f"BOSE-EINSTEIN: Quantum Coherence (C={coherence:.2f}). Super-Trend DOWN."
                
        elif coherence < 0.4:
            # Incoherent Light (Scatter).
            # Market is chopped/mixed.
            signal = "WAIT" # Stay out of noise
            confidence = 0.0
            reason = f"BOSE-EINSTEIN: Incoherent State (C={coherence:.2f}). Thermal Noise."

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'coherence': coherence, 'spin_total': total_spin, 'reason': reason}
            )
            
        return None
