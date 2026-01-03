
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("MaxwellSwarm")

class MaxwellSwarm(SubconsciousUnit):
    """
    Phase 88: The Maxwell Swarm (Electrodynamic Induction).
    
    Applies Maxwell's Equations and Lenz's Law to Market Dynamics.
    
    Physics:
    - A changing magnetic flux induces an Electromotive Force (EMF) that OPPOSES the change.
    - Lenz's Law: E = - d(Phi)/dt
    - Phi (Flux) = B (Field Strength) * A (Area).
    
    Logic:
    - We map Market Flux (Phi) as: Price Momentum (B) * Volume (A).
    - We calculate the Induced EMF (E) as the rate of change of Flux.
    - High E (Rapid Spike in Price/Vol) -> Strong Back-EMF -> REJECTION (Reversal).
    - Low E (Gradual Rise) -> Weak Back-EMF -> CONTINUATION.
    
    This explains why "Parabolic" moves often fail (High Inductance) while "Grinding" moves sustain.
    """
    def __init__(self):
        super().__init__("Maxwell_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 20: return None
        
        closes = df_m5['close'].values
        volumes = df_m5['volume'].values
        
        # 1. Calculate Field Strength (B) -> Momentum
        # We use a short period momentum
        momentum = np.diff(closes) 
        # Pad the first element
        momentum = np.insert(momentum, 0, 0)
        
        # 2. Calculate Flus (Phi) -> Momentum * Volume
        # This represents the "Energy Flow" through the surface of time.
        flux = momentum * volumes
        
        # 3. Calculate Induced EMF (dPhi / dt)
        # Rate of change of Flux.
        # We look at the last few candles.
        
        d_flux_dt = np.diff(flux[-5:]) # Change over last 5 bars
        
        # Current EMF is the most recent change
        current_emf = d_flux_dt[-1]
        
        # Normalize EMF relative to recent history average
        avg_flux_change = np.mean(np.abs(np.diff(flux[-50:])))
        if avg_flux_change == 0: avg_flux_change = 1
        
        normalized_emf = current_emf / avg_flux_change
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 4. Decision Logic (Lenz's Law)
        # If EMF is MASSIVE (e.g. > 5 sigma), Expect Opposition.
        
        if abs(normalized_emf) > 5.0:
            # High Inductance Event.
            # The market changed state too fast. Back-EMF will push back.
            
            direction = "UP" if current_emf > 0 else "DOWN"
            
            if direction == "UP":
                signal = "SELL" # Fade the Spike
                confidence = 85.0
                reason = f"MAXWELL: High Back-EMF (E={normalized_emf:.1f}). Inductive Rejection Expected."
            else:
                signal = "BUY" # Fade the Crash
                confidence = 85.0
                reason = f"MAXWELL: High Back-EMF (E={normalized_emf:.1f}). Inductive Rejection Expected."
                
        elif abs(normalized_emf) < 1.0 and abs(normalized_emf) > 0.2:
            # Low Inductance, steady flow.
            # Continuation.
             pass #Maxwell mainly detects Rejections. Let others handle Trends.
        
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'emf': normalized_emf, 'flux': flux[-1], 'reason': reason}
            )
            
        return None
