
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import kurtosis, skew
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("EventHorizonSwarm")

class EventHorizonSwarm(SubconsciousUnit):
    """
    The Event Horizon (Fat-Tail Detector).
    Phase 31 Innovation.
    Logic:
    1. Analyzes the Statistical Distribution of Returns (M1/M5).
    2. Calculates Kurtosis (Fat Tails) and Skewness (Asymmetry).
    3. Normal Market ~ Kurtosis 3.0.
    4. Crisis/Crash/Moon ~ Kurtosis > 10.0.
    5. If Event Horizon detected (Extreme Non-Gaussian), we likely want to:
       - VETO standard Mean Reversion trades (they get run over).
       - ALLOW Momentum trades (Ride the Black Swan).
    """
    def __init__(self):
        super().__init__("EventHorizonSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m1')
        if df is None or len(df) < 100: return None
        
        # Calculate Returns
        returns = df['close'].pct_change().dropna()
        
        # Rolling Window for Adaptability (Last 60 mins)
        recent_returns = returns.tail(60)
        
        if len(recent_returns) < 30: return None
        
        # Stats
        k = kurtosis(recent_returns)
        s = skew(recent_returns)
        
        # Interpretation
        # High Kurtosis = "Price jumps are more frequent than Normal Dist predicts"
        # This implies Risk of Stop Hunts.
        
        signal = "WAIT"
        confidence = 0.0
        details = f"Kurt: {k:.2f} | Skew: {s:.2f}"
        
        if k > 6.0:
            # Extreme Fat Tails.
            # Strategy: CAUTION.
            # If we are contrarian, this kills us.
            # We issue a "VETO" for any fragile strategies.
            signal = "VETO"
            confidence = 80.0
            reason = f"Event Horizon: Fat Tails Detected (K={k:.2f}). Markets are unstable."
            
            return SwarmSignal(
                source="EventHorizonSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={"reason": reason, "kurtosis": k}
            )
            
        elif k < 2.0:
            # Platykurtic (Thin Tails). Market is boring/ranging.
            # Safe for Mean Reversion.
            pass

        return None
