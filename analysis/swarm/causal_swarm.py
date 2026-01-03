
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("CausalSwarm")

class CausalSwarm(SubconsciousUnit):
    """
    The Causal Graph (Granger Causality).
    Phase 36 Innovation.
    Logic:
    1. Distinguishes 'Leading' from 'Lagging' factors.
    2. Tests if Variable X (e.g., Volume) Granger-Causes Variable Y (Price).
    3. Methodology (Simplified Granger):
       - Does past X help predict current Y better than past Y alone?
       - We compare variance of residuals.
    4. Reasoning:
       - If Volume CAUSES Price -> Trust Gravity/Whale Swarms.
       - If Price CAUSES Volume -> Ignore Volume (it's just chasing).
    """
    def __init__(self):
        super().__init__("CausalSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        # Data Prep
        price = df['close'].values
        volume = df['volume'].values
        
        # We need differences to ensure stationarity (usually)
        price_diff = np.diff(price)
        vol_diff = np.diff(volume)
        
        if len(price_diff) < 30: return None
        
        # Simplified Granger Test Lag 1
        # Model 1: P(t) = a*P(t-1) + E1
        # Model 2: P(t) = b*P(t-1) + c*V(t-1) + E2
        # If Var(E2) < Var(E1) significantly, then V causes P.
        
        # Prepare arrays
        # Target: P(t) from index 1 to end
        Y = price_diff[1:] 
        
        # Predictor 1: P(t-1) from index 0 to end-1
        X_price = price_diff[:-1]
        
        # Predictor 2: V(t-1)
        X_vol = vol_diff[:-1]
        
        # Solve least squares (Polyfit degree 1 or simple matrix)
        # Model 1 (Univariate autoregression)
        # y = m*x + c
        A1 = np.vstack([X_price, np.ones(len(X_price))]).T
        m1, c1 = np.linalg.lstsq(A1, Y, rcond=None)[0]
        preds1 = m1*X_price + c1
        residuals1 = Y - preds1
        rss1 = np.sum(residuals1**2)
        
        # Model 2 (Bivariate)
        # y = m1*x1 + m2*x2 + c
        A2 = np.vstack([X_price, X_vol, np.ones(len(X_price))]).T
        m_p, m_v, c2 = np.linalg.lstsq(A2, Y, rcond=None)[0]
        preds2 = m_p*X_price + m_v*X_vol + c2
        residuals2 = Y - preds2
        rss2 = np.sum(residuals2**2)
        
        # F-Test Logic (Simplified)
        # Did adding Volume reduce error?
        # RSS2 should be less than RSS1
        
        if rss1 == 0: return None
        
        improvement_ratio = (rss1 - rss2) / rss1
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # If improvement > 5%, we say Volume has Causal Power
        if improvement_ratio > 0.05:
            # Volume is Leading.
            # Analyze Volume Direction.
            # If Volume is rising and Causality is strong -> Move is supported.
            # If Volume is falling and Causality is strong -> Move is weak.
            
            # Simple Logic: Trust the Volume Trend.
            # recent_vol_trend = np.mean(vol_diff[-5:])
            
            # If we know Volume causes Price, we look at Volume Impulse.
            last_vol_impulse = vol_diff[-1]
            last_price_impulse = price_diff[-1]
            
            if last_vol_impulse > 0:
                 # Volume Spike. 
                 # If price is up -> Buy (Supported)
                 # If price is down -> Sell (Supported)
                 # Causal link confirms the move.
                 
                 if last_price_impulse > 0:
                     signal = "BUY"
                     confidence = 70.0 + (improvement_ratio * 100) # Boost confidence by causal strength
                     reason = f"Causal Link Confirmed: Volume Leads Price (Imp: {improvement_ratio:.2%})"
                 elif last_price_impulse < 0:
                     signal = "SELL"
                     confidence = 70.0 + (improvement_ratio * 100)
                     reason = f"Causal Link Confirmed: Volume Leads Price (Imp: {improvement_ratio:.2%})"
                     
        elif improvement_ratio < -0.05:
             # Strange case where adding volume made it worse? Overfitting?
             # Usually means Volume is noise.
             pass
        else:
             # Improvement ~ 0. No Causality.
             # Volume is irrelevant. Ignore GravitySwarm?
             pass
             
        if signal != "WAIT":
            # Cap confidence
            confidence = min(95.0, confidence)
            
            return SwarmSignal(
                source="CausalSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={"reason": reason, "causal_improvement": improvement_ratio}
            )
            
        return None
