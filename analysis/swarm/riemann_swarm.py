
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("RiemannSwarm")

class RiemannSwarm(SubconsciousUnit):
    """
    Phase 90: The Riemann Swarm (Curvature of Spacetime).
    
    Treats the Market as a Curved Riemannian Manifold.
    
    Physics:
    - In General Relativity, gravity is the curvature of spacetime.
    - Positive Curvature (Sphere): Parallel geodesics converge. (Gravity / Mean Reversion).
    - Negative Curvature (Saddle/Hyperbolic): Parallel geodesics diverge. (Expansion / Trend).
    
    Logic:
    - We track two 'geodesics' (paths):
      1. Slow Path (SMA 20) - The 'inertial' frame.
      2. Fast Path (EMA 20) - The 'accelerated' frame.
    - Separation Vector (Jacobi Field) J = Fast - Slow.
    - Sectional Curvature K is related to the second derivative of separation.
      d2J/dt2 + K*J = 0 (Jacobi Equation).
      So K ~ -(Acceleration of Separation) / Separation.
      
    - If Separation is accelerating AWAY (Diverging faster than linear) -> K is Negative (Hyperbolic).
      -> SIGNAL: TREND EXPANSION.
      
    - If Separation is accelerating TOWARDS (Converging) -> K is Positive (Spherical).
      -> SIGNAL: MEAN REVERSION / CONSOLIDATION.
    """
    def __init__(self):
        super().__init__("Riemann_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        closes = df_m5['close']
        
        # 1. Define Geodesics
        sma = closes.rolling(window=20).mean()
        ema = closes.ewm(span=20, adjust=False).mean()
        
        # 2. Calculate Separation Vector (J)
        # J = EMA - SMA
        separation = ema - sma
        
        # We need the derivatives of separation Magnitude |J|
        # J_t = first derivative (Velocity of separation)
        # J_tt = second derivative (Acceleration of separation)
        
        sep_values = separation.values
        
        # Use last 10 points for smooth derivative
        if len(sep_values) < 10: return None
        
        # First Derivative (Velocity of Spread)
        vel_sep = np.gradient(sep_values)
        
        # Second Derivative (Acceleration of Spread)
        acc_sep = np.gradient(vel_sep)
        
        # 3. Estimate Sectional Curvature (K)
        # From Jacobi Eq: J'' + KJ = 0  => K = -J'' / J
        # We clamp J to avoid division by zero
        
        current_J = sep_values[-1]
        current_acc = acc_sep[-1]
        
        if abs(current_J) < 0.0001: 
            K = 0 # Flat space
        else:
            K = -current_acc / current_J
            
        # 4. Interpret Curvature
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        meta_data = {}
        
        # Determine Trend Direction from EMA
        trend_direction = "BUY" if ema.iloc[-1] > ema.iloc[-2] else "SELL"
        
        # Case A: K is Negative (Hyperbolic)
        # Separation is accelerating away. (J and J'' have same sign? No, K = -J''/J. If J'' and J same sign, K negative).
        # Example: J > 0 and accelerating up (J'' > 0) -> K < 0.
        # This means Divergence -> TREND.
        
        if K < -0.1:
            # Hyperbolic Expansion.
            # The space is stretching.
            signal = trend_direction
            confidence = 85.0
            reason = f"RIEMANN: Negative Curvature (K={K:.2f}). Hyperbolic Expansion. Trend accelerating."
            
        # Case B: K is Positive (Spherical)
        # Separation is decelerating or reversing. (J > 0 but J'' < 0).
        # Gravity is pulling them back together.
        
        elif K > 0.1:
            # Spherical Closure.
            # The space is closing in.
            # This suggests the Trend is ending or Reverting.
            
            signal = "SELL" if trend_direction == "BUY" else "BUY" # Reversal
            confidence = 80.0
            reason = f"RIEMANN: Positive Curvature (K={K:.2f}). Spherical Closure. Geodesics converging (Reversal)."
            
        else:
            # Flat Space (Euclidean)
            # Constant velocity separation?
            pass

        meta_data = {
            'curvature_K': K,
            'separation': current_J,
            'reason': reason
        }

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data=meta_data
            )
            
        return None
