
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("ManifoldSwarm")

class ManifoldSwarm(SubconsciousUnit):
    """
    The Geometric Mind.
    Uses concepts from Riemannian Geometry to detect Market Curvature.
    
    Logic:
    - Market moves on a 'Manifold' (Surface), not a straight line.
    - We estimate the 'Curvature' (Ricci Scalar) of the price path.
    - Extreme Curvature = Reversal Imminent (Geodesic Deviation).
    - Flat Space = Trend Continuation.
    """
    def __init__(self):
        super().__init__("Manifold_Swarm")
        self.window = 14
        
    async def process(self, context) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        if df_m1 is None or len(df_m1) < 30: return None
        
        # 1. Embed Data into 2D Manifold (Time, Price)
        # Normalize Price to be comparable to Time steps
        prices = df_m1['close'].values[-30:]
        
        # Normalization (Z-Score)
        mean_p = np.mean(prices)
        std_p = np.std(prices)
        if std_p == 0: return None
        
        norm_prices = (prices - mean_p) / std_p
        
        # Time steps (0, 1, 2...)
        time_steps = np.arange(len(norm_prices))
        
        # 2. Calculate Discrete Curvature (Menger Curvature)
        # Curvature of 3 points (A, B, C) = 4 * Area / (AB * BC * CA)
        # We calculate curvature for the latest cluster of points.
        
        # Use last 3 points
        A = np.array([time_steps[-3], norm_prices[-3]])
        B = np.array([time_steps[-2], norm_prices[-2]])
        C = np.array([time_steps[-1], norm_prices[-1]])
        
        curvature = self._menger_curvature(A, B, C)
        
        # 3. Interpret Curvature
        # High Curvature (> 0.5) implies a sharp turn/corner.
        # Low Curvature (< 0.1) implies a straight line.
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Determine Direction of the Turn
        # Vector AB and BC
        AB = B - A
        BC = C - B
        # Cross product (2D analog) implies convexity
        cross_prod = AB[0]*BC[1] - AB[1]*BC[0]
        
        # Thresholds
        if curvature > 0.4:
            # SHARP TURN DETECTED
            # If cross_prod > 0: Counter-Clockwise Turn (Bottoming? or Topping?)
            # Actually, check Price acceleration direction.
            # If Price was falling and turned -> Buy.
            
            # Simple Logic: 
            # If Middle Point (B) is lower than A and C -> Bottom -> BUY
            if B[1] < A[1] and B[1] < C[1]: 
                signal = "BUY"
                confidence = 75
                reason = f"Manifold Bend: Bottom Singularity (k={curvature:.2f})"
            # If Middle Point (B) is higher than A and C -> Top -> SELL
            elif B[1] > A[1] and B[1] > C[1]:
                signal = "SELL"
                confidence = 75
                reason = f"Manifold Bend: Top Singularity (k={curvature:.2f})"
        
        elif curvature < 0.05:
            # FLAT SPACE (Geodesic is straight)
            # Continuation of Momentum
            # Vector AC
            AC = C - A
            slope = AC[1] / (AC[0] + 1e-9)
            
            if slope > 0.1:
                signal = "BUY"
                confidence = 60
                reason = f"Flat Geodesic: Uptrend (Slope={slope:.2f})"
            elif slope < -0.1:
                signal = "SELL"
                confidence = 60
                reason = f"Flat Geodesic: Downtrend (Slope={slope:.2f})"

        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'curvature': curvature, 'reason': reason}
            )

        return None

    def _menger_curvature(self, A, B, C):
        # Area of triangle
        # Area = 0.5 |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        area = 0.5 * abs(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))
        
        # Side lengths
        ab = np.linalg.norm(A - B)
        bc = np.linalg.norm(B - C)
        ca = np.linalg.norm(C - A)
        
        if ab * bc * ca == 0: return 0.0
        
        # curvature = 4 * Area / (ab * bc * ca)
        return 4 * area / (ab * bc * ca)
