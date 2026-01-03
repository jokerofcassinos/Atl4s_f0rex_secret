
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger("LorentzSwarm")

class LorentzSwarm(SubconsciousUnit):
    """
    Phase 79: The Lorentz Swarm (Chaos Theory & Bifurcations).
    
    Uses Phase Space Reconstruction to detect Market Regime Changes (Bifurcations).
    
    Physics:
    - Markets are Chaotic Systems (Deterministic Chaos).
    - They orbit 'Strange Attractors' (Hidden Equilibrium states).
    - A Trend Reversal is a 'Bifurcation' (Jump from Attractor A to Attractor B).
    
    Logic:
    - We reconstruct the Phase Space using Time Delay Embedding.
    - We measure the distance between current trajectory and recent 'Attractor Center'.
    - If distance exceeds the 'Lyapunov Threshold', the orbit has destabilized.
    - SIGNAL: EXIT IMMEDIATELY if Orbit Violates Position Logic.
    """
    def __init__(self):
        super().__init__("Lorentz_Swarm")
        self.embedding_dim = 3
        self.delay = 3 # Lag

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        tick = context.get('tick')
        
        if df_m5 is None or len(df_m5) < 50: return None
        
        closes = df_m5['close'].values
        
        # 1. Phase Space Reconstruction (Time Delay Embedding)
        # Vector X(t) = [x(t), x(t-tau), x(t-2tau)]
        # We need the last N vectors to define the "Current Attractor"
        
        N = 20 # Lookback for attractor shape
        tau = self.delay
        
        if len(closes) < (N + (self.embedding_dim * tau)): return None
        
        # Create Trajectory Matrix
        trajectory = []
        for i in range(N):
            # Index backwards from -1
            idx = -1 - i
            # Point = [P(t), P(t-tau), P(t-2tau)]
            point = [
                closes[idx], 
                closes[idx - tau], 
                closes[idx - 2*tau]
            ]
            trajectory.append(point)
            
        trajectory = np.array(trajectory) 
        # trajectory[0] is the current point (t). 
        # trajectory[1..N] is history.
        
        # 2. Estimate Attractor Stability (Centroid Distance)
        # Where is the "Center" of the recent motion?
        centroid = np.mean(trajectory[1:], axis=0) # Mean of history (excluding current if we want pure deviation)
        
        # Current Deviation
        current_point = trajectory[0]
        deviation_vec = current_point - centroid
        deviation_norm = np.linalg.norm(deviation_vec)
        
        # Average Radius of the Attractor (Standard Deviation of distances)
        # distances from centroid to all historical points
        history_points = trajectory[1:]
        dists = np.linalg.norm(history_points - centroid, axis=1)
        attractor_radius = np.mean(dists) + (2.0 * np.std(dists)) # 2 Sigma boundary
        
        # 3. Bifurcation Check (Escape from Attractor)
        # If Current Deviation > Attractor Radius, we have BIFURCATED.
        # The system has ejected from the previous equilibrium.
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        bifurcation = False
        if deviation_norm > attractor_radius:
            bifurcation = True
            
        # 4. Directionality of Bifurcation
        # Did we break UP or DOWN relative to the Attractor?
        # Use the Z-axis (Price itself) or the vector difference?
        # Simple Logic: Is Current Price > Centroid Price?
        
        current_price = current_point[0]
        centroid_price = centroid[0]
        
        delta = current_price - centroid_price
        
        positions = tick.get('positions', 0)
        
        if bifurcation:
            # CHAOS DETECTED.
            if delta > 0:
                # Upward Bifurcation (Bullish Breakout)
                # If we are Short, this is bad. 
                signal = "BUY"
                confidence = 85.0
                reason = f"LORENTZ: Bifurcation Detected (UP). Orbit escaped Attractor. Delta={delta:.2f}"
                
                if positions < 0: # We are Short
                     # EMERGENCY FLIP
                     signal = "EXIT_SHORT"
                     confidence = 95.0
                     reason = f"LORENTZ: Chaos Breakout UP. Short Thesis Invalidated."
                     
            else:
                # Downward Bifurcation (Bearish Crash)
                # If we are Long, this is FATAL.
                signal = "SELL"
                confidence = 85.0
                reason = f"LORENTZ: Bifurcation Detected (DOWN). Orbit escaped Attractor. Delta={delta:.2f}"
                
                if positions > 0: # We are Long
                     # EMERGENCY FLIP / EXIT
                     # User complaint: "Market dropping, buy not closed".
                     # This solves it.
                     signal = "EXIT_LONG"
                     confidence = 98.0 # Critical Priority
                     reason = f"LORENTZ: Chaos Crash DOWN. Long Thesis Invalidated (Bifurcation)."

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'deviation': deviation_norm, 'radius': attractor_radius, 'reason': reason}
            )
            
        return None
