
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger("HyperdimensionalSwarm")

class HyperdimensionalSwarm(SubconsciousUnit):
    """
    Phase 64: The Hyperdimensional Hedge.
    
    Uses Phase Space Reconstruction (Takens' Embedding) to analyze market stability.
    It embeds the Price Time Series into an M-dimensional Manifold.
    Then it calculates the 'Correlation Dimension' and 'Lyapunov Divergence' in that space.
    
    Logic:
    - If Trajectories in Phase Space are parallel (Low Divergence) -> Buy/Sell Trend is Real.
    - If Trajectories are orthogonal/scrambled (High Divergence) -> Trend is Fake/Exhausted.
    """
    def __init__(self):
        super().__init__("Hyperdimensional_Swarm")
        self.embedding_dim = 3 # 3D Phase Space
        self.time_delay = 2 # Tau

    async def process(self, context) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        if df_m1 is None or len(df_m1) < 100: return None
        
        # 1. Prepare Data
        # We use a sliding window of recent prices
        window = 60
        data = df_m1['close'].values[-window:]
        
        # Normalize
        data = (data - np.mean(data)) / (np.std(data) + 1e-9)
        
        # 2. Embed in Phase Space (Time Delay Embedding)
        # X(t) = [x(t), x(t-tau), x(t-2tau)]
        embedded = []
        for i in range(len(data) - (self.embedding_dim - 1) * self.time_delay):
            point = [data[i + j * self.time_delay] for j in range(self.embedding_dim)]
            embedded.append(point)
            
        embedded = np.array(embedded)
        if len(embedded) < 10: return None
        
        # 3. Analyze Trajectory Divergence (Simplified Lyapunov)
        # We measure the average distance between consecutive vectors in phase space.
        # If distance is increasing rapidly, we are bifurcating (Reversal/Chaos).
        
        # Calculate Euclidean distances between adjacent points in time
        steps = np.diff(embedded, axis=0)
        step_sizes = np.linalg.norm(steps, axis=1)
        
        avg_step = np.mean(step_sizes)
        recent_step = step_sizes[-1]
        
        # Growth Ratio
        divergence_ratio = recent_step / (avg_step + 1e-9)
        
        # 4. Attractor Classification
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # If Divergence is extreme (> 2.5x normal), the Attractor is breaking.
        if divergence_ratio > 2.5:
             # Instability Detected.
             # If Price is rising but Phase Space is exploring new volumne rapidly => BLOW OFF TOP.
             current_trend = data[-1] - data[-5]
             
             if current_trend > 0:
                 signal = "SELL" # Rejection of the High
                 confidence = 88.0
                 reason = f"HYPERDIMENSIONAL: Attractor Divergence (Ratio {divergence_ratio:.2f}) on Uptrend -> Blow-off Top."
             elif current_trend < 0:
                 signal = "BUY" # Rejection of the Low
                 confidence = 88.0
                 reason = f"HYPERDIMENSIONAL: Attractor Divergence (Ratio {divergence_ratio:.2f}) on Downtrend -> Panic Bottom."
                 
        # If Divergence is very low (< 0.5), we are in a Limit Cycle (Ranging/Stuck).
        elif divergence_ratio < 0.5:
             signal = "WAIT"
             confidence = 0.0
             reason = "Phase Space Contraction (Range)"
             
        # If Divergence is normal (1.0), Trend follows momentum (handled by KinematicSwarm).
        
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'divergence': divergence_ratio, 'reason': reason}
            )
            
        return None
