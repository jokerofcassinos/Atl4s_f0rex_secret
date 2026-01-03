
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("MirrorSwarm")

class MirrorSwarm(SubconsciousUnit):
    """
    The Mirror Swarm (Imitation Learning / Inverse RL).
    Phase 44 Innovation.
    Logic:
    1. "Monkey See, Monkey Do".
    2. Maintains a "Hall of Fame": A collection of market states that resulted in wins recently.
    3. If current state is similar to a Hall of Fame state, copy the action.
    4. Focuses on 'Style' rather than just stats.
    """
    def __init__(self):
        super().__init__("MirrorSwarm")
        # Hall of Fame: List of (Vector, Action, Confidence)
        # Vector is a simple feature set: [RSI, ADX, Volatility, Slope]
        self.hall_of_fame = [] 
        self.max_memory = 50

    def learn(self, vector, action, reward):
        """
        Called when a trade closes profitably.
        """
        if reward > 0:
            self.hall_of_fame.append({'vector': vector, 'action': action, 'reward': reward})
            # Keep top by reward
            self.hall_of_fame.sort(key=lambda x: x['reward'], reverse=True)
            if len(self.hall_of_fame) > self.max_memory:
                self.hall_of_fame = self.hall_of_fame[:self.max_memory]

    def _extract_features(self, df):
        # Create a signature vector
        close = df['close'].values
        if len(close) < 20: return None
        
        # 1. Norm Slope
        slope = (close[-1] - close[-5]) / close[-5] * 1000
        # 2. Volatility
        vol = np.std(np.diff(close)[-20:])
        # 3. Relative Position in Range (Stochastic-like)
        low = np.min(close[-20:])
        high = np.max(close[-20:])
        rng = high - low if high != low else 1.0
        rel_pos = (close[-1] - low) / rng
        
        return np.array([slope, vol, rel_pos])

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        # Note: In a real system, 'learn' would be called by the Neuroplasticity system.
        # For now, we simulate the 'Existence' of the Hall of Fame by pre-seeding it 
        # with 'Ideal' vectors if empty, or we rely on runtime learning (which starts empty).
        # Let's Pre-seed with generic logic to make it active immediately.
        
        if not self.hall_of_fame:
             # Pre-seed: [Slope, Vol, RelPos]
             # Bullish Seed: High Slope, Low Vol, High RelPos
             self.hall_of_fame.append({'vector': np.array([2.0, 0.5, 0.9]), 'action': 'BUY', 'reward': 100})
             # Bearish Seed: Negative Slope, Low Vol, Low RelPos
             self.hall_of_fame.append({'vector': np.array([-2.0, 0.5, 0.1]), 'action': 'SELL', 'reward': 100})
        
        df = context.get('df_m5')
        if df is None: return None
        
        current_vec = self._extract_features(df)
        if current_vec is None: return None
        
        # Find Nearest Neighbor in Hall of Fame
        best_match = None
        min_dist = float('inf')
        
        for memory in self.hall_of_fame:
            mem_vec = memory['vector']
            # Euclidean Distance
            dist = np.linalg.norm(current_vec - mem_vec)
            
            if dist < min_dist:
                min_dist = dist
                best_match = memory
                
        # Threshold for imitation
        # Distance depends on scale. Slope is ~2.0, RelPos ~0.9.
        # Dist < 1.0 is decent.
        
        if best_match and min_dist < 1.5:
            action = best_match['action']
            # Confidence based on closeness + reward size?
            # Just closeness for now.
            confidence = max(0, 100 - (min_dist * 40)) # 1.5 dist -> 40 conf. 0 dist -> 100 conf.
            
            if confidence > 50:
                return SwarmSignal(
                    source="MirrorSwarm",
                    signal_type=action,
                    confidence=confidence,
                    timestamp=0,
                    meta_data={
                        "imitation_dist": float(min_dist),
                        "model_action": action,
                        "inspiration": "HallOfFame"
                    }
                )
                
        return None
