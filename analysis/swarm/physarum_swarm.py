
import logging
import numpy as np
import time
import random
from typing import Dict, Any, List, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("PhysarumSwarm")

class PhysarumSwarm(SubconsciousUnit):
    """
    Phase 99: The Physarum Swarm (Biological Liquidity Network).
    
    Mimics Physarum polycephalum (Slime Mold) behavior.
    Solves for the optimal path to 'Food' (Liquidity/Profit) by reinforcing 
    successful flow tubes and establishing a network.
    
    Mathematical Model:
    - Conductivity (C_ij) of edge (i,j)
    - Flux (Q_ij) through edge = D_ij * (P_i - P_j)
    - Adaptation: dC/dt = |Q| - decay * C
    """
    def __init__(self):
        super().__init__("Physarum_Swarm")
        # Simplified Model:
        # We model 3 main "Tubes": 
        # 1. Trend Continuation (Forward)
        # 2. Reversal (Backward)
        # 3. Stagnation (Lateral)
        
        # Initial Conductivity
        self.tubes = {
            'continuation': 0.5,
            'reversal': 0.5,
            'stagnation': 0.5
        }
        self.decay_rate = 0.05
        self.growth_rate = 0.1
        
        self.last_price = 0.0
        self.last_trend = 0 # 1=Up, -1=Down
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        tick = context.get('tick')
        if not tick: return None
        
        current_price = tick['bid']
        
        # 0. Initialization
        if self.last_price == 0:
            self.last_price = current_price
            return None
            
        # 1. Determine "Nutrient Flow" (What actually happened)
        # Did the market Continue, Reverse, or Stagnate?
        
        delta = current_price - self.last_price
        threshold = 0.5 # Minimum movement to count as flow
        
        actual_event = 'stagnation'
        
        if abs(delta) > threshold:
            current_direction = 1 if delta > 0 else -1
            
            if self.last_trend != 0:
                if current_direction == self.last_trend:
                    actual_event = 'continuation'
                else:
                    actual_event = 'reversal'
            
            self.last_trend = current_direction
        else:
             actual_event = 'stagnation'

        self.last_price = current_price

        # 2. Adaptation Step (Grow/Shrink Tubes)
        # The tube that matched reality gets a surge of flux (Growth)
        # All tubes decay naturally
        
        total_conductivity = sum(self.tubes.values())
        
        for tube_name in self.tubes:
            # Decay
            self.tubes[tube_name] *= (1.0 - self.decay_rate)
            
            # Growth if matched
            if tube_name == actual_event:
                self.tubes[tube_name] += self.growth_rate
                
            # Clamp
            self.tubes[tube_name] = max(0.01, min(self.tubes[tube_name], 5.0))
            
        # 3. Prediction based on Dominant Tube
        # The Slime Mold predicts that the thickest tube is the path of least resistance.
        
        dominant_tube = max(self.tubes, key=self.tubes.get)
        max_cond = self.tubes[dominant_tube]
        
        # Normalize dominance
        dominance_score = max_cond / sum(self.tubes.values())
        
        if dominance_score < 0.4: return None # No clear path
        
        signal = "WAIT"
        conf = 0.0
        meta = {'tubes': self.tubes}
        
        # Interpret Tube into Signal
        # If Continuation is dominant -> Follow current trend
        # If Reversal is dominant -> Bet against current trend
        
        if self.last_trend == 0: return None
        
        if dominant_tube == 'continuation':
            signal = "BUY" if self.last_trend > 0 else "SELL"
            conf = 75.0 + (dominance_score * 20)
            meta['reason'] = f"Physarum Network: Robust Continuation Tube ({dominance_score:.2f})"
            
        elif dominant_tube == 'reversal':
            signal = "SELL" if self.last_trend > 0 else "BUY"
            conf = 75.0 + (dominance_score * 20)
            meta['reason'] = f"Physarum Network: Robust Reversal Tube ({dominance_score:.2f})"
            
        elif dominant_tube == 'stagnation':
            signal = "WAIT"
            
        if signal == "WAIT": return None
        
        return SwarmSignal(
            source=self.name,
            signal_type=signal,
            confidence=min(conf, 98.0),
            timestamp=time.time(),
            meta_data=meta
        )
