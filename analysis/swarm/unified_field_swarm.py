
import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("UnifiedFieldSwarm")

class UnifiedFieldSwarm(SubconsciousUnit):
    """
    Swarm Intelligence 2.0: The Unified Field.
    Restored from legacy 'ScalpSwarm'.
    
    Models market price as a particle moving through a fluid field of Order Flow.
    Integrates:
    1. Kinematic Velocity (Particle Speed)
    2. Field Pressure (Order Flow/Alpha Potential)
    3. Strange Attractors (Chaos/Lyapunov)
    4. Entropy-Dynamic Weighting
    """
    def __init__(self):
        super().__init__("Unified_Field_Swarm")
        self.cooldown = 0
        self.threshold = 0.5
        self.last_trade_time = 0

    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        # Extract Context
        tick = context.get('tick')
        df_m5 = context.get('df_m5')
        df_m1 = context.get('df_m1')
        
        # We need these scores. If not available, we assume 0 or look in context.
        # Swarm Orchestrator might need to publish these to context first?
        # Or we act on raw data.
        
        # Unified Field relies on pre-calculated scores (alpha, tech, phy).
        # We can approximate them from our own logic or wait for V2 updates.
        # For now, we self-calculate simplified versions.
        
        if not tick or df_m5 is None or df_m1 is None: return None
        
        # 1. Kinematics (Velocity)
        closes = df_m1['close'].values
        if len(closes) < 5: return None
        
        v = (closes[-1] - closes[-5]) / closes[-5] * 1000.0 # Normalized Slope
        v = np.clip(v, -1.0, 1.0)
        
        # 2. Field Pressure (Volume/Imbalance)
        # Simplified Impulse
        vol = df_m1['tick_volume'].values[-1]
        avg_vol = np.mean(df_m1['tick_volume'].values[-5:])
        pressure = (vol / avg_vol) - 1.0 if avg_vol > 0 else 0
        P = np.clip(pressure, -1.0, 1.0)
        
        # 3. Attractor (Mean Reversion vs Trend)
        # We'll use RSI as proxy for Entropy/Attractor for now
        # OR we check if 'KinematicSwarm' output is in context 'thoughts'?
        # We can peek at bus!
        
        # Let's use simple logic: If V is High, Attractor pulls back (Mean Rev) 
        # unless Pressure is High (Breakout).
        
        A = 0
        if abs(v) > 0.8:
            A = -v * 0.5 # Drag
        
        # 4. Unified Vector S
        # w_v=0.4, w_p=0.4, w_a=0.2
        S = (0.4 * v) + (0.4 * P) + (0.2 * A)
        
        # Signal Inversion logic from original file?
        # "S = -S" was user request.
        S = -S 
        
        # Decision
        threshold = 0.4
        
        signal = "WAIT"
        conf = 0.0
        
        if S > threshold:
            signal = "BUY"
            conf = min(99.0, (S - threshold) * 200) # Scale to conf
        elif S < -threshold:
            signal = "SELL"
            conf = min(99.0, (abs(S) - threshold) * 200)
            
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=conf + 50.0, # Base confidence
                timestamp=time.time(),
                meta_data={'S': S, 'v': v, 'P': P, 'A': A}
            )
            
        return None
