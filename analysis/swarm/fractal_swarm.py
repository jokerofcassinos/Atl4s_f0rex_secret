
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("FractalSwarm")

class FractalMicroClimax(SubconsciousUnit):
    """
    Phase 146: Fractal Micro-Climax (The Microscope).
    
    System #16: High-Fidelity Perception.
    Detects exhaustion patterns at the fractal level (Tick/M1).
    Used for 'Sniper' entries to minimize drawdown.
    """
    def __init__(self):
        super().__init__("Fractal_Swarm")
        self.tick_buffer = []
        self.max_ticks = 1000
        
    def process_ticks(self, ticks: List[float]):
        """
        Ingests raw tick prices.
        """
        self.tick_buffer.extend(ticks)
        if len(self.tick_buffer) > self.max_ticks:
            self.tick_buffer = self.tick_buffer[-self.max_ticks:]
            
    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m1 = context.get('data_map', {}).get('M1')
        if df_m1 is None or len(df_m1) < 50: return None
        
        # 1. Calculate Micro-Velocity (Price change per minute)
        # First derivative of price
        velocity = df_m1['close'].diff()
        
        # 2. Calculate Micro-Acceleration (Change in velocity)
        # Second derivative
        acceleration = velocity.diff()
        
        current_vel = velocity.iloc[-1]
        current_acc = acceleration.iloc[-1]
        
        # 3. Detect "Climax" Signature
        # High Velocity + Decelerating Acceleration = Exhaustion
        # Example: Price shoots up (Vel > 0), but slowing down (Acc < 0)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        atr = df_m1['high'].iloc[-1] - df_m1['low'].iloc[-1] # Simple Range
        noise_threshold = atr * 0.1
        
        if abs(current_vel) > noise_threshold:
            # Bullish Climax?
            if current_vel > 0 and current_acc < 0:
                # Price is rising, but 'gravity' is pulling it back (Deceleration)
                if abs(current_acc) > abs(current_vel) * 0.5:
                    signal = "SELL"
                    confidence = 75.0
                    reason = "Fractal Climax: Bullish Exhaustion (Decelerating Rise)"
                    
            # Bearish Climax?
            elif current_vel < 0 and current_acc > 0:
                # Price is falling, but braking (Deceleration/Positive Acc)
                if abs(current_acc) > abs(current_vel) * 0.5:
                    signal = "BUY"
                    confidence = 75.0
                    reason = "Fractal Climax: Bearish Exhaustion (Decelerating Fall)"
                    
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={
                    "velocity": float(current_vel),
                    "acceleration": float(current_acc),
                    "reason": reason
                }
            )
            
        return None
