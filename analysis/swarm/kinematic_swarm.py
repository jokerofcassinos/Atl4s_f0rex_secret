
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("KinematicSwarm")

class KinematicSwarm(SubconsciousUnit):
    """
    Phase 48: The Kinematic Swarm.
    Analyzes the Physics of Market Motion.
    
    Concepts:
    - Velocity (v): Speed of price change (Momentum).
    - Acceleration (a): Rate of change of velocity (Force).
    - Jerk (j): Rate of change of acceleration (Shock).
    
    Logic:
    - Falling Knife: v < 0, a < 0 (Accelerating Down) -> SELL / EXIT BUY.
    - Rocket: v > 0, a > 0 (Accelerating Up) -> BUY / EXIT SELL.
    - Exhaustion: v > 0, a < 0 (Going up but slowing down) -> PREPARE REVERSAL.
    """
    def __init__(self):
        super().__init__("KinematicSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m1') # Lowest latency
        if df is None or len(df) < 10: return None
        
        # 1. Calculate Physics
        closes = df['close']
        
        # Velocity (First Derivative)
        velocity = closes.diff()
        
        # Acceleration (Second Derivative)
        acceleration = velocity.diff()
        
        # Jerk (Third Derivative)
        jerk = acceleration.diff()
        
        # Current State (Last closed candle)
        v = velocity.iloc[-1]
        a = acceleration.iloc[-1]
        j = jerk.iloc[-1]
        
        # Real-time Component (Tick vs Last Close)
        tick = context.get('tick', {})
        current_bid = tick.get('bid', 0)
        last_close = closes.iloc[-1]
        
        realtime_v = 0
        if current_bid > 0:
            realtime_v = current_bid - last_close
            
        # Combine Historical and Realtime
        # If Realtime contradicts History strongly, trust Realtime (The "Jerk" of the Now)
        
        signal = None
        confidence = 0.0
        reason = ""
        
        # --- LOGIC GATES ---
        
        # 1. STRONG DOWNTREND (Crash/Correction)
        # Velocity Negative AND Accelerating Down OR Realtime dropping hard
        if (v < 0 and a < 0) or (realtime_v < -5.0): # $5 drop instantly
            signal = "SELL"
            confidence = 85.0
            if realtime_v < -10.0: confidence = 95.0 # Panic Drop
            reason = f"Kinematics: Accelerating Down (v={v:.2f}, a={a:.2f}, rt_v={realtime_v:.2f})"
            
        # 2. STRONG UPTREND (Breakout)
        elif (v > 0 and a > 0) or (realtime_v > 5.0):
            signal = "BUY"
            confidence = 85.0
            if realtime_v > 10.0: confidence = 95.0 # Moon Shot
            reason = f"Kinematics: Accelerating Up (v={v:.2f}, a={a:.2f}, rt_v={realtime_v:.2f})"
            
        # 3. EXHAUSTION (Turnaround)
        elif v > 0 and a < 0:
            # Going up, but slowing down (Gravity taking over)
            # Weak Signal or Exit Signal
            # For now, let's just not Buy.
            pass
            
        if signal:
            return SwarmSignal(
                source="KinematicSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'velocity': v, 'acceleration': a, 'jerk': j, 'reason': reason}
            )
            
        return None
