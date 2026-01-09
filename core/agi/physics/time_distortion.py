
import logging
import time
import numpy as np
from collections import deque
from typing import Dict, Any

logger = logging.getLogger("TimeDistortion")

class TimeDistortionEngine:
    """
    Phase 19: Time Distortion Engine (The Pulse).
    Measures the subjective 'speed' of market time relative to clock time.
    
    Concepts:
    - Tick Velocity: Ticks per second (TPS).
    - Time Warp: When TPS exceeds statistical norms (HFT Activity).
    - Dilated Time: Low TPS (Dead Market).
    """
    def __init__(self, window_seconds: int = 20):
        self.window_seconds = window_seconds
        self.timestamps = deque()
        self.velocities = deque(maxlen=60) # 1 minute history of velocities
        self.baseline_velocity = 1.0
        
    def process_tick(self, tick: Dict) -> Dict:
        """
        Ingests a tick and returns the current Time State.
        """
        now = time.time()
        self.timestamps.append(now)
        
        # Prune old timestamps
        while self.timestamps and (now - self.timestamps[0] > self.window_seconds):
            self.timestamps.popleft()
            
        # Calculate Instant Velocity (TPS)
        count = len(self.timestamps)
        # Avoid division by zero if window is empty? No, window starts filling.
        # We normalize to window size. If window is 20s, and we have 20 ticks, that's 1 TPS.
        # Wait, if we just started, time delta might be small.
        # Let's just use count / window_seconds as a rolling average approximation (assuming full window).
        # Actually better: count / actual_time_span if span > 1s
        
        velocity = 0.0
        if len(self.timestamps) > 1:
            span = self.timestamps[-1] - self.timestamps[0]
            if span > 1.0:
                 velocity = count / span
            else:
                 velocity = count # Surge in <1s
        
        self.velocities.append(velocity)
        
        # Update Baseline (Moving Average of velocities)
        if len(self.velocities) > 0:
            self.baseline_velocity = float(np.mean(self.velocities))
            
        # Avoid zero baseline
        if self.baseline_velocity < 0.1: self.baseline_velocity = 0.1
        
        # Calculate Warp Factor (Ratio instant / baseline)
        if self.baseline_velocity > 0:
            warp_factor = velocity / self.baseline_velocity
        else:
            warp_factor = 1.0
            
        # Determine State
        state = "NORMAL"
        if warp_factor > 3.0:
            state = "WARP_EVENT" # HFT Surge
        elif warp_factor > 1.5:
            state = "ACCELERATED"
        elif warp_factor < 0.5:
            state = "DILATED" # Dead
            
        # Special HFT Detection
        # If absolute velocity > 10 TPS (depends on broker)
        if velocity > 5.0:
             state = "HFT_ACTIVITY"
             
        return {
            'velocity': velocity,
            'baseline': self.baseline_velocity,
            'warp_factor': warp_factor,
            'time_state': state
        }
