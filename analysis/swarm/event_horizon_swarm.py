
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("EventHorizonSwarm")

class EventHorizonSwarm(SubconsciousUnit):
    """
    Phase 77: The Event Horizon Swarm (General Relativity).
    
    Models the market using Orbital Mechanics and General Relativity.
    The VWAP (Volume Weighted Average Price) is the 'Black Hole' or 'Sun' (Center of Gravity).
    
    Physics:
    - Mass (M): Accumulated Volume recently.
    - Radius (r): Distance from Price to VWAP.
    - Gravity (g): Pull towards VWAP = G * M / r^2.
    - Escape Velocity (ve): Sqrt(2 * G * M / r).
    
    Logic:
    - If Price Momentum (Velocity) > Escape Velocity, we have a TRUE BREAKOUT (Escape).
    - If Price Momentum < Escape Velocity, gravity wins, and Price returns to VWAP (Mean Reversion).
    """
    def __init__(self):
        super().__init__("Event_Horizon_Swarm")
        self.big_g = 0.0001 # Gravitational Constant (tuned)

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Calculate Center of Mass (VWAP)
        # We assume the 'Star' is the session VWAP or rolling VWAP (50 periods)
        closes = df_m5['close'].values
        
        if 'tick_volume' in df_m5.columns:
             volumes = df_m5['tick_volume'].replace(0, 1).values
        elif 'Volume' in df_m5.columns:
             volumes = df_m5['Volume'].replace(0, 1).values
        else:
             volumes = np.ones(len(closes)) # Assume uniform mass if unknown
             
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        typical_price = (highs + lows + closes) / 3
        
        # Cumulative VWAP over last 50 candles (Local Gravity Well)
        # sum(price * vol) / sum(vol)
        vwap_numerator = np.sum(typical_price[-50:] * volumes[-50:])
        vwap_denominator = np.sum(volumes[-50:])
        vwap = vwap_numerator / vwap_denominator
        
        # 2. Physics Parameters
        
        # Mass (M): Total Volume in the system (Strength of the Gravity Well)
        # Normalized relative to average volume to keep G constant effective
        avg_vol = np.mean(volumes[-200:]) if len(volumes) > 200 else np.mean(volumes)
        mass = vwap_denominator / avg_vol # Relative Mass
        
        # Radius (r): Distance from Current Price to VWAP
        current_price = closes[-1]
        r = abs(current_price - vwap)
        
        # Avoid division by zero (Singularity)
        if r < 0.00001: r = 0.00001
        
        # 3. Calculate Escape Velocity
        # ve = sqrt(2 * G * M / r)
        # Note: In physics, ve drops as r increases.
        # Here: The further away we are, the 'weaker' gravity is?
        # Actually financial gravity (Mean Reversion) acts like a Spring (Hubble's Law / Hooke's Law).
        # Force increases with distance? NO, usually VWAP pull is strong when close, but elastic limit breaks.
        # Let's stick to standard Orbital Mechanics.
        # Gravity is stronger closer to VWAP.
        
        # BUT, for a breakout, we need to escape the "Orbit".
        # Let's use Potential Energy. U = -GM/r.
        
        # Simplified:
        # To break away, Kinetic Energy > Potential Energy.
        # 1/2 v^2 > G*M / r_base?
        
        # Actually, let's look at it as:
        # Gravity Force Fg = G * M * r (Hooke's Law for Markets) instead of 1/r^2?
        # Markets act like springs (Mean Reverting).
        # So Force pulls HARDER the further you get, until it snaps.
        
        # Let's model Escape Velocity from the "Elastic Band" model.
        # Breakout Energy required = k * x^2
        
        # Alternative: Event Horizon logic.
        # If r < Schwarzschild Radius, light cannot escape.
        # Rs = 2GM / c^2.
        
        # Let's use the standard "Orbital" check.
        # Are we in Orbit (Reversion) or Escaping (Trend)?
        
        # Current Velocity (v)
        velocity = abs(closes[-1] - closes[-2])
        
        # Escape Velocity (Empirical)
        # If we are far (r is large), Gravity is HIGH (Spring).
        # We need massive velocity to continue.
        
        escape_vel = self.big_g * mass * r # Force proportional to distance (Spring)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 4. Relativistic Logic
        
        if velocity > escape_vel:
            # We have enough momentum to stretch the spring further or break it.
            # ESCAPE TRAJECTORY (Breakout)
            direction = "UP" if closes[-1] > closes[-2] else "DOWN"
            
            if direction == "UP" and current_price > vwap:
                 signal = "BUY"
                 confidence = 85.0
                 reason = f"EVENT HORIZON: Escape Velocity Occured. (v={velocity:.5f} > ve={escape_vel:.5f}). Breakout."
            elif direction == "DOWN" and current_price < vwap:
                 signal = "SELL"
                 confidence = 85.0
                 reason = f"EVENT HORIZON: Escape Velocity Occured. (v={velocity:.5f} > ve={escape_vel:.5f}). Breakout."
                 
        else:
            # Velocity is too low to fight Gravity.
            # ORBITAL DECAY (Mean Reversion)
            # Price should fall back to VWAP.
            
            # Only trade this if we are significantly far away (Potential Energy is high)
            volatility_atr = np.mean(highs[-14:] - lows[-14:])
            
            if r > (2 * volatility_atr): # 2 ATR away from VWAP
                 # Imminent Collapse back to center
                 if current_price > vwap:
                      signal = "SELL" # Revert to VWAP
                      confidence = 80.0
                      reason = f"EVENT HORIZON: Orbital Decay. Gravity ({escape_vel:.5f}) > Velocity ({velocity:.5f}). Reversion."
                 else:
                      signal = "BUY" # Revert to VWAP
                      confidence = 80.0
                      reason = f"EVENT HORIZON: Orbital Decay. Gravity ({escape_vel:.5f}) > Velocity ({velocity:.5f}). Reversion."

        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'escape_vel': escape_vel, 'velocity': velocity, 'reason': reason}
            )
            
        return None
