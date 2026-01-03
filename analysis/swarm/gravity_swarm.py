
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("GravitySwarm")

class GravitySwarm(SubconsciousUnit):
    """
    The Gravity Well (Physics Engine).
    Phase 33 Innovation.
    Logic:
    1. Models Market as a Gravity System.
    2. Mass = Volume at Price (Volume Profile).
    3. POC (Point of Control) = Supermassive Black Hole (Strongest Attractor).
    4. Kinetic Energy = Volatility * Volume.
    
    States:
    - ORBIT: Price is stuck in a Gravity Well (Range). -> Mean Reversion to POC.
    - ESCAPE: Price validates Escape Velocity (Breakout). -> Trend Following.
    - FREE FALL: Price is moving to next Attractor.
    """
    def __init__(self):
        super().__init__("GravitySwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_h1') # Use H1 for Macro Gravity
        if df is None or len(df) < 100: return None
        
        tick = context.get('tick')
        current_price = tick.get('bid')
        
        # 1. Calculate Volume Profile (Simplified)
        # We bin prices into 50 zones.
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, 50)
        
        # Digitize
        df['bin'] = np.digitize(df['close'], bins)
        
        # Group by bin and sum volume
        volume_profile = df.groupby('bin')['volume'].sum()
        
        # Find Point of Control (POC) - The bin with max volume
        poc_bin = volume_profile.idxmax()
        # Approximate price of POC (midpoint of bin)
        bin_width = bins[1] - bins[0]
        poc_price = bins[poc_bin-1] + (bin_width / 2)
        
        # 2. Calculate Gravity Force
        # F = G * (M1 * M2) / r^2
        # M1 (Market Mass) = Total Volume of POC
        # M2 (Price Mass) = 1 (Unit)
        # r = Distance from POC
        
        dist = current_price - poc_price
        r = abs(dist)
        if r == 0: r = 0.0001
        
        mass = volume_profile.max()
        gravity = mass / (r**2) # Simplified Force
        
        # 3. Calculate Kinetic Energy (Momentum)
        # KE = 0.5 * m * v^2
        # Velocity = Recent Change
        velocity = df['close'].diff().tail(3).mean()
        ke = 0.5 * abs(velocity)**2
        
        # 4. Decision Logic
        signal = "WAIT"
        confidence = 0.0
        details = ""
        
        # Thresholds (Conceptual)
        escape_velocity_threshold = 5.0 # Need dynamic calibration
        
        if r < (bin_width * 3):
            # We are near POC (In the Well)
            if ke < escape_velocity_threshold:
                # Low Energy inside Well -> ORBIT -> Revert to POC
                # If price > POC, Sell. If price < POC, Buy.
                if dist > 0: 
                    signal = "SELL"
                    confidence = 60.0
                    details = "Gravity Pull (Orbiting POC)"
                else: 
                    signal = "BUY"
                    confidence = 60.0
                    details = "Gravity Pull (Orbiting POC)"
            else:
                # High Energy inside Well -> Potential ESCAPE
                # Don't fade it.
                pass
        else:
            # We are far from POC.
            # Are we escaping or falling back?
            # If Velocity matches Direction away from POC -> ESCAPE
            
            moving_away = (dist > 0 and velocity > 0) or (dist < 0 and velocity < 0)
            
            if moving_away and ke > escape_velocity_threshold:
                 signal = "BUY" if velocity > 0 else "SELL"
                 confidence = 85.0
                 details = "ESCAPE VELOCITY CONFIRMED. Leaving Gravity Well."
            elif not moving_away:
                 # Falling back?
                 # Could be a deep mean reversion or just a pullback.
                 pass

        if signal != "WAIT":
            return SwarmSignal(
                source="GravitySwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={"reason": details, "POC": poc_price, "KE": ke}
            )
            
        return None
