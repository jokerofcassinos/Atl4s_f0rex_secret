
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("NavierStokesSwarm")

class NavierStokesSwarm(SubconsciousUnit):
    """
    Phase 73: The Navier-Stokes Swarm (Fluid Dynamics).
    
    Treats the market as a Fluid Medium.
    Uses the Reynolds Number (Re) to classify Market Regime into:
    1. Laminar Flow (Smooth Trend, Low Re).
    2. Turbulent Flow (Choppy Chaos, High Re).
    
    Formula: Re = (Density * Velocity * Length) / Viscosity
    - Density (rho): Volume / Time (Market Thickness).
    - Velocity (v): Price Change / Time (Momentum).
    - Length (L): ATR (Characteristic Scale).
    - Viscosity (mu): Spread + Resistance (Friction).
    """
    def __init__(self):
        super().__init__("Navier_Stokes_Swarm")
        self.critical_re_lower = 2000 # Limit for Laminar
        self.critical_re_upper = 4000 # Onset of Turbulence

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        tick = context.get('tick')
        
        if df_m5 is None or len(df_m5) < 20: return None
        
        # 1. Physics Parameters
        
        # Velocity (v): Log Return of close
        last_close = df_m5['close'].iloc[-1]
        prev_close = df_m5['close'].iloc[-2]
        velocity = abs(np.log(last_close / prev_close)) * 10000 # Scale up for float
        
        # Characteristic Length (L): ATR-14
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        closes = df_m5['close'].values
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        length_scale = np.mean(tr[-14:])
        
        # Density (rho): Volume (Flow Rate)
        # Verify if volume exists, else use 1.0 (Incompressible Flow approximation)
        if 'tick_volume' in df_m5.columns:
            volume = df_m5['tick_volume'].iloc[-1]
        else:
            volume = 1000.0 # Default density
            
        density = np.log1p(volume) # Log density to dampen massive spikes
        
        # Viscosity (mu): Friction
        # Spread is the primary friction.
        spread = tick['ask'] - tick['bid'] if tick else 0.0001
        viscosity = max(spread, 0.0001) * 10000 # Scale to match velocity
        
        # 2. Reynolds Number Calculation
        # Re = (rho * v * L) / mu
        # Dimensions analysis is tricky with financial data, so we look for relative values.
        
        numerator = density * velocity * length_scale
        denominator = viscosity
        
        reynolds_number = numerator / denominator if denominator > 0 else 99999
        
        # 3. Flow Regime Classification
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        regime = "TRANSITIONAL"
        
        if reynolds_number < self.critical_re_lower:
            regime = "LAMINAR"
            # Laminar Flow = Smooth Trend.
            # Strategy: Follow the Velocity Vector.
            
            # Recalculate signed velocity to know direction
            signed_v = last_close - prev_close
            
            if signed_v > 0:
                signal = "BUY"
                confidence = 75.0
                reason = f"NAVIER-STOKES: Laminar Flow (Re={reynolds_number:.0f}). Smooth Uptrend."
            elif signed_v < 0:
                signal = "SELL"
                confidence = 75.0
                reason = f"NAVIER-STOKES: Laminar Flow (Re={reynolds_number:.0f}). Smooth Downtrend."
                
        elif reynolds_number > self.critical_re_upper:
            regime = "TURBULENT"
            # Turbulent Flow = Chaotic Eddies.
            # Strategy: Mean Reversion or Avoid.
            # Momentum will likely dissipate into heat (losses).
            
            signal = "WAIT" # Adviser: High Risk
            confidence = 0.0 
            # Or Mean Reversion Signal?
            # Creating a Veto Signal for Momentum Strategies
            reason = f"NAVIER-STOKES: Turbulent Flow (Re={reynolds_number:.0f}). Chaos Detected."
            
        else:
            # Transitional
            pass

        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'reynolds': reynolds_number, 'regime': regime, 'reason': reason}
            )
        
        # If Turbulent, we might want to return a Neutral Signal with Metadata to warn others
        if regime == "TURBULENT":
             return SwarmSignal(
                source=self.name,
                signal_type="WAIT", # Or "HOLD"
                confidence=50.0,
                timestamp=time.time(),
                meta_data={'reynolds': reynolds_number, 'regime': regime, 'reason': reason}
            )

        return None
