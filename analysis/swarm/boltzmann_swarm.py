
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time
from scipy.stats import entropy

logger = logging.getLogger("BoltzmannSwarm")

class BoltzmannSwarm(SubconsciousUnit):
    """
    Phase 82: The Boltzmann Swarm (Statistical Mechanics).
    
    Models the market using Thermodynamics.
    Calculates the Helmholtz Free Energy (F) available to perform 'Work' (Price Movement).
    
    Formula: F = U - T*S
    
    Variables:
    - Internal Energy (U): Directed Momentum (The force pushing the trend).
    - Temperature (T): Volatility (The vibrating energy of particles).
    - Entropy (S): Disorder (The randomness of the path).
    
    Logic:
    - High U, Low S -> High Free Energy -> Spontaneous, Clean Trend. (Super-conductive).
    - Low U, High S -> Low Free Energy -> Heat Death (Chop/Range).
    - High T (Volatility) can amplify S (Disorder), killing the trend.
    """
    def __init__(self):
        super().__init__("Boltzmann_Swarm")
        self.lookback = 20

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 30: return None
        
        closes = df_m5['close'].values
        
        # 1. Calculate Internal Energy (U) - Directed Momentum
        # U = Abs(Net Displacement) / Time
        displacement = closes[-1] - closes[-self.lookback]
        U = abs(displacement)
        
        # 2. Calculate Temperature (T) - Volatility
        # T = Standard Deviation of returns or ATR
        # Let's use Sum of Absolute Ranges (Path Length) vs Displacement
        # Actually, T is 'Average Kinetic Energy'.
        # Let's say T = Average True Range / Average Price * 10000 (Basis points)
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        T = np.mean(tr[-self.lookback:]) # Average Volatility
        
        # 3. Calculate Entropy (S) - Disorder
        # We can use the Efficiency Ratio concept inverted, or Shannon Entropy of returns.
        # Let's use Shannon Entropy of the Price Distribution in the window.
        window_closes = closes[-self.lookback:]
        hist, _ = np.histogram(window_closes, bins=10, density=True)
        # Normalize sum to 1 just in case
        hist_probs = hist / np.sum(hist) + 1e-9
        S = entropy(hist_probs)
        
        # Normalize S (0 to 1 scaling roughly)
        max_S = np.log(10)
        S_norm = S / max_S
        
        # 4. Calculate Helmholtz Free Energy (F)
        # F = U - (T * S_factor)
        # We need to balance dimensions.
        # U is in Price Units. T is in Price Units. S is dimensionless.
        # F = U - (T * S_norm * K_boltzmann)
        # We usually want F to be positive for a strong trend? 
        # Actually in physics, systems minimize Free Energy. 
        # But here we invoke "Work Capacity". 
        # Let's define "Available Trend Energy" = U * (1 - S_norm).
        # Less Entropy = More Energy available for Trend.
        # If High Volatility (T) is present, it actually increases disorder's impact?
        
        # Let's implement specific Gibbs logic:
        # Effective Trend Strength = U - (T * S_norm * 2.0)
        # If Trend Strength > 0, we have Order.
        # If Trend Strength < 0, Entropy dominates (Chop).
        
        F = U - (T * S_norm * 2.0)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 5. Thermodynamics Decision
        
        if F > 0:
            # High Free Energy (Ordered State)
            # The trend has enough energy to overcome entropy.
            
            # Determine Direction from Displacement
            direction = "UP" if displacement > 0 else "DOWN"
            
            # Z-Score of F to determine strength
            # Just raw heuristic for now.
            # If F is positive, it means Displacement > (Volatility * Entropy * 2).
            # This requires a very clean move.
            
            if direction == "UP":
                signal = "BUY"
                confidence = 85.0 + (min(F/T, 1.0) * 10) # Scale conf by Energy Ratio
                reason = f"BOLTZMANN: High Free Energy (F={F:.2f}). U={U:.2f} > TS={T*S_norm*2.0:.2f}. Spontaneous Trend."
            else:
                signal = "SELL"
                confidence = 85.0 + (min(F/T, 1.0) * 10)
                reason = f"BOLTZMANN: High Free Energy (F={F:.2f}). U={U:.2f} > TS={T*S_norm*2.0:.2f}. Spontaneous Trend."
                
        else:
            # Low Free Energy (Heat Death)
            # Entropy/Volatility is killing the momentum.
            # Avoid trading or Exit.
            
            # If F is deeply negative, it's total chaos.
            signal = "WAIT"
            confidence = 0.0
            reason = f"BOLTZMANN: Heat Death (F={F:.2f}). Entropy/Vol killing Trend."
            
        # 6. Second Law Check (Entropy Increasing?)
        # If S is rising fast, trend is ending.
        
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'FreeEnergy': F, 'U': U, 'TS': T*S_norm*2.0, 'reason': reason}
            )
            
        return None
