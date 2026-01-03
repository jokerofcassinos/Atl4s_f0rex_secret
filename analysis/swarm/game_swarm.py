
import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("GameSwarm")

class GameSwarm(SubconsciousUnit):
    """
    Phase 92: The Nash Swarm 2.0 (Evolutionary Game Theory).
    
    Models the market as an Evolutionary Game with Replicator Dynamics.
    
    Physics/Math:
    - Replicator Equation: dx_i/dt = x_i * (f_i(x) - phi(x))
    - x_i: Frequency of strategy i in population.
    - f_i: Fitness of strategy i (Payoff).
    - phi: Average fitness of population.
    
    Market Application:
    - Species: Bulls (B) and Bears (S).
    - Population State (x): Share of Volume Flow.
    - Fitness (f): Realized Price Movement. 
      - If Price UP: Bulls gain fitness, Bears lose.
      - If Price DOWN: Bears gain fitness, Bulls lose.
      
    Signals:
    1. ESS (Evolutionary Stable Strategy):
       - Dominant species has Positive Growth (dx/dt > 0).
       - MEANS: The trend is reinforced by outcomes. (STRONG TREND).
       
    2. Mutant Invasion (Instability):
       - Dominant species has Negative Growth (dx/dt < 0).
       - MEANS: Despite volume dominance, the payout is failing. (DIVERGENCE/REVERSAL).
    """
    def __init__(self):
        super().__init__("Game_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 30: return None
        
        # 1. Determine Populations (x)
        # We estimate Bull/Bear volume split using close-open delta
        # Pure buy volume vs Pure sell volume is hard without tick data,
        # so we approximate:
        # If Close > Open: Bull Vol = Volume, Bear Vol = 0
        # If Close < Open: Bear Vol = Volume, Bull Vol = 0
        # Smooth this over time to get a "Population" state.
        
        closes = df_m5['close'].values
        opens = df_m5['open'].values
        volumes = df_m5['volume'].values
        
        rolling_window = 10
        
        bull_vols = np.where(closes > opens, volumes, 0)
        bear_vols = np.where(closes < opens, volumes, 0)
        
        # Population Fractions (x) - Moving Average
        # We look at the SEQUENCE of x states to determine dx/dt
        
        limit = len(closes)
        if limit < rolling_window + 5: return None
        
        # Calculate Rolling Sums
        bull_series = pd.Series(bull_vols).rolling(window=rolling_window).sum()
        bear_series = pd.Series(bear_vols).rolling(window=rolling_window).sum()
        total_series = bull_series + bear_series
        
        # Avoid div by zero
        x_B = (bull_series / total_series.replace(0, 1)).values
        x_S = (bear_series / total_series.replace(0, 1)).values
        
        # 2. Determine Fitness (f)
        # Fitness is simply the Price Return over the same window.
        # But specifically:
        # f_B (Bull Fitness) = % Change. (Positive if Up, Negative if Down)
        # f_S (Bear Fitness) = -% Change. (Positive if Down, Negative if Up)
        
        price_change = pd.Series(closes).pct_change(periods=rolling_window).values * 100
        
        # 3. Analyze Dynamics at current time
        idx = -1
        
        curr_xB = x_B[idx]
        curr_xS = x_S[idx]
        
        curr_fB = price_change[idx]
        curr_fS = -price_change[idx]
        
        # Average Fitness phi
        phi = curr_xB * curr_fB + curr_xS * curr_fS
        
        # Replicator Dynamics (Growth Rates)
        # dx/dt = x * (f - phi)
        # We don't strictly need to compute this if we just check f vs phi, 
        # but let's calculate the "Pressure".
        
        growth_B = curr_xB * (curr_fB - phi)
        growth_S = curr_xS * (curr_fS - phi)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        meta_data = {}
        
        # 4. Signal Logic
        
        # Case A: Bulls Dominant (x_B > 0.6)
        if curr_xB > 0.6:
            if growth_B > 0:
                # ESS: Bulls are dominant AND growing in fitness.
                # Trend is Healthy.
                signal = "BUY"
                confidence = 80 + (curr_xB * 10)
                reason = f"NASH ESS: Stable Bull Dominance (x={curr_xB:.2f}, growth={growth_B:.4f})"
            elif growth_B < -0.01:
                # INVASION: Bulls are dominant BUT losing fitness.
                # Price is likely stalling or dropping despite Bull Volume.
                # Divergence.
                signal = "SELL" # Contrarian Reversal
                confidence = 75.0
                reason = f"NASH INVASION: Bully Trap. Dominance x={curr_xB:.2f} but Growth Negative."
                
        # Case B: Bears Dominant (x_S > 0.6)
        elif curr_xS > 0.6:
            if growth_S > 0:
                # ESS: Bears are dominant AND growing.
                signal = "SELL"
                confidence = 80 + (curr_xS * 10)
                reason = f"NASH ESS: Stable Bear Dominance (x={curr_xS:.2f}, growth={growth_S:.4f})"
            elif growth_S < -0.01:
                # INVASION: Bear Trap.
                signal = "BUY"
                confidence = 75.0
                reason = f"NASH INVASION: Bear Trap. Dominance x={curr_xS:.2f} but Growth Negative."
                
        meta_data = {
            'x_bull': curr_xB,
            'x_bear': curr_xS,
            'growth_bull': growth_B,
            'reason': reason
        }

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data=meta_data
            )
            
        return None
