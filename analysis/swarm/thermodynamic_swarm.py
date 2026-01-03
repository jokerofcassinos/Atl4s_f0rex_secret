
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit
from scipy.stats import entropy

logger = logging.getLogger("ThermoSwarm")

class ThermodynamicSwarm(SubconsciousUnit):
    """
    Maxwell's Demon (Information Theory & Thermodynamics).
    Phase 42 Innovation.
    Logic:
    1. Measures the Shannon Entropy of the Price Action.
    2. Concept:
       - Low Entropy (Order) = Compressible Data = Strong Trend.
       - High Entropy (Disorder) = Random Data = Chop/Noise.
    3. Metrics:
       - Shannon Entropy (H)
       - Market Temperature (Vol * Entropy)
    """
    def __init__(self):
        super().__init__("ThermodynamicSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        # 1. Prepare Returns Stream
        close = df['close'].values
        returns = np.diff(close)
        
        # 2. Discretize Returns (Binning) for Histogram
        # We need a probability distribution p(x)
        # 100 bins to capture granularity
        hist, bin_edges = np.histogram(returns, bins=50, density=True)
        
        # Normalize histogram to be a probability mass function (sum=1)
        # Note: density=True makes area=1, but we need sum(p)=1 for discrete entropy
        # Simple approach: count occurrences
        counts, _ = np.histogram(returns, bins=50)
        probs = counts / np.sum(counts)
        # Filter zero probs to avoid log(0)
        probs = probs[probs > 0]
        
        # 3. Calculate Shannon Entropy H(X)
        # Max Entropy (Uniform Distribution) = log(N_bins) = log(50) approx 3.9
        S = entropy(probs)
        max_S = np.log(len(counts))
        
        # Normalized Entropy (Efficiency)
        # 0.0 = Perfectly Ordered (One bin)
        # 1.0 = Perfectly Random (Uniform)
        efficiency = S / max_S
        
        # Efficiency < 0.6 implies Order (Trend)
        # Efficiency > 0.9 implies Chaos
        
        # 4. Market Temperature (Kinetic Energy)
        # Volatility scales the impact of entropy.
        vol = np.std(returns)
        temperature = efficiency * vol
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Logic: Maxwell's Demon sorts the molecules (ticks)
        
        # Condition 1: Low Entropy (Crystallization/Order)
        if efficiency < 0.75:
            # The market is ORDERED. Trend Following is safe.
            # Determine direction by mean return
            mean_ret = np.mean(returns[-10:])
            
            if mean_ret > 0:
                signal = "BUY"
                # Lower Entropy = Higher Confidence
                confidence = (1.0 - efficiency) * 100.0 + 30 # Base boost
                reason = f"Ordered State (Entropy {efficiency:.2f}): Upward Flow"
            elif mean_ret < 0:
                signal = "SELL"
                confidence = (1.0 - efficiency) * 100.0 + 30
                reason = f"Ordered State (Entropy {efficiency:.2f}): Downward Flow"
                
        # Condition 2: High Entropy (Gas/Plasma)
        else:
            # Market is DISORDERED. 
            # If Temperature is High -> Explosion/Breakout possible? Or just Noise?
            # Usually Noise.
            reason = f"High Entropy ({efficiency:.2f}): Thermal Noise"
            confidence = 10.0 # Low confidence
            
        # Boost Confidence if "Cold" (Low Temp) and Ordered (Low Entropy)
        # Super-Conductive State
        
        if signal != "WAIT":
            return SwarmSignal(
                source="ThermodynamicSwarm",
                signal_type=signal,
                confidence=min(100.0, confidence),
                timestamp=0,
                meta_data={
                    "entropy": efficiency,
                    "temperature": temperature,
                    "state": "Ordered" if efficiency < 0.75 else "Disordered"
                }
            )
            
        return None
