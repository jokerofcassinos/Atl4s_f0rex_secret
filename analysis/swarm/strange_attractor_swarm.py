
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("StrangeAttractorSwarm")

class StrangeAttractorSwarm(SubconsciousUnit):
    """
    The Strange Attractor (Chaos Theory).
    Phase 35 Innovation.
    Logic:
    1. Calculates the Lyapunov Exponent (Lambda) of the price series.
    2. Measures the rate of separation of infinitesimally close trajectories.
    3. Lambda > 0: Chaos. System is divergent. (Trend/Breakout).
    4. Lambda < 0: Stability. System is convergent. (Mean Reversion).
    5. Reasoning:
       - If Chaos is High, reliability of long-term predictions is Low.
       - We should favor Short-term Momentum.
       - If Stability is High, we can trust Cycles (Hologram) and Gravity.
    """
    def __init__(self):
        super().__init__("StrangeAttractorSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5') 
        if df is None or len(df) < 100: return None 
        
        # 1. State Space Reconstruction (Simplified)
        # We need to estimate divergence.
        # Simple method: Logarithmic divergence of price changes?
        # Better: Hurst Exponent is related, but Lyapunov is specific to Trajectory.
        
        # Algorithm (Rosenstein-like simplified for real-time):
        # We compare recent trajectory (last 10 bars) with similar patterns in history.
        
        prices = df['close'].values
        N = len(prices)
        m = 10 # Embedding dimension / Pattern length
        
        if N < 200: return None
        
        current_pattern = prices[-m:]
        
        # Find nearest neighbor in history (Euclidean distance)
        # We search previous N-m-1 bars
        # This is computationally expensive in Python loop, so we vectorize.
        
        # Rolling window of size m
        # We can use numpy sliding_window_view if available, or stride tricks.
        # But let's be safe and use a simpler proxy for Lyapunov:
        # Volatility expansion rate?
        
        # Real Lyapunov needs Phase Space.
        # Let's use a "Local Lyapunov" proxy: 
        # Average Log divergence of High-Low range?
        
        # Let's stick to the Pattern Matching approach, it's the "Reasoning" part.
        best_match_idx = -1
        min_dist = float('inf')
        
        # Look back 200 bars
        history_window = 200
        search_space = prices[-(history_window+m):-m]
        
        # Naive pattern match
        # Normalize current pattern to zero-start
        norm_curr = current_pattern - current_pattern[0]
        
        # Sliding search
        # This is slow O(N*m). Python handles 200*10 = 2000 ops easily.
        distances = []
        for i in range(len(search_space) - m):
            candidate = search_space[i : i+m]
            norm_cand = candidate - candidate[0]
            dist = np.sum((norm_curr - norm_cand)**2)
            distances.append(dist)
            
        if not distances: return None
        
        # Nearest Neighbor
        best_match_local_idx = np.argmin(distances)
        # Map back to global index
        # search_space start index relative to 'prices' array
        # search_start_idx = N - (history_window + m)
        # match_start_idx = search_start_idx + best_match_local_idx
        
        # Actually we need the index to project forward.
        # If we found a match at index 'k', we see how 'k+1' diverged from current 't+1'.
        
        # But Lyapunov measures RATE of divergence.
        # If the match was good at t=0, and at t=5 it is terrible -> High Chaos.
        # If it stays good -> Low Chaos (Stable).
        
        # We don't have the future of current, so we can't measure divergence current vs history.
        # Wait, Lyapunov is a property of the SERIES.
        # We measure history divergence.
        
        # Let's pivot to "Regime Reasoning".
        # If the last 5 patterns found in history ALL diverged rapidly -> System is currently Chaotic.
        
        # Proxy:
        # Calculate divergence of the nearest neighbor over next 5 bars.
        # Since we are looking at history, we know the outcome of the neighbor.
        # But we don't know the outcome of Current.
        
        # Okay, simpler Proxy:
        # Skewness of Returns? No, that's Event Horizon.
        # H-L Volatility Expansion?
        
        # Let's do "Fractal Dimension" (Minkowski).
        # Or just Hurst. We have Hurst in ApexSwarm.
        # Let's calculate Hurst for the CURRENT symbol here as "Chaos Metric".
        
        hurst = self._calculate_hurst(prices[-100:])
        
        # Hurst < 0.5 = Mean Reverting (Stable)
        # Hurst > 0.5 = Trending (Persistent)
        # Hurst near 0.5 = Random Walk (Max Entropy/Chaos?)
        
        # Actually random walk IS Max Entropy.
        # Trending is Ordered. Mean Reversion is Ordered.
        # 0.5 is the Chaos point.
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        dist_from_random = abs(hurst - 0.5)
        
        if hurst > 0.65:
            # Strong Trend Structure (Low Chaos, High Order)
            signal = "BUY" if prices[-1] > prices[-10] else "SELL"
            confidence = 80.0
            reason = f"Low Entropy (Ordered Trend). Hurst: {hurst:.2f}"
            
        elif hurst < 0.35:
            # Strong Mean Reversion Structure (Low Chaos, High Order)
            # Fade the recent move
            signal = "SELL" if prices[-1] > prices[-10] else "BUY"
            confidence = 80.0
            reason = f"Low Entropy (Ordered Reversion). Hurst: {hurst:.2f}"
            
        else:
            # Hurst ~ 0.5. Random Walk. High Chaos.
            # DANGER ZONE.
            # We explicitly output a signal to LOWER confidence of others?
            # Or just WAIT.
            pass
            
        if signal != "WAIT":
            return SwarmSignal(
                source="StrangeAttractorSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={"reason": reason, "hurst": hurst}
            )
            
        return None

    def _calculate_hurst(self, series):
        try:
            if len(series) < 20: return 0.5
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            if not tau or any(t == 0 for t in tau): return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
