
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("HDCSwarm")

class HyperdimensionalSwarm(SubconsciousUnit):
    """
    The Hyperdimensional Brain (Vector Symbolic Architecture).
    Phase 43 Innovation.
    Logic:
    1. Projects low-dimensional features (Price, RSI, Vol) into 10,000-dim Hypervectors.
    2. Operations:
       - MAP (Encoding): Feature Value -> Hypervector.
       - BUNDLE (Superposition): Combining features into a State Vector.
       - BIND (XOR): Binding Variable to Value.
    3. Reasoning:
       - Compare State Vector with "Archetypes" (Ideal Bull, Ideal Bear).
       - Metric: Cosine Similarity (or Hamming Distance).
    """
    def __init__(self):
        super().__init__("HyperdimensionalSwarm")
        self.dim = 10000
        # Initialize Random Projection Matrix (Orthogonal Memory)
        # We need simpler approach efficiently in Python.
        # Fixed Codebook for features.
        np.random.seed(42)
        self.archetypes = {
            "BULL": self._random_hv(),
            "BEAR": self._random_hv(),
            "CHAOS": self._random_hv()
        }
        
        # We need Item Memories for encoding continuous values
        # We'll use a simple bucket method: 10 levels of RSI, 10 levels of Trend.
        self.rsi_memory = [self._random_hv() for _ in range(10)]
        self.trend_memory = [self._random_hv() for _ in range(10)]
        
        # Learn Superposition (Bundling)
        # Ideally we learn these over time, but we pre-seed them for now.
        # "Ideal Bull" = High RSI + Positive Trend
        # We manually construct the archetypes by bundling properties.
        
        # Construct BULL Archetype: RSI > 70 (indices 7,8,9) + Trend Positive (indices 6-9)
        bull_props = self.rsi_memory[7] + self.rsi_memory[8] + self.trend_memory[7] + self.trend_memory[8]
        self.archetypes['BULL'] = self._binarize(bull_props)
        
        # Construct BEAR Archetype: RSI < 30 (indices 0,1,2) + Trend Negative (indices 0-3)
        bear_props = self.rsi_memory[1] + self.rsi_memory[2] + self.trend_memory[1] + self.trend_memory[2]
        self.archetypes['BEAR'] = self._binarize(bear_props)

    def _random_hv(self):
        # Bipolar {-1, 1} is better for bundling than {0, 1}
        return np.random.choice([-1, 1], size=self.dim)
        
    def _binarize(self, vector):
        # Threshold at 0 for bipolar
        return np.where(vector > 0, 1, -1)
        
    def _cosine_sim(self, v1, v2):
        # For bipolar vectors, cosine sim is related to Hamming distance
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        # Feature Extraction
        # 1. RSI (Approximate)
        closes = df['close'].values
        deltas = np.diff(closes)
        gain = deltas[deltas > 0].mean() if len(deltas[deltas > 0]) > 0 else 0
        loss = -deltas[deltas < 0].mean() if len(deltas[deltas < 0]) > 0 else 1e-9
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Trend Slope (Normalized)
        slope = (closes[-1] - closes[-10]) / (closes[-10] + 1e-9) * 100
        # Normalize slope roughly -0.5% to +0.5%
        
        # Encoding (Mapping to Indices 0-9)
        rsi_idx = int(np.clip(rsi / 10, 0, 9))
        
        # Slope mapping: -0.2 to 0.2
        # -0.2 -> 0, 0 -> 5, +0.2 -> 9
        slope_norm = (slope + 0.2) / 0.4 * 10
        slope_idx = int(np.clip(slope_norm, 0, 9))
        
        # Construct Query Vector
        # Q = RSI_Vec + Trend_Vec
        query_raw = self.rsi_memory[rsi_idx] + self.trend_memory[slope_idx]
        query_hv = self._binarize(query_raw)
        
        # Associative Search (Compare with Archetypes)
        sim_bull = self._cosine_sim(query_hv, self.archetypes['BULL'])
        sim_bear = self._cosine_sim(query_hv, self.archetypes['BEAR'])
        
        signal = "WAIT"
        confidence = 0.0
        
        # Similarity threshold for HDC (High dimensionality implies orthogonality is ~0)
        # Similar vectors will have > 0.4 similarity clearly.
        
        if sim_bull > 0.3 and sim_bull > sim_bear:
            signal = "BUY"
            confidence = float(sim_bull * 100.0) + 20 # HDC is robust
        elif sim_bear > 0.3 and sim_bear > sim_bull:
            signal = "SELL"
            confidence = float(sim_bear * 100.0) + 20
            
        if signal != "WAIT":
             return SwarmSignal(
                source="HyperdimensionalSwarm",
                signal_type=signal,
                confidence=min(100.0, confidence),
                timestamp=0,
                meta_data={
                    "hdc_bull_sim": float(sim_bull),
                    "hdc_bear_sim": float(sim_bear),
                    "concept": "Bullish State" if signal == "BUY" else "Bearish State"
                }
            )
            
        return None
