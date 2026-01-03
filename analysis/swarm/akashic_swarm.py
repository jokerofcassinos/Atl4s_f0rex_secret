
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SwarmSignal, SubconsciousUnit
from scipy.spatial.distance import cdist

logger = logging.getLogger("AkashicSwarm")

class AkashicSwarm(SubconsciousUnit):
    """
    The Akashic Records (Vector Memory).
    Phase 39 Innovation.
    Logic:
    1. Analogical Reasoning: "History doesn't repeat, but it rhymes."
    2. Encodes Market State into a Vector: [RSI, Volatility, Hurst, Gravity, CyclePhase].
    3. Stores vectors in a dynamic memory bank.
    4. Retrieval: Finds k-Nearest Neighbors (Euclidean Distance).
    5. Prediction: Weighted average of the *outcome* of those neighbors.
    """
    def __init__(self):
        super().__init__("AkashicSwarm")
        self.memory_bank = [] # List of vectors
        self.outcomes = [] # List of resulting price moves (10 bars later)
        self.pending_states = [] # Queue to assign outcomes later
        self.vector_dim = 5 # RSI, ATR, Hurst, Gravity, Phase

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        # 1. Feature Extraction (The State Vector)
        # We need normalized features (0-1 range roughly) for Euclidean distance to work well.
        
        # Feature A: RSI (0-100 -> 0-1)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        feat_rsi = rsi / 100.0
        
        # Feature B: Volatility (ATR%) normalized?
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        price = df['close'].iloc[-1]
        feat_vol = (atr / price) * 1000 # Scaling factor
        
        # Feature C: Trend Slope (Linear Reg)
        y = df['close'].tail(10).values
        x = np.arange(10)
        slope = np.polyfit(x, y, 1)[0]
        feat_slope = slope / price * 10000
        
        # Feature D: Relative Position in Range (Stoch-like)
        low_50 = df['low'].tail(50).min()
        high_50 = df['high'].tail(50).max()
        if high_50 == low_50: return None
        feat_pos = (price - low_50) / (high_50 - low_50)
        
        # Feature E: Volume Intensity
        vol_avg = df['volume'].tail(20).mean()
        vol_curr = df['volume'].iloc[-1]
        feat_vol_int = vol_curr / vol_avg if vol_avg > 0 else 0
        
        current_vector = np.array([feat_rsi, feat_vol, feat_slope, feat_pos, feat_vol_int])
        
        # 2. Memory Management (Learning)
        # Store current state in "Pending", to be resolved in future
        # We want to know outcome 5 bars later.
        self.pending_states.append({
            'vector': current_vector,
            'price_at_entry': price,
            'timestamp': df.index[-1],
            'bars_unseen': 5
        })
        
        # Resolve Pending States
        # Check old pending states
        # In live run, we check time. In backtest, we just use deque logic.
        # Here we assume 'process' is called sequentially.
        # But wait, we can't resolve "Future" instantly in live.
        # We need to resolve *Previous* pendings using *Current* price.
        
        resolved_indices = []
        for i, p in enumerate(self.pending_states):
            p['bars_unseen'] -= 1
            if p['bars_unseen'] <= 0:
                # Outcome revealed!
                outcome = (price - p['price_at_entry']) / p['price_at_entry'] * 10000 # Basis points
                self.memory_bank.append(p['vector'])
                self.outcomes.append(outcome)
                resolved_indices.append(i)
                
        # Remove resolved (reverse order)
        for i in sorted(resolved_indices, reverse=True):
            del self.pending_states[i]
            
        # Limit Memory Size (Sliding Window of Experience)
        max_mem = 2000
        if len(self.memory_bank) > max_mem:
            self.memory_bank = self.memory_bank[-max_mem:]
            self.outcomes = self.outcomes[-max_mem:]
            
        # 3. Recall (k-NN)
        if len(self.memory_bank) < 10: return None # Need data
        
        # Convert to matrix
        memory_matrix = np.array(self.memory_bank)
        
        # Calculate Distances
        # cdist expects 2D arrays
        dists = cdist([current_vector], memory_matrix, metric='euclidean')[0]
        
        # Find k nearest
        k = 5
        # argpartition is faster than sort
        if len(dists) < k: k = len(dists)
        nearest_indices = np.argpartition(dists, k)[:k]
        
        # 4. Prediction
        nearby_outcomes = [self.outcomes[i] for i in nearest_indices]
        avg_outcome = np.mean(nearby_outcomes)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Thresholds (Basis points)
        if avg_outcome > 5.0: # Expect +5 bps move
            signal = "BUY"
            confidence = min(90.0, 50.0 + abs(avg_outcome)*2)
            reason = f"Analogical Recall: Positive Precedents (Avg: +{avg_outcome:.2f} bps)"
        elif avg_outcome < -5.0:
            signal = "SELL"
            confidence = min(90.0, 50.0 + abs(avg_outcome)*2)
            reason = f"Analogical Recall: Negative Precedents (Avg: {avg_outcome:.2f} bps)"
            
        if signal != "WAIT":
            return SwarmSignal(
                source="AkashicSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={"reason": reason, "neighbors": k, "avg_outcome": avg_outcome}
            )
            
        return None
