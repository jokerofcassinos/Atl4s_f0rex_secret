import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import json
import os
from sklearn.neighbors import NearestNeighbors
import pickle

@dataclass
class RealitySnapshot:
    """
    DNA of a Market Moment.
    Capture TUDO physical, contextual, and internal state.
    """
    # 1. Physical Identifiers (The Body)
    timestamp: float
    price_open: float
    price_high: float
    price_low: float
    price_close: float
    volume: float
    
    # 2. Morphological (The Shape)
    body_size: float
    upper_wick: float
    lower_wick: float
    total_range: float
    
    # 3. Kinematic (The Motion)
    rsi_14: float
    atr_14: float
    sma_200_dist: float  # Distance % from SMA 200
    
    # 4. Chronometric (The Time)
    hour: int
    minute: int
    day_of_week: int
    is_8min_cycle: bool  # True if timestamp % 8 minutes == 0
    
    # 5. Internal/Psychological (The Mind) - Optional placeholders for now
    consensus_score: float = 0.0
    sentiment_score: float = 0.0
    
    # 6. Future Outcome (The Destiny) - Populated initially as None, filled later
    future_10m_change: float = 0.0
    future_max_drawdown: float = 0.0
    future_max_profit: float = 0.0
    outcome_label: str = "UNKNOWN" # 'BULL_WIN', 'BEAR_WIN', 'NEUTRAL'

    def to_vector(self) -> np.array:
        """Returns the numerical feature vector for searching."""
        return np.array([
            self.rsi_14,
            self.body_size,
            self.upper_wick,
            self.lower_wick,
            self.sma_200_dist,
            float(self.hour),
            float(self.minute),
            1.0 if self.is_8min_cycle else 0.0
        ])

class AkashicCore:
    """
    The Memory Bank.
    Stores RealitySnapshots and allows 'Time Travel' querying.
    """
    def __init__(self, storage_path: str = "data/memory/akashic_records_v1.parquet"):
        self.storage_path = storage_path
        self.snapshots: List[RealitySnapshot] = []
        self.feature_matrix = None
        self.knn_engine = None
        self.is_trained = False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        self.load_memory()

    def record_moment(self, snapshot: RealitySnapshot):
        """Adds a moment to short-term memory."""
        self.snapshots.append(snapshot)

    def save_memory(self):
        """Persists memory to disk."""
        if not self.snapshots:
            return
            
        # Convert to DataFrame
        data = [asdict(s) for s in self.snapshots]
        df = pd.DataFrame(data)
        
        df.to_parquet(self.storage_path)
        print(f"🧠 [AKASHIC] Memory saved with {len(df)} records.")
        
        # Re-train search engine
        self.rebuild_index()

    def load_memory(self):
        """Loads memory from disk."""
        if os.path.exists(self.storage_path):
            try:
                df = pd.read_parquet(self.storage_path)
                # Reconstruct snapshots
                self.snapshots = [RealitySnapshot(**row) for row in df.to_dict('records')]
                print(f"🧠 [AKASHIC] Loaded {len(self.snapshots)} ancient memories.")
                self.rebuild_index()
            except Exception as e:
                print(f"⚠️ [AKASHIC] Corrupted memory or empty: {e}")

    def rebuild_index(self):
        """Builds the KD-Tree/Ball-Tree for fast searching."""
        if not self.snapshots:
            return

        # Extract vectors
        vectors = [s.to_vector() for s in self.snapshots]
        self.feature_matrix = np.array(vectors)
        
        # Initialize KNN (High Performance Search)
        # Using BallTree for higher dimensions efficiency
        self.knn_engine = NearestNeighbors(n_neighbors=50, algorithm='ball_tree')
        self.knn_engine.fit(self.feature_matrix)
        self.is_trained = True
        print("🧠 [AKASHIC] Search Index Rebuilt (Synapses Firing).")

    def consult_the_oracle(self, current_snapshot: RealitySnapshot, n_neighbors: int = 50) -> Dict:
        """
        Finds the 'Twins' in history and predicts the future.
        """
        if not self.is_trained:
            return {"status": "NOT_TRAINED", "confidence": 0.0}

        # Query vector
        query_vec = current_snapshot.to_vector().reshape(1, -1)
        
        # Find matches
        distances, indices = self.knn_engine.kneighbors(query_vec, n_neighbors=n_neighbors)
        
        # Analyze outcomes of the twins
        twins = [self.snapshots[i] for i in indices[0]]
        
        bull_wins = sum(1 for t in twins if t.outcome_label == 'BULL_WIN')
        bear_wins = sum(1 for t in twins if t.outcome_label == 'BEAR_WIN')
        
        total = len(twins)
        bull_prob = bull_wins / total
        bear_prob = bear_wins / total
        
        # Calculate Average Drawdown and Profit from twins
        avg_drawdown = np.mean([t.future_max_drawdown for t in twins])
        avg_profit = np.mean([t.future_max_profit for t in twins])
        
        direction = "NEUTRAL"
        confidence = 0.0
        
        if bull_prob > 0.6:
            direction = "BUY"
            confidence = bull_prob
        elif bear_prob > 0.6:
            direction = "SELL"
            confidence = bear_prob
            
        return {
            "status": "SUCCESS",
            "direction": direction,
            "confidence": confidence,
            "avg_drawdown": avg_drawdown,
            "avg_profit": avg_profit,
            "similar_scenarios_found": total,
            "top_match_similarity": 1.0 / (1.0 + distances[0][0]) # Pseudo-similarity score
        }
