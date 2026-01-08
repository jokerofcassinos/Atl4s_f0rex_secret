
import logging
import numpy as np
import json
import os
from typing import List, Dict, Any
from scipy.spatial.distance import cosine

logger = logging.getLogger("HolographicMemory")

class HolographicMemory:
    """
    The Akasha.
    Long-term Vector Memory for Pattern Recall.
    Stores: [Feature_Vector] -> Outcome (PnL)
    """
    def __init__(self, memory_file="brain/memory_bank.json"):
        self.memory_file = memory_file
        self.vectors = [] # List of np.arrays
        self.outcomes = [] # List of floats
        self.meta = [] # List of dicts
        self.load_memory()
        
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.vectors.append(np.array(item['vector']))
                        self.outcomes.append(item['outcome'])
                        self.meta.append(item['meta'])
                logger.info(f"Holographic Memory Loaded: {len(self.vectors)} experiences.")
            except Exception as e:
                logger.error(f"Memory Corruption: {e}")
                
    def save_memory(self):
        # Periodic save (or on shutdown)
        data = []
        for i in range(len(self.vectors)):
            data.append({
                'vector': self.vectors[i].tolist(),
                'outcome': self.outcomes[i],
                'meta': self.meta[i]
            })
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to Save Memory: {e}")

    def store_experience(self, feature_vector, outcome, meta={}):
        """
        Encodes a lived moment into the Hologram.
        """
        # Normalize vector?
        # Assuming feature_vector is already scaled 0-1 or similar.
        self.vectors.append(np.array(feature_vector))
        self.outcomes.append(outcome)
        self.meta.append(meta)
        
    def recall(self, current_vector, k=5):
        """
        Retrieves the 5 most similar past moments.
        Returns: Average Expected Outcome, Confidence score.
        """
        if len(self.vectors) < k:
            return 0.0, 0.0 # Tabula Rasa
            
        distances = []
        target = np.array(current_vector).astype(np.float32)
        
        # Calculate Cosine Distances
        # Opt: Use matrix math if vectors list is large (e.g. np.dot)
        # For now, simple loop is fine for < 10000 items.
        
        for i, vec in enumerate(self.vectors):
            vec_f = vec.astype(np.float32)
            dot = np.dot(target, vec_f)
            norm_a = np.linalg.norm(target)
            norm_b = np.linalg.norm(vec_f)
            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot / (norm_a * norm_b)
                
            distances.append((similarity, self.outcomes[i]))
            
        # Sort by similarity (Highest first)
        distances.sort(key=lambda x: x[0], reverse=True)
        
        # Top K
        top_k = distances[:k]
        
        # Weighted Outcome
        total_sim = sum([x[0] for x in top_k])
        if total_sim == 0: return 0.0, 0.0
        
        weighted_outcome = sum([x[0] * x[1] for x in top_k]) / total_sim
        
        # Confidence = Average Similarity of the match
        confidence = total_sim / k 
        
        return weighted_outcome, confidence

    def construct_hologram(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Transforms market state dict into a dense holographic vector.
        Uses Random Projection (simulated) to encode high-dimensional features.
        """
        # Feature Extraction
        price_norm = np.log(market_state.get('bid', 1.0) + 1)
        vol_norm = np.log(market_state.get('volume', 1.0) + 1)
        rsi = market_state.get('rsi', 50) / 100.0
        metrics = market_state.get('metrics', {})
        if isinstance(metrics, dict):
            entropy = metrics.get('entropy', 0.5)
        else:
            # Assume it's a PerformanceMetrics object or similar
            entropy = getattr(metrics, 'entropy', 0.5)
        
        # Base Vector (Deterministic)
        vec = np.array([price_norm, vol_norm, rsi, entropy])
        
        # simulated "Holographic Projection" (Random Matrix Expansion)
        # In real world, we'd use a fixed random matrix. Here we just expand functionally.
        hologram = np.concatenate([
            vec, 
            np.sin(vec * np.pi), 
            np.cos(vec * np.pi),
            vec * vec,
            np.tanh(vec)
        ])
        
        # Normalize
        norm = np.linalg.norm(hologram)
        return hologram / (norm + 1e-9)

    def retrieve_associative_memory(self, market_state: Dict[str, Any], k=5) -> Dict[str, Any]:
        """
        Pulls 'Historical Rhymes' from the Akasha.
        Returns: { 'expected_outcome': float, 'confidence': float, 'similar_scenarios': int }
        """
        query_vec = self.construct_hologram(market_state)
        outcome, conf = self.recall(query_vec, k)
        
        return {
            "expected_outcome": outcome,
            "confidence": conf,
            "memory_depth": len(self.vectors),
            "insight": "Historical Pattern Recognition" if conf > 0.7 else "Unmapped Territory"
        }

    def recall_associative(self, query_vector, k=10) -> List[Dict[str, Any]]:
        """
        Active Inference Recall: Retrieves full associative memory of past states.
        Returns list of {vector, outcome, meta} for detailed generative modeling.
        """
        if not self.vectors: return []
        
        target = np.array(query_vector).astype(np.float32)
        results = []
        
        for i, vec in enumerate(self.vectors):
            vec_f = vec.astype(np.float32)
            dot = np.dot(target, vec_f)
            norm_a = np.linalg.norm(target)
            norm_b = np.linalg.norm(vec_f)
            similarity = 0 if (norm_a == 0 or norm_b == 0) else dot / (norm_a * norm_b)
            
            if similarity > 0.5: # Relevance threshold
                results.append({
                    'similarity': similarity,
                    'outcome': self.outcomes[i],
                    'meta': self.meta[i],
                    'vector': self.vectors[i] 
                })
        
        # Sort and return top K
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def retrieve_intuition(self, vector):
        """
        Alias for recall, used by some AGI components for semantic clarity.
        """
        return self.recall(vector)
