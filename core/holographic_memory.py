
import logging
import numpy as np
import json
import os
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
        target = np.array(current_vector)
        
        # Calculate Cosine Distances
        # Opt: Use matrix math if vectors list is large (e.g. np.dot)
        # For now, simple loop is fine for < 10000 items.
        
        for i, vec in enumerate(self.vectors):
            # Cosine distance is 1 - similarity. Low is good.
            # dist = cosine(target, vec) # scipy cosine is generic
            # Manually: 1 - (A.B / |A||B|)
            dot = np.dot(target, vec)
            norm_a = np.linalg.norm(target)
            norm_b = np.linalg.norm(vec)
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
