import numpy as np
import logging

logger = logging.getLogger("Atl4s-Topology")

class TopologyEngine:
    def __init__(self):
        self.embedding_dim = 3
        self.delay = 5 

    def analyze_persistence(self, data):
        """
        Analyzes the Topological Persistence of the data.
        Returns:
            loop_score (float): Strength of the largest hole (Cycle).
            betti_1 (int): Number of significant loops.
        """
        if data is None or len(data) < 50:
            return 0, 0
            
        # Point Cloud via Time-Delay Embedding (Takens' Theorem)
        # Vector v_t = [x_t, x_{t-delay}, x_{t-2*delay}]
        
        prices = np.array(data)
        n = len(prices)
        
        # Create Point Cloud
        points = []
        for i in range(n - (self.embedding_dim * self.delay)):
            vec = []
            for d in range(self.embedding_dim):
                vec.append(prices[i + d * self.delay])
            points.append(vec)
            
        points = np.array(points)
        if len(points) < 10: return 0, 0
        
        # Normalize cloud to unit box
        points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-9)
        
        # Simplified Vietoris-Rips Filtration (Heuristic)
        # We want to know if points form a circle (Loop) or a line.
        # A Loop implies recurrence. A Line implies Trend.
        
        # Calculate Center of Mass
        center = np.mean(points, axis=0)
        
        # Calculate distances from center
        dists = np.linalg.norm(points - center, axis=1)
        
        # Variance of distances. 
        # If points are on a circle, dists are constant -> Variance is Low.
        # If points are a blob/line, dists vary -> Variance is High.
        
        dist_std = np.std(dists)
        avg_dist = np.mean(dists)
        
        # Topology Score: Higher if it looks like a clean loop/orbit
        # (Low variance of radius)
        
        # But we also need spread (it shouldn't be a single point).
        spread = np.std(points)
        
        if spread < 0.01: return 0, 0
        
        # Heuristic "Loopiness": 1 / dist_std (normalized by avg_dist)
        # Loop Score = avg_dist / (dist_std + epsilon)
        loop_score = avg_dist / (dist_std + 1e-9)
        
        # Threshold for "Significant Loop"
        # Empirically, > 3.0 usually indicates a cyclic orbit in phase space
        betti_1 = 1 if loop_score > 3.0 else 0
        
        return loop_score, betti_1
