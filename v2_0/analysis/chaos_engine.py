import numpy as np
import logging

logger = logging.getLogger("Atl4s-Chaos")

class ChaosEngine:
    def __init__(self):
        self.embedding_dim = 3
        self.lag = 2

    def calculate_lyapunov(self, df):
        """
        Estimates the Largest Lyapunov Exponent (LLE) using Rosenstein's algorithm (simplified).
        Positive LLE = Chaotic (Sensitive to initial conditions).
        Negative LLE = Stable (Attracting to equilibrium).
        """
        if df is None or len(df) < 100:
            return 0.0
            
        # Use returns or log prices
        data = np.log(df['close'].values)
        
        N = len(data)
        M = N - (self.embedding_dim - 1) * self.lag
        if M < 10: return 0.0
        
        # Reconstruct Phase Space
        orbit = []
        for i in range(M):
            point = [data[i + j * self.lag] for j in range(self.embedding_dim)]
            orbit.append(point)
        orbit = np.array(orbit)
        
        # Find Nearest Neighbors (Euclidean distance)
        # Optimization: We check a subset of points to keep it fast (Real-time constraint)
        div_sum = 0
        count = 0
        
        # Check last 20 points against history
        for i in range(M - 20, M - 1): 
            # Find nearest neighbor for point i, excluding temporal neighbors (i-lag to i+lag)
            min_dist = float('inf')
            nearest_idx = -1
            
            p_i = orbit[i]
            
            for j in range(M - 25): # Look in past history
                if abs(i - j) < self.lag: continue
                
                dist = np.linalg.norm(p_i - orbit[j])
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    nearest_idx = j
            
            if nearest_idx != -1 and min_dist > 0:
                # Evolve both points by 'step' steps
                step = 1 # 1 step forward divergence
                if i + step < M and nearest_idx + step < M:
                    dist_next = np.linalg.norm(orbit[i+step] - orbit[nearest_idx+step])
                    # Lyapunov ~ ln(dist_next / dist_initial)
                    if dist_next > 0:
                        div_sum += np.log(dist_next / min_dist)
                        count += 1
                        
        if count == 0:
            return 0.0
            
        # Average divergence per step
        lyapunov = div_sum / count
        
        # Interpretation:
        # > 0.01: Highly Chaotic (Unpredictable)
        # 0.0 to 0.01: Weakly Chaotic (Trendable)
        # < 0: Stable (Mean Reversion)
        
        return lyapunov
