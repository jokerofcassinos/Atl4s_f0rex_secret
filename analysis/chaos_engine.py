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
        
        # Find Nearest Neighbors (Vectorized)
        # Check last 20 points against history
        # Create mask for self.lag exclusion
        indices = np.arange(M)
        
        for i in range(M - 20, M - 1): 
            p_i = orbit[i]
            
            # Vectorized distance calculation
            # Use only history (up to M-25)
            history_orbit = orbit[:M-25]
            if len(history_orbit) == 0: continue

            dists = np.linalg.norm(history_orbit - p_i, axis=1)
            
            # Mask out temporal neighbors (i-lag to i+lag)
            # Since we iterate i in (M-20..M-1), and history is up to M-25,
            # |i - j| is always > 5 > lag(2). So we technically don't need the check if lag is small.
            # But to be safe and generic:
            mask = np.abs(indices[:M-25] - i) > self.lag
            
            valid_dists = dists[mask]
            valid_indices = indices[:M-25][mask]
            
            if len(valid_dists) > 0:
                min_idx_in_valid = np.argmin(valid_dists)
                min_dist = valid_dists[min_idx_in_valid]
                nearest_idx = valid_indices[min_idx_in_valid]
            else:
                nearest_idx = -1
                min_dist = 0

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
