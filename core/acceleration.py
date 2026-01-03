
import numpy as np
try:
    from numba import jit, njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Mock jit if not available to prevent crashes
    def jit(*args, **kwargs):
        # Case 1: Used as @jit (decorator without args) -> func is first arg
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
            
        # Case 2: Used as @jit(...) (factory with args) -> return decorator
        def decorator(func):
            return func
        return decorator
        
    njit = jit
    prange = range

# --- OPTIMIZED KERNELS ---

@njit(fastmath=True)
def jit_monte_carlo_walk(last_price, daily_vol, dt, steps, simulations):
    """
    Simulates N Geometric Brownian Motion paths.
    Raw Calculation Speed: C++ Level.
    """
    results = np.zeros(simulations)
    
    # Pre-calculate constants
    drift = 0 # Assume neutral drift for short term
    # shock = vol * sqrt(dt) * Z
    
    # We only need the ENDPOINT for the probability cloud
    # Optimization: Don't store the full path if not needed
    
    sqrtdt = np.sqrt(dt)
    
    for i in prange(simulations):
        price = last_price
        for _ in range(steps):
             shock = np.random.standard_normal() * daily_vol * sqrtdt
             price = price + (price * shock)
        results[i] = price
        
    return results

@njit(fastmath=True)
def jit_lyapunov_search(orbit, M, lag, embedding_dim):
    """
    Optimized Nearest Neighbor Search for Rosenstein Algorithm.
    O(N^2) complexity reduced by raw compilation speed.
    """
    div_sum = 0.0
    count = 0
    
    # Limit search scope for speed
    search_limit = min(50, M) 
    
    for i in range(M - search_limit, M - 1):
        p_i = orbit[i]
        min_dist = 1e9
        nearest_idx = -1
        
        # Look backwards
        for j in range(max(0, i-200), i - lag):
            # Manual Euclidean Distance (Faster than np.linalg.norm in a loop sometimes)
            dist = 0.0
            for k in range(embedding_dim):
                d = p_i[k] - orbit[j, k]
                dist += d*d
            dist = np.sqrt(dist)
            
            if dist < min_dist and dist > 1e-9:
                min_dist = dist
                nearest_idx = j
                
        if nearest_idx != -1:
            step = 1
            if i + step < M and nearest_idx + step < M:
                 # Dist Next
                 dist_next = 0.0
                 for k in range(embedding_dim):
                    d = orbit[i+step, k] - orbit[nearest_idx+step, k]
                    dist_next += d*d
                 dist_next = np.sqrt(dist_next)
                 
                 if dist_next > 1e-9:
                     div_sum += np.log(dist_next / min_dist)
                     count += 1
                     
    if count == 0: return 0.0
    return div_sum / count

@njit(fastmath=True)
def jit_dtw_distance(s1, s2):
    """
    Simple Dynamic Time Warping Matrix Calculation.
    O(N*M)
    """
    n = len(s1)
    m = len(s2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        dtw_matrix[i, 0] = 1e9
    for i in range(m+1):
        dtw_matrix[0, i] = 1e9
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            # min(insertion, deletion, match)
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix[n, m]
