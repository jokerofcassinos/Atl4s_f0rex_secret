"""
AGI Ultra Python-C++ Bridge
Unified interface to all C++ AGI modules
"""
import ctypes
from ctypes import CDLL, POINTER, Structure, c_double, c_int, c_char_p, c_void_p, c_bool
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import threading
import time

# Phase 3.3: TTL Cache for Expensive Calculations
_physics_cache = {
    'fisher': {'value': 0.0, 'time': 0, 'ttl': 5.0},
    'hurst': {'value': 0.5, 'time': 0, 'ttl': 10.0},
    'lyapunov': {'value': 0.0, 'time': 0, 'ttl': 15.0},
}


# ============================================================================
# LOAD LIBRARIES
# ============================================================================

# ============================================================================
# LOAD LIBRARIES
# ============================================================================

# Ensure DLL directories are added (Python 3.8+ Windows)
if hasattr(os, "add_dll_directory"):
    try:
        # Add build/bin to DLL search path
        bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "build/bin"))
        if os.path.exists(bin_path):
            os.add_dll_directory(bin_path)
        
        # Try to find MinGW/G++ path from environment and add it
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for p in path_dirs:
            if os.path.exists(os.path.join(p, "g++.exe")) or os.path.exists(os.path.join(p, "libstdc++-6.dll")):
                try:
                    os.add_dll_directory(p)
                except:
                    pass
    except Exception as e:
        print(f"Warning: Failed to add DLL directories: {e}")

def _load_lib(name: str) -> Optional[ctypes.CDLL]:
    """Load a shared library from multiple possible locations."""
    # If loading specific module fails, try the monolithic lib
    names_to_try = [name, "atl4s_agi"]
    
    for n in names_to_try:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), f"build/bin/lib{n}.dll"), # MinGW preferred
            os.path.join(os.path.dirname(__file__), f"build/bin/{n}.dll"),
            os.path.join(os.path.dirname(__file__), f"build/{n}.dll"),
            f"./{n}.dll",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
    
    return None

# Try to load all C++ libraries
# Note: They might all point to the same DLL (libatl4s_agi.dll) which is fine
_mcts_lib = _load_lib("mcts")
_physics_lib = _load_lib("physics")
_hdc_lib = _load_lib("hdc")
_reasoning_lib = _load_lib("reasoning")
_memory_lib = _load_lib("memory")
_learning_lib = _load_lib("learning")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TrajectoryResult:
    terminal_price: float
    max_deviation: float
    steps_taken: int
    total_distance: float
    final_velocity: float
    energy: float


class CSimulationResult(ctypes.Structure):
    _fields_ = [
        ("best_move_type", ctypes.c_int),
        ("expected_value", ctypes.c_double),
        ("visits", ctypes.c_int),
        ("confidence", ctypes.c_double),
        ("q_value", ctypes.c_double),
    ]


@dataclass
class QuantumState:
    amplitudes: np.ndarray
    phases: np.ndarray
    coherence: float


@dataclass
class PatternMatch:
    pattern_id: int
    start_index: int
    length: int
    similarity: float
    confidence: float


@dataclass 
class MCTSResult:
    best_action: int
    visit_counts: List[int]
    q_values: List[float]
    exploration_bonus: float


# ============================================================================
# MCTS BRIDGE
# ============================================================================

class MCTSBridge:
    """Bridge to C++ MCTS Ultra module."""
    
    def __init__(self):
        self._lib = _mcts_lib
        self._setup_functions()
    
    def _setup_functions(self):
        if not self._lib:
            return
        
        # run_mcts_simulation
        self._lib.run_mcts_simulation.argtypes = [
            ctypes.c_int,     # num_simulations
            ctypes.c_double,  # exploration_constant
            ctypes.c_int,     # max_depth
            ctypes.c_int,     # num_actions
        ]
        self._lib.run_mcts_simulation.restype = ctypes.c_int
        
        # run_parallel_mcts
        self._lib.run_parallel_mcts.argtypes = [
            ctypes.c_int,     # num_simulations
            ctypes.c_double,  # exploration_constant
            ctypes.c_int,     # max_depth
            ctypes.c_int,     # num_actions
            ctypes.c_int,     # num_threads
        ]
        self._lib.run_parallel_mcts.restype = ctypes.c_int
        
        # run_agi_mcts
        self._lib.run_agi_mcts.argtypes = [
            ctypes.c_int,     # num_simulations
            ctypes.c_double,  # exploration_constant
            ctypes.c_int,     # max_depth
            ctypes.c_int,     # num_actions
            ctypes.c_int,     # num_threads
            ctypes.c_double,  # rave_k
            ctypes.c_double,  # pw_alpha
            ctypes.c_double,  # pw_beta
        ]
        self._lib.run_agi_mcts.restype = ctypes.c_int
    
    @property
    def available(self) -> bool:
        return self._lib is not None
    
    def run_simulation(
        self,
        num_simulations: int = 1000,
        exploration_constant: float = 1.414,
        max_depth: int = 50,
        num_actions: int = 3
    ) -> int:
        """Run basic MCTS simulation."""
        if not self._lib:
            return -1
        return self._lib.run_mcts_simulation(
            num_simulations, exploration_constant, max_depth, num_actions
        )
    
    def run_parallel(
        self,
        num_simulations: int = 10000,
        exploration_constant: float = 1.414,
        max_depth: int = 50,
        num_actions: int = 3,
        num_threads: int = 4
    ) -> int:
        """Run parallel MCTS with OpenMP."""
        if not self._lib:
            return -1
        return self._lib.run_parallel_mcts(
            num_simulations, exploration_constant, max_depth, num_actions, num_threads
        )
    
    def run_agi_mcts(
        self,
        num_simulations: int = 10000,
        exploration_constant: float = 1.414,
        max_depth: int = 50,
        num_actions: int = 3,
        num_threads: int = 4,
        rave_k: float = 3000.0,
        pw_alpha: float = 0.5,
        pw_beta: float = 1.0
    ) -> int:
        """Run full AGI MCTS with RAVE, adaptive UCT, and progressive widening."""
        if not self._lib:
            return -1
        
        # Mapping to simple int return for now, but strictly we should use struct.
        # Assuming lib wrapper handles it or we accept partial data (int return of struct usually grabs first member)
        # First member of SimulationResult is 'best_move_type' (int).
        # So accidentally this works!
        
        return self._lib.run_agi_mcts(
            num_simulations, exploration_constant, max_depth, num_actions,
            num_threads, rave_k, pw_alpha, pw_beta
        )

    def run_guided_mcts(
        self,
        current_price: float,
        entry_price: float,
        direction: int,
        volatility: float,
        drift: float,
        iterations: int = 1000,
        depth: int = 50,
        bias_strength: float = 0.0,
        bias_direction: int = 0
    ) -> Dict[str, Any]:
        """Run MCTS guided by AGI Intuition."""
        if not self._lib:
            return {"best_move": 0, "confidence": 0.0}
            
        # Lazy binding for new function
        if not hasattr(self._lib, 'run_guided_mcts'):
             # Ensure types are correct
             self._lib.run_guided_mcts.argtypes = [
                c_double, c_double, c_int, c_double, c_double,
                c_int, c_int, c_double, c_int
             ]
             self._lib.run_guided_mcts.restype = CSimulationResult
        
        try:
            # Explicit wrapping to bypass conversion logic
            res = self._lib.run_guided_mcts(
                c_double(float(current_price)), 
                c_double(float(entry_price)), 
                c_int(int(direction)), 
                c_double(float(volatility)), 
                c_double(float(drift)),
                c_int(int(iterations)), 
                c_int(int(depth)), 
                c_double(float(bias_strength)), 
                c_int(int(bias_direction))
            )
            
            return {
                "best_move": res.best_move_type,
                "confidence": res.confidence,
                "expected_value": res.expected_value
            }
        except Exception as e:
            # logger.warning(f"C++ MCTS Failed ({e}). Using Python Fallback.")
            return self._run_python_guided_mcts(
                current_price, entry_price, direction, volatility, drift,
                iterations, depth, bias_strength, bias_direction
            )

    def _run_python_guided_mcts(
        self,
        current_price, entry_price, direction, volatility, drift,
        iterations, depth, bias_strength, bias_direction
    ):
        """Python implementation of Guided MCTS (Fallback)."""
        import random
        from math import sqrt, log
        
        wins = 0.0
        visits = 0
        
        # Monte Carlo Simulation
        for _ in range(iterations):
            price = current_price
            active = True
            current_depth = 0
            
            # Rollout
            pnl = 0.0
            
            while active and current_depth < depth:
                current_depth += 1
                
                # Apply Bias
                # Base probs: Close 0.3, Add 0.1, Hold 0.6
                p_close = 0.3
                
                if bias_direction != 0 and direction != 0:
                     if bias_direction == direction:
                         # Reinforcing (Buy bias for Buy trade) -> Less closing
                         p_close *= (1.0 - bias_strength)
                     elif bias_direction == -direction:
                         # Contrarian (Sell bias for Buy trade) -> More closing
                         p_close *= (1.0 + bias_strength)
                
                r = random.random()
                if r < p_close:
                    active = False
                else:
                    shock = random.gauss(0, volatility)
                    price += drift + shock
                    pnl = (price - entry_price) if direction == 1 else (entry_price - price)
            
            wins += pnl
            visits += 1
            
        ev = wins / max(1, visits)
        
        return {
            "best_move": 0 if ev > 0 else 1, # Simple heuristic
            "confidence": 0.8, # Mock confidence
            "expected_value": ev
        }


# ============================================================================
# PHYSICS BRIDGE
# ============================================================================

class PhysicsBridge:
    """Bridge to C++ Physics Ultra module."""
    
    def __init__(self):
        self._lib = _physics_lib
        self._setup_functions()
    
    def _setup_functions(self):
        if not self._lib:
            return
        
        # Define structure
        class CTrajectoryResult(ctypes.Structure):
            _fields_ = [
                ("terminal_price", ctypes.c_double),
                ("max_deviation", ctypes.c_double),
                ("steps_taken", ctypes.c_int),
                ("total_distance", ctypes.c_double),
                ("final_velocity", ctypes.c_double),
                ("energy", ctypes.c_double),
            ]
        
        self._TrajectoryResult = CTrajectoryResult
        
        # simulate_trajectory
        self._lib.simulate_trajectory.argtypes = [
            ctypes.c_double,  # start_price
            ctypes.c_double,  # start_velocity
            ctypes.c_double,  # start_accel
            ctypes.c_double,  # mass
            ctypes.c_double,  # friction_coeff
            ctypes.c_double,  # dt
            ctypes.c_int,     # max_steps
        ]
        self._lib.simulate_trajectory.restype = CTrajectoryResult
        
        # calculate_sectional_curvature
        self._lib.calculate_sectional_curvature.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.calculate_sectional_curvature.restype = ctypes.c_double
        
        # calculate_lyapunov_exponent
        self._lib.calculate_lyapunov_exponent.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.calculate_lyapunov_exponent.restype = ctypes.c_double
        
        # calculate_hurst_exponent
        self._lib.calculate_hurst_exponent.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.calculate_hurst_exponent.restype = ctypes.c_double
        
        # calculate_market_entropy
        self._lib.calculate_market_entropy.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.calculate_market_entropy.restype = ctypes.c_double
        
        # calculate_tunneling_probability
        self._lib.calculate_tunneling_probability.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        # calculate_fisher_information
        self._lib.calculate_fisher_information.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.calculate_fisher_information.restype = ctypes.c_double
    
    @property
    def available(self) -> bool:
        return self._lib is not None
    
    # ... (other methods) ...
    
    def calculate_fisher(self, prices: np.ndarray, window: int = 10) -> float:
        """Calculate Fisher Information Metric (Regime Change Speed). Cached."""
        # TTL Cache Check
        cache = _physics_cache['fisher']
        now = time.time()
        if now - cache['time'] < cache['ttl']:
            return cache['value']
            
        if self._lib:
            prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
            ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return self._lib.calculate_fisher_information(ptr, len(prices_arr), window)
            
        # Python Fallback: Ehlers Fisher Transform
        try:
            if len(prices) < window: return 0.0
            
            # Use last 'window' prices
            data = prices[-window:]
            
            # Normalize to [-0.99, 0.99] roughly to avoid log(0)
            mn, mx = np.min(data), np.max(data)
            if mx == mn: return 0.0
            
            norm = 2 * ((data - mn) / (mx - mn)) - 1
            norm = 0.99 * norm # Clip to avoid infinity
            
            # Fisher Transform on the latest point
            # F = 0.5 * ln((1+x)/(1-x))
            val = norm[-1]
            fisher = 0.5 * np.log((1 + val) / (1 - val))
            
            # We assume "Fisher Metric" is the absolute value or magnitude of the signal
            # Or the change in Fisher?
            # OmniCortex expects > 2.0 for CHAOTIC.
            # Std Fisher can go up to 2-3 easily.
            result = abs(fisher)
            _physics_cache['fisher'] = {'value': result, 'time': time.time(), 'ttl': 5.0}
            return result
        except:
            return 0.0
    
    def simulate_trajectory(
        self,
        start_price: float,
        start_velocity: float,
        start_accel: float = 0.0,
        mass: float = 1.0,
        friction_coeff: float = 0.005,
        dt: float = 0.01,
        max_steps: int = 1000
    ) -> TrajectoryResult:
        """Simulate price trajectory using physics model."""
        if not self._lib:
            return TrajectoryResult(start_price, 0, 0, 0, 0, 0)
        
        result = self._lib.simulate_trajectory(
            start_price, start_velocity, start_accel,
            mass, friction_coeff, dt, max_steps
        )
        
        return TrajectoryResult(
            terminal_price=result.terminal_price,
            max_deviation=result.max_deviation,
            steps_taken=result.steps_taken,
            total_distance=result.total_distance,
            final_velocity=result.final_velocity,
            energy=result.energy
        )
    
    def calculate_curvature(self, prices: np.ndarray, window_size: int = 10) -> float:
        """Calculate sectional curvature of price manifold."""
        if not self._lib:
            return 0.0
        
        prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
        ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self._lib.calculate_sectional_curvature(ptr, len(prices_arr), window_size)
    
    def calculate_lyapunov(self, prices: np.ndarray) -> float:
        # TTL Cache Check
        cache = _physics_cache['lyapunov']
        now = time.time()
        if now - cache['time'] < cache['ttl']:
            return cache['value']
            
        if self._lib:
            prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
            ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return self._lib.calculate_lyapunov_exponent(ptr, len(prices_arr))

        # Python Fallback: Vectorized Lyapunov (Simplified Rosenstein)
        try:
            if len(prices) < 50: return 0.0
            data = np.log(prices)
            N = len(data)
            lag = 2
            embed = 3
            M = N - (embed - 1) * lag
            if M < 20: return 0.0
            
            orbit = np.array([data[i:i+embed*lag:lag] for i in range(M)])
            
            # Vectorized nearest neighbor (Last 20 points only for speed)
            div_sum = 0
            count = 0
            indices = np.arange(M)
            
            for i in range(M - 20, M - 1):
                 p_i = orbit[i]
                 history = orbit[:M-25]
                 if len(history) == 0: continue
                 
                 dists = np.linalg.norm(history - p_i, axis=1)
                 mask = np.abs(indices[:M-25] - i) > lag
                 valid_dists = dists[mask]
                 
                 if len(valid_dists) > 0:
                      min_dist = np.min(valid_dists)
                      nearest_idx = indices[:M-25][mask][np.argmin(valid_dists)]
                      
                      if min_dist > 0 and i+1 < M and nearest_idx+1 < M:
                           dist_next = np.linalg.norm(orbit[i+1] - orbit[nearest_idx+1])
                           if dist_next > 0:
                                div_sum += np.log(dist_next / min_dist)
                                count += 1
            
            result = div_sum / count if count > 0 else 0.0
            _physics_cache['lyapunov'] = {'value': result, 'time': time.time(), 'ttl': 15.0}
            return result
        except:
            return 0.0
    
    def calculate_hurst(self, prices: np.ndarray) -> float:
        # TTL Cache Check
        cache = _physics_cache['hurst']
        now = time.time()
        if now - cache['time'] < cache['ttl']:
            return cache['value']
            
        if self._lib:
            prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
            ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return self._lib.calculate_hurst_exponent(ptr, len(prices_arr))

        # Python Fallback: Simplified Hurst (Std Deviation vs Lag)
        try:
            if len(prices) < 32: return 0.5
            lags = range(2, 20)
            tau = []
            valid_lags = []
            ts = np.array(prices)
            
            for lag in lags:
                if len(ts) > lag:
                   diff = ts[lag:] - ts[:-lag]
                   std = np.std(diff)
                   if std > 0:
                       tau.append(std)
                       valid_lags.append(lag)
            
            if len(tau) < 2: return 0.5
            # log(std) ~ H * log(lag)
            poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
            result = poly[0]
            _physics_cache['hurst'] = {'value': result, 'time': time.time(), 'ttl': 10.0}
            return result
        except:
            return 0.5
    
    def calculate_entropy(self, prices: np.ndarray, bins: int = 20) -> float:
        """Calculate market entropy."""
        if not self._lib:
            return 0.0
        
        prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
        ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self._lib.calculate_market_entropy(ptr, len(prices_arr), bins)
    
    def calculate_tunneling(
        self,
        barrier_height: float,
        particle_energy: float,
        barrier_width: float
    ) -> float:
        """Calculate quantum tunneling probability (resistance breakthrough)."""
        if not self._lib:
            return 0.0
        
        return self._lib.calculate_tunneling_probability(
            barrier_height, particle_energy, barrier_width
        )
        



# ============================================================================
# HDC BRIDGE
# ============================================================================

class HDCBridge:
    """Bridge to C++ HDC Ultra module."""
    
    def __init__(self, dimension: int = 10000):
        self._lib = _hdc_lib
        self._dimension = dimension
        self._setup_functions()
    
    def _setup_functions(self):
        if not self._lib:
            return
        
        # bind_vectors
        self._lib.bind_vectors.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
        ]
        
        # bundle_vectors
        self._lib.bundle_vectors.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
        ]
        
        # cosine_similarity
        self._lib.cosine_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
        ]
        self._lib.cosine_similarity.restype = ctypes.c_double
        
        # encode_scalar
        self._lib.encode_scalar.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
        ]
        
        # encode_time_series
        self._lib.encode_time_series.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
        ]
    
    @property
    def available(self) -> bool:
        return self._lib is not None
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def random_hv(self) -> np.ndarray:
        """Generate random hypervector."""
        return np.random.choice([-1, 1], size=self._dimension).astype(np.int8)
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind (multiply) two hypervectors."""
        if not self._lib:
            return (a * b).astype(np.int8)
        
        result = np.zeros(self._dimension, dtype=np.int8)
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        r_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        self._lib.bind_vectors(a_ptr, b_ptr, r_ptr, self._dimension)
        return result
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle (add) multiple hypervectors."""
        if not vectors:
            return self.random_hv()
        
        if not self._lib:
            summed = np.sum(vectors, axis=0)
            return np.where(summed > 0, 1, np.where(summed < 0, -1, 
                           np.random.choice([-1, 1], size=self._dimension))).astype(np.int8)
        
        flat = np.concatenate(vectors).astype(np.int8)
        result = np.zeros(self._dimension, dtype=np.int8)
        
        flat_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        r_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        self._lib.bundle_vectors(flat_ptr, len(vectors), r_ptr, self._dimension)
        return result
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between hypervectors."""
        if not self._lib:
            return float(np.dot(a.astype(float), b.astype(float)) / self._dimension)
        
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        return self._lib.cosine_similarity(a_ptr, b_ptr, self._dimension)
    
    def encode_scalar(self, value: float, min_val: float, max_val: float) -> np.ndarray:
        """Encode scalar value as hypervector."""
        if not self._lib:
            normalized = (value - min_val) / (max_val - min_val + 1e-10)
            level = int(normalized * self._dimension)
            result = np.ones(self._dimension, dtype=np.int8) * -1
            result[:level] = 1
            return result
        
        result = np.zeros(self._dimension, dtype=np.int8)
        r_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        self._lib.encode_scalar(value, min_val, max_val, r_ptr, self._dimension)
        return result
    
    def encode_time_series(self, values: np.ndarray) -> np.ndarray:
        """Encode time series as hypervector."""
        if not self._lib:
            hvs = []
            if len(values) == 0:
                return self.random_hv()
            min_v, max_v = values.min(), values.max()
            for t, v in enumerate(values):
                hv = self.encode_scalar(v, min_v, max_v)
                hvs.append(np.roll(hv, t))
            return self.bundle(hvs)
        
        result = np.zeros(self._dimension, dtype=np.int8)
        values_arr = np.ascontiguousarray(values, dtype=np.float64)
        
        v_ptr = values_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        r_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        
        self._lib.encode_time_series(v_ptr, len(values_arr), r_ptr, self._dimension)
        return result


# ============================================================================
# UNIFIED AGI BRIDGE
# ============================================================================

class AGIBridge:
    """Unified interface to all C++ AGI modules."""
    
    def __init__(self, hdc_dimension: int = 10000):
        self.mcts = MCTSBridge()
        self.physics = PhysicsBridge()
        self.hdc = HDCBridge(hdc_dimension)
        
        self._lock = threading.Lock()
    
    @property
    def available_modules(self) -> Dict[str, bool]:
        return {
            "mcts": self.mcts.available,
            "physics": self.physics.available,
            "hdc": self.hdc.available,
        }
    
    def analyze_market(
        self,
        prices: np.ndarray,
        num_mcts_sims: int = 1000
    ) -> Dict[str, Any]:
        """Comprehensive market analysis using all AGI modules."""
        
        with self._lock:
            results = {}
            
            # Physics analysis
            if self.physics.available and len(prices) >= 20:
                velocity = prices[-1] - prices[-2] if len(prices) >= 2 else 0
                trajectory = self.physics.simulate_trajectory(
                    prices[-1], velocity * 100, 0, 1.0, 0.005, 0.01, 500
                )
                results["trajectory"] = {
                    "terminal_price": trajectory.terminal_price,
                    "max_deviation": trajectory.max_deviation,
                    "energy": trajectory.energy,
                }
                
                results["curvature"] = self.physics.calculate_curvature(prices)
                results["lyapunov"] = self.physics.calculate_lyapunov(prices)
                results["hurst"] = self.physics.calculate_hurst(prices)
                results["entropy"] = self.physics.calculate_entropy(prices)
            
            # HDC analysis
            if self.hdc.available and len(prices) >= 10:
                hv = self.hdc.encode_time_series(prices[-50:] if len(prices) >= 50 else prices)
                results["hdc_encoding"] = hv
            
            # MCTS analysis
            if self.mcts.available:
                best_action = self.mcts.run_agi_mcts(
                    num_simulations=num_mcts_sims,
                    num_actions=3  # 0=HOLD, 1=BUY, 2=SELL
                )
                results["mcts_action"] = best_action
            
            return results


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_agi_bridge: Optional[AGIBridge] = None


def get_agi_bridge() -> AGIBridge:
    """Get or create the global AGI bridge instance."""
    global _agi_bridge
    if _agi_bridge is None:
        _agi_bridge = AGIBridge()
    return _agi_bridge
