"""
AGI Ultra Python-C++ Bridge
Unified interface to all C++ AGI modules
"""
import ctypes
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import threading


# ============================================================================
# LOAD LIBRARIES
# ============================================================================

def _load_lib(name: str) -> Optional[ctypes.CDLL]:
    """Load a shared library from multiple possible locations."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), f"build/{name}.dll"),
        os.path.join(os.path.dirname(__file__), f"build/{name}.so"),
        os.path.join(os.path.dirname(__file__), f"build/lib{name}.so"),
        os.path.join(os.path.dirname(__file__), f"{name}.dll"),
        f"./{name}.dll",
        f"./{name}.so",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError as e:
                print(f"Warning: Could not load {path}: {e}")
    
    return None


# Try to load all C++ libraries
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
        return self._lib.run_agi_mcts(
            num_simulations, exploration_constant, max_depth, num_actions,
            num_threads, rave_k, pw_alpha, pw_beta
        )


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
        self._lib.calculate_tunneling_probability.restype = ctypes.c_double
    
    @property
    def available(self) -> bool:
        return self._lib is not None
    
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
        """Calculate Lyapunov exponent (chaos measure)."""
        if not self._lib:
            return 0.0
        
        prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
        ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self._lib.calculate_lyapunov_exponent(ptr, len(prices_arr))
    
    def calculate_hurst(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent (long-term memory)."""
        if not self._lib:
            return 0.5
        
        prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
        ptr = prices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self._lib.calculate_hurst_exponent(ptr, len(prices_arr))
    
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
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
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
