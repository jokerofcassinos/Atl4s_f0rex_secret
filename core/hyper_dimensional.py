import numpy as np
import logging
import threading
from cpp_core.agi_bridge import get_agi_bridge

logger = logging.getLogger("HyperCore")

# Dimensionality (Must match C++ build if fixed, but flexible usually)
D = 10000 

class HyperVector:
    """
    Wrapper for a 10,000-dimensional Hypervector.
    Delegate heavy lifting to C++ via AGIBridge.
    """
    def __init__(self, data=None):
        self.bridge = get_agi_bridge().hdc
        if data is None:
            # Force Numpy Generation
            if False and self.bridge.available and hasattr(self.bridge, 'random_hv'):
                self.values = self.bridge.random_hv()
            else:
                self.values = np.random.choice([-1, 1], size=D).astype(np.int8)
        else:
            self.values = data.astype(np.int8)

    def bind(self, other):
        """XOR (Multiplication)."""
        if self.bridge.available:
            res = self.bridge.bind(self.values, other.values)
            return HyperVector(res)
        else:
            # Python Fallback
            return HyperVector(self.values * other.values)

    def bundle(self, other):
        """Superposition (Addition)."""
        if self.bridge.available:
            res = self.bridge.bundle([self.values, other.values])
            return HyperVector(res)
        else:
            res = self.values + other.values
            res[res > 0] = 1
            res[res < 0] = -1
            # Randomize zeros
            zeros = (res == 0)
            if np.any(zeros):
                res[zeros] = np.random.choice([-1, 1], size=np.sum(zeros))
            return HyperVector(res)

    def permute(self, k=1):
        """
        Temporal Binding via Cyclic Shift (Permutation).
        V_t = Permute(V_t-1)
        """
        # Try C++ first
        if self.bridge.available and hasattr(self.bridge, 'permute'):
            try:
                res = self.bridge.permute(self.values, k)
                return HyperVector(res)
            except Exception as e:
                # Fallback on error
                pass
        
        # Python cyclic shift fallback
        return HyperVector(np.roll(self.values, k))

    def similarity(self, other):
        """Cosine Similarity."""
        if self.bridge.available and hasattr(self.bridge, 'cosine_similarity'):
           try:
               return self.bridge.cosine_similarity(self.values, other.values)
           except:
               pass
               
        # Python fallback: dot product
        # CAST TO FLOAT/INT32 to avoid int8 overflow with D=10000
        v1 = self.values.astype(np.float32)
        v2 = other.values.astype(np.float32)
        dot = np.dot(v1, v2)
        return dot / D

class TimeEncoder:
    """
    Encodes temporal sequences into Holographic Memory.
    Sequence = V_t + P(V_t-1) + P^2(V_t-2) ...
    """
    def __init__(self):
        pass

    def encode_sequence(self, vectors: list) -> HyperVector:
        """
        Encodes a list of HyperVectors into a single Sequence Vector due to Permutation.
        Latest vector is 'most prominent', older vectors are shifted further.
        """
        if not vectors: return HyperVector()
        
        sequence_vector = HyperVector(np.zeros(D))
        for i, vec in enumerate(reversed(vectors)):
            # S = V_0 + P(V_-1) + P(P(V_-2)) ...
            shifted = vec.permute(k=i)
            sequence_vector = sequence_vector.bundle(shifted)
            
        return sequence_vector


class HDCEncoder:
    """
    Encodes market features into HyperVectors.
    """
    def __init__(self):
        self.bridge = get_agi_bridge().hdc
        # Base vectors can be generated on fly or cached
        self._cache = {}
        
        # Fallback Basis Vectors
        np.random.seed(42)
        self.min_basis = np.random.choice([-1, 1], size=D).astype(np.int8)
        self.max_basis = np.random.choice([-1, 1], size=D).astype(np.int8)

    def encode_value(self, val_0_100):
        """Encodes a scalar 0-100 into a HyperVector"""
        # Force Python Fallback to ensure consistency (Bridge seems flaky)
        if False and self.bridge.available:
            try:
                res = self.bridge.encode_scalar(val_0_100, 0, 100)
                return HyperVector(res)
            except:
                pass # Fallback

        # Python Fallback: Linear Interpolation (Thermometer Code)
        # Mix min_basis and max_basis
        ratio = max(0, min(100, val_0_100)) / 100.0
        split_idx = int(D * ratio)
        
        # Construct vector: First part from MAX, Second from MIN
        # (This is just one way to create correlated vectors)
        new_vals = np.concatenate((self.max_basis[:split_idx], self.min_basis[split_idx:]))
        return HyperVector(new_vals) 

    def encode_state(self, close_pct, vol_pct, rsi):
        """
        Encodes state S = (CLOSE * Level(close)) + (VOL * Level(vol)) ...
        """
        v_close = self.encode_value(close_pct)
        v_vol = self.encode_value(vol_pct)
        v_rsi = self.encode_value(rsi)
        
        # We need identifying vectors for the fields (ID Vectors)
        # Ideally these are constant.
        # For simplicity, we just bundle the values themselves for now, 
        # OR we bind them to random ID vectors if we were strict.
        # Let's bind to ID vectors to differentiate fields.
        
        # ID Vectors
        if 'ID_CLOSE' not in self._cache: self._cache['ID_CLOSE'] = HyperVector()
        if 'ID_VOL' not in self._cache: self._cache['ID_VOL'] = HyperVector()
        if 'ID_RSI' not in self._cache: self._cache['ID_RSI'] = HyperVector()
        
        f_close = self._cache['ID_CLOSE'].bind(v_close)
        f_vol = self._cache['ID_VOL'].bind(v_vol)
        f_rsi = self._cache['ID_RSI'].bind(v_rsi)
        
        return f_close.bundle(f_vol).bundle(f_rsi)

class HyperDimensionalEngine:
    """
    Orchestrates HDC operations for the Swarm.
    """
    def __init__(self):
        self.encoder = HDCEncoder()
        
    def encode(self, state_dict: dict) -> HyperVector:
        return self.encoder.encode_state(
            state_dict.get('close_pct', 50),
            state_dict.get('vol_pct', 50),
            state_dict.get('rsi', 50)
        )
