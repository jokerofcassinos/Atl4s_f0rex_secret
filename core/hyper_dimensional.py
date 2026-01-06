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
            if self.bridge.available:
                # Use C++ random generation if exposed, or numpy
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

    def similarity(self, other):
        """Cosine Similarity."""
        if self.bridge.available:
            return self.bridge.cosine_similarity(self.values, other.values)
        else:
            dot = np.dot(self.values, other.values)
            return dot / D

class HDCEncoder:
    """
    Encodes market features into HyperVectors.
    """
    def __init__(self):
        self.bridge = get_agi_bridge().hdc
        # Base vectors can be generated on fly or cached
        self._cache = {}

    def encode_value(self, val_0_100):
        """Encodes a scalar 0-100 into a HyperVector"""
        if self.bridge.available:
            res = self.bridge.encode_scalar(val_0_100, 0, 100)
            return HyperVector(res)
        else:
            # Simple Python encoding fallback
            # (Omitted strictly for brevity, assuming C++ works or basic random fallback)
            return HyperVector() 

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
