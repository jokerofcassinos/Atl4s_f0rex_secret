
import numpy as np
import logging

logger = logging.getLogger("HyperCore")

# Dimensionality
D = 10000 


# Try Load C++ Lib
_hdc_lib = None
try:
    import ctypes
    from core.cpp_loader import load_dll
    
    _hdc_lib = load_dll("hdc_core.dll")
    
    # void bind_vectors(const int8_t* val_a, const int8_t* val_b, int8_t* result, int size)
    # void bind_vectors(const int8_t* val_a, const int8_t* val_b, int8_t* result, int size)
    _hdc_lib.bind_vectors.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), 
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int
    ]
    
    # void bundle_vectors(const int8_t* inputs, int num, int8_t* result, int size)
    _hdc_lib.bundle_vectors.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int, 
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int
    ]
    
    # double cosine_similarity(const int8_t* val_a, const int8_t* val_b, int size)
    _hdc_lib.cosine_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_int
    ]
    _hdc_lib.cosine_similarity.restype = ctypes.c_double
        
        # logger.info("HDC: Silicon Memory Active (C++)")
except Exception as e:
    # logger.warning(f"HDC: C++ Load Failed ({e})")
    pass

class HyperVector:
    """
    10,000-bit vector wrapper for HDC operations.
    Using 'dense' bipolar representation (-1, 1) or binary (0, 1).
    We use Bipolar (-1, 1) for easier math (Multiplication = XOR).
    """
    def __init__(self, data=None):
        if data is None:
            # Random initialization (Orthogonal by high probability)
            self.values = np.random.choice([-1, 1], size=D).astype(np.int8)
        else:
            self.values = data.astype(np.int8)

    def bind(self, other):
        """XOR (Multiplication in bipolar)."""
        if _hdc_lib:
            res = np.zeros(D, dtype=np.int8)
            c_a = self.values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            c_b = other.values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            c_res = res.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
            
            _hdc_lib.bind_vectors(c_a, c_b, c_res, D)
            return HyperVector(res)
        else:
            return HyperVector(self.values * other.values)

    def bundle(self, other):
        """Superposition (Addition)."""
        if _hdc_lib:
             # Fast Bundle
             inputs = np.concatenate([self.values, other.values])
             res = np.zeros(D, dtype=np.int8)
             c_in = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
             c_res = res.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
             _hdc_lib.bundle_vectors(c_in, 2, c_res, D)
             return HyperVector(res)
        else:
            res = self.values + other.values
            res[res > 0] = 1
            res[res < 0] = -1
            res[res == 0] = np.random.choice([-1, 1], size=np.sum(res==0))
            return HyperVector(res)
    
    def permute(self, shifts=1):
        """Cyclic Shift (Roll)."""
        return HyperVector(np.roll(self.values, shifts))

    def similarity(self, other):
        """Cosine Similarity."""
        if _hdc_lib:
             c_a = self.values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
             c_b = other.values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
             return _hdc_lib.cosine_similarity(c_a, c_b, D)
        else:
            dot = np.dot(self.values, other.values)
            return dot / D

    @staticmethod
    def batch_bundle(vectors):
        if not vectors: return HyperVector()
        
        if _hdc_lib:
             # Flatten
             raw_arrays = [v.values for v in vectors]
             flat = np.concatenate(raw_arrays)
             res = np.zeros(D, dtype=np.int8)
             
             c_in = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
             c_res = res.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
             
             _hdc_lib.bundle_vectors(c_in, len(vectors), c_res, D)
             return HyperVector(res)
        else:
            sum_vec = np.zeros(D, dtype=np.int32)
            for v in vectors:
                sum_vec += v.values
            
            res = np.zeros(D, dtype=np.int8)
            res[sum_vec > 0] = 1
            res[sum_vec < 0] = -1
            res[res == 0] = np.random.choice([-1, 1], size=np.sum(res==0))
            return HyperVector(res)

# Pre-computed Base Vectors (Item Memory)
class HDCEncoder:
    def __init__(self):
        self.features = {
            'CLOSE': HyperVector(),
            'VOLUME': HyperVector(),
            'RSI': HyperVector(),
            'TREND': HyperVector()
        }
        # Continuous value mapping requires Level Hypervectors
        self.levels = []
        base = HyperVector()
        
        # Create 100 interpolated vectors
        self.level_vecs = []
        current = np.copy(base.values)
        flip_mask = np.arange(D)
        np.random.shuffle(flip_mask)
        
        chunk = D // 100
        
        for i in range(101):
            self.level_vecs.append(HyperVector(np.copy(current)))
            # Flip chunk bits
            start = i * chunk
            end = i * chunk + chunk
            if end > D: end = D
            current[flip_mask[start:end]] *= -1 # Flip sign
            
    def encode_value(self, val_0_100):
        """Encodes a scalar 0-100 into a HyperVector"""
        idx = int(np.clip(val_0_100, 0, 100))
        return self.level_vecs[idx]

    def encode_state(self, close_pct, vol_pct, rsi):
        """
        Encodes state S = (CLOSE * Level(close)) + (VOL * Level(vol)) ...
        """
        # CLOSE * Val
        v_close = self.features['CLOSE'].bind(self.encode_value(close_pct))
        v_vol = self.features['VOLUME'].bind(self.encode_value(vol_pct))
        v_rsi = self.features['RSI'].bind(self.encode_value(rsi))
        
        # Bundle
        state = HyperVector.batch_bundle([v_close, v_vol, v_rsi])
        return state

class HyperDimensionalEngine:
    """
    Orchestrates HDC operations for the Swarm.
    """
    def __init__(self):
        self.encoder = HDCEncoder()
        
    def encode(self, state_dict):
        # Wrapper for encoding
        return self.encoder.encode_state(
            state_dict.get('close_pct', 50),
            state_dict.get('vol_pct', 50),
            state_dict.get('rsi', 50)
        )
