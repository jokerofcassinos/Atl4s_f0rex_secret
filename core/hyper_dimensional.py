
import numpy as np
import logging

logger = logging.getLogger("HyperCore")

# Dimensionality
D = 10000 

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
        """
        XOR (Multiplication in bipolar).
        Preserves distance. Invertible.
        Used to bind Variable to Value (e.g. Price * High).
        """
        return HyperVector(self.values * other.values)

    def bundle(self, other):
        """
        Superposition (Addition).
        Result is similar to both inputs.
        """
        # Element-wise sum of bipolar vectors
        # Normalization (Majority rule) happens contextually or we keep the integers for a bit
        # Simple Majority Rule:
        res = self.values + other.values
        # Threshold back to -1, 1
        res[res > 0] = 1
        res[res < 0] = -1
        # Random breaking of ties (zeros)
        res[res == 0] = np.random.choice([-1, 1], size=np.sum(res==0))
        return HyperVector(res)
    
    def permute(self, shifts=1):
        """
        Cyclic Shift (Roll).
        Encodes Sequence / Time.
        """
        return HyperVector(np.roll(self.values, shifts))

    def similarity(self, other):
        """
        Cosine Similarity (Normalized Dot Product).
        For Bipolar vectors, this is 1 - 2*HammingDist/D.
        Range: -1 to 1.
        """
        dot = np.dot(self.values, other.values)
        return dot / D

    @staticmethod
    def batch_bundle(vectors):
        if not vectors: return HyperVector()
        sum_vec = np.zeros(D, dtype=np.int32)
        for v in vectors:
            sum_vec += v.values
        
        # Initial majority
        res = np.zeros(D, dtype=np.int8)
        res[sum_vec > 0] = 1
        res[sum_vec < 0] = -1
        res[res == 0] = np.random.choice([-1, 1], size=np.sum(res==0))
        return HyperVector(res)

# Pre-computed Base Vectors (Item Memory)
# We need base vectors for Features (Close, Vol, RSI) and Values (High, Low, Up, Down)
class HDCEncoder:
    def __init__(self):
        self.features = {
            'CLOSE': HyperVector(),
            'VOLUME': HyperVector(),
            'RSI': HyperVector(),
            'TREND': HyperVector()
        }
        # Continuous value mapping requires Level Hypervectors
        # We generate a "Thermometer code" or interpolated vectors
        self.levels = []
        base = HyperVector()
        destination = HyperVector()
        # Create 100 interpolated vectors
        # Simple random flipping approach for now (Orthogonal Levels)
        # Actually, for continuous values, we want Sim(L1, L2) to be high.
        # So we flip bits progressively.
        
        self.level_vecs = []
        current = np.copy(base.values)
        # Flip 100 bits per step? D=10000, 100 steps.
        flip_mask = np.arange(D)
        np.random.shuffle(flip_mask)
        
        chunk = D // 100
        
        for i in range(101):
            self.level_vecs.append(HyperVector(np.copy(current)))
            # Flip chunk bits
            start = i * chunk
            end = (i+1) * chunk
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
