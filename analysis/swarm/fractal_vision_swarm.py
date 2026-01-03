
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal
from scipy.spatial.distance import euclidean

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

logger = logging.getLogger("FractalVisionSwarm")

class FractalVisionSwarm(SubconsciousUnit):
    """
    The Eye Reborn.
    Uses Dynamic Time Warping to see Shapes in the Chaos.
    """
    def __init__(self):
        super().__init__("Fractal_Vision_Swarm")
        self.patterns = self._load_patterns()

    def _load_patterns(self):
        # Define some basic shapes normalized (0 to 1)
        # 10 point sequences
        patterns = {}
        
        # Bull Flag (Impulse up, consolidation down)
        patterns['BULL_FLAG'] = np.array([0.0, 0.5, 1.0, 0.9, 0.8, 0.85, 0.8, 0.9, 0.95, 1.2])
        
        # Bear Flag (Impulse down, consolidation up)
        patterns['BEAR_FLAG'] = np.array([1.0, 0.5, 0.0, 0.1, 0.2, 0.15, 0.2, 0.1, 0.05, -0.2])
        
        # Head and Shoulders (Top)
        patterns['HEAD_SHOULDERS'] = np.array([0.0, 0.5, 0.2, 0.8, 0.2, 0.5, 0.0])
        
        return patterns

    def _normalize(self, sequence):
        # Min-Max Normalization to 0-1
        seq = np.array(sequence)
        result = (seq - seq.min()) / (seq.max() - seq.min())
        return result

    async def process(self, context) -> SwarmSignal:
        if fastdtw is None: 
            return None # Library missing logic handling
            
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 30: return None
        
        # Extract last N candles (matches pattern length approx)
        # We try matching different window sizes: 10, 20
        closes = df_m5['close'].values
        
        best_match = None
        best_dist = float('inf')
        
        windows = [10, 15, 20] # Scales
        
        for name, template in self.patterns.items():
            for w in windows:
                if len(closes) < w: continue
                
                segment = closes[-w:]
                # Resample segment to match template length using interpolation
                # Or Resample template to match segment?
                # Easier: Interpolate segment to 10 points (template size)
                
                x_old = np.linspace(0, 1, len(segment))
                x_new = np.linspace(0, 1, len(template))
                segment_resampled = np.interp(x_new, x_old, segment)
                
                segment_norm = self._normalize(segment_resampled)
                
                # Use Accelerated Kernel
                from core.acceleration import jit_dtw_distance
                # JIT DTW expects flat arrays, which we have.
                distance = jit_dtw_distance(segment_norm, template)
                
                if distance < best_dist:
                    best_dist = distance
                    best_match = name
                    
        # Threshold for validation
        # Dist depends on length, but normalized 0-1.
        # usually < 1.0 or 2.0 is good match.
        
        if best_match and best_dist < 2.0:
            signal = "WAIT"
            confidence = 60 # Base
            
            # Map Pattern to Signal
            if best_match == 'BULL_FLAG': signal = "BUY"
            elif best_match == 'BEAR_FLAG': signal = "SELL"
            elif best_match == 'HEAD_SHOULDERS': signal = "SELL"
            
            # Confidence inverse to distance
            confidence = max(50, 90 - (best_dist * 10))
            
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'pattern': best_match, 'dtw_dist': best_dist}
            )
            
        return None
