
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("WaveletSwarm")

class WaveletSwarm(SubconsciousUnit):
    """
    The Filter.
    Uses Discrete Wavelet Transform (Haar) to de-noise price action.
    """
    def __init__(self):
        super().__init__("Wavelet_Swarm")

    def haar_transform(self, data):
        """
        Simple 1-Level Haar DWT.
        Returns (Approximation, Detail) arrays.
        """
        data = np.array(data)
        if len(data) % 2 != 0:
            data = data[:-1] # Truncate odd
            
        n = len(data) // 2
        approx = np.zeros(n)
        detail = np.zeros(n)
        
        for i in range(n):
            approx[i] = (data[2*i] + data[2*i+1]) / np.sqrt(2)
            detail[i] = (data[2*i] - data[2*i+1]) / np.sqrt(2)
            
        return approx, detail

    def inverse_haar(self, approx, detail):
        """
        Reconstructs signal.
        """
        n = len(approx)
        rec = np.zeros(n * 2)
        for i in range(n):
            rec[2*i] = (approx[i] + detail[i]) / np.sqrt(2)
            rec[2*i+1] = (approx[i] - detail[i]) / np.sqrt(2)
        return rec

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 32: return None
        
        close = df_m5['close'].values
        
        # 1. Decompose (Level 1)
        a1, d1 = self.haar_transform(close)
        
        # 2. Decompose (Level 2) - Extract Trend
        a2, d2 = self.haar_transform(a1)
        
        # 3. Filter Noise (Kill Detail)
        # We zero out d1 (High freq noise) and d2 (Medium freq noise)
        # Reconstruct purely from A2 (Trend)
        d2_clean = np.zeros_like(d2)
        d1_clean = np.zeros_like(d1) # Zero noise
        
        rec_a1 = self.inverse_haar(a2, d2_clean) # Reconstruct Level 1 approx
        if len(rec_a1) < len(a1): # Pad check if lengths differ due to odd/even
             pass 
             
        rec_final = self.inverse_haar(rec_a1, d1_clean) # Reconstruct Final
        
        # Match lengths
        actual_len = len(rec_final)
        
        # 4. Analyze 'Clean' Trend
        # Slope of the last few points of the clean signal
        slope = rec_final[-1] - rec_final[-3]
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        if slope > 0:
            signal = "BUY"
            confidence = 80
            reason = "Wavelet Trend (De-noised) Positive"
        elif slope < 0:
            signal = "SELL"
            confidence = 80
            reason = "Wavelet Trend (De-noised) Negative"
            
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'slope': slope}
            )
            
        return None
