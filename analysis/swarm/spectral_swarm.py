
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal
import numpy.fft as fft

logger = logging.getLogger("SpectralSwarm")

class SpectralSwarm(SubconsciousUnit):
    """
    The Time Lord.
    Uses FFT to identify dominant market cycles and phase alignment.
    """
    def __init__(self):
        super().__init__("Spectral_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 128: return None
        
        # 1. Prepare Data
        # Detrending is crucial for FFT
        close = df_m5['close'].values
        # Simple detrend: price - linear regression
        x = np.arange(len(close))
        p = np.polyfit(x, close, 1)
        trend = np.polyval(p, x)
        detrended = close - trend
        
        # 2. Perform FFT
        n = len(detrended)
        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        spectrum = fft.fft(detrended * window)
        frequencies = fft.fftfreq(n)
        
        # 3. Find Dominant Frequency
        # We look at the magnitude spectrum for positive freq
        magnitude = np.abs(spectrum[:n//2])
        freqs = frequencies[:n//2]
        
        # Ignore DC component (0) and very low freqs
        magnitude[0] = 0
        
        peak_idx = np.argmax(magnitude)
        dominant_freq = freqs[peak_idx]
        
        if dominant_freq == 0: return None
        
        period = 1 / dominant_freq # In bars
        
        # 4. Phase Analysis
        # Where are we in the cycle? 0 - 2pi
        phase = np.angle(spectrum[peak_idx])
        
        # Reconstruct just this wave: A * cos(wt + phi)
        # We check the slope of the reconstruction at the end
        t_next = n
        cycle_val = magnitude[peak_idx] * np.cos(2 * np.pi * dominant_freq * t_next + phase)
        cycle_prev = magnitude[peak_idx] * np.cos(2 * np.pi * dominant_freq * (t_next-1) + phase)
        
        slope = cycle_val - cycle_prev
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        # Logic: If Cycle is turning up -> Buy
        if slope > 0 and cycle_prev < 0:
            # Turning point up (Trough)
            signal = "BUY"
            confidence = 75
            reason = f"Cycle Trough (Period {period:.1f} bars)"
            
        elif slope < 0 and cycle_prev > 0:
            # Turning point down (Peak)
            signal = "SELL"
            confidence = 75
            reason = f"Cycle Peak (Period {period:.1f} bars)"
            
        # Refinement: Add Harmonic check (Secondary peaks)
        
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'period': period, 'phase': phase}
            )
            
        return None
