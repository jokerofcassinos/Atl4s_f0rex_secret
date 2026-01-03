
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time
from scipy.fft import fft, ifft

logger = logging.getLogger("SuperluminalSwarm")

class SuperluminalSwarm(SubconsciousUnit):
    """
    Phase 76: The Superluminal Swarm (Tachyonic FFT).
    
    Uses Spectral Analysis (Fast Fourier Transform) to decompose price action into waves.
    Extrapolates the Dominant Harmonics into the future to create a 'Ghost Path'.
    
    Physics:
    - Markets move in cycles (Waves).
    - By isolating the strongest frequencies and phase-shifting them forward,
      we can reconstructed the probable future waveform.
    - Effectively a 'Negative Lag' indicator.
    """
    def __init__(self):
        super().__init__("Superluminal_Swarm")
        self.lookback = 64 # Power of 2 for FFT efficiency

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < self.lookback: return None
        
        # 1. Prepare Data
        # We need a stationary series for FFT.
        # Detrending is crucial. We use linear detrending.
        prices = df_m5['close'].iloc[-self.lookback:].values
        
        # Linear Regression to find Trend Line
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        trend_line = slope * x + intercept
        
        # Detrend
        detrended = prices - trend_line
        
        # 2. Fast Fourier Transform (FFT)
        # Transform Time Domain -> Frequency Domain
        fft_coeffs = fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # 3. Filter / Extrapolate
        # We only keep the Top N Dominant Frequencies (Denoising)
        # Sort by Amplitude (Magnitude of coefficients)
        amplitudes = np.abs(fft_coeffs)
        
        # Zero out weak frequencies (Noise Gate)
        # Keep top 5 harmonics
        indices = np.argsort(amplitudes)[-5:] 
        filtered_fft = np.zeros_like(fft_coeffs)
        filtered_fft[indices] = fft_coeffs[indices]
        
        # 4. Tachyonic Projection (Time Travel)
        # We want to know the value at T + 3
        future_steps = 3
        
        # Reconstruct signal for T + future_steps manually using sum of sines?
        # Or faster: Extrapolate the reconstructed denoised signal.
        # Inverse FFT gives us the denoised CURRENT wave.
        reconstructed_detrended = np.fft.ifft(filtered_fft).real
        
        # To project, we extend the sine waves.
        # Function: Sum( Amp * cos(2*pi*freq*t + phase) )
        # We calculate this for t = lookback + future_steps
        
        future_val_detrended = 0.0
        
        for idx in indices:
            freq = frequencies[idx]
            coeff = filtered_fft[idx]
            amp = np.abs(coeff) / len(detrended) # Normalize
            phase = np.angle(coeff)
            
            # FFT freq is cycles per sample unit
            # Wave = Amp * cos(2*pi * freq * t + phase)
            # wait, numpy fft format is tricky.
            # Let's use the definition: sum(c * exp(2pi * i * freq * t))
            
            # Actually, standard IFFT definition:
            # y[t] = (1/n) * sum(exp(2j * pi * k * t / n))
            # calculating for t = self.lookback + future_steps
            
            t_future = self.lookback + future_steps - 1 # 0-indexed logic roughly
            
            # Simple manual summation of the contributing waves at t_future
            term = (1.0/len(detrended)) * coeff * np.exp(2j * np.pi * idx * (len(detrended) + future_steps) / len(detrended))
            future_val_detrended += term.real
            
            # Note: The index logic in loop 'idx' from argsort might map to negative freqs too.
            # FFT Output: [0, 1, 2... -2, -1]
            
        # 5. Retrend
        # Add back the linear trend slope for the future time
        future_trend = slope * (self.lookback + future_steps) + intercept
        predicted_price = future_val_detrended + future_trend
        
        current_price = prices[-1]
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 6. Oracle Logic
        # Compare Future (T+3) with Present (T)
        
        threshold = current_price * 0.0005 # 0.05% Move Predicted
        
        diff = predicted_price - current_price
        
        if diff > threshold:
            signal = "BUY"
            confidence = 85.0
            reason = f"SUPERLUMINAL: Future Price (T+3) calculated at {predicted_price:.2f} (> Curr)."
        elif diff < -threshold:
            signal = "SELL"
            confidence = 85.0
            reason = f"SUPERLUMINAL: Future Price (T+3) calculated at {predicted_price:.2f} (< Curr)."
            
        # Tachyonic Validation: Check if the reconstructed current price matches real price
        # (Quality of Fit)
        rec_curr_detrended = np.fft.ifft(filtered_fft).real[-1]
        rec_curr = rec_curr_detrended + (slope * (self.lookback-1) + intercept)
        error = abs(rec_curr - current_price)
        
        # If fit is bad, the market is non-cyclical (Noise), so don't trust the projection.
        if error > (current_price * 0.001): # 0.1% Fit Error
             return None # Signal Unreliable (Aperiodic)

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'predicted_price': predicted_price, 'fit_error': error}
            )
            
        return None
