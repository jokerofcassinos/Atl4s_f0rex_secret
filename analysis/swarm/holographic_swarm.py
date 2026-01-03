
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("HolographicSwarm")

class HolographicSwarm(SubconsciousUnit):
    """
    The Quantum Hologram (Spectral Analysis).
    Phase 34 Innovation.
    Logic:
    1. Transforms Time-Series Price Data into Frequency Domain using FFT (Fast Fourier Transform).
    2. Identifies the "Dominant Cycle" (The wave with the highest amplitude).
    3. Filters out high-frequency noise (Smoothing).
    4. Projects the Phase of the Dominant Cycle.
       - Phase 0 or 2pi = Cycle Trough (Potential Bottom) -> BUY
       - Phase pi = Cycle Peak (Potential Top) -> SELL
    """
    def __init__(self):
        super().__init__("HolographicSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_h1') # Use H1 for Cyclical Stability
        if df is None or len(df) < 128: return None # FFT likes powers of 2 ideally, but >100 is needed
        
        # 1. Prepare Data (Detrending)
        # FFT requires stationary data usually, or we just look for cycles in the detrended series.
        close = df['close'].values
        # Simple linear detrending
        x = np.arange(len(close))
        poly = np.polyfit(x, close, 1)
        trend = np.polyval(poly, x)
        detrended = close - trend
        
        # 2. Perform FFT
        # We use a window of last 128 candles for spectral clarity
        # If we use too long, cycles shift. 128 hours ~ 1 week (Forex) is good.
        N = 128
        if len(detrended) < N: N = len(detrended)
        y = detrended[-N:]
        
        fft_vals = np.fft.rfft(y)
        fft_freq = np.fft.rfftfreq(N)
        
        # 3. Find Dominant Frequency
        # We ignore the 0 freq (DC component)
        amplitudes = np.abs(fft_vals)
        # Zero out low frequencies that are just trend remnants
        amplitudes[0:3] = 0 
        
        peak_idx = np.argmax(amplitudes)
        dominant_freq = fft_freq[peak_idx]
        
        if dominant_freq == 0: return None
        
        cycle_period = 1 / dominant_freq
        
        # 4. Phase Analysis
        # What is the phase of this specific frequency at the last data point?
        # Angle of the complex number
        phase_angle = np.angle(fft_vals[peak_idx])
        
        # The FFT tells us the phase at t=0 (start of window). 
        # We need phase at t=N (end of window).
        # Phase_t = Phase_0 + 2*pi*f*t
        current_phase = phase_angle + (2 * np.pi * dominant_freq * (N-1))
        
        # Normalize to 0 - 2pi
        current_phase = current_phase % (2 * np.pi)
        
        # 5. Interpretation
        # Sine wave: 
        # 0 = Center going Up (Bullish Momentum)
        # pi/2 = Peak (Overbought) -> Reversal Down
        # pi = Center going Down (Bearish Momentum)
        # 3pi/2 = Trough (Oversold) -> Reversal Up
        
        signal = "WAIT"
        confidence = 0.0
        bias = ""
        
        # We target the TURNING POINTS
        # Peak (Sell): Around pi/2 (1.57)
        # Trough (Buy): Around 3pi/2 (4.71)
        
        # Tolerance window
        tol = 0.5
        
        if abs(current_phase - (3 * np.pi / 2)) < tol:
            signal = "BUY"
            confidence = 75.0
            bias = "Cycle Trough"
        elif abs(current_phase - (np.pi / 2)) < tol:
            signal = "SELL"
            confidence = 75.0
            bias = "Cycle Peak"
        
        # Or Momentum logic?
        # If phase is between 3pi/2 and pi/2 (via 0) -> Rising
        # If phase is between pi/2 and 3pi/2 -> Falling
        
        # Let's stick to Reversal Logic for this Swarm (Counter-cyclical)
        
        if signal != "WAIT":
            return SwarmSignal(
                source="HolographicSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={
                    "reason": f"{bias} detected. Period: {cycle_period:.1f} bars",
                    "phase": current_phase,
                    "period": cycle_period
                }
            )
            
        return None
