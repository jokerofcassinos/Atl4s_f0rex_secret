
import numpy as np
import logging
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("RiemannSwarm")

class RiemannSwarm(SubconsciousUnit):
    """
    Phase 115: The Riemann Zeta Swarm (Prime Harmonics).
    
    "The Market is Music. Primes are the Notes."
    
    Uses Spectral Analysis (FFT) to detect if market cycles align with Prime Numbers.
    - Coherence: Power(Primes) / Power(Total).
    - Signal: Phase of the Dominant Prime Harmonic.
    """
    
    def __init__(self):
        super().__init__("Riemann_Swarm")
        self.primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        self.min_data_points = 60 # Need at least 60 candles for FFT clarity
        
        try:
             from core.cpp_loader import load_dll
             load_dll("physics_core.dll")
             logger.info("RIEMANN ENGINE: C++ CORE ACTIVE [TURBO MODE]")
        except:
             logger.info("RIEMANN ENGINE: PYTHON FALLBACK [STANDARD MODE]")
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        candles = context.get('candles')
        if candles is None or len(candles) < self.min_data_points:
            return None
            
        # Extract Closing Prices
        closes = candles['close'].values
        
        # 1. Detrending (Remove linear trend to find cycles)
        x = np.arange(len(closes))
        p = np.polyfit(x, closes, 1)
        trend = np.polyval(p, x)
        detrended = closes - trend
        
        # 2. Fast Fourier Transform (FFT)
        # Transform Time Domain to Frequency Domain
        fft_vals = np.fft.rfft(detrended)
        fft_freq = np.fft.rfftfreq(len(detrended))
        
        # Power Spectrum
        power = np.abs(fft_vals) ** 2
        
        # 3. Find Dominant Cycles
        # Convert Freq to Period (Candles)
        # Period = 1 / Freq
        # Ignore DC component (index 0)
        
        valid_indices = np.where(fft_freq > 0)[0]
        periods = 1 / fft_freq[valid_indices]
        powers = power[valid_indices]
        
        # Sort by Power
        # Get Top 3 Dominant Cycles
        sorted_indices = np.argsort(powers)[::-1]
        top_indices = sorted_indices[:3]
        
        total_power = np.sum(powers)
        prime_power = 0.0
        dominant_signal = "WAIT"
        dominant_conf = 0.0
        
        # Track buy/sell pressure from primes
        score_buy = 0.0
        score_sell = 0.0
        
        reason_parts = []
        
        # 4. Check Prime Alignment (Riemann Hypothesis heuristic)
        for idx in top_indices:
            cycle_len = periods[idx]
            cycle_power = powers[idx]
            
            # Check proximity to nearest Prime
            nearest_prime = min(self.primes, key=lambda x: abs(x - cycle_len))
            error = abs(cycle_len - nearest_prime)
            
            # If error is small (< 15%), we have Harmonic Resonance
            if error < (nearest_prime * 0.15):
                score = (cycle_power / total_power) * 100
                prime_power += score
                
                # Determine Phase (Are we at Top or Bottom of this Prime Wave?)
                # Phase angle of FFT component
                phase = np.angle(fft_vals[valid_indices[idx]])
                
                # Derivative (Slope) at t_last determines direction
                t_last = len(detrended) - 1
                freq = fft_freq[valid_indices[idx]]
                omega = 2 * np.pi * freq
                slope = -omega * np.sin(omega * t_last + phase)
                
                direction = "UP" if slope > 0 else "DOWN"
                reason_parts.append(f"P({nearest_prime})->{direction}")
                
                if slope > 0:
                     score_buy += score
                else:
                     score_sell += score
        
        # 5. Synthesis
        curvature = self._calculate_curvature_bridge(closes)
        
        # Curvature Logic (General Relativity market analogy)
        # K > 0 (Spherical): Space is closing on itself -> Trend Exhaustion/Reversal
        # K < 0 (Hyperbolic): Space is expanding -> Trend Acceleration
        
        curvature_bias = "NEUTRAL"
        if curvature > 0.5:
             curvature_bias = "REVERSAL"
             # If Signal is BUY, but Curvature says Reversal -> Weaken confidence
             # If Signal is SELL, and Curvature says Reversal (of a uptrend) -> Strengthen?
             # Simplified: High Positive K acts as "Gravity" pulling price back.
        elif curvature < -0.5:
             curvature_bias = "EXPANSION"
             # Trend is accelerating.
             
        if prime_power > 10.0: # 10% of energy in Primes
            if score_buy > score_sell:
                dominant_signal = "BUY"
                dominant_conf = score_buy * 2.0 # Amplify impact
                
                if curvature_bias == "EXPANSION": dominant_conf += 10.0
                if curvature_bias == "REVERSAL": dominant_conf -= 20.0
                
            elif score_sell > score_buy:
                dominant_signal = "SELL"
                dominant_conf = score_sell * 2.0
                
                if curvature_bias == "EXPANSION": dominant_conf += 10.0
                if curvature_bias == "REVERSAL": dominant_conf -= 20.0
            
            meta = {
                'prime_energy': prime_power,
                'harmonics': ", ".join(reason_parts),
                'curvature': curvature,
                'geometry': curvature_bias
            }
            logger.info(f"RIEMANN ZETA: Primes({prime_power:.1f}%) | K={curvature:.4f} ({curvature_bias})")
            
            return SwarmSignal(
                source=self.name,
                signal_type=dominant_signal,
                confidence=min(dominant_conf, 95.0), 
                timestamp=None, 
                meta_data=meta
            )
            
        return None

    def _calculate_curvature_bridge(self, closes: np.ndarray) -> float:
        """Calculates Sectional Curvature (C++ or Py Fallback)"""
        try:
            import ctypes
            from core.cpp_loader import load_dll
            
            lib = load_dll("physics_core.dll")
            
            # double calculate_sectional_curvature(double* prices, int length, int window_size)
            lib.calculate_sectional_curvature.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
            ]
            lib.calculate_sectional_curvature.restype = ctypes.c_double
            
            # Prepare Array
            array_type = ctypes.c_double * len(closes)
            c_closes = array_type(*closes)
                 
            k = lib.calculate_sectional_curvature(c_closes, len(closes), 20)
            return k
        except Exception:
            pass
            
        # Python Fallback (Simplified Discrete Curvature)
        # K ~ y'' / (1 + y'^2)^1.5
        # We average K over the last 20 points
        if len(closes) < 20: return 0.0
        
        window = closes[-20:]
        total_k = 0.0
        
        for i in range(2, len(window)):
             # Scale derivatives to reasonable values (Price changes are small)
             # Multiply by 100 or 1000 to make 'dx' meaningful? 
             # Let's assume dx=1 (1 candle)
             
             y_now = window[i]
             y_prev = window[i-1]
             y_prev2 = window[i-2]
             
             dy = y_now - y_prev
             ddy = (y_now - y_prev) - (y_prev - y_prev2)
             
             denom = pow(1.0 + dy*dy, 1.5)
             if denom == 0: denom = 0.0001
             
             k = ddy / denom
             total_k += k
             
        return total_k / (len(window) - 2)
