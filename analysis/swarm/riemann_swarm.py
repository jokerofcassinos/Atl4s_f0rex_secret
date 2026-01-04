
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
        if prime_power > 10.0: # 10% of energy in Primes
            if score_buy > score_sell:
                dominant_signal = "BUY"
                dominant_conf = score_buy * 2.0 # Amplify impact
            elif score_sell > score_buy:
                dominant_signal = "SELL"
                dominant_conf = score_sell * 2.0
            
            meta = {
                'prime_energy': prime_power,
                'harmonics': ", ".join(reason_parts)
            }
            logger.info(f"RIEMANN ZETA: Prime Harmonics (Energy: {prime_power:.1f}%). {meta['harmonics']}")
            
            return SwarmSignal(
                source=self.name,
                signal_type=dominant_signal,
                confidence=min(dominant_conf, 95.0), 
                timestamp=None, 
                meta_data=meta
            )
            
        return None
