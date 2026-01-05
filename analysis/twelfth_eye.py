import logging
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Any

logger = logging.getLogger("Atl4s-TwelfthEye")

class TwelfthEye:
    """
    The Twelfth Eye: Chronos (The Time-Crystal Engine).
    
    Role:
    - Analyzes market periodicity and "Time Crystals" (repeating structures in time).
    - Uses the Kuramoto Model to measure synchronization between different timeframes.
    - Predicts "Time to Impact" (TTI) phase transitions.
    
    Physics:
    - Market oscillators (M5, M15, H1) are treated as coupled pendulums.
    - When they sync (Order Parameter r -> 1), a violent move is guaranteed.
    """
    def __init__(self):
        self.name = "Chronos (Time Keeper)"
        self.phases = {
            'M5': 0.0,
            'M15': 0.0,
            'H1': 0.0
        }
        self.natural_frequencies = {
            'M5': 2 * np.pi / 5,
            'M15': 2 * np.pi / 15,
            'H1': 2 * np.pi / 60
        }
        self.coupling_constant = 0.5 # K
        
    def calculate_sync_index(self, data_map: Dict[str, Any]) -> float:
        """
        Calculates the Kuramoto Order Parameter (r).
        Range: 0 (Chaos/Noise) to 1 (Perfect Synchronization/Singularity).
        """
        complex_sum = 0 + 0j
        count = 0
        
        # We need "Phase" data. 
        # Proxy: Normalized position of price within RSI cycle or Stochastic cycle in that timeframe.
        # This is accurate enough for a trading bot approximation of phase.
        
        for timeframe, df in data_map.items():
            if not isinstance(df, pd.DataFrame) or df.empty: continue
            if timeframe not in self.phases: continue
            
            # Extract Phase (0 to 2pi)
            # Use RSI because it is a natural oscillator
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0
            
            # Map RSI 0-100 to Phase 0-2pi
            # RSI 0 = 0 (Bottom), RSI 50 = pi (Middle), RSI 100 = 2pi (Top)?
            # Actually: A full cycle is low -> high -> low. 
            # Simple map: phase = (rsi / 100) * 2 * pi
            phase = (rsi / 100.0) * 2 * np.pi
            
            self.phases[timeframe] = phase
            
            # Sum complex vectors e^(i*theta)
            complex_sum += np.exp(1j * phase)
            count += 1
            
        if count == 0: return 0.0
        
        # r = |(1/N) * sum(e^itheta)|
        mean_vector = complex_sum / count
        r = np.abs(mean_vector) # Magnitude
        
        return float(r)

    def detect_time_crystal(self, prices: np.ndarray) -> Dict:
        """
        Detects Discrete Time Symmetry Breaking (Time Crystals).
        Looks for Period Doubling Bifurcations in the return series.
        """
        if len(prices) < 64: return None
        
        # 1. FFT
        fft_vals = np.fft.fft(prices)
        fft_freq = np.fft.fftfreq(len(prices))
        
        # Filter positive freqs
        mask = fft_freq > 0
        power_spectrum = np.abs(fft_vals[mask])**2
        freqs = fft_freq[mask]
        
        # 2. Find Peak
        peak_idx = np.argmax(power_spectrum)
        dominant_freq = freqs[peak_idx]
        
        if dominant_freq == 0: return None
        
        period = 1.0 / dominant_freq
        
        # 3. Check for Sub-Harmonics (Period Doubling)
        # A Time Crystal beats at f/2, f/4...
        
        # Look for significant power at exactly half the dominant frequency
        target_f = dominant_freq / 2.0
        
        # Find closest frequency bin
        idx_sub = (np.abs(freqs - target_f)).argmin()
        
        sub_power = power_spectrum[idx_sub]
        peak_power = power_spectrum[peak_idx]
        
        # Ratio
        ratio = sub_power / peak_power
        
        is_crystal = False
        description = "Noise"
        
        if ratio > 0.3: # Significant sub-harmonic energy
             is_crystal = True
             description = f"TIME CRYSTAL: Period Doubling Detected (T={period:.1f}->{period*2:.1f})"
             
        return {
            'is_crystal': is_crystal,
            'period': period,
            'sub_harmonic_ratio': ratio,
            'desc': description
        }

    def analyze(self, data_map: Dict[str, Any]) -> Dict:
        # 1. Sync
        r = self.calculate_sync_index(data_map)
        
        # 2. Crystal
        df_m5 = data_map.get('M5')
        crystal_data = None
        if df_m5 is not None:
             closes = df_m5['close'].values
             # Detrend for FFT
             returns = np.diff(closes)
             crystal_data = self.detect_time_crystal(returns)
             
        return {
            'synchronization_index': r,
            'time_crystal': crystal_data,
            'human_readout': f"Sync: {r:.2f} | {crystal_data['desc'] if crystal_data else 'No Data'}"
        }
