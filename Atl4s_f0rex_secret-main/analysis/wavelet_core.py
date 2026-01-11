import numpy as np
import logging

logger = logging.getLogger("Atl4s-Wavelet")

class WaveletCore:
    def __init__(self):
        # Scales correspond to frequency bands. 
        # Low scale = High Frequency (Fast noise)
        # High scale = Low Frequency (Trend)
        self.scales = np.arange(1, 31) 

    def morlet(self, t, s):
        """
        Simplified Real-valued Morlet Wavelet for convolution.
        """
        w = np.exp(-t**2 / (2 * s**2)) * np.cos(5 * t / s)
        return w

    def decompose(self, data):
        """
        Performs Continuous Wavelet Transform (CWT) using convolution.
        Returns:
            power_spectrum (dict): Energy at different scales.
            dominant_scale (float): Scale with max energy.
            coherence (float): Measure of signal clarity.
        """
        if data is None or len(data) < 50:
            return {'energy_fast': 0, 'energy_slow': 0, 'dominant_scale': 0, 'coherence': 0}
            
        # Normalize data
        x = np.array(data)
        x = (x - np.mean(x)) / (np.std(x) + 1e-9)
        
        n = len(x)
        energies = []
        
        # We only analyze the recent window for performance
        window_size = min(n, 60)
        recent_x = x[-window_size:]
        
        t = np.arange(-window_size//2, window_size//2)
        
        total_energy = 0
        max_e = 0
        dom_s = 0
        
        energy_fast = 0 # Scales 1-10
        energy_slow = 0 # Scales 20-30
        
        for s in self.scales:
            # Generate wavelet
            w = self.morlet(t, s)
            # Normalize wavelet energy
            w = w / (np.sqrt(np.sum(w**2)) + 1e-9)
            
            # Convolve (Valid mode to avoid edge effects, take last point)
            # Actually we want the 'instantaneous' transform at the end
            # So dot product of flipped wavelet with recent data matches convolution at lag 0
            
            # Ensure dims match
            if len(w) > len(recent_x):
                w = w[-len(recent_x):]
                
            # Dot product is the convolution response at the current time t
            response = np.dot(recent_x, w) 
            power = response ** 2
            
            energies.append(power)
            total_energy += power
            
            if power > max_e:
                max_e = power
                dom_s = s
                
            if s <= 10: energy_fast += power
            elif s >= 20: energy_slow += power
            
        # Coherence: How concentrated is the energy?
        # High coherence means the market is "singing" a clear note (Cycle)
        # Low coherence means white noise
        coherence = max_e / (total_energy + 1e-9)
        
        return {
            'energy_fast': energy_fast,
            'energy_slow': energy_slow,
            'dominant_scale': dom_s,
            'coherence': coherence
        }
