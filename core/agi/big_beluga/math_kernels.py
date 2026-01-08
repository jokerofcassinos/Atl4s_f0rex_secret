
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger("BigBelugaMath")

class BigBelugaMath:
    @staticmethod
    def lorentzian_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula a distância Lorentziana entre dois vetores.
        Fórmula: log(1 + |x - y|)
        """
        diff = np.abs(x - y)
        return np.sum(np.log(1 + diff))

    @staticmethod
    def gaussian_filter_2pole(data: np.ndarray, period: int) -> np.ndarray:
        """
        Filtro Gaussiano de 2 Pólos para suavização com lag mínimo.
        """
        beta = 2.415 * (1 - np.cos(2 * np.pi / period))
        alpha = -beta + np.sqrt(beta * (beta + 2))
        
        output = np.zeros_like(data)
        if len(data) < 2: return data
        
        output[0] = data[0]
        for i in range(1, len(data)):
            output[i] = (alpha * data[i]) + ((1 - alpha) * output[i-1])
            
        return output

    @staticmethod
    def kalman_filter(data: np.ndarray, process_noise: float = 1e-5, measurement_noise: float = 1e-3) -> np.ndarray:
        """
        Modified 1D Kalman Filter for price smoothing.
        Reduces lag compared to standard MAs by dynamically adjusting gain.
        """
        n = len(data)
        if n == 0: return data
        
        # State initialization
        x_est = data[0]
        p_est = 1.0
        
        output = np.zeros(n)
        output[0] = x_est
        
        for i in range(1, n):
            # Prediction
            x_pred = x_est
            p_pred = p_est + process_noise
            
            # Update
            k_gain = p_pred / (p_pred + measurement_noise)
            x_est = x_pred + k_gain * (data[i] - x_pred)
            p_est = (1 - k_gain) * p_pred
            
            output[i] = x_est
            
        return output

    @staticmethod
    def hurst_exponent(time_series: np.ndarray) -> float:
        """
        Calculates the Hurst Exponent using Rescaled Range (R/S) analysis.
        H < 0.5: Mean Reverting
        H = 0.5: Random Walk (Brownian)
        H > 0.5: Trending (Persistent)
        """
        ts = time_series
        n = len(ts)
        if n < 20: return 0.5 # Not enough data
        
        # Calculate log returns
        # Avoid zero division or log of zero issues by adding small epsilon
        returns = np.diff(np.log(ts + 1e-9))
        
        # Split into chunks (simplified R/S for speed)
        # We'll use a single range calculation for the whole series for efficiency in this bot context
        mean = np.mean(returns)
        cumulative_deviation = np.cumsum(returns - mean)
        r_range = np.max(cumulative_deviation) - np.min(cumulative_deviation)
        s_std = np.std(returns)
        
        if s_std == 0: return 0.5
        
        rs = r_range / s_std
        # Hurst = log(R/S) / log(n)
        # This is a simplified point estimate, not a full regression over multiple scales
        hurst = np.log(rs) / np.log(n)
        
        return float(hurst)

    @staticmethod
    def action_wave(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
        """
        ActionWave: Volatility-weighted polynomial regression wave.
        Identifies 'Kinetic Energy' in price movement.
        """
        n = len(price)
        if n < 10: return np.zeros(n)
        
        # 1. Volatility Weighting
        vol_mean = np.mean(vol) + 1e-9
        vol_weights = vol / vol_mean
        
        # 2. Weighted Moving Average (Kinetic Base)
        # Simple weighted convolution for speed
        weighted_price = price * vol_weights
        
        # 3. Polynomial Fit (Trend Direction)
        # fit last 20 points
        lookback = min(n, 20)
        y_segment = weighted_price[-lookback:]
        x_segment = np.arange(lookback)
        
        coeffs = np.polyfit(x_segment, y_segment, 2)
        trend_curve = np.polyval(coeffs, x_segment)
        
        # Return the curve padded to match original length (mostly generic zeros for history)
        full_curve = np.zeros(n)
        full_curve[-lookback:] = trend_curve
        
        return full_curve
