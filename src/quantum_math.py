import numpy as np
import pandas as pd
from scipy.stats import entropy

class QuantumMath:
    """
    Advanced mathematical models for market analysis ("Quantum" Layer).
    Includes Entropy, Hurst Exponent, and Kalman Filters.
    """

    @staticmethod
    def calculate_entropy(series: pd.Series, window=14):
        """
        Calculates Shannon Entropy to measure market disorder.
        High Entropy = Chaos/Sideways.
        Low Entropy = Order/Trend.
        """
        def _entropy(data):
            # Discretize data into bins to calculate probability distribution
            hist, bin_edges = np.histogram(data, bins='auto', density=True)
            # hist contains probabilities
            return entropy(hist)

        return series.rolling(window=window).apply(_entropy)

    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, window=100):
        """
        Estimates Hurst Exponent to determine if series is random walk, trending, or mean reverting.
        H < 0.5: Mean Reverting
        H = 0.5: Random Walk (Brownian Motion)
        H > 0.5: Trending
        Note: Simplified R/S analysis implementation for rolling window.
        """
        def _hurst(ts):
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0 

        # Rolling Hurst is computationally expensive, use with care.
        # This is a simplified placeholder. Ideal Hursts require log-log plots over longer ranges.
        return series.rolling(window=window).apply(_hurst) # This might be slow for real-time M5 if window is huge

    @staticmethod
    def kalman_filter(series: pd.Series, q=0.0001, r=0.1):
        """
        Applies a 1D Kalman Filter to smooth price and find 'True Value'.
        q: Process variance (Reaction speed)
        r: Measurement variance (Noise dampening)
        """
        n_iter = len(series)
        sz = (n_iter,) # size of array
        
        # Allocate space for arrays
        xhat = np.zeros(sz)      # a posteri estimate of x
        P = np.zeros(sz)         # a posteri error estimate
        xhatminus = np.zeros(sz) # a priori estimate of x
        Pminus = np.zeros(sz)    # a priori error estimate
        K = np.zeros(sz)         # gain or blending factor

        # Initial guesses
        xhat[0] = series.iloc[0]
        P[0] = 1.0

        values = series.values

        for k in range(1, n_iter):
            # Time update
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + q

            # Measurement update
            K[k] = Pminus[k] / (Pminus[k] + r)
            xhat[k] = xhatminus[k] + K[k] * (values[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        return pd.Series(xhat, index=series.index)

    @staticmethod
    def z_score(series: pd.Series, window=20):
        """Calculates Rolling Z-Score."""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std
