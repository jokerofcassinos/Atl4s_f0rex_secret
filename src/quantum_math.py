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
        Estimates Hurst Exponent using a more robust DFA-like approach.
        H < 0.5: Mean Reverting (Rough Volatility)
        H = 0.5: Random Walk
        H > 0.5: Trending (Persistent)
        """
        def _robust_hurst(ts):
            if len(ts) < 20: return 0.5
            lags = range(2, min(len(ts)//2, 30))
            # Calculate the variance of the differences at different lags
            tau = []
            for lag in lags:
                diffs = ts[lag:] - ts[:-lag]
                std = np.std(diffs)
                tau.append(max(std, 1e-6)) # Avoid zero std
            
            # Log-log slope
            try:
                # Use Hurst Exponent based on Average Range (Simplified R/S)
                # We Use the variance of increments at different scales
                log_lags = np.log(lags)
                log_variances = np.log([np.var(ts[lag:] - ts[:-lag]) for lag in lags])
                
                # Hurst = Slope / 2
                mask = np.isfinite(log_lags) & np.isfinite(log_variances)
                if np.sum(mask) < 2: return 0.5
                
                poly = np.polyfit(log_lags[mask], log_variances[mask], 1)
                h = poly[0] / 2.0
                return max(0.01, min(0.99, h))
            except:
                return 0.5

        return series.rolling(window=window).apply(_robust_hurst)

    @staticmethod
    def fisher_information_curvature(series: pd.Series, window=20):
        """
        Calculates the Scalar Curvature of the Statistical Manifold (Fisher Information).
        This measures the 'rate of change of the market distribution's geometry'.
        Spikes in curvature often precede regime transitions.
        """
        def _fisher_r(data):
            if len(data) < 10: return 0
            # We model the distribution as Gaussian (mu, sigma)
            # The Fisher Metric for Gaussian is g = [[1/sigma^2, 0], [0, 2/sigma^2]]
            # Scalar Curvature R for this manifold is -1 (constant negative curvature)
            # HOWEVER, we want the VELOCITY of change on this manifold (Fisher Distance)
            
            # Split window into two halves to measure informational distance
            mid = len(data) // 2
            h1 = data[:mid]
            h2 = data[mid:]
            
            mu1, sigma1 = np.mean(h1), np.std(h1) + 1e-9
            mu2, sigma2 = np.mean(h2), np.std(h2) + 1e-9
            
            # Fisher Distance between two Gaussians
            # d^2 = 2 * log( ( (mu1-mu2)^2 + 2*(sigma1^2 + sigma2^2) ) / (4 * sigma1 * sigma2) )
            distance = np.sqrt(max(0, 2 * np.log(((mu1 - mu2)**2 + 2*(sigma1**2 + sigma2**2)) / (4 * sigma1 * sigma2))))
            return distance

        return series.rolling(window=window).apply(_fisher_r)

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
