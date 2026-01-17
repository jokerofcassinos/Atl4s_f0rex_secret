import numpy as np
import logging
from scipy.stats import entropy

logger = logging.getLogger("Atl4s-MathCore")

class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

class FractalGeometry:
    def __init__(self):
        pass

    def calculate_hurst_exponent(self, time_series):
        """
        Calculates the Hurst Exponent to determine long-term memory of time series.
        H < 0.5: Mean Reverting
        H ~ 0.5: Random Walk
        H > 0.5: Trending
        """
        lags = range(2, 20)
        
        # Calculate Tau (Standard Deviation of differences)
        # We need a list because time_series can be varying length
        ts_array = np.array(time_series)
        tau = []
        valid_lags = []
        
        for lag in lags:
            if len(ts_array) > lag:
                diff = np.subtract(ts_array[lag:], ts_array[:-lag])
                std = np.std(diff)
                if std > 0:
                    tau.append(std)
                    valid_lags.append(lag)
        
        # Avoid empty
        if len(tau) < 2:
             return 0.5

        # Use polyfit to estimate the slope of the log-log plot
        # log(std) = H * log(lag) + C
        # Note: This is a simplified method. R/S analysis is more robust but heavier.
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return poly[0] # The slope usually approximates H directly or H*const. Standard diffusion is H.

    def fractal_dimension(self, time_series):
        """
        Estimates Fractal Dimension (D).
        D = 2 - H
        """
        h = self.calculate_hurst_exponent(time_series)
        return 2 - h

class EntropyEngine:
    def __init__(self):
        pass

    def shannon_entropy(self, time_series, num_bins=10):
        """
        Calculates Shannon Entropy of the price distribution.
        Higher entropy = More randomness/Chaos.
        Lower entropy = More Order/structure.
        """
        try:
            hist, _ = np.histogram(time_series, bins=num_bins, density=False)
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                probs = hist / hist_sum
                probs = probs[probs > 0]
                return entropy(probs)
            else:
                return 0.0
        except Exception:
            return 0.0

    def approximate_entropy(self, U, m=2, r=0.2):
        """
        Approximate Entropy (ApEn) to detect regularity.
        U: Time series
        m: Embedding dimension
        r: Tolerance (multiplier of std)
        """
        try:
            U = np.array(U)
            N = len(U)
            std = np.std(U)
            if std == 0: return 0
            
            # This is O(N^2), can be slow for large N. Keep N small (~100).
            def _phi(m):
                x = np.array([U[j:j+m] for j in range(N-m+1)])
                # Pairwise distances
                # We need to optimize this for performance if N is large
                # For N=100, N^2 = 10000, feasible.
                C = []
                for i in range(len(x)):
                    # Chebyshev distance
                    dist = np.max(np.abs(x - x[i]), axis=1)
                    count = np.sum(dist <= r * std)
                    C.append(count / (N - m + 1))
                
                return np.sum(np.log(C)) / (N-m+1)
            
            # Limit precision errors
            if len(U) < 10: return 0.0
            
            return abs(_phi(m) - _phi(m+1))
        except Exception as e:
            # logger.error(f"ApEn Error: {e}")
            return 0.0

class BayesianRegime:
    def __init__(self):
        self.prior_trend = 0.5
        # Likelihoods updated with more math-heavy metrics
        self.likelihoods = {
            'high_hurst': {'trend': 0.8, 'range': 0.2},
            'low_hurst': {'trend': 0.2, 'range': 0.8},
            'low_entropy': {'trend': 0.7, 'range': 0.3},
            'high_entropy': {'trend': 0.3, 'range': 0.7},
            'high_kalman_vel': {'trend': 0.9, 'range': 0.1}
        }

    def update(self, evidence_list):
        posterior_trend = self.prior_trend
        for evidence in evidence_list:
            if evidence in self.likelihoods:
                l_trend = self.likelihoods[evidence]['trend']
                l_range = self.likelihoods[evidence]['range']
                p_evidence = l_trend * posterior_trend + l_range * (1 - posterior_trend)
                if p_evidence > 0:
                    posterior_trend = (l_trend * posterior_trend) / p_evidence
        
        posterior_trend = 0.9 * posterior_trend + 0.1 * 0.5
        self.prior_trend = posterior_trend
        return posterior_trend

class CycleMath:
    def __init__(self):
        pass

    def detect_dominant_cycle(self, prices):
        if len(prices) < 32: return 0, 0.0
        x = np.arange(len(prices))
        p = np.polyfit(x, prices, 1)
        detrended = prices - (p[0] * x + p[1])
        window = np.hanning(len(prices))
        windowed = detrended * window
        fft_vals = np.fft.rfft(windowed)
        power = np.abs(fft_vals) ** 2
        power[0] = 0
        dominant_idx = np.argmax(power)
        if dominant_idx == 0: return 0, 0.0
        fft_freq = np.fft.rfftfreq(len(windowed))
        dominant_freq = fft_freq[dominant_idx]
        if dominant_freq == 0: return 0, 0.0
        return 1 / dominant_freq, power[dominant_idx] / np.sum(power)

class MathCore:
    def __init__(self):
        self.bayes = BayesianRegime()
        self.cycle = CycleMath()
        self.fractal = FractalGeometry()
        self.entropy = EntropyEngine()
        self.kalman = KalmanFilter()

    def analyze(self, df):
        if df is None or df.empty:
            return {'regime_prob': 0.5, 'hurst': 0.5, 'entropy': 0, 'kalman_err': 0}

        closes = df['close'].values
        
        # 1. Kalman Filter Smoothing
        # Feed last few points to stabilize (in real-time we'd maintain state)
        # For this stateless implementation, we run on recent history
        # We assume 1D state for simplicity
        k_est = closes[0]
        # Re-initialize to avoid stale state from previous calls if we reused the object without reset
        # But here we keep the object alive. Let's just run update on the new points if possible,
        # but since 'analyze' is called stateless-ish with a new DF potentially, we might want to iterate.
        # Ideally, main loop calls 'update' per tick.
        # Here we simulate by running on the last 20 points.
        
        subset = closes[-20:] if len(closes) > 20 else closes
        for p in subset:  
             k_est = self.kalman.update(p)
        
        kalman_diff = closes[-1] - k_est
        
        # 2. Fractal Analysis
        # Hurst requires some length
        if len(closes) > 50:
            hurst = self.fractal.calculate_hurst_exponent(closes[-100:]) 
        else:
            hurst = 0.5
            
        # 3. Entropy Analysis
        if len(closes) > 20:
            s_entropy = self.entropy.shannon_entropy(closes[-50:])
        else:
            s_entropy = 0
            
        # 4. Bayesian Evidence
        evidence = []
        if hurst > 0.55: evidence.append('high_hurst')
        elif hurst < 0.45: evidence.append('low_hurst')
        
        if s_entropy < 1.5 and s_entropy > 0: evidence.append('low_entropy') 
        elif s_entropy >= 1.5: evidence.append('high_entropy')
        
        # Velocity via Kalman
        if abs(kalman_diff) > np.std(subset): 
             evidence.append('high_kalman_vel')
             
        regime_prob = self.bayes.update(evidence)
        
        # 5. Cycle Analysis
        period, strength = self.cycle.detect_dominant_cycle(closes)

        return {
            'regime_prob': regime_prob,
            'hurst': hurst,
            'entropy': s_entropy,
            'kalman_price': k_est,
            'kalman_diff': kalman_diff,
            'cycle_period': period,
            'cycle_strength': strength
        }
