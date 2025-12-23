import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import logging

logger = logging.getLogger("Atl4s-MacroMath")

class MacroMath:
    @staticmethod
    def garch_11_forecast(returns, horizon=5):
        """
        Simplified GARCH(1,1) parameter estimation and volatility forecast.
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        if len(returns) < 20: return np.std(returns)
        
        # Returns must be centered
        resids = returns - np.mean(returns)
        
        # Likelihood function for GARCH(1,1)
        def log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
                return 1e10
            
            n = len(resids)
            sq_resids = resids**2
            sigmas_sq = np.zeros(n)
            sigmas_sq[0] = np.var(resids) # Initial variance
            
            for t in range(1, n):
                sigmas_sq[t] = omega + alpha * sq_resids[t-1] + beta * sigmas_sq[t-1]
                
            # Log-likelihood of Normal distribution
            loglik = -0.5 * np.sum(np.log(2 * np.pi * sigmas_sq) + sq_resids / sigmas_sq)
            return -loglik

        # Initial guess
        initial_params = [0.1 * np.var(resids), 0.1, 0.8]
        
        try:
            res = minimize(log_likelihood, initial_params, method='L-BFGS-B', 
                          bounds=((1e-6, None), (0, 1), (0, 1)))
            
            if not res.success: return np.std(returns)
            
            omega, alpha, beta = res.x
            
            # Forecast sigma^2 for next steps
            last_sigma_sq = np.var(resids)
            last_resid_sq = resids[-1]**2
            
            # 1-step forecast
            forecast_sq = omega + alpha * last_resid_sq + beta * last_sigma_sq
            
            # Multi-step (converges to long-run variance)
            long_run_var = omega / (1 - alpha - beta)
            
            # Return current forecasted volatility (annualized or per-period std)
            return np.sqrt(forecast_sq)
        except:
            return np.std(returns)

    @staticmethod
    def wavelet_haar_mra(s1, s2):
        """
        Simplified Multi-Resolution Analysis (Haar Wavelet Proxy)
        Calculates coherence between two series at different scales (High/Low frequency).
        """
        def haar_transform(x):
            if len(x) % 2 != 0: x = x[:-1]
            avgs = (x[0::2] + x[1::2]) / np.sqrt(2)
            coeffs = (x[0::2] - x[1::2]) / np.sqrt(2)
            return avgs, coeffs

        # Normalize
        s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-9)
        s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-9)
        
        n = min(len(s1), len(s2))
        s1, s2 = s1[-n:], s2[-n:]
        
        a1, c1 = haar_transform(s1)
        a2, c2 = haar_transform(s2)
        
        # Low Frequency Correlation
        low_f_corr = np.corrcoef(a1, a2)[0, 1] if len(a1) > 2 else 0
        # High Frequency Correlation (Noise)
        high_f_corr = np.corrcoef(c1, c2)[0, 1] if len(c1) > 2 else 0
        
        return {
            'trend_coherence': low_f_corr,
            'noise_coherence': high_f_corr
        }

    @staticmethod
    def calculate_cointegration(y, x):
        """
        Engle-Granger simplified cointegration test.
        1. OLS: y = beta * x + alpha + residuals
        2. Check residuals for stationarity.
        """
        if len(y) != len(x): return 0
        
        # OLS
        poly = np.polyfit(x, y, 1)
        beta, alpha = poly[0], poly[1]
        
        target = beta * x + alpha
        residuals = y - target
        
        # Simplified Stationarity Check: Autocorrelation of residuals
        # Stationary series have low/decaying autocorrelation.
        # We check the correlation between res and res[1:]
        if len(residuals) < 10: return {'beta': beta, 'stat_score': 0, 'z_score': curr_z}
        
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        # StatScore: Reverses autocorrelation (high autocorr = non-stationary/random walk)
        stat_score = max(0, 1.0 - autocorr)
        
        # Z-Score of current divergence
        curr_z = (residuals[-1]) / (np.std(residuals) + 1e-9)
        
        return {
            'beta': beta,
            'stat_score': stat_score, # Higher = more mean-reverting
            'z_score': curr_z
        }

    @staticmethod
    def bayesian_regime_detect(series, prev_prob=0.5):
        """
        Recursive Bayesian filter to detect "Expansion" vs "Contraction".
        Likelihood based on recent direction vs global volatility.
        """
        if len(series) < 10: return 0.5
        
        change = (series[-1] - series[-5]) / series[-5]
        vol = np.std(np.diff(series) / series[:-1])
        
        # Likelihoods (Sensitive to window drift)
        # Expansion: Positive drift over 4 bars
        # Contraction: Negative drift over 4 bars
        window_drift = 0.01 # Expect 1% move over 4 bars in expansion
        p_data_expansion = norm.pdf(change, loc=window_drift, scale=vol * 2 + 1e-6)
        p_data_contraction = norm.pdf(change, loc=-window_drift, scale=vol * 2 + 1e-6)
        
        # Bayes Rule
        unnorm_expansion = p_data_expansion * prev_prob
        unnorm_contraction = p_data_contraction * (1 - prev_prob)
        
        # Normalize and prevent underflow
        total = unnorm_expansion + unnorm_contraction
        if total < 1e-20: return 0.5
        
        post_prob = unnorm_expansion / total
        # Clamp to avoid total extinction of a state
        return max(0.01, min(0.99, post_prob))
