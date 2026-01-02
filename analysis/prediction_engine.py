import numpy as np
import logging

logger = logging.getLogger("Atl4s-Prediction")

class PredictionEngine:
    def __init__(self):
        self.simulations = 1000
        self.steps = 50 # Look ahead 50 candles (approx 4 hours on M5)

    def monte_carlo_simulation(self, current_price, volatility, drift, steps=50, simulations=1000):
        """
        Runs Monte Carlo simulations for future price paths.
        Returns: paths (simulations x steps)
        """
        # Random shocks
        shocks = np.random.normal(0, 1, (simulations, steps))
        
        # Price paths container
        paths = np.zeros((simulations, steps + 1))
        paths[:, 0] = current_price
        
        # Simulate
        for t in range(1, steps + 1):
            # Geometric Brownian Motion: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            # Simplified: S_t = S_{t-1} + S_{t-1} * (drift + vol * shock)
            
            # Using simple returns for speed
            # drift is expected return per step
            # volatility is std dev of returns per step
            
            returns = drift + volatility * shocks[:, t-1]
            paths[:, t] = paths[:, t-1] * (1 + returns)
            
        return paths

    def calculate_skewness(self, data):
        """
        Calculates Skewness of the distribution.
        Positive Skew: Tail on the right (Upside Potential).
        Negative Skew: Tail on the left (Downside Risk).
        """
        n = len(data)
        if n < 3: return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0: return 0
        
        # Third moment
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew

    def analyze(self, df):
        """
        Main analysis method.
        """
        if df is None or len(df) < 100:
            return {'prob_bullish': 0.5, 'prob_bearish': 0.5, 'skew': 0, 'kurtosis': 0, 'expected_price': 0}
            
        current_price = df.iloc[-1]['close']
        
        # Calculate Drift and Volatility (Log Returns)
        closes = df['close']
        prices = closes.values
        log_returns = np.diff(np.log(prices))
        
        drift = np.mean(log_returns)
        volatility = np.std(log_returns)
        
        # Run Simulation
        paths = self.monte_carlo_simulation(current_price, volatility, drift, self.steps, self.simulations)
        
        # Analyze Outcomes (The "Quantum Cloud")
        final_prices = paths[:, -1]
        
        # Probabilities
        bullish_outcomes = np.sum(final_prices > current_price)
        bearish_outcomes = np.sum(final_prices < current_price)
        
        prob_bullish = bullish_outcomes / self.simulations
        prob_bearish = bearish_outcomes / self.simulations
        
        # Skewness (Directional Bias of the Cloud)
        skew = self.calculate_skewness(final_prices)
        
        # Kurtosis (Fat Tails - not implemented fully formulaic here, simplified concepts only for now)
        # We focus on Skew for "Aggression"
        
        # Calculate Expected Value
        expected_price = np.mean(final_prices)
        
        return {
            'prob_bullish': prob_bullish,
            'prob_bearish': prob_bearish,
            'skew': skew,
            'expected_price': expected_price,
            'volatility': volatility
        }

    def rapid_forecast(self, df, steps=10, simulations=2000):
        """
        Quantum Short-Path: High-Frequency Prediction.
        Focuses on immediate price trajectory (next 10 candles).
        Returns: {prob_up, prob_down, velocity_score}
        """
        if df is None or len(df) < 50:
            return {'prob_up': 0.5, 'prob_down': 0.5, 'velocity': 0}
            
        current_price = df.iloc[-1]['close']
        
        # Calculate Short-Term Drift/Vol (Last 20 candles)
        subset = df.iloc[-20:]
        closes = subset['close'].values
        log_returns = np.diff(np.log(closes))
        
        if len(log_returns) == 0: return {'prob_up': 0.5, 'prob_down': 0.5, 'velocity': 0}
        
        drift = np.mean(log_returns)
        volatility = np.std(log_returns)
        
        # Fast Vectorized Monte Carlo
        # Shape: (simulations, steps)
        shocks = np.random.normal(0, 1, (simulations, steps))
        returns = drift + volatility * shocks
        
        # Cumulative returns -> Price paths
        # P_t = P_0 * exp(cumsum(returns)) approx P_0 * prod(1+r)
        # Using cumprod(1+r)
        price_paths = current_price * np.cumprod(1 + returns, axis=1)
        
        # Analyze Final State (at step 10)
        final_prices = price_paths[:, -1]
        
        bullish_count = np.sum(final_prices > current_price)
        bearish_count = np.sum(final_prices < current_price)
        
        prob_up = bullish_count / simulations
        prob_down = bearish_count / simulations
        
        # Calculate Probability Velocity (Acceleration of Probability)
        # Check step 5 vs step 10
        mid_prices = price_paths[:, steps // 2]
        prob_up_mid = np.sum(mid_prices > current_price) / simulations
        
        # If prob increases from mid to final, we have conviction
        # Velocity = (Prob_Final - Prob_Mid) / Steps
        velocity = (prob_up - prob_up_mid) * 100 # Scaled score
        
        return {
            'prob_up': prob_up,
            'prob_down': prob_down,
            'velocity': velocity,
            'volatility': volatility
        }
