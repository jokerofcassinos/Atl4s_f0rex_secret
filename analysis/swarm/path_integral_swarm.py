
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("PathIntegralSwarm")

class PathIntegralSwarm(SubconsciousUnit):
    """
    The Feynman Machine.
    Calculates the 'Sum Over Histories' to predict the Probability Density Function (PDF)
    of the future price.
    Principle: The market explores all paths, but paths of 'Least Action' dominate.
    """
    def __init__(self):
        super().__init__("Path_Integral_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Calibrate Physics Engine
        # We need Drift (mu) and Diffusion (sigma) from recent history
        returns = df_m5['close'].pct_change().dropna().values[-50:]
        
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        current_price = df_m5['close'].iloc[-1]
        
        # 2. Path Simulation (The multiverse)
        # We simulate N paths for T steps
        N_PATHS = 500
        T_STEPS = 10
        
        # Standard Brownian Motion: dS = mu*S*dt + sigma*S*dW
        # We use a vectorized numpy approach
        dt = 1 # 1 bar
        
        # Random shocks
        dW = np.random.normal(0, 1, size=(N_PATHS, T_STEPS))
        
        # Cumulative paths
        # paths[i, t] = Price at step t for path i
        paths = np.zeros((N_PATHS, T_STEPS + 1))
        paths[:, 0] = current_price
        
        for t in range(1, T_STEPS + 1):
            # Geometric Brownian Motion step
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * dW[:, t-1]
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)
            
        # 3. Calculate Action (S)
        # In QFT, Probability ~ exp(iS). In Diffusion (Wick rotated), Prob ~ exp(-S).
        # We define Action S as the "Cost" of the path.
        # Cost = Deviation from Trend + Excessive Volatility penalty.
        # Simplified: S = Sum of (Return - ExpectedReturn)^2
        # This penalizes wild implementations, favoring "smooth" paths of least resistance.
        
        # Calculate log returns of paths
        path_returns = np.diff(np.log(paths), axis=1) # (N, T)
        
        # Action S for each path: Sum of squared z-scores (Standardized deviations)
        # z = (ret - mu) / sigma
        z_scores = (path_returns - mu) / sigma
        action = np.sum(z_scores**2, axis=1)
        
        # 4. Calculate Propagator (Weights)
        # Weight = exp(-S / Temperature)
        # Temperature controls how much we tolerate volatility.
        temperature = 2.0 * T_STEPS 
        weights = np.exp(-action / temperature)
        
        # Normalize weights
        weights /= np.sum(weights)
        
        # 5. Collapse Wavefunction
        # Weighted Average of the Final Price
        final_prices = paths[:, -1]
        expected_future_price = np.sum(final_prices * weights)
        
        # 6. Interpret
        predicted_return = (expected_future_price - current_price) / current_price
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        sigma_threshold = sigma * np.sqrt(T_STEPS) # 1 SD move over T steps
        
        if predicted_return > sigma_threshold * 0.5:
            signal = "BUY"
            confidence = 85
            reason = f"Feynman Path: +{predicted_return*100:.2f}% (Least Action)"
        elif predicted_return < -sigma_threshold * 0.5:
            signal = "SELL"
            confidence = 85
            reason = f"Feynman Path: {predicted_return*100:.2f}% (Least Action)"
            
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'expected_price': expected_future_price, 'action_min': np.min(action)}
            )
            
        return None
