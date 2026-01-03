
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

logger = logging.getLogger("GPSwarm")

class GaussianProcessSwarm(SubconsciousUnit):
    """
    The Gaussian Process (Non-Parametric Probabilistic Logic).
    Phase 41 Innovation.
    Logic:
    1. Models the asset price as a Gaussian Process.
    2. Kernel: C(1.0) * Matern(nu=1.5).
    3. Fits to last 50 bars.
    4. Predicts next 5 bars + Uncertainty (Sigma).
    5. Confidence inverse to Uncertainty.
    """
    def __init__(self):
        super().__init__("GaussianProcessSwarm")
        # Switch to Matern Kernel (nu=1.5 handles roughness better than RBF)
        # Widened bounds to prevent ConvergenceWarning (1e-9 for noise handling)
        self.kernel = C(1.0, (1e-3, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-9, 1e5), nu=1.5)
        # alpha=1.0 explicitly adds noise handling. normalize_y=True helps with scaling.
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=3, alpha=1.0, normalize_y=True)

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 100: return None
        
        # Data Prep (Normalized)
        # We model the Price relative to a moving average to detrend slightly, or just pure price?
        # Let's model the "Diff" (Velocity) to keep it stationary.
        
        closes = df['close'].values
        velocity = np.diff(closes)[-60:] # Last 60 deltas
        
        X = np.atleast_2d(np.arange(len(velocity))).T
        y = velocity
        
        # Fit GP
        # Warning: GP is expensive O(N^3). 60 points is fast enough.
        try:
            self.gp.fit(X, y)
        except Exception as e:
            logger.warning(f"GP Fit Error: {e}")
            return None
            
        # Predict Future
        X_pred = np.atleast_2d(np.arange(len(velocity), len(velocity)+3)).T
        y_pred, sigma = self.gp.predict(X_pred, return_std=True)
        
        # Reasoning
        # Sum of predicted velocity = Projected Price Change
        projected_chg = np.sum(y_pred)
        avg_sigma = np.mean(sigma)
        
        # Confidence
        # If sigma is high (Uncertainty), Confidence Low.
        # If sigma is low (Tight fit), Confidence High.
        
        # Heuristic for Confidence:
        # Standard Deviation of the input data vs Prediction Sigma
        data_std = np.std(y)
        conf_ratio = 1.0 - min(1.0, avg_sigma / (data_std + 1e-9))
        
        confidence = conf_ratio * 100.0
        
        signal = "WAIT"
        
        # Direction
        if projected_chg > 0.5: # Significant Upward Velocity
            signal = "BUY"
            confidence = min(100, confidence + 20) # Boost if trend confirms
        elif projected_chg < -0.5:
            signal = "SELL"
            confidence = min(100, confidence + 20)
            
        if signal != "WAIT":
            return SwarmSignal(
                source="GaussianProcessSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={
                    "projected_delta": projected_chg,
                    "uncertainty_sigma": avg_sigma,
                    "logic": "Non-Parametric Kernel Regression"
                }
            )
            
        return None
