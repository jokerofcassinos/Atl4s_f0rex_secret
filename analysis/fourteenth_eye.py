import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger("Atl4s-FourteenthEye")

class FourteenthEye:
    """
    The Fourteenth Eye: The Manifold Engine.
    
    Physics:
    - Treats the market as a Riemannian Manifold M.
    - Metric Tensor g_ij is derived from the covariance of state variables.
    - Calculates Scalar Curvature R (Ricci Scalar).
    
    Interpretation:
    - R > 0 (Spherical): Space is compact. Geodesics converge. Mean Reversion.
    - R < 0 (Hyperbolic): Space is expanding. Geodesics diverge. Trend Acceleration.
    - R -> Infinity: Singularity. Market Crash/Melt-up.
    """
    def __init__(self):
        self.name = "Manifold Engine (Ricci Flow)"
        self.lookback = 30
        
    def _construct_metric_tensor(self, data: pd.DataFrame) -> np.ndarray:
        """
        Constructs the Metric Tensor g_ij at the current point t.
        We define the manifold coordinates as (Return, LogVolatility, LogVolume).
        g_ij is the Inverse Covariance Matrix (Information Metric) or simple Covariance?
        
        In Information Geometry (Fisher Information Metric), g_ij ~ Inverse Covariance.
        Here we treat the local geometry defined by the correlation structure.
        """
        if len(data) < self.lookback: return None
        
        window = data.iloc[-self.lookback:].copy()
        
        # 1. State Variables (Coordinates)
        # x1 = Returns
        window['ret'] = window['close'].pct_change().fillna(0)
        # x2 = Volatility (Parkinson or simple StdDev)
        window['vol'] = window['ret'].rolling(5).std().fillna(0)
        # x3 = Relative Volume
        window['v_rel'] = (window['volume'] / (window['volume'].rolling(20).mean() + 1))
        
        # Matrix X [n_samples, 3]
        X = window[['ret', 'vol', 'v_rel']].dropna().values
        
        # Normalize to avoid scale issues
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        
        # Covariance Matrix (Local Geometry)
        cov = np.cov(X.T) # 3x3
        
        # Metric Tensor g is often modeled as the Inverse Covariance (Precision Matrix)
        # representing the "cost" of moving in that direction.
        try:
             g = np.linalg.inv(cov + np.eye(3)*1e-6) # Regularize
        except np.linalg.LinAlgError:
             g = np.eye(3)
             
        return g

    def calculate_curvature(self, data_map: Dict[str, Any]) -> Dict:
        """
        Calculates the Ricci Scalar Curvature R.
        Since we don't have a continuous differentiable manifold, we estimate discrete curvature.
        Ollivier-Ricci Curvature or Sectional Curvature proxy.
        
        Proxy Method:
        Measure the divergence of trajectories (Geodesics).
        If neighbors move apart faster than Euclidean -> Negative Curvature (Hyperbolic).
        If neighbors move closer -> Positive Curvature (Spherical).
        """
        df_m5 = data_map.get('M5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        closes = df_m5['close'].values
        
        # We look at "trajectories" of length N starting from close points.
        # But this is chaos theory (Lyapunov). Curvature is related.
        # R ~ - Lambda (Lyapunov Exponent).
        
        # Let's use the Metric Volume method.
        # Volume of a ball in curved space V_g vs Euclidean V_e.
        # V_g / V_e ~ 1 - (R / 6(n+2)) * r^2
        
        # We estimate "Volume" as the determinant of the Covariance Matrix (Generalized Variance).
        
        window_size = 20
        t_now = df_m5.iloc[-window_size:]
        t_prev = df_m5.iloc[-window_size*2:-window_size]
        
        vol_now = self._calculate_phase_volume(t_now)
        vol_prev = self._calculate_phase_volume(t_prev)
        
        if vol_prev == 0: return None
        
        # If Volume is shrinking -> Positive Curvature ( Sphere)
        # If Volume is expanding -> Negative Curvature (Hyperbolic)
        
        ratio = vol_now / vol_prev
        
        # R is inverse to Volume change
        # R > 0 => Volume shrinks (Ratio < 1)
        # R < 0 => Volume expands (Ratio > 1)
        
        ricci_scalar = (1.0 - ratio) * 10.0 # Scaling Factor
        
        # Refinement: Trend Acceleration
        # If Trend is accelerating, Curvature is Negative (Explosion).
        
        return {
            'ricci_scalar': ricci_scalar,
            'metric_volume': vol_now,
            'expansion_ratio': ratio
        }

    def _calculate_phase_volume(self, df: pd.DataFrame) -> float:
        # Calculate Determinant of Covariance Matrix of (Price, Vol, Volatility)
        if len(df) < 10: return 0.0
        
        ret = df['close'].pct_change().fillna(0)
        vol = df['high'] - df['low']
        vrel = df['volume']
        
        data = np.column_stack((ret, vol, vrel))
        # Normalize
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)
        
        cov = np.cov(data.T)
        det = np.linalg.det(cov)
        return det
