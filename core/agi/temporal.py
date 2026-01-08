
import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger("Temporal")

class FractalTimeScaleIntegrator:
    """
    System 24: Fractal Time Scale Integrator.
    "Time is not linear, it is fractal."
    Ensures coherence across M1, M5, H1, D1 execution frames.
    """
    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.weights = {'1m': 0.1, '5m': 0.2, '15m': 0.2, '1h': 0.3, '4h': 0.2}

    def calculate_fractal_coherence(self, market_data_map: Dict[str, Any]) -> float:
        """
        Returns a score from -1.0 (Full Bearish Coherence) to +1.0 (Full Bullish Coherence).
        0.0 means Chaos/Noise (Timeframes disagree).
        """
        score = 0.0
        total_weight = 0.0
        
        for tf, weight in self.weights.items():
            if tf in market_data_map:
                df = market_data_map[tf]
                if df is not None and not df.empty:
                    # Simple Trend Proxy: SMA20 vs SMA50 or just Price > SMA50
                    # For prototype, we use last candle direction + sma slope if available
                    # Assuming df has 'close', 'open'
                    
                    last = df.iloc[-1]
                    trend = 0.0
                    
                    # Candle Direction
                    if last['close'] > last['open']: trend += 0.5
                    elif last['close'] < last['open']: trend -= 0.5
                    
                    # Moving Average Check (if exists)
                    if 'sma_50' in df.columns:
                         if last['close'] > last['sma_50']: trend += 0.5
                         else: trend -= 0.5
                    
                    score += trend * weight
                    total_weight += weight
                    
        if total_weight == 0: return 0.0
        return score / total_weight

    def detect_temporal_dilation(self, market_data_map: Dict[str, Any]) -> str:
        """
        Detects if "Time is speeding up" (Lower TFs exploding relative to Higher TFs).
        High Volatility on M1 while H1 is flat = Pre-Breakout or Noise.
        """
        if '1m' not in market_data_map or '1h' not in market_data_map:
             return "NORMAL"
             
        df_m1 = market_data_map.get('1m')
        df_h1 = market_data_map.get('1h')
        
        if df_m1 is None or df_h1 is None: return "NORMAL"
        
        # Calculate ATR/Range relative to price
        def get_rel_range(df):
            if df.empty: return 0
            dev = (df['high'] - df['low']).mean()
            return dev / df['close'].mean()
            
        vol_m1 = get_rel_range(df_m1.tail(10)) * 60 # Annualize/Normalize to Hour
        vol_h1 = get_rel_range(df_h1.tail(10))
        
        ratio = vol_m1 / (vol_h1 + 1e-9)
        
        if ratio > 5.0:
            return "DILATION_FAST" # M1 is screaming, H1 sleeping
        elif ratio < 0.2:
            return "DILATION_SLOW" # H1 moving, M1 dead
            
        return "SYNCHRONIZED"
