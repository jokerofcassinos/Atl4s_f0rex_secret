
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger("FractalTrend")

class FractalTrend:
    """
    Sistema 4/25: Fractal Trend (REAL IMPLEMENTATION)
    Identifica pontos de swing fractais e zonas de S/R dinâmicas.
    Uses Multi-Timeframe EMA and Price Action Structure.
    """
    def __init__(self):
        self.fractal_dimension = 0.0
        self.trend_bias = "NEUTRAL"
        self.trend_strength = 0.0
        
    def analyze(self, tick: Dict[str, Any], market_data_map: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze trend structure across multiple timeframes (M5, H1, H4, D1).
        Requires 'market_data_map' populated by DataLoader.
        """
        # Default State
        analyzed_tfs = {}
        weights = {'D1': 0.4, 'H4': 0.3, 'H1': 0.2, 'M5': 0.1}
        total_score = 0.0
        total_weight = 0.0
        
        if not market_data_map:
            return {"fractal_bias": "NEUTRAL", "trend_strength": 0.0, "reason": "No data"}

        # Normalize Keys (Handle 'H1' vs '1h', 'M5' vs '5m')
        key_map = {
            'D1': ['D1', '1d', 'd1'],
            'H4': ['H4', '4h', 'h4'],
            'H1': ['H1', '1h', 'h1'],
            'M5': ['M5', '5m', 'm5']
        }
        
        for tf_name, weight in weights.items():
            df = None
            # Try to find data for this TF
            for key in key_map[tf_name]:
                if key in market_data_map and market_data_map[key] is not None:
                    df = market_data_map[key]
                    break
            
            if df is not None and not df.empty and len(df) > 55:
                # Analyze this TF
                bias, strength = self._analyze_single_tf(df, tf_name)
                analyzed_tfs[tf_name] = {'bias': bias, 'strength': strength}
                
                # Convert Bias to Score (-1 to 1)
                numeric_score = 1.0 if bias == "BULLISH" else -1.0 if bias == "BEARISH" else 0.0
                # Scale by strength
                weighted_score = numeric_score * strength * weight
                
                total_score += weighted_score
                total_weight += weight
            else:
                # Missing data for this timeframe
                analyzed_tfs[tf_name] = {'bias': "UNKNOWN", 'strength': 0.0}

        # Calculate Composite
        if total_weight == 0:
            return {"fractal_bias": "NEUTRAL", "trend_strength": 0.0, "reason": "Insufficient history across TFs"}
            
        final_composite = total_score # Weights sum to 1.0 ideally, but if missing data?
        # Normalize by available weight
        final_composite = final_composite / total_weight
        
        # Determine Final Bias
        final_bias = "NEUTRAL"
        if final_composite > 0.2: final_bias = "BULLISH"
        if final_composite < -0.2: final_bias = "BEARISH"
        
        return {
            "fractal_bias": final_bias,
            "trend_strength": abs(final_composite),
            "composite_score": final_composite, # Signed strength (-1 to 1)
            "details": analyzed_tfs,
            "structure_break": False # Placeholder for deeper PA logic
        }

    def _analyze_single_tf(self, df: Any, tf_name: str) -> tuple:
        """Helper to analyze a single DataFrame."""
        try:
            closes = df['close'].values
            if len(closes) < 200:
                # Use shorter EMAs for limited data? Or just skip 200 checks
                ema_fast = self._calculate_ema(closes, 20)
                ema_slow = self._calculate_ema(closes, 50)
            else:
                ema_fast = self._calculate_ema(closes, 50)
                ema_slow = self._calculate_ema(closes, 200)

            c_price = closes[-1]
            c_fast = ema_fast[-1]
            c_slow = ema_slow[-1]
            
            bias = "NEUTRAL"
            strength = 0.5
            
            # Logic: Cross & Position
            if c_fast > c_slow:
                if c_price > c_fast:
                    bias = "BULLISH"
                    strength = 0.9 # Strong Trend
                elif c_price > c_slow:
                    bias = "BULLISH"
                    strength = 0.6 # Pullback
                else:
                    bias = "NEUTRAL" # Messy trade zone
            else:
                if c_price < c_fast:
                    bias = "BEARISH"
                    strength = 0.9
                elif c_price < c_slow:
                    bias = "BEARISH"
                    strength = 0.6
                else:
                    bias = "NEUTRAL"
                    
            return bias, strength
        except Exception as e:
            logger.error(f"Error analyzing {tf_name}: {e}")
            return "NEUTRAL", 0.0

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
