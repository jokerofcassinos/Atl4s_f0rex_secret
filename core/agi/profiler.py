import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("AGIProfiler")

class AGIProfiler:
    """
    The Interviewer v2.0 (Real Data Analysis).
    Analyzes historical data to recommend the optimal profile.
    """
    def __init__(self, data_loader=None):
        self.data_loader = data_loader
        
    def analyze_market_conditions(self):
        """
        Runs a comprehensive 'Pre-Flight Check'.
        Returns a recommendation Dict based on Real Metrics (ATR, Entropy).
        """
        logger.info("AGI PROFILER: Scanning Market Matrix...")
        
        # Defaults
        volatility_score = 50.0
        entropy_score = 0.5
        atr_value = 0.0
        
        if self.data_loader:
            try:
                # Fetch fresh data for the default symbol (usually ETHUSD or XAUUSD)
                # We need Daily for ATR and H1 for Entropy
                # get_data returns a map: {'D1': df, 'H1': df, ...}
                data_map = self.data_loader.get_data() 
                
                # 1. Calculate ATR (Daily) - The "Heartbeat"
                if 'D1' in data_map and data_map['D1'] is not None:
                     df_d1 = data_map['D1']
                     atr_value = self._calculate_atr(df_d1)
                     # Normalize Volatility Score (0-100)
                     # XAUUSD typical ATR is $25. Low is $15, High is $40.
                     # ETHUSD typical ATR is $100.
                     # This is heuristic normalization
                     volatility_score = min(100, (atr_value / df_d1['close'].iloc[-1]) * 100 * 20) 
                     
                # 2. Calculate Entropy (H1) - The "Chaos"
                if 'H1' in data_map and data_map['H1'] is not None:
                     df_h1 = data_map['H1']
                     entropy_score = self._calculate_shannon_entropy(df_h1)
                     
                logger.info(f"AGI METRICS: ATR={atr_value:.2f} | VolScore={volatility_score:.1f} | Entropy={entropy_score:.2f}")

            except Exception as e:
                logger.error(f"Profiler Analysis Failed: {e}")

        # Recommendation Logic
        rec = {
            "mode": "SNIPER",
            "risk_profile": "STANDARD",
            "reason": "Market is balanced.",
            "metrics": {
                "atr": atr_value,
                "entropy": entropy_score,
                "vol_score": volatility_score
            }
        }
        
        # Logic: High Chaos (Entropy > 0.8) -> SNIPER (Defensive)
        if entropy_score > 0.8 or volatility_score > 80:
            rec['mode'] = "SNIPER"
            rec['risk_profile'] = "CAUTIOUS"
            rec['reason'] = f"High Chaos Detected (Entropy {entropy_score:.2f}). Recommendation: Defensive Sniping."
            
        # Logic: Low Chaos (Trend) -> WOLF_PACK (Aggressive)
        elif entropy_score < 0.6 and volatility_score > 20:
            rec['mode'] = "WOLF_PACK" 
            rec['risk_profile'] = "AGGRESSIVE"
            rec['reason'] = f"Stable Trend Detected (Entropy {entropy_score:.2f}). Recommendation: Pack Hunting."
            
        else:
            rec['mode'] = "SNIPER"
            rec['reason'] = "Market Ambiguous. Defaulting to Sniper."
            
        return rec

    def _calculate_atr(self, df, period=14):
        """Average True Range"""
        if len(df) < period + 1: return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _calculate_shannon_entropy(self, df, bins=20):
        """
        Calculates Shannon Entropy of price returns.
        Higher = More Chaos/Noise. Lower = More Order/Trend.
        """
        if len(df) < 50: return 0.5
        
        # Log Returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Histogram
        hist, _ = np.histogram(returns, bins=bins, density=True)
        
        # Filter zeros
        hist = hist[hist > 0]
        
        # Entropy = -Sum(p * log(p))
        entropy = -np.sum(hist * np.log(hist))
        
        # Normalize roughly to 0-1 range (for comparison)
        # Max entropy for uniform dist is log(bins)
        max_entropy = np.log(bins)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
