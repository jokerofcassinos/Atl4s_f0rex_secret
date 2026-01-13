import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from core.agi.module_thought_adapter import AGIModuleAdapter, ModuleThoughtResult

logger = logging.getLogger("Atl4s-Quant")


class Quant:
    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "M5"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.agi_adapter = AGIModuleAdapter(module_name="Quant")

    def analyze(self, df):
        """
        Analyzes Statistical Probability (Z-Score, Distribution).
        Returns:
            score (int): Confidence score (0-100)
            direction (int): 1 (Buy), -1 (Sell), 0 (Neutral)
        """
        if df is None or len(df) < 50:
            return 0, 0, "NEUTRAL"

        df = df.copy()

        # Calculate Z-Score of Close relative to 20-period Mean/StdDev
        period = 20
        df['mean'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        
        # Z = (Price - Mean) / Std
        # Z = (Price - Mean) / Std
        # Ensure scalars
        close_s = df['close']
        if isinstance(close_s, pd.DataFrame): close_s = close_s.iloc[:, 0]
        
        mean_s = df['mean']
        if isinstance(mean_s, pd.DataFrame): mean_s = mean_s.iloc[:, 0]
        
        std_s = df['std']
        if isinstance(std_s, pd.DataFrame): std_s = std_s.iloc[:, 0]
        
        current_close = float(close_s.iloc[-1])
        current_mean = float(mean_s.iloc[-1])
        current_std = float(std_s.iloc[-1])
        
        if current_std == 0:
            return 0.0, 0, "NEUTRAL"
            
        z_score = (current_close - current_mean) / current_std
        
        score = 0
        direction = 0
        
        # Mean Reversion Logic
        # If Z-Score is extreme (> 2 or < -2), probability of reversion is high.
        
        signal_type = "NEUTRAL"

        # Overbought (Sell Signal)
        if z_score > 2.0:
            # 2 Sigma = ~95% probability
            score += 40
            if z_score > 3.0:
                score += 20 # Extreme extension
            
            direction = -1 # Expect reversion to mean (Sell)
            signal_type = "REVERSION_SELL"
            logger.info(f"Quant: Z-Score {z_score:.2f} (Overbought)")

        # Oversold (Buy Signal)
        elif z_score < -2.0:
            score += 40
            if z_score < -3.0:
                score += 20
            
            direction = 1 # Expect reversion to mean (Buy)
            signal_type = "REVERSION_BUY"
            logger.info(f"Quant: Z-Score {z_score:.2f} (Oversold)")
            
        # --- PULLBACK LOGIC (Predator Mode) ---
        # "Discount" Zone: Price is slightly below mean (-1.5 to -0.5)
        elif -1.5 <= z_score <= -0.5:
            score += 20
            direction = 1 # Potential Dip Buy (needs Trend confirmation)
            signal_type = "PULLBACK_BUY"
            logger.info(f"Quant: Z-Score {z_score:.2f} (Discount/Pullback Zone)")
            
        # "Premium" Zone: Price is slightly above mean (0.5 to 1.5)
        elif 0.5 <= z_score <= 1.5:
            score += 20
            direction = -1 # Potential Rally Sell (needs Trend confirmation)
            signal_type = "PULLBACK_SELL"
            logger.info(f"Quant: Z-Score {z_score:.2f} (Premium/Pullback Zone)")
            
        # Camada AGI: transformar em evento de memória + pensamento
        raw_output: Dict[str, Any] = {
            "score": score,
            "direction": direction,
            "z_score": float(z_score),
            "signal_type": signal_type,
        }

        market_state: Dict[str, Any] = {
            "price": current_close,
            "mean": current_mean,
            "std": current_std,
            "z_score": float(z_score),
        }

        thought: ModuleThoughtResult = self.agi_adapter.think_on_analysis(
            symbol=self.symbol,
            timeframe=self.timeframe,
            market_state=market_state,
            raw_module_output=raw_output,
        )

        # Mantém API legada (score, direction, signal_type) mas injeta AGI em meta
        logger.debug(
            "Quant AGI: decision=%s score=%.1f root=%s",
            thought.decision,
            thought.score,
            thought.thought_root_id,
        )

        return score, direction, signal_type

    def calculate_hurst(self, series, min_window=10):
        """
        Calculates the Hurst Exponent to determine time series memory.
        H < 0.5: Mean Reverting
        H = 0.5: Random Walk
        H > 0.5: Trending
        """
        try:
            series = np.array(series)
            if len(series) < 100: return 0.5
            
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            
            # Use linear regression to estimate the slope
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0 
        except:
            return 0.5

    def calculate_entropy(self, series, base=2):
        """
        Calculates Shannon Entropy to measure market disorder/chaos.
        Higher Entropy = More Random/Noise.
        """
        try:
            series = np.array(series)
            data = pd.Series(series).value_counts(normalize=True, bins=10) # Discretize
            entropy = stats.entropy(data, base=base)
            return entropy
        except:
            return 0

    def analyze_advanced(self, df):
        """
        Returns advanced metrics: Hurst, Entropy, Volatility Cone status.
        """
        if df is None or len(df) < 100:
            return {'hurst': 0.5, 'entropy': 0, 'volatility_rank': 0.5}
            
        close_prices = df['close'].values
        
        # Hurst
        hurst = self.calculate_hurst(close_prices)
        
        # Entropy (on returns)
        returns = df['close'].pct_change().dropna()
        entropy = self.calculate_entropy(returns)
        
        # Volatility Rank (Percentile)
        current_vol = returns.std()
        hist_vol = returns.rolling(window=100).std()
        vol_rank = stats.percentileofscore(hist_vol.dropna(), current_vol) / 100.0
        
        logger.info(f"Quant Advanced: Hurst={hurst:.2f}, Entropy={entropy:.2f}, VolRank={vol_rank:.2f}")
        
        return {
            'hurst': hurst,
            'entropy': entropy,
            'volatility_rank': vol_rank
        }
