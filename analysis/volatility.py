import pandas as pd
import ta
import logging

logger = logging.getLogger("Atl4s-Volatility")

class VolatilityGuard:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Analyzes Volatility (Bollinger Bands, ATR).
        Returns:
            score (int): Confidence score (0-100) - Higher means SAFE to trade (good volatility).
            status (str): "EXPANDING", "CONTRACTING", "SQUEEZE"
        """
        if df is None or len(df) < 20:
            return 0, "UNKNOWN"

        df = df.copy()

        # Bollinger Bands
        # Ensure inputs are 1D Series
        close_series = df['close'].squeeze()
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
            
        bb_indicator = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2.0)
        df['bb_width'] = bb_indicator.bollinger_wband()
        
        bandwidth = float(df['bb_width'].iloc[-1])
        
        # ATR
        high_series = df['high'].squeeze()
        low_series = df['low'].squeeze()
        
        if isinstance(high_series, pd.DataFrame): high_series = high_series.iloc[:, 0]
        if isinstance(low_series, pd.DataFrame): low_series = low_series.iloc[:, 0]
            
        atr_indicator = ta.volatility.AverageTrueRange(high=high_series, low=low_series, close=close_series, window=14)
        df['ATR'] = atr_indicator.average_true_range()
        atr = float(df['ATR'].iloc[-1])
        
        score = 0
        status = "NORMAL"
        
        # Thresholds (Need calibration for M5 XAUUSD)
        # XAUUSD M5 typical ATR might be 0.5 - 2.0 points.
        # If ATR is too low, it's dead market.
        
        current_price = float(close_series.iloc[-1])
        min_atr = 0.5 if current_price > 500 else 0.00005 # 0.5 for Gold, 0.5 pips (0.00005) for Forex
        
        if atr > min_atr: # Minimum volatility requirement
            score += 50
        else:
            # logger.info(f"Low Volatility detected. ATR: {atr:.4f} < {min_atr}")
            # Don't fail completely, just lower score? 
            # Actually, return LOW_VOL but maybe Consensus can override perfectly valid 0.5 pip scalps?
            # Let's keep it but make it very sensitive (0.5 pips).
            return 20, "LOW_VOL" # Allow it but low score
            
        # Bandwidth Expansion
        # If bandwidth is increasing, volatility is expanding.
        # Bandwidth Expansion
        # If bandwidth is increasing, volatility is expanding.
        prev_bandwidth = float(df['bb_width'].iloc[-2]) if not isinstance(df['bb_width'], pd.DataFrame) else float(df['bb_width'].iloc[:, 0].iloc[-2])
        
        if bandwidth > prev_bandwidth:
            status = "EXPANDING"
            score += 30
        elif bandwidth < 1.0: # Arbitrary low threshold for squeeze
            status = "SQUEEZE"
            # Squeeze is good for breakout preparation, but risky for trend following if not broken yet.
            score += 10 
        else:
            status = "CONTRACTING"
            score += 20
            
        return score, status
