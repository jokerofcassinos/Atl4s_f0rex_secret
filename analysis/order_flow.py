import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Atl4s-OrderFlow")

class OrderFlowEngine:
    """
    BigBeluga / Whale Watcher.
    Analyzes Volume Delta and Micro-Structure.
    """
    
    def calculate_delta(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximates Delta Volume using Tick Volume and Price Action.
        Delta = Buy Vol - Sell Vol.
        Approximation: If Close > Open, Vol is mostly Buy. Else Sell.
        Or use High-Low proportional split.
        """
        # Improved approximation:
        # Vol * (2*(Close - Low) - (High - Low)) / (High - Low)
        # This scales volume based on where the close is within the candle.
        # Close at High -> +Vol. Close at Low -> -Vol. Mid -> 0.
        
        range_len = df['high'] - df['low']
        range_len = range_len.replace(0, 1e-9)
        
        position = (df['close'] - df['low']) / range_len # 0 to 1
        
        # Map 0..1 to -1..1
        intensity = (position * 2) - 1
        
        delta = df['volume'] * intensity
        return delta

    def analyze_flow(self, df: pd.DataFrame):
        """
        Returns Flow State.
        """
        if len(df) < 50: return None
        
        delta = self.calculate_delta(df)
        cum_delta = delta.rolling(20).sum()
        
        # 1. Whale Detection (BigBeluga)
        # Volume > 3x Average
        avg_vol = df['volume'].rolling(50).mean()
        current_vol = df['volume'].iloc[-1]
        
        is_whale = False
        if current_vol > avg_vol.iloc[-1] * 3.0:
            is_whale = True
            
        # 2. Divergence
        # Price making Highs, Delta making Lows?
        
        # 3. Absorption
        # High Volume but small Candle Body (Doji)
        # Indicates Limit Orders soaking up market orders.
        body = abs(df['close'] - df['open'])
        range_len = df['high'] - df['low']
        is_absorption = False
        if current_vol > avg_vol.iloc[-1] * 2.0 and body.iloc[-1] < range_len.iloc[-1] * 0.3:
            is_absorption = True
            
        return {
            'delta': delta.iloc[-1],
            'cum_delta': cum_delta.iloc[-1],
            'is_whale': is_whale,
            'is_absorption': is_absorption,
            'vol_ratio': current_vol / (avg_vol.iloc[-1] + 1)
        }
