
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("SNRMatrix")

class SNRLevel:
    def __init__(self, price: float, type: str, source: str, age: int = 0):
        self.price = price
        self.type = type # 'SUPPORT', 'RESISTANCE'
        self.source = source # 'SWING_HIGH', 'PSYCH', 'ORDER_BLOCK'
        self.strength = 1.0
        self.tests = 0
        self.creation_time = age # Index or Timestamp

class SNRMatrix:
    """
    Phase 21: SNR Matrix (Structural Nexus Resonance).
    Identifies the "Invisible Walls" of the market.
    """
    def __init__(self):
        self.levels: List[SNRLevel] = []
        
    def scan_structure(self, df: pd.DataFrame) -> List[SNRLevel]:
        """
        Scans DataFrame for structural levels.
        """
        self.levels = []
        if df is None or len(df) < 50:
            return []
            
        # 1. Identify Fractals (Bill Williams / Swing Points)
        # 5-bar fractal: High is higher than 2 left and 2 right
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # We need a window to look back and forward.
        # Since we are live, we usually look back. 
        # A completed fractal needs 2 closed candles to the right.
        
        for i in range(2, len(df) - 2):
            # Swing High
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                   lvl = SNRLevel(highs[i], 'RESISTANCE', 'SWING_HIGH', age=i)
                   self.levels.append(lvl)
                   
            # Swing Low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                   lvl = SNRLevel(lows[i], 'SUPPORT', 'SWING_LOW', age=i)
                   self.levels.append(lvl)
                   
        # 2. Psychological Levels (Based on current price)
        current_price = closes[-1]
        # Generate nearest 00 and 50 levels
        base = int(current_price * 100) / 100.0 # Truncate to cents
        # This logic depends on the asset symbol formatting.
        # For XAUUSD (~2000.00), Psych levels are 2000, 2010, 2020 (Whole Numbers)
        # For EURUSD (~1.0500), Psych levels are 1.0500, 1.0550 (Big Figures)
        
        # Generic "Big Figure" generator around current price
        # Simply add them as generic levels for now, refined by Symbol type ideally.
        
        return self.levels

    def get_nearest_levels(self, current_price: float, threshold: float = 0.001) -> Dict:
        """
        Returns nearest Support and Resistance to current price.
        """
        supports = [l for l in self.levels if l.price < current_price]
        resistances = [l for l in self.levels if l.price > current_price]
        
        nearest_sup = max(supports, key=lambda x: x.price) if supports else None
        nearest_res = min(resistances, key=lambda x: x.price) if resistances else None
        
        return {
            'nearest_support': nearest_sup.price if nearest_sup else 0.0,
            'nearest_resistance': nearest_res.price if nearest_res else 999999.0,
            'sup_dist': (current_price - nearest_sup.price) if nearest_sup else 999.0,
            'res_dist': (nearest_res.price - current_price) if nearest_res else 999.0
        }
