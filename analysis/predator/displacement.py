
import pandas as pd
import numpy as np

class DisplacementEngine:
    """
    Displacement / Momentum Engine.
    
    Concept:
    Institutional moves are characterized by "Displacement":
    - Large bodies relative to wicks
    - Large range relative to recent average
    - Fast velocity
    """
    
    def detect_displacement(self, df: pd.DataFrame) -> dict:
        """
        Analyzes the last few candles for displacement characteristics.
        """
        if df is None or len(df) < 20:
            return {'detected': False}
            
        # Check last 3 candles for the "Move"
        recent = df.iloc[-3:]
        
        # Calculate Average True Range or Average Body Size for baseline
        avg_body = abs(df['close'] - df['open']).rolling(20).mean().iloc[-1]
        
        if avg_body == 0: return {'detected': False}
        
        best_disp = {'detected': False, 'strength': 0}
        
        for _, candle in recent.iterrows():
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0: continue
            
            # Displacement Criteria 1: Body Dominance
            # The body should be huge part of the candle (little wicks)
            body_ratio = body / total_range
            
            # Displacement Criteria 2: Relative Size
            # Needs to be significantly larger than average churn
            size_ratio = body / avg_body
            
            # Thresholds:
            # Body > 65% of range
            # Size > 1.5x Average
            if body_ratio > 0.65 and size_ratio > 1.5:
                direction = "BUY" if candle['close'] > candle['open'] else "SELL"
                
                # Update if this is stronger than previous found
                if size_ratio > best_disp['strength']:
                    best_disp = {
                        'detected': True,
                        'direction': direction,
                        'strength': size_ratio,
                        'body_ratio': body_ratio,
                        'time': candle.name
                    }
                    
        return best_disp
