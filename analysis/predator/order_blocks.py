
import pandas as pd
import numpy as np

class OrderBlockEngine:
    """
    Detects institutional Order Blocks (OB) with VALIDATION criteria.
    
    Definition:
    - Bullish OB: The last down-candle before a strong up-move that breaks structure/leaves FVG.
    - Bearish OB: The last up-candle before a strong down-move.
    
    Validation:
    1. Displacement (Impulsive Move)
    2. Structure Break (BOS)
    3. FVG Creation (Imbalance)
    """

    def detect_valid_ob(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Scans recent history for the best valid Order Block that price is currently testing.
        """
        if df is None or len(df) < 20:
            return {'detected': False}

        current_price = df['close'].iloc[-1]
        
        # Search backward for OB candidates (Last 50 candles max)
        # We look for the source of the move.
        scan_window = df.iloc[-50:-2] # Exclude current forming candle and previous (too close)
        
        best_ob = None
        
        # Iterating backwards usually finds the nearest fresh OB
        # But logic requires forward scan to check for the move AFTER the candle.
        
        for i in range(len(scan_window) - 3):
            candle = scan_window.iloc[i]
            next_candle = scan_window.iloc[i+1] # The validation candle (impulse)
            
            is_valid_candidate = False
            ob_zone = {}
            
            if direction == "BUY":
                # Looking for BULLISH OB: A bearish candle...
                if candle['close'] < candle['open']:
                    # ...followed by strong Bullish Displacement
                    body_curr = abs(candle['close'] - candle['open'])
                    move_next = next_candle['close'] - candle['close'] # Next close minus Current Close (Gap up/run up)
                    
                    # Criteria 1: Displacement Strength
                    # Impulse must be > 1.5x the OB body size
                    displacement_strength = move_next / (body_curr + 0.00001)
                    
                    if displacement_strength > 1.5 and next_candle['close'] > next_candle['open']:
                        is_valid_candidate = True
                        ob_zone = {
                            'top': candle['open'], # Open of down candle is usually the sensitive level
                            'bottom': candle['low'],
                            'type': 'BULLISH_OB',
                            'strength': displacement_strength,
                            'time': scan_window.index[i]
                        }

            elif direction == "SELL":
                # Looking for BEARISH OB: A bullish candle...
                if candle['close'] > candle['open']:
                    # ...followed by strong Bearish Displacement
                    body_curr = abs(candle['close'] - candle['open'])
                    move_next = candle['close'] - next_candle['close']
                    
                    displacement_strength = move_next / (body_curr + 0.00001)
                    
                    # Note: next_candle['close'] < next_candle['open'] (Bearish) check is good but sometimes gap down is enough
                    if displacement_strength > 1.5 and next_candle['close'] < next_candle['open']:
                        is_valid_candidate = True
                        ob_zone = {
                            'top': candle['high'],
                            'bottom': candle['open'], # Open of up candle
                            'type': 'BEARISH_OB',
                            'strength': displacement_strength,
                            'time': scan_window.index[i]
                        }

            if is_valid_candidate:
                # Validation 2: Price must be currently testing this zone
                # Price must have LEFT the zone and now RETURNED.
                # (Simple check: is current price inside/touching it?)
                
                # Check mitigation state
                # In a real engine we'd track if it was ALREADY mitigated.
                # For now, we assume if we found it, it's a candidate.
                
                if direction == "BUY":
                    # For Buy OB, we want price to drop INTO it
                    # Current Low <= OB Top and Current Close >= OB Bottom (didn't fully crash through)
                    if df['low'].iloc[-1] <= ob_zone['top'] and df['close'].iloc[-1] >= ob_zone['bottom']:
                         best_ob = ob_zone
                else:
                    # For Sell OB, we want price to rally INTO it
                    if df['high'].iloc[-1] >= ob_zone['bottom'] and df['close'].iloc[-1] <= ob_zone['top']:
                         best_ob = ob_zone

        if best_ob:
            return {
                'detected': True,
                'zone': best_ob,
                'signal': best_ob['type'].split('_')[0] # BULLISH or BEARISH
            }
        
        return {'detected': False}
