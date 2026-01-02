import pandas as pd
import logging

logger = logging.getLogger("Atl4s-Patterns")

class PatternRecon:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Analyzes the latest candles for high-probability reversal patterns.
        Returns:
            score (int): Confidence score (0-100)
            direction (int): 1 (Buy), -1 (Sell), 0 (Neutral)
            pattern_name (str): Name of the detected pattern
        """
        if df is None or len(df) < 5:
            return 0, 0, "None"
        
        df = df.copy()
        
        # Get last 3 candles for context
        c1 = df.iloc[-1] # Current (or just closed)
        c2 = df.iloc[-2] # Previous
        c3 = df.iloc[-3] # Pre-Previous
        
        # Helper: Candle Properties
        def get_body(c): return abs(c['close'] - c['open'])
        def get_upper_wick(c): return c['high'] - max(c['close'], c['open'])
        def get_lower_wick(c): return min(c['close'], c['open']) - c['low']
        def is_bullish(c): return c['close'] > c['open']
        def is_bearish(c): return c['close'] < c['open']
        
        score = 0
        direction = 0
        pattern = "None"
        
        # --- Pattern 1: Hammer (Bullish Reversal) ---
        # Logic: Small body, Long lower wick (2x body), Little/No upper wick.
        # Context: Must be at bottom of a short-term downtrend (c2, c3 bearish or lower lows)
        
        body1 = get_body(c1)
        upper1 = get_upper_wick(c1)
        lower1 = get_lower_wick(c1)
        
        is_hammer = lower1 > (2 * body1) and upper1 < body1
        
        # Context: Downtrend?
        is_downtrend = c2['close'] < c3['close']
        
        if is_hammer and is_downtrend:
            score = 70
            direction = 1
            pattern = "Hammer"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 2: Shooting Star (Bearish Reversal) ---
        # Logic: Small body, Long upper wick (2x body), Little/No lower wick.
        # Context: Must be at top of uptrend.
        
        is_star = upper1 > (2 * body1) and lower1 < body1
        
        # Context: Uptrend?
        is_uptrend = c2['close'] > c3['close']
        
        if is_star and is_uptrend:
            score = 70
            direction = -1
            pattern = "Shooting Star"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 3: Bullish Engulfing ---
        # Logic: c1 is Green, c2 is Red. c1 body engulfs c2 body.
        # Context: Downtrend.
        
        is_bull_engulf = is_bullish(c1) and is_bearish(c2) and \
                         c1['close'] > c2['open'] and c1['open'] < c2['close']
                         
        if is_bull_engulf and is_downtrend:
            score = 80 # Stronger signal
            direction = 1
            pattern = "Bullish Engulfing"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 4: Bearish Engulfing ---
        # Logic: c1 is Red, c2 is Green. c1 body engulfs c2 body.
        # Context: Uptrend.
        
        is_bear_engulf = is_bearish(c1) and is_bullish(c2) and \
                         c1['close'] < c2['open'] and c1['open'] > c2['close']
                         
        if is_bear_engulf and is_uptrend:
            score = 80
            direction = -1
            pattern = "Bearish Engulfing"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 5: Inside Bar (The Bible Strategy) ---
        # Logic: c1 is completely inside c2 (High < c2 High, Low > c2 Low).
        # Context: Often a continuation or breakout setup.
        # We treat it as a "Potential Reversal" if at a level (Consensus handles level).
        
        is_inside = c1['high'] < c2['high'] and c1['low'] > c2['low']
        
        if is_inside:
            # Inside Bar is neutral/continuation by itself, but powerful at levels.
            # We return it with a lower score, relying on Confluence to boost it.
            score = 50 
            direction = 0 # Neutral until breakout, but we flag it.
            # Actually, if we are in a trend, it's often continuation.
            # Let's return direction based on previous candle color (continuation)
            direction = 1 if is_bullish(c2) else -1
            pattern = "Inside Bar"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 6: Morning Star (Bullish Reversal) ---
        # Logic: c3 Red, c2 Small Body (Star), c1 Green (closes > 50% of c3).
        # Context: Downtrend.
        
        body3 = get_body(c3)
        midpoint3 = c3['open'] + (c3['close'] - c3['open']) / 2 # For Red, Open > Close
        midpoint3 = c3['close'] + (body3 / 2) # Correct midpoint calc
        
        is_morning_star = is_bearish(c3) and \
                          get_body(c2) < (body3 * 0.3) and \
                          is_bullish(c1) and \
                          c1['close'] > (c3['close'] + body3 * 0.5)
                          
        if is_morning_star and is_downtrend:
            score = 85 # Very Strong
            direction = 1
            pattern = "Morning Star"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 7: Evening Star (Bearish Reversal) ---
        # Logic: c3 Green, c2 Small Body, c1 Red (closes > 50% of c3).
        # Context: Uptrend.
        
        is_evening_star = is_bullish(c3) and \
                          get_body(c2) < (get_body(c3) * 0.3) and \
                          is_bearish(c1) and \
                          c1['close'] < (c3['close'] - get_body(c3) * 0.5)
                          
        if is_evening_star and is_uptrend:
            score = 85 # Very Strong
            direction = -1
            pattern = "Evening Star"
            logger.info(f"Pattern Detected: {pattern}")
            return score, direction, pattern

        # --- Pattern 8: Micro-Momentum (Fallback) ---
        # 3 Consecutive Candles of same color
        if is_bullish(c1) and is_bullish(c2) and is_bullish(c3):
            return 20, 1, "Micro-Momentum (Bull)"
        elif is_bearish(c1) and is_bearish(c2) and is_bearish(c3):
            return 20, -1, "Micro-Momentum (Bear)"

        return 0, 0, "None"
