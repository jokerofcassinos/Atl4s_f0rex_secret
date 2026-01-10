"""
M8 Fibonacci Triple Validation System

A scoring-based decision system using:
- Layer 1 (BIAS): H1/H4 macro trend alignment (+3 points max)
- Layer 2 (TRIGGER): M8 candlestick patterns (+5 points max)
- Layer 3 (REFINE): M2 micro-momentum confirmation (+2 points max)
- Time Bonus: M8 Quarters (+2 to -999 based on cycle position)

Threshold: >= 7 points to execute
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("M8FibonacciSystem")


# =============================================================================
# PATTERN DETECTION FUNCTIONS
# =============================================================================

def detect_engulfing(df: pd.DataFrame) -> str:
    """Detects Engulfing pattern on last candle."""
    if df is None or len(df) < 2:
        return "NONE"
    
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Bullish Engulfing
    if prev['close'] < prev['open']:  # Previous is bearish
        if curr['close'] > curr['open']:  # Current is bullish
            if curr['close'] > prev['open'] and curr['open'] < prev['close']:
                return "BUY"
    
    # Bearish Engulfing
    if prev['close'] > prev['open']:  # Previous is bullish
        if curr['close'] < curr['open']:  # Current is bearish
            if curr['close'] < prev['open'] and curr['open'] > prev['close']:
                return "SELL"
    
    return "NONE"


def detect_break_of_structure(df: pd.DataFrame, lookback: int = 5) -> str:
    """Detects Break of Structure (new high/low)."""
    if df is None or len(df) < lookback + 1:
        return "NONE"
    
    recent_high = df['high'].iloc[-lookback-1:-1].max()
    recent_low = df['low'].iloc[-lookback-1:-1].min()
    current_close = df['close'].iloc[-1]
    
    if current_close > recent_high:
        return "BUY"  # Bullish BOS
    elif current_close < recent_low:
        return "SELL"  # Bearish BOS
    
    return "NONE"


def detect_fvg_entry(df: pd.DataFrame, direction: str) -> bool:
    """Detects if price entered Fair Value Gap."""
    if df is None or len(df) < 3:
        return False
    
    candle_2_back = df.iloc[-3]
    current = df.iloc[-1]
    
    if direction == "BUY":
        if current['low'] <= candle_2_back['high']:
            return True
    else:
        if current['high'] >= candle_2_back['low']:
            return True
    
    return False


def detect_ema_crossover(df: pd.DataFrame, fast: int = 8, slow: int = 21) -> str:
    """Detects EMA crossover."""
    if df is None or len(df) < slow + 2:
        return "NONE"
    
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    
    # Current state
    curr_above = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    prev_above = ema_fast.iloc[-2] > ema_slow.iloc[-2]
    
    if curr_above and not prev_above:
        return "BUY"  # Bullish crossover
    elif not curr_above and prev_above:
        return "SELL"  # Bearish crossover
    
    return "NONE"


def detect_swing_structure(df: pd.DataFrame, lookback: int = 10) -> str:
    """Detects swing structure (Higher Highs/Lower Lows)."""
    if df is None or len(df) < lookback * 2:
        return "NEUTRAL"
    
    # Get recent swings
    period1_high = df['high'].iloc[-lookback*2:-lookback].max()
    period1_low = df['low'].iloc[-lookback*2:-lookback].min()
    period2_high = df['high'].iloc[-lookback:].max()
    period2_low = df['low'].iloc[-lookback:].min()
    
    # Higher Highs + Higher Lows = Bullish
    if period2_high > period1_high and period2_low > period1_low:
        return "BULLISH"
    # Lower Highs + Lower Lows = Bearish
    elif period2_high < period1_high and period2_low < period1_low:
        return "BEARISH"
    
    return "NEUTRAL"


# =============================================================================
# LAYER SCORE CALCULATIONS
# =============================================================================

def calculate_bias_score(df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> Tuple[str, int]:
    """
    Layer 1: BIAS (H1/H4 Macro Direction)
    Returns: (bias_direction, score, h1_h4_aligned)
    
    v5.0: Added h1_h4_aligned flag for Dynamic Threshold.
          - If H1 and H4 agree -> Threshold 7 (relaxed)
          - If only H4 is clear -> Threshold 8 (strict)
    """
    if df_h1 is None or df_h4 is None:
        return ("NEUTRAL", 0, False)
    
    if len(df_h1) < 50 or len(df_h4) < 10:
        return ("NEUTRAL", 1, False)  # Partial data
    
    # H1 Trend via EMA
    ema20 = df_h1['close'].ewm(span=20).mean().iloc[-1]
    ema50 = df_h1['close'].ewm(span=50).mean().iloc[-1]
    price = df_h1['close'].iloc[-1]
    
    h1_trend = "NEUTRAL"
    if price > ema20 > ema50:
        h1_trend = "BULLISH"
    elif price < ema20 < ema50:
        h1_trend = "BEARISH"
    
    # H4 Structure via Swing Points
    h4_structure = detect_swing_structure(df_h4)
    
    # v5.0: Check if H1 and H4 agree
    h1_h4_aligned = (h1_trend == h4_structure and h1_trend != "NEUTRAL")
    
    # Score calculation
    # GATE 1: BINARY BIAS (Sniper Protocol v4.0)
    # If H4 says BULLISH, we ONLY allow BUYS.
    # If H4 says BEARISH, we ONLY allow SELLS.
    # If H4 is NEUTRAL, we BLOCK everything.
    
    if h4_structure == "BULLISH":
        return ("BULLISH", 3, h1_h4_aligned)
    elif h4_structure == "BEARISH":
        return ("BEARISH", 3, h1_h4_aligned)
    else:
        # H1 might be trending, but if H4 is choppy, we stay out.
        # Strict Filter.
        return ("NEUTRAL", 0, False)


def calculate_trigger_score(df_m8: pd.DataFrame, bias_direction: str) -> Tuple[str, int]:
    """
    Layer 2: TRIGGER (M8 Patterns)
    Returns: (signal, score)
    """
    if bias_direction == "NEUTRAL":
        return ("WAIT", 0)
    
    if df_m8 is None or len(df_m8) < 5:
        return ("WAIT", 0)
    
    score = 0
    signal = "WAIT"
    expected_signal = "BUY" if bias_direction == "BULLISH" else "SELL"
    
    # 1. Engulfing Pattern (+2)
    engulf = detect_engulfing(df_m8)
    if engulf == expected_signal:
        score += 2
        signal = expected_signal
    
    # 2. Break of Structure (+2)
    bos = detect_break_of_structure(df_m8)
    if bos == expected_signal:
        score += 2
        signal = expected_signal
    
    # 3. FVG Entry (+1)
    fvg = detect_fvg_entry(df_m8, expected_signal)
    if fvg:
        score += 1
    
    # 4. EMA Crossover (+1)
    ema_cross = detect_ema_crossover(df_m8)
    if ema_cross == expected_signal:
        score += 1
    
    return (signal, min(score, 5))


def calculate_refine_score(df_m2: pd.DataFrame, trigger_signal: str) -> int:
    """
    Layer 3: REFINE (M2 Micro-Momentum)
    Returns: score 0-2
    """
    if trigger_signal == "WAIT":
        return 0
    
    if df_m2 is None or len(df_m2) < 3:
        return 1  # Neutral if no data
    
    # Check last 3 M2 candles momentum
    last_3 = df_m2.iloc[-3:]
    closes = last_3['close'].values
    
    momentum = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            momentum += 1
        elif closes[i] < closes[i-1]:
            momentum -= 1
    
    if trigger_signal == "BUY":
        if momentum >= 1:
            return 2  # Pullback complete
        elif momentum == 0:
            return 1  # Neutral
        else:
            return 0  # Still falling
    
    elif trigger_signal == "SELL":
        if momentum <= -1:
            return 2
        elif momentum == 0:
            return 1
        else:
            return 0
    
    return 0


def get_m8_gate(current_time: datetime = None) -> dict:
    """
    M8 QUARTERS GATE SYSTEM (Corrected)
    
    Gate Q1 (0-2 min):   DEAD ZONE - Score -999 (VETO)
    Gate Q2 (2-4 min):   PENALTY   - Score -2 (Breakout Only)
    Gate Q3 (4-6 min):   GOLDEN    - Score +2 (Reversal Zone)
    Gate Q4 (6-8 min):   DECAY     - Score 0 (End of Cycle)
    """
    if current_time is None:
        current_time = datetime.now()
    
    minutes = current_time.minute
    seconds = current_time.second
    # M8 block starts at minutes multiple of 8: 0, 8, 16, 24, 32, 40, 48, 56
    block_start_minute = (minutes // 8) * 8
    seconds_into_m8 = ((minutes - block_start_minute) * 60) + seconds
    
    gate_info = {
        'gate': 'UNKNOWN',
        'score': 0,
        'tradeable': True,
        'reason': ''
    }
    
    if seconds_into_m8 < 120:
        # Q1: 0-2 mins - DEAD ZONE / ACCUMULATION
        gate_info.update({
            'gate': 'Q1',
            'score': -999,
            'tradeable': False,
            'reason': 'Q1 Dead Zone (Accumulation)'
        })
    elif seconds_into_m8 < 240:
        # Q2: 2-4 mins - MANIPULATION / TRAP
        # Penalty applied. High conviction only.
        gate_info.update({
            'gate': 'Q2',
            'score': -2,
            'tradeable': True,
            'reason': 'Q2 Manipulation (Penalty -2)'
        })
    elif seconds_into_m8 < 360:
        # Q3: 4-6 mins - GOLDEN / DISTRIBUTION
        # The ideal time for the move.
        gate_info.update({
            'gate': 'Q3',
            'score': +2,
            'tradeable': True,
            'reason': 'Q3 Golden Zone (Bonus +2)'
        })
    else:
        # Q4: 6-8 mins - DECAY / CONTINUATION
        # Move is often done or fading.
        gate_info.update({
            'gate': 'Q4',
            'score': 0,
            'tradeable': True,
            'reason': 'Q4 Decay Phase'
        })
        
    return gate_info


# =============================================================================
# MAIN SYSTEM CLASS
# =============================================================================

class M8FibonacciSystem:
    """
    M8 Fibonacci Triple Validation System.
    Replaces cascade veto system with weighted scoring.
    """
    
    def __init__(self, threshold: int = 7):
        self.threshold = threshold
        self.max_score = 12  # 3 + 5 + 2 + 2
    
    def evaluate(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        df_m8: pd.DataFrame,
        df_m2: pd.DataFrame = None,
        current_time: datetime = None
    ) -> Dict:
        """
        Evaluates if trade should be executed.
        
        Returns:
            {
                'execute': bool,
                'signal': 'BUY' | 'SELL' | 'WAIT',
                'confidence': float (0-100),
                'total_score': int,
                'breakdown': {...},
                'reason': str
            }
        """
        result = {
            'execute': False,
            'signal': 'WAIT',
            'confidence': 0.0,
            'total_score': 0,
            'breakdown': {},
            'reason': ''
        }
        
        # Layer 1: BIAS (v5.0: Now returns 3-tuple)
        bias_direction, bias_score, h1_h4_aligned = calculate_bias_score(df_h1, df_h4)
        result['breakdown']['bias'] = {
            'direction': bias_direction, 
            'score': bias_score,
            'h1_h4_aligned': h1_h4_aligned  # v5.0: For dynamic threshold
        }
        
        if bias_score == 0:
            result['reason'] = 'H1/H4 Conflict - No clear macro direction'
            return result
        
        # Layer 2: TRIGGER
        trigger_signal, trigger_score = calculate_trigger_score(df_m8, bias_direction)
        result['breakdown']['trigger'] = {'signal': trigger_signal, 'score': trigger_score}
        
        if trigger_score == 0:
            result['reason'] = 'No valid M8 setup'
            return result
        
        # Layer 3: REFINE (Optional)
        refine_score = calculate_refine_score(df_m2, trigger_signal)
        result['breakdown']['refine'] = {'score': refine_score}
        
        # Time Gate
        gate_info = get_m8_gate(current_time)
        result['breakdown']['gate'] = gate_info
        
        gate_score = gate_info['score']
        
        # Hard VETO Check
        if not gate_info['tradeable'] or gate_score <= -900:
            result['reason'] = f"{gate_info['gate']} Block: {gate_info['reason']}"
            return result
        
        # TOTAL SCORE
        total_score = bias_score + trigger_score + refine_score + gate_score
        result['total_score'] = total_score
        
        # v5.0: DYNAMIC THRESHOLD
        # If H1 and H4 agree (strong conviction) -> Threshold 7
        # If only H4 is clear (weaker conviction) -> Threshold 8
        dynamic_threshold = 7 if h1_h4_aligned else 8
        result['breakdown']['dynamic_threshold'] = dynamic_threshold
        
        # THRESHOLD CHECK (Using Dynamic Threshold)
        if total_score >= dynamic_threshold and trigger_signal != "WAIT":
            result['execute'] = True
            result['signal'] = trigger_signal
            
            # Map score to confidence (7-12 -> 50%-95%)
            confidence = 50.0 + ((total_score - dynamic_threshold) / 5.0) * 45.0
            result['confidence'] = min(95.0, confidence)
            result['reason'] = f'Triple Validation passed: {total_score}/{self.max_score}'
        else:
            if total_score >= dynamic_threshold and trigger_signal == "WAIT":
                 result['reason'] = f'Score {total_score} passed, but NO TRIGGER (Signal=WAIT)'
            else:
                 result['reason'] = f'Score {total_score} < threshold {dynamic_threshold}'
        
        logger.info(f"M8 FIBONACCI: {result['signal']} | Score: {total_score} | {result['reason']}")
        
        return result
