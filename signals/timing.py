"""
Atl4s Timing Module v2.0

Implements institutional timing theories:
1. Quarterly Theory (90-minute cycles with Q1-Q4)
2. M8 Fibonacci System (8-minute cycles)
3. Time Macros (20-minute vortex)
4. Session Opens (Midnight/NY)
5. Initial Balance

Based on ICT/SMC concepts and institutional order flow patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from enum import Enum
import logging

logger = logging.getLogger("Atl4s-Timing")


class Quarter(Enum):
    """90-minute cycle quarters."""
    Q1_ACCUMULATION = "Q1"
    Q2_MANIPULATION = "Q2"  # The Judas Swing
    Q3_DISTRIBUTION = "Q3"  # Golden Zone
    Q4_CONTINUATION = "Q4"


class MarketPhase(Enum):
    """AMD Power of Three phases."""
    ACCUMULATION = "A"
    MANIPULATION = "M"
    DISTRIBUTION = "D"


@dataclass
class TimingSignal:
    """Result from timing analysis."""
    tradeable: bool
    phase: str
    score: int  # -10 to +10
    reason: str
    session: str  # London, NY, Asian
    quarter: Optional[Quarter] = None
    time_to_next_phase: int = 0  # minutes
    
    # Advanced
    is_golden_zone: bool = False
    is_manipulation_zone: bool = False
    macro_active: bool = False


class QuarterlyTheory:
    """
    Implements the 90-Minute Quarterly Theory.
    
    Each 90-minute cycle is divided into 4 quarters of 22.5 minutes:
    - Q1 (0-22.5 min): Accumulation - Price fakes or goes lateral
    - Q2 (22.5-45 min): Manipulation - Judas Swing (false breakout)
    - Q3 (45-67.5 min): Distribution - The REAL move
    - Q4 (67.5-90 min): Continuation or Reversal
    
    Key insight: If Q2 shows a sweep opposite to the trend, Q3 will go with trend.
    """
    
    # 90-minute cycle start times (based on NY midnight)
    CYCLE_STARTS = [0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, 990, 1080, 1170, 1260, 1350]
    
    # Quarter durations in minutes
    QUARTER_DURATION = 22.5
    
    def __init__(self):
        self.last_q2_sweep_direction: Optional[str] = None
        self.cycle_high: float = 0.0
        self.cycle_low: float = float('inf')
        self.current_cycle_start: Optional[datetime] = None
    
    def get_current_quarter(self, current_time: datetime) -> Tuple[Quarter, int]:
        """
        Get the current quarter and minutes into that quarter.
        
        Returns: (Quarter, minutes_into_quarter)
        """
        # Minutes since midnight
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        
        # Find which 90-minute cycle we're in
        cycle_index = minutes_since_midnight // 90
        minutes_into_cycle = minutes_since_midnight % 90
        
        # Determine quarter
        if minutes_into_cycle < 22.5:
            quarter = Quarter.Q1_ACCUMULATION
        elif minutes_into_cycle < 45:
            quarter = Quarter.Q2_MANIPULATION
        elif minutes_into_cycle < 67.5:
            quarter = Quarter.Q3_DISTRIBUTION
        else:
            quarter = Quarter.Q4_CONTINUATION
        
        quarter_start = (minutes_into_cycle // 22.5) * 22.5
        minutes_into_quarter = minutes_into_cycle - quarter_start
        
        return quarter, int(minutes_into_quarter)
    
    def analyze(self, 
                current_time: datetime, 
                df: pd.DataFrame,
                trend_bias: str = "NEUTRAL") -> TimingSignal:
        """
        Analyze current timing position in the 90-minute cycle.
        
        Args:
            current_time: Current datetime
            df: Recent M1 or M5 data
            trend_bias: Higher timeframe bias (BULLISH/BEARISH/NEUTRAL)
        
        Returns:
            TimingSignal with trading recommendation
        """
        quarter, minutes_in = self.get_current_quarter(current_time)
        
        # Calculate time to next phase
        time_to_next = int(22.5 - minutes_in)
        
        # Get session
        session = self._get_session(current_time)
        
        # Base scoring
        score = 0
        tradeable = False
        reason = ""
        is_golden = False
        is_manip = False
        
        if quarter == Quarter.Q1_ACCUMULATION:
            # Accumulation - DO NOT TRADE
            score = -5
            tradeable = False
            reason = "Q1 Accumulation: Wait for structure to form"
            
        elif quarter == Quarter.Q2_MANIPULATION:
            # Manipulation Zone - Look for Judas Swing
            is_manip = True
            
            # Check if we've had a sweep
            if df is not None and len(df) > 10:
                recent_high = df['high'].iloc[-10:].max()
                recent_low = df['low'].iloc[-10:].min()
                current_price = df['close'].iloc[-1]
                
                # Detect sweep pattern
                if current_price < recent_low * 0.9998:  # Swept low
                    self.last_q2_sweep_direction = "DOWN"
                    reason = "Q2 Manipulation: Liquidity swept BELOW. Expect reversal UP in Q3"
                elif current_price > recent_high * 1.0002:  # Swept high
                    self.last_q2_sweep_direction = "UP"
                    reason = "Q2 Manipulation: Liquidity swept ABOVE. Expect reversal DOWN in Q3"
                else:
                    reason = "Q2 Manipulation: No clear sweep yet. Waiting..."
            
            score = -2
            tradeable = False  # Still wait
            
        elif quarter == Quarter.Q3_DISTRIBUTION:
            # THE GOLDEN ZONE - Maximum opportunity
            is_golden = True
            
            if self.last_q2_sweep_direction == "DOWN" and trend_bias in ["BULLISH", "NEUTRAL"]:
                # Q2 swept low, now we BUY
                score = 8
                tradeable = True
                reason = f"Q3 GOLDEN ZONE: Q2 swept LOW, entering LONG (Trend: {trend_bias})"
                
            elif self.last_q2_sweep_direction == "UP" and trend_bias in ["BEARISH", "NEUTRAL"]:
                # Q2 swept high, now we SELL
                score = 8
                tradeable = True
                reason = f"Q3 GOLDEN ZONE: Q2 swept HIGH, entering SHORT (Trend: {trend_bias})"
                
            else:
                score = 3
                tradeable = True
                reason = f"Q3 Distribution: No clear Q2 sweep, but golden zone active"
                
        elif quarter == Quarter.Q4_CONTINUATION:
            # Late cycle - Reduced opportunity
            score = 0
            tradeable = True
            reason = "Q4 Continuation: Cycle winding down, reduced position size recommended"
        
        return TimingSignal(
            tradeable=tradeable,
            phase=quarter.value,
            score=score,
            reason=reason,
            session=session,
            quarter=quarter,
            time_to_next_phase=time_to_next,
            is_golden_zone=is_golden,
            is_manipulation_zone=is_manip
        )
    
    def _get_session(self, current_time: datetime) -> str:
        """Determine current trading session."""
        hour = current_time.hour
        
        # UTC times
        if 0 <= hour < 8:
            return "ASIAN"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 17:
            return "NY_OVERLAP"  # Best session
        elif 17 <= hour < 22:
            return "NY"
        else:
            return "ASIAN"


class M8FibonacciSystem:
    """
    8-Minute Fibonacci Timing System v2.0
    
    Enhanced version with:
    - Quarter gates (Q1-Q4 within 8 minutes)
    - Multi-timeframe confluence
    - Dynamic threshold based on volatility
    """
    
    def __init__(self, threshold: int = 7):
        self.threshold = threshold
        self.max_score = 12
    
    def get_m8_position(self, current_time: datetime) -> Dict:
        """
        Get position within the 8-minute cycle.
        
        Gates:
        - Q1 (0-2 min): DEAD ZONE - No trades
        - Q2 (2-4 min): PENALTY - Only high conviction
        - Q3 (4-6 min): GOLDEN - Ideal entry window
        - Q4 (6-8 min): DECAY - Cycle ending
        """
        minutes = current_time.minute
        seconds = current_time.second
        
        # M8 block alignment (0, 8, 16, 24, 32, 40, 48, 56)
        block_start = (minutes // 8) * 8
        seconds_into_block = ((minutes - block_start) * 60) + seconds
        
        if seconds_into_block < 120:  # 0-2 min
            return {
                'gate': 'Q1',
                'score': -999,  # VETO
                'tradeable': False,
                'reason': 'Q1 Dead Zone: Accumulation phase'
            }
        elif seconds_into_block < 240:  # 2-4 min
            return {
                'gate': 'Q2',
                'score': -2,
                'tradeable': True,
                'reason': 'Q2 Manipulation: -2 penalty applied'
            }
        elif seconds_into_block < 360:  # 4-6 min
            return {
                'gate': 'Q3',
                'score': 2,
                'tradeable': True,
                'reason': 'Q3 Golden Zone: +2 bonus'
            }
        else:  # 6-8 min
            return {
                'gate': 'Q4',
                'score': 0,
                'tradeable': True,
                'reason': 'Q4 Decay: Neutral timing'
            }
    
    def evaluate(self,
                 df_h1: pd.DataFrame,
                 df_h4: pd.DataFrame,
                 df_m8: pd.DataFrame,
                 df_m2: pd.DataFrame = None,
                 current_time: datetime = None) -> Dict:
        """
        Full M8 system evaluation with triple validation.
        
        Layer 1 (BIAS): H1/H4 macro direction (+3 max)
        Layer 2 (TRIGGER): M8 patterns (+5 max)
        Layer 3 (REFINE): M2 momentum (+2 max)
        Layer 4 (TIME): M8 quarter gate (+2 to -999)
        
        Threshold: >= 7 (or 8 if H1/H4 not aligned)
        """
        if current_time is None:
            current_time = datetime.now()
        
        result = {
            'execute': False,
            'signal': 'WAIT',
            'confidence': 0.0,
            'total_score': 0,
            'breakdown': {},
            'reason': ''
        }
        
        # Layer 1: BIAS
        bias, bias_score, aligned = self._calculate_bias(df_h1, df_h4)
        result['breakdown']['bias'] = {
            'direction': bias,
            'score': bias_score,
            'aligned': aligned
        }
        
        if bias_score == 0:
            result['reason'] = 'H1/H4 conflict: No clear macro direction'
            return result
        
        # Layer 2: TRIGGER
        trigger_signal, trigger_score = self._calculate_trigger(df_m8, bias)
        result['breakdown']['trigger'] = {
            'signal': trigger_signal,
            'score': trigger_score
        }
        
        if trigger_score == 0:
            result['reason'] = 'No valid M8 trigger pattern'
            return result
        
        # Layer 3: REFINE
        refine_score = self._calculate_refine(df_m2, trigger_signal)
        result['breakdown']['refine'] = {'score': refine_score}
        
        # Layer 4: TIME GATE
        gate_info = self.get_m8_position(current_time)
        result['breakdown']['gate'] = gate_info
        
        # Hard veto check
        if not gate_info['tradeable'] or gate_info['score'] <= -900:
            result['reason'] = f"{gate_info['gate']}: {gate_info['reason']}"
            return result
        
        # Calculate total score
        total_score = bias_score + trigger_score + refine_score + gate_info['score']
        result['total_score'] = total_score
        
        # Dynamic threshold
        threshold = 7 if aligned else 8
        result['breakdown']['threshold'] = threshold
        
        # Final decision
        if total_score >= threshold and trigger_signal != "WAIT":
            result['execute'] = True
            result['signal'] = trigger_signal
            
            # Confidence mapping (7-12 -> 50-95%)
            confidence = 50.0 + ((total_score - threshold) / 5.0) * 45.0
            result['confidence'] = min(95.0, confidence)
            result['reason'] = f'Triple validation passed: {total_score}/{self.max_score}'
        else:
            result['reason'] = f'Score {total_score} < threshold {threshold}'
        
        return result
    
    def _calculate_bias(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> Tuple[str, int, bool]:
        """Calculate H1/H4 trend bias."""
        if df_h1 is None or df_h4 is None:
            return ("NEUTRAL", 0, False)
        
        if len(df_h1) < 50 or len(df_h4) < 10:
            return ("NEUTRAL", 1, False)
        
        # H1 EMA trend
        close = df_h1['close']
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        price = close.iloc[-1]
        
        h1_trend = "NEUTRAL"
        if price > ema20 > ema50:
            h1_trend = "BULLISH"
        elif price < ema20 < ema50:
            h1_trend = "BEARISH"
        
        # H4 structure
        h4_trend = self._detect_swing_structure(df_h4)
        
        aligned = (h1_trend == h4_trend and h1_trend != "NEUTRAL")
        
        if h4_trend == "BULLISH":
            return ("BULLISH", 3, aligned)
        elif h4_trend == "BEARISH":
            return ("BEARISH", 3, aligned)
        else:
            return ("NEUTRAL", 0, False)
    
    def _detect_swing_structure(self, df: pd.DataFrame, lookback: int = 10) -> str:
        """Detect HH/HL or LH/LL structure."""
        if len(df) < lookback * 2:
            return "NEUTRAL"
        
        p1_high = df['high'].iloc[-lookback*2:-lookback].max()
        p1_low = df['low'].iloc[-lookback*2:-lookback].min()
        p2_high = df['high'].iloc[-lookback:].max()
        p2_low = df['low'].iloc[-lookback:].min()
        
        if p2_high > p1_high and p2_low > p1_low:
            return "BULLISH"
        elif p2_high < p1_high and p2_low < p1_low:
            return "BEARISH"
        
        return "NEUTRAL"
    
    def _calculate_trigger(self, df_m8: pd.DataFrame, bias: str) -> Tuple[str, int]:
        """Calculate M8 trigger patterns."""
        if bias == "NEUTRAL" or df_m8 is None or len(df_m8) < 5:
            return ("WAIT", 0)
        
        score = 0
        signal = "WAIT"
        expected = "BUY" if bias == "BULLISH" else "SELL"
        
        # 1. Engulfing
        if self._detect_engulfing(df_m8) == expected:
            score += 2
            signal = expected
        
        # 2. Break of Structure
        if self._detect_bos(df_m8) == expected:
            score += 2
            signal = expected
        
        # 3. FVG Entry
        if self._detect_fvg(df_m8, expected):
            score += 1
        
        # 4. EMA Crossover
        if self._detect_ema_cross(df_m8) == expected:
            score += 1
        
        return (signal, min(score, 5))
    
    def _calculate_refine(self, df_m2: pd.DataFrame, trigger: str) -> int:
        """Calculate M2 momentum confirmation."""
        if trigger == "WAIT" or df_m2 is None or len(df_m2) < 3:
            return 1  # Neutral
        
        closes = df_m2['close'].iloc[-3:].values
        momentum = sum(1 if closes[i] > closes[i-1] else -1 for i in range(1, len(closes)))
        
        if trigger == "BUY":
            return 2 if momentum >= 1 else (1 if momentum == 0 else 0)
        else:
            return 2 if momentum <= -1 else (1 if momentum == 0 else 0)
    
    def _detect_engulfing(self, df: pd.DataFrame) -> str:
        """Detect engulfing pattern."""
        if len(df) < 2:
            return "NONE"
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Bullish engulfing
        if prev['close'] < prev['open']:  # Bearish prev
            if curr['close'] > curr['open']:  # Bullish curr
                if curr['close'] > prev['open'] and curr['open'] < prev['close']:
                    return "BUY"
        
        # Bearish engulfing
        if prev['close'] > prev['open']:  # Bullish prev
            if curr['close'] < curr['open']:  # Bearish curr
                if curr['close'] < prev['open'] and curr['open'] > prev['close']:
                    return "SELL"
        
        return "NONE"
    
    def _detect_bos(self, df: pd.DataFrame, lookback: int = 5) -> str:
        """Detect break of structure."""
        if len(df) < lookback + 1:
            return "NONE"
        
        recent_high = df['high'].iloc[-lookback-1:-1].max()
        recent_low = df['low'].iloc[-lookback-1:-1].min()
        current_close = df['close'].iloc[-1]
        
        if current_close > recent_high:
            return "BUY"
        elif current_close < recent_low:
            return "SELL"
        
        return "NONE"
    
    def _detect_fvg(self, df: pd.DataFrame, direction: str) -> bool:
        """Detect FVG entry."""
        if len(df) < 3:
            return False
        
        candle_2_back = df.iloc[-3]
        current = df.iloc[-1]
        
        if direction == "BUY":
            return current['low'] <= candle_2_back['high']
        else:
            return current['high'] >= candle_2_back['low']
    
    def _detect_ema_cross(self, df: pd.DataFrame, fast: int = 8, slow: int = 21) -> Tuple[str, int]:
        """Detect EMA crossover."""
        if df is None or len(df) < slow:
            return "WAIT", 0
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # Check recent cross
        if len(df) < 3:
            return "WAIT", 0
        
        curr_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
        prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
        
        if curr_diff > 0 and prev_diff <= 0:
            return "BUY", 1
        elif curr_diff < 0 and prev_diff >= 0:
            return "SELL", 1
        else:
            return "WAIT", 0
    
    def analyze(self, df_m8: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Simplified analyze() method for Genesis integration.
        
        Args:
            df_m8: M8 or M5 dataframe
            current_time: Current timestamp
        
        Returns:
            Dict with signal, confidence, and gate info
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Get M8 gate position
        gate_info = self.get_m8_position(current_time)
        
        # Simple result structure
        result = {
            'signal': 'WAIT',
            'confidence': 0,
            'gate': gate_info['gate'],
            'tradeable': gate_info['tradeable'],
            'reason': gate_info['reason'],
            'score': gate_info['score']
        }
        
        # If in dead zone, return immediately
        if not gate_info['tradeable']:
            return result
        
        # Try simple pattern detection if we have M8 data
        if df_m8 is not None and len(df_m8) >= 10:
            try:
                # Detect engulfing pattern (_detect_engulfing returns string: "BUY", "SELL", or "NONE")
                direction = self._detect_engulfing(df_m8)
                
                if direction in ['BUY', 'SELL']:
                    result['signal'] = direction
                    result['confidence'] = 70  # Good engulfing signal
                    result['reason'] = f"{gate_info['gate']}: Engulfing {direction}"
            except Exception as e:
                logger.warning(f"M8 pattern detection error: {e}")
        
        return result


class TimeMacroFilter:
    """
    20-Minute Macro Filter (xx:50 to xx:10)
    
    The algorithm rebalances between minute 50 and minute 10 of the next hour.
    Highs/lows made in this window are considered "fragile" and likely to be swept.
    """
    
    def __init__(self):
        self.macro_high: float = 0.0
        self.macro_low: float = float('inf')
        self.macro_active: bool = False
    
    def is_in_macro_window(self, current_time: datetime) -> bool:
        """Check if we're in the 20-minute macro window."""
        minute = current_time.minute
        return minute >= 50 or minute < 10
    
    def analyze(self, current_time: datetime, df: pd.DataFrame) -> Dict:
        """
        Analyze macro timing.
        
        If we're in the macro window and price makes a high/low,
        mark it as fragile (likely to be swept).
        """
        in_macro = self.is_in_macro_window(current_time)
        
        result = {
            'in_macro': in_macro,
            'fragile_high': None,
            'fragile_low': None,
            'recommendation': 'NORMAL'
        }
        
        if in_macro and df is not None and len(df) > 5:
            current_high = df['high'].iloc[-5:].max()
            current_low = df['low'].iloc[-5:].min()
            
            # Track macro extremes
            if current_high > self.macro_high:
                self.macro_high = current_high
                result['fragile_high'] = current_high
            
            if current_low < self.macro_low:
                self.macro_low = current_low
                result['fragile_low'] = current_low
            
            result['recommendation'] = 'CAUTION: Macro window active - extremes may be swept'
        else:
            # Reset tracking outside macro
            self.macro_high = 0.0
            self.macro_low = float('inf')
        
        return result


class InitialBalanceFilter:
    """
    Initial Balance (IB) Filter
    
    The first hour of London or NY defines the day's character:
    - If price stays in IB range: Range Day (mean reversion)
    - If price breaks IB with force: Trend Day (continuation)
    """
    
    def __init__(self):
        self.ib_high: float = 0.0
        self.ib_low: float = float('inf')
        self.ib_calculated: bool = False
        self.day_type: str = "UNKNOWN"
    
    def calculate_ib(self, df: pd.DataFrame, session_start_hour: int = 8) -> Dict:
        """
        Calculate Initial Balance from first hour data.
        
        Args:
            df: M5 or M1 data with datetime index
            session_start_hour: Hour when session starts (8 for London, 13 for NY)
        """
        if df is None or len(df) < 12:  # Need at least 1 hour of M5 data
            return {'calculated': False}
        
        # Filter for first hour
        first_hour_mask = df.index.hour == session_start_hour
        first_hour = df[first_hour_mask]
        
        if len(first_hour) == 0:
            return {'calculated': False}
        
        self.ib_high = first_hour['high'].max()
        self.ib_low = first_hour['low'].min()
        self.ib_calculated = True
        
        return {
            'calculated': True,
            'ib_high': self.ib_high,
            'ib_low': self.ib_low,
            'ib_range': self.ib_high - self.ib_low
        }
    
    def analyze(self, current_price: float) -> Dict:
        """
        Analyze price position relative to Initial Balance.
        """
        if not self.ib_calculated:
            return {'status': 'IB not calculated'}
        
        # Determine day type
        if current_price > self.ib_high:
            self.day_type = "TREND_UP"
            return {
                'day_type': 'TREND',
                'direction': 'UP',
                'recommendation': 'Trade continuation breakouts to upside'
            }
        elif current_price < self.ib_low:
            self.day_type = "TREND_DOWN"
            return {
                'day_type': 'TREND',
                'direction': 'DOWN',
                'recommendation': 'Trade continuation breakouts to downside'
            }
        else:
            self.day_type = "RANGE"
            return {
                'day_type': 'RANGE',
                'direction': 'NEUTRAL',
                'recommendation': 'Mean reversion: Buy at IB low, Sell at IB high'
            }
