"""
Laplace Demon - Correlation Analysis Module v2.0

Implements cross-asset analysis:

1. SMT Divergence (Smart Money Technique)
   - Detects divergence between correlated pairs (EURUSD/GBPUSD)
   - The weaker pair reveals institutional intent

2. Power of One (Standard Deviation)
   - Price expansion from session open
   - 68% of action within 1 SD
   - Beyond 2.5 SD = Mathematical exhaustion

3. Inversion FVG
   - When support becomes resistance (and vice versa)
   - Highest conviction reversal signal
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger("Laplace-Correlation")


@dataclass
class SMTSignal:
    """SMT Divergence signal."""
    detected: bool
    strong_pair: str
    weak_pair: str
    divergence_type: str  # "BULLISH_DIV" or "BEARISH_DIV"
    confidence: float
    trade_action: str
    reason: str


class SMTDivergence:
    """
    Smart Money Technique - Cross-Asset Divergence
    
    When EURUSD and GBPUSD are correlated (+), but one makes a higher high
    while the other fails, it reveals institutional weakness.
    
    Trade the WEAKER asset in the direction the STRONGER is pointing.
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.correlation_window = 50
        
    def analyze(self, 
                df_primary: pd.DataFrame, 
                df_secondary: pd.DataFrame,
                primary_symbol: str = "GBPUSD",
                secondary_symbol: str = "EURUSD") -> SMTSignal:
        """
        Analyze SMT divergence between two correlated pairs.
        
        If EURUSD makes higher high but GBPUSD fails = SELL GBPUSD
        If EURUSD makes lower low but GBPUSD fails = BUY GBPUSD
        """
        if df_primary is None or df_secondary is None:
            return SMTSignal(
                detected=False, strong_pair="", weak_pair="",
                divergence_type="", confidence=0, 
                trade_action="WAIT", reason="Insufficient data"
            )
        
        if len(df_primary) < self.lookback or len(df_secondary) < self.lookback:
            return SMTSignal(
                detected=False, strong_pair="", weak_pair="",
                divergence_type="", confidence=0,
                trade_action="WAIT", reason="Need more candles"
            )
        
        # Get recent swing points
        p_highs = df_primary['high'].iloc[-self.lookback:]
        p_lows = df_primary['low'].iloc[-self.lookback:]
        s_highs = df_secondary['high'].iloc[-self.lookback:]
        s_lows = df_secondary['low'].iloc[-self.lookback:]
        
        # Split into two halves for comparison
        mid = self.lookback // 2
        
        # Primary swing structure
        p_prev_high = p_highs.iloc[:mid].max()
        p_curr_high = p_highs.iloc[mid:].max()
        p_prev_low = p_lows.iloc[:mid].min()
        p_curr_low = p_lows.iloc[mid:].min()
        
        # Secondary swing structure
        s_prev_high = s_highs.iloc[:mid].max()
        s_curr_high = s_highs.iloc[mid:].max()
        s_prev_low = s_lows.iloc[:mid].min()
        s_curr_low = s_lows.iloc[mid:].min()
        
        # Detect divergence
        # Bearish SMT: Secondary makes HH, Primary fails (LH)
        if s_curr_high > s_prev_high and p_curr_high < p_prev_high:
            return SMTSignal(
                detected=True,
                strong_pair=secondary_symbol,
                weak_pair=primary_symbol,
                divergence_type="BEARISH_DIV",
                confidence=75.0,
                trade_action=f"SELL {primary_symbol}",
                reason=f"{secondary_symbol} made Higher High, {primary_symbol} failed. Weakness confirmed."
            )
        
        # Bullish SMT: Secondary makes LL, Primary fails (HL)
        if s_curr_low < s_prev_low and p_curr_low > p_prev_low:
            return SMTSignal(
                detected=True,
                strong_pair=secondary_symbol,
                weak_pair=primary_symbol,
                divergence_type="BULLISH_DIV",
                confidence=75.0,
                trade_action=f"BUY {primary_symbol}",
                reason=f"{secondary_symbol} made Lower Low, {primary_symbol} held. Strength confirmed."
            )
        
        # Check reverse (Primary strong, Secondary weak)
        if p_curr_high > p_prev_high and s_curr_high < s_prev_high:
            return SMTSignal(
                detected=True,
                strong_pair=primary_symbol,
                weak_pair=secondary_symbol,
                divergence_type="BEARISH_DIV",
                confidence=75.0,
                trade_action=f"SELL {secondary_symbol}",
                reason=f"{primary_symbol} made Higher High, {secondary_symbol} failed. Sell the laggard."
            )
        
        if p_curr_low < p_prev_low and s_curr_low > s_prev_low:
            return SMTSignal(
                detected=True,
                strong_pair=primary_symbol,
                weak_pair=secondary_symbol,
                divergence_type="BULLISH_DIV",
                confidence=75.0,
                trade_action=f"BUY {secondary_symbol}",
                reason=f"{primary_symbol} made Lower Low, {secondary_symbol} held. Buy the laggard."
            )
        
        return SMTSignal(
            detected=False, strong_pair="", weak_pair="",
            divergence_type="", confidence=0,
            trade_action="WAIT", reason="No divergence detected"
        )
    
    def calculate_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate rolling correlation between two pairs."""
        if len(df1) < self.correlation_window or len(df2) < self.correlation_window:
            return 0.0
        
        # Align by index
        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) < self.correlation_window:
            return 0.0
        
        close1 = df1.loc[common_idx]['close'].iloc[-self.correlation_window:]
        close2 = df2.loc[common_idx]['close'].iloc[-self.correlation_window:]
        
        return close1.corr(close2)


class PowerOfOne:
    """
    Power of One - Standard Deviation Bands
    
    Statistical price boundaries based on session open.
    - 68% of price action within ±1 SD
    - 95% within ±2 SD
    - Beyond ±2.5 SD = Mathematical exhaustion (reversal zone)
    """
    
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        self.session_open: float = 0.0
        self.daily_volatility: float = 0.0
        
    def calculate_bands(self, 
                        df_daily: pd.DataFrame,
                        session_open: float = None) -> Dict:
        """
        Calculate Standard Deviation bands from session open.
        
        Returns bands at ±1, ±2, ±2.5 SD levels.
        """
        if df_daily is None or len(df_daily) < self.lookback_days:
            return {'error': 'Insufficient daily data'}
        
        if session_open is None:
            session_open = df_daily['close'].iloc[-1]
        
        self.session_open = session_open
        
        # Calculate daily volatility (daily range as % of close)
        daily_ranges = (df_daily['high'] - df_daily['low']).iloc[-self.lookback_days:]
        avg_range = daily_ranges.mean()
        
        # Use ATR-based SD
        self.daily_volatility = avg_range
        
        # Calculate bands
        bands = {
            'open': session_open,
            'sd': self.daily_volatility,
            '+1SD': session_open + (1.0 * self.daily_volatility),
            '-1SD': session_open - (1.0 * self.daily_volatility),
            '+2SD': session_open + (2.0 * self.daily_volatility),
            '-2SD': session_open - (2.0 * self.daily_volatility),
            '+2.5SD': session_open + (2.5 * self.daily_volatility),
            '-2.5SD': session_open - (2.5 * self.daily_volatility),
        }
        
        return bands
    
    def analyze_exhaustion(self, current_price: float, bands: Dict) -> Dict:
        """
        Analyze if price is in exhaustion zone.
        
        Beyond ±2.5 SD = HARD REVERSAL zone
        Beyond ±2 SD = Caution zone
        """
        if not bands or 'open' not in bands:
            return {'exhausted': False}
        
        deviation = (current_price - bands['open']) / bands['sd'] if bands['sd'] > 0 else 0
        
        result = {
            'current_sd': round(deviation, 2),
            'exhausted': False,
            'zone': 'NORMAL',
            'action': 'Normal trading'
        }
        
        if deviation >= 2.5:
            result['exhausted'] = True
            result['zone'] = 'EXTREME_OVERBOUGHT'
            result['action'] = 'HARD REVERSAL ZONE: Only SHORTS allowed. Mathematically unsustainable.'
            result['bias'] = 'SELL'
        elif deviation >= 2.0:
            result['zone'] = 'OVERBOUGHT'
            result['action'] = 'Caution zone. Reduce LONG exposure. Look for reversal signs.'
        elif deviation >= 1.0:
            result['zone'] = 'UPPER_NORMAL'
            result['action'] = 'Above average range. Monitor for exhaustion.'
        elif deviation <= -2.5:
            result['exhausted'] = True
            result['zone'] = 'EXTREME_OVERSOLD'
            result['action'] = 'HARD REVERSAL ZONE: Only LONGS allowed. Mathematically unsustainable.'
            result['bias'] = 'BUY'
        elif deviation <= -2.0:
            result['zone'] = 'OVERSOLD'
            result['action'] = 'Caution zone. Reduce SHORT exposure. Look for reversal signs.'
        elif deviation <= -1.0:
            result['zone'] = 'LOWER_NORMAL'
            result['action'] = 'Below average range. Monitor for exhaustion.'
        
        return result


class InversionFVG:
    """
    Inversion Fair Value Gap
    
    When a bullish FVG (support) is broken and becomes resistance,
    it's the clearest sign of directional change.
    
    The algorithm validates the inversion before continuing.
    """
    
    def __init__(self):
        self.active_fvgs: List[Dict] = []
        self.inverted_fvgs: List[Dict] = []
    
    def track_fvg(self, df: pd.DataFrame):
        """Track all FVGs in the data."""
        if df is None or len(df) < 4:
            return
        
        for i in range(len(df) - 3):
            c1 = df.iloc[i]
            c2 = df.iloc[i + 1]
            c3 = df.iloc[i + 2]
            
            # Bullish FVG
            if c3['low'] > c1['high']:
                fvg = {
                    'type': 'BULLISH',
                    'top': c3['low'],
                    'bottom': c1['high'],
                    'timestamp': c2.name if hasattr(c2, 'name') else datetime.now(),
                    'inverted': False
                }
                self.active_fvgs.append(fvg)
            
            # Bearish FVG
            if c3['high'] < c1['low']:
                fvg = {
                    'type': 'BEARISH',
                    'top': c1['low'],
                    'bottom': c3['high'],
                    'timestamp': c2.name if hasattr(c2, 'name') else datetime.now(),
                    'inverted': False
                }
                self.active_fvgs.append(fvg)
        
        # Keep only recent
        self.active_fvgs = self.active_fvgs[-50:]
    
    def detect_inversion(self, df: pd.DataFrame) -> Dict:
        """
        Detect FVG inversion.
        
        A Bullish FVG that gets closed below = Now resistance (SELL signal)
        A Bearish FVG that gets closed above = Now support (BUY signal)
        """
        if not self.active_fvgs or df is None or len(df) < 2:
            return {'detected': False}
        
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        for fvg in self.active_fvgs:
            if fvg['inverted']:
                continue
            
            # Bullish FVG broken (closed below its bottom)
            if fvg['type'] == 'BULLISH':
                if prev_close > fvg['bottom'] and current_close < fvg['bottom']:
                    fvg['inverted'] = True
                    self.inverted_fvgs.append(fvg)
                    
                    return {
                        'detected': True,
                        'type': 'BULLISH_TO_BEARISH',
                        'level': fvg['top'],  # Now use top as resistance
                        'entry_zone': (fvg['bottom'], fvg['top']),
                        'action': 'SELL limit order at old FVG top',
                        'sl': fvg['top'] + 0.0015,  # 15 pips above
                        'reason': 'Bullish FVG inverted to resistance'
                    }
            
            # Bearish FVG broken (closed above its top)
            if fvg['type'] == 'BEARISH':
                if prev_close < fvg['top'] and current_close > fvg['top']:
                    fvg['inverted'] = True
                    self.inverted_fvgs.append(fvg)
                    
                    return {
                        'detected': True,
                        'type': 'BEARISH_TO_BULLISH',
                        'level': fvg['bottom'],  # Now use bottom as support
                        'entry_zone': (fvg['bottom'], fvg['top']),
                        'action': 'BUY limit order at old FVG bottom',
                        'sl': fvg['bottom'] - 0.0015,  # 15 pips below
                        'reason': 'Bearish FVG inverted to support'
                    }
        
        return {'detected': False}


class MeanThreshold:
    """
    Mean Threshold (50% Rule)
    
    Institutions defend the 50% level of their order blocks.
    If price closes beyond 50%, the institutional intent has changed.
    
    Entry: 50% of the block
    Invalidation: Close beyond 50%
    """
    
    def calculate_mean_threshold(self, 
                                  block_high: float, 
                                  block_low: float,
                                  block_type: str) -> Dict:
        """
        Calculate optimal entry at 50% of the block.
        """
        midpoint = (block_high + block_low) / 2
        
        if block_type == "BULLISH":
            return {
                'optimal_entry': midpoint,
                'entry_type': 'BUY_LIMIT',
                'sl': block_low - 0.0005,  # 5 pips below block
                'invalidation': f'Close below {midpoint:.5f}',
                'reason': '50% Mean Threshold entry for maximum R:R'
            }
        else:
            return {
                'optimal_entry': midpoint,
                'entry_type': 'SELL_LIMIT',
                'sl': block_high + 0.0005,  # 5 pips above block
                'invalidation': f'Close above {midpoint:.5f}',
                'reason': '50% Mean Threshold entry for maximum R:R'
            }
    
    def check_invalidation(self, 
                           current_close: float,
                           block_high: float,
                           block_low: float,
                           block_type: str) -> bool:
        """
        Check if block has been invalidated (closed beyond 50%).
        """
        midpoint = (block_high + block_low) / 2
        
        if block_type == "BULLISH":
            return current_close < midpoint
        else:
            return current_close > midpoint


class AMDPowerOfThree:
    """
    AMD (Accumulation - Manipulation - Distribution) Detector
    
    The Power of Three pattern:
    1. A (Accumulation): Sideways, building positions
    2. M (Manipulation): Fake breakout (Judas Swing)
    3. D (Distribution): The real move
    
    Key rule: NEVER trade the first breakout (it's usually fake)
    """
    
    def __init__(self):
        self.asia_range: Tuple[float, float] = (0.0, 0.0)
        self.london_first_move: str = None
        self.manipulation_detected: bool = False
    
    def set_asia_range(self, df: pd.DataFrame, asia_start: int = 0, asia_end: int = 8):
        """
        Set the Asian session range (accumulation zone).
        
        London typically uses this as the manipulation target.
        """
        if df is None or len(df) < 20:
            return
        
        # Filter for Asian hours
        asia_mask = (df.index.hour >= asia_start) & (df.index.hour < asia_end)
        asia_data = df[asia_mask]
        
        if len(asia_data) > 0:
            self.asia_range = (asia_data['low'].min(), asia_data['high'].max())
    
    def detect_phase(self, 
                     df: pd.DataFrame, 
                     current_time: datetime = None) -> Dict:
        """
        Detect current AMD phase.
        """
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        
        # Asian session (0-8 UTC): Accumulation
        if 0 <= hour < 8:
            return {
                'phase': 'A',
                'name': 'ACCUMULATION',
                'action': 'WAIT. Range building. Do not trade breakouts.',
                'asia_high': self.asia_range[1],
                'asia_low': self.asia_range[0]
            }
        
        # London open (8-11 UTC): Manipulation
        if 8 <= hour < 11:
            if df is not None and len(df) > 5:
                current_price = df['close'].iloc[-1]
                
                # Check if first move broke Asia
                broke_high = current_price > self.asia_range[1]
                broke_low = current_price < self.asia_range[0]
                
                if broke_high:
                    self.london_first_move = "UP"
                    self.manipulation_detected = True
                    return {
                        'phase': 'M',
                        'name': 'MANIPULATION',
                        'first_move': 'UP',
                        'action': 'JUDAS SWING! First breakout UP is likely FALSE. Prepare to SELL.',
                        'expected_real_move': 'DOWN'
                    }
                elif broke_low:
                    self.london_first_move = "DOWN"
                    self.manipulation_detected = True
                    return {
                        'phase': 'M',
                        'name': 'MANIPULATION',
                        'first_move': 'DOWN',
                        'action': 'JUDAS SWING! First breakout DOWN is likely FALSE. Prepare to BUY.',
                        'expected_real_move': 'UP'
                    }
            
            return {
                'phase': 'M',
                'name': 'MANIPULATION',
                'action': 'Watching for first breakout (likely fake)'
            }
        
        # NY overlap (11+ UTC): Distribution
        if hour >= 11:
            if self.manipulation_detected:
                expected = 'DOWN' if self.london_first_move == 'UP' else 'UP'
                return {
                    'phase': 'D',
                    'name': 'DISTRIBUTION',
                    'action': f'Distribution phase. Trade {expected} (opposite of manipulation).',
                    'bias': expected,
                    'high_conviction': True
                }
            
            return {
                'phase': 'D',
                'name': 'DISTRIBUTION',
                'action': 'Distribution phase. Follow the flow.'
            }
        
        return {'phase': 'UNKNOWN'}
    
    def get_anti_first_breakout_signal(self,
                                        df: pd.DataFrame,
                                        current_time: datetime = None) -> Dict:
        """
        Generate signal to fade the first breakout.
        
        This is the core AMD strategy.
        """
        phase_info = self.detect_phase(df, current_time)
        
        if phase_info.get('phase') != 'M' or not self.manipulation_detected:
            return {'signal': None}
        
        if self.london_first_move == "UP":
            return {
                'signal': 'SELL',
                'confidence': 80.0,
                'entry': 'On first reversal candle',
                'sl': self.asia_range[1] + 0.0020,  # 20 pips above Asia high
                'tp1': self.asia_range[0],  # Asia low
                'tp2': self.asia_range[0] - 0.0030,  # 30 pips below
                'reason': 'AMD: Fading Judas Swing UP'
            }
        
        elif self.london_first_move == "DOWN":
            return {
                'signal': 'BUY',
                'confidence': 80.0,
                'entry': 'On first reversal candle',
                'sl': self.asia_range[0] - 0.0020,  # 20 pips below Asia low
                'tp1': self.asia_range[1],  # Asia high
                'tp2': self.asia_range[1] + 0.0030,  # 30 pips above
                'reason': 'AMD: Fading Judas Swing DOWN'
            }
        
        return {'signal': None}
