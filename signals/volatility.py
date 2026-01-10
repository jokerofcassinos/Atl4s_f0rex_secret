"""
Laplace Demon - Volatility Analysis Module v2.0

Implements volatility analysis:
- ATR analysis with dynamic bands
- Bollinger Bands with squeeze detection
- Volatility regime classification
- Displacement detection (institutional moves)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger("Laplace-Volatility")


@dataclass
class VolatilityState:
    """Current volatility state."""
    regime: str  # LOW, NORMAL, HIGH, EXTREME
    atr: float
    atr_percentile: float
    expanding: bool
    contracting: bool
    recommendation: str


class VolatilityAnalyzer:
    """
    Complete volatility analysis suite.
    """
    
    def __init__(self, atr_period: int = 14, bb_period: int = 20, bb_std: float = 2.0):
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Full volatility analysis."""
        if df is None or len(df) < 50:
            return {'error': 'Insufficient data'}
        
        result = {
            'atr': self._calculate_atr_analysis(df),
            'bollinger': self._calculate_bollinger(df),
            'regime': None,
            'displacement': self._detect_displacement(df)
        }
        
        result['regime'] = self._classify_regime(result)
        
        return result
    
    def _calculate_atr_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate ATR with historical context (Vectorized)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Current ATR
        atr_series = tr.rolling(window=self.atr_period).mean()
        current_atr = atr_series.iloc[-1]
        
        # Historical percentile (Vectorized)
        if len(atr_series) > self.atr_period:
            valid_atr = atr_series.dropna()
            percentile = (valid_atr < current_atr).mean() * 100
        else:
            percentile = 50.0
        
        # Trend
        if len(atr_series) > self.atr_period * 2:
            prev_atr = atr_series.iloc[-self.atr_period-1] # Approx previous period
        else:
            prev_atr = current_atr
            
        expanding = current_atr > prev_atr * 1.1
        contracting = current_atr < prev_atr * 0.9
        
        return {
            'value': float(round(current_atr, 6)),
            'value_pips': float(round(current_atr * 10000, 1)),
            'percentile': float(round(percentile, 1)),
            'expanding': bool(expanding),
            'contracting': bool(contracting)
        }
    
    def _calculate_bollinger(self, df: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands with squeeze detection (Vectorized)."""
        close = df['close']
        
        # Bands
        middle = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        
        # Band width (Vectorized)
        bandwidth_series = (upper - lower) / middle * 100
        current_bandwidth = bandwidth_series.iloc[-1]
        
        # Squeeze detection (Vectorized)
        if len(bandwidth_series) > self.bb_period:
            valid_bw = bandwidth_series.dropna()
            percentile = (valid_bw < current_bandwidth).mean() * 100
        else:
            percentile = 50.0
        
        squeeze = percentile < 20  # Bands in bottom 20%
        
        # Position in bands
        if current_close > current_upper:
            position = 'ABOVE_UPPER'
            signal = 'OVERBOUGHT'
        elif current_close < current_lower:
            position = 'BELOW_LOWER'
            signal = 'OVERSOLD'
        elif current_close > current_middle:
            position = 'UPPER_HALF'
            signal = 'BULLISH'
        else:
            position = 'LOWER_HALF'
            signal = 'BEARISH'
        
        return {
            'upper': float(round(current_upper, 5)),
            'middle': float(round(current_middle, 5)),
            'lower': float(round(current_lower, 5)),
            'band_width': float(round(current_bandwidth, 2)),
            'squeeze': bool(squeeze),
            'position': position,
            'signal': signal
        }
    
    def _detect_displacement(self, df: pd.DataFrame) -> Dict:
        """
        Detect institutional displacement.
        
        A candle with body > 2x ATR and 70%+ body ratio = Displacement.
        This is institutional intent, not noise.
        """
        if len(df) < 15:
            return {'detected': False}
        
        atr = self._calculate_atr_analysis(df)['value']
        last_candle = df.iloc[-1]
        
        body = abs(last_candle['close'] - last_candle['open'])
        full_range = last_candle['high'] - last_candle['low']
        
        if full_range == 0:
            return {'detected': False}
        
        body_ratio = body / full_range
        atr_ratio = body / atr if atr > 0 else 0
        
        if atr_ratio >= 2.0 and body_ratio >= 0.7:
            direction = "UP" if last_candle['close'] > last_candle['open'] else "DOWN"
            
            return {
                'detected': True,
                'direction': direction,
                'atr_ratio': round(atr_ratio, 2),
                'body_ratio': round(body_ratio * 100, 1),
                'action': f'DISPLACEMENT {direction}: High-probability institutional move',
                'follow_up': 'Look for pullback to FVG or OB for entry'
            }
        
        return {'detected': False}
    
    def _classify_regime(self, analysis: Dict) -> VolatilityState:
        """Classify current volatility regime."""
        atr_data = analysis['atr']
        bb_data = analysis['bollinger']
        
        percentile = atr_data['percentile']
        
        if percentile >= 90:
            regime = 'EXTREME'
            recommendation = 'REDUCE SIZE: Extreme volatility. Wide stops required.'
        elif percentile >= 70:
            regime = 'HIGH'
            recommendation = 'Trending environment. Trade breakouts and momentum.'
        elif percentile >= 30:
            regime = 'NORMAL'
            recommendation = 'Standard volatility. All strategies valid.'
        else:
            regime = 'LOW'
            if bb_data['squeeze']:
                recommendation = 'SQUEEZE DETECTED: Prepare for expansion. Range strategies only.'
            else:
                recommendation = 'Low volatility. Consider mean reversion strategies.'
        
        return VolatilityState(
            regime=regime,
            atr=atr_data['value'],
            atr_percentile=percentile,
            expanding=atr_data['expanding'],
            contracting=atr_data['contracting'],
            recommendation=recommendation
        )


class DisplacementCandle:
    """
    Displacement Candle Detection
    
    Identifies candles that show clear institutional intent:
    - Large body (> 2x ATR)
    - High body ratio (> 70%)
    - Creates FVG (imbalance)
    """
    
    def __init__(self, atr_multiplier: float = 2.0, body_ratio_min: float = 0.7):
        self.atr_multiplier = atr_multiplier
        self.body_ratio_min = body_ratio_min
    
    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all displacement candles in recent data."""
        if df is None or len(df) < 20:
            return []
        
        # Calculate ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-14:])
        
        displacements = []
        
        for i in range(15, len(df)):
            candle = df.iloc[i]
            body = abs(candle['close'] - candle['open'])
            full_range = candle['high'] - candle['low']
            
            if full_range == 0:
                continue
            
            body_ratio = body / full_range
            atr_ratio = body / atr if atr > 0 else 0
            
            if atr_ratio >= self.atr_multiplier and body_ratio >= self.body_ratio_min:
                direction = "BULLISH" if candle['close'] > candle['open'] else "BEARISH"
                
                # Check for FVG
                has_fvg = False
                if i >= 2:
                    prev2 = df.iloc[i - 2]
                    if direction == "BULLISH":
                        has_fvg = candle['low'] > prev2['high']
                    else:
                        has_fvg = candle['high'] < prev2['low']
                
                displacements.append({
                    'index': i,
                    'time': candle.name if hasattr(candle, 'name') else None,
                    'direction': direction,
                    'body': body,
                    'atr_ratio': round(atr_ratio, 2),
                    'body_ratio': round(body_ratio * 100, 1),
                    'has_fvg': has_fvg,
                    'origin': candle['open'],
                    'destination': candle['close']
                })
        
        return displacements[-5:]  # Return last 5


class VolatilityFilter:
    """
    Volatility-based trade filter.
    
    Adjusts trade parameters based on current volatility regime.
    """
    
    def __init__(self, base_sl_pips: float = 15.0, base_tp_pips: float = 30.0):
        self.base_sl = base_sl_pips
        self.base_tp = base_tp_pips
    
    def adjust_for_volatility(self, volatility_regime: str, atr_pips: float) -> Dict:
        """
        Adjust SL/TP based on volatility regime.
        """
        if volatility_regime == 'EXTREME':
            sl_multiplier = 2.0
            tp_multiplier = 2.0
            size_multiplier = 0.5  # Half size
        elif volatility_regime == 'HIGH':
            sl_multiplier = 1.5
            tp_multiplier = 1.5
            size_multiplier = 0.75
        elif volatility_regime == 'LOW':
            sl_multiplier = 0.75
            tp_multiplier = 0.75
            size_multiplier = 1.25  # Slightly larger size
        else:  # NORMAL
            sl_multiplier = 1.0
            tp_multiplier = 1.0
            size_multiplier = 1.0
        
        # Use ATR-based stops if larger than base
        atr_sl = atr_pips * 1.5
        atr_tp = atr_pips * 2.5
        
        adjusted_sl = max(self.base_sl * sl_multiplier, atr_sl)
        adjusted_tp = max(self.base_tp * tp_multiplier, atr_tp)
        
        return {
            'sl_pips': round(adjusted_sl, 1),
            'tp_pips': round(adjusted_tp, 1),
            'size_multiplier': size_multiplier,
            'regime': volatility_regime,
            'note': f'{volatility_regime} volatility: Adjusted SL={adjusted_sl:.1f}, TP={adjusted_tp:.1f}'
        }


class BalancedPriceRange:
    """
    Balanced Price Range (BPR) Detection
    
    When a bullish FVG overlaps with a bearish FVG, the price is "balanced".
    This zone acts as a hard barrier - price should reject from it.
    """
    
    def detect_bpr(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Balanced Price Ranges.
        
        BPR = Overlap of bullish and bearish FVGs
        """
        if df is None or len(df) < 10:
            return []
        
        # Find FVGs
        bullish_fvgs = []
        bearish_fvgs = []
        
        for i in range(2, len(df)):
            c1 = df.iloc[i - 2]
            c3 = df.iloc[i]
            
            # Bullish FVG
            if c3['low'] > c1['high']:
                bullish_fvgs.append({
                    'bottom': c1['high'],
                    'top': c3['low'],
                    'index': i
                })
            
            # Bearish FVG
            if c3['high'] < c1['low']:
                bearish_fvgs.append({
                    'bottom': c3['high'],
                    'top': c1['low'],
                    'index': i
                })
        
        # Find overlaps
        bprs = []
        
        for b_fvg in bullish_fvgs:
            for s_fvg in bearish_fvgs:
                # Check for overlap
                overlap_bottom = max(b_fvg['bottom'], s_fvg['bottom'])
                overlap_top = min(b_fvg['top'], s_fvg['top'])
                
                if overlap_bottom < overlap_top:
                    bprs.append({
                        'bottom': overlap_bottom,
                        'top': overlap_top,
                        'midpoint': (overlap_bottom + overlap_top) / 2,
                        'bullish_fvg': b_fvg,
                        'bearish_fvg': s_fvg,
                        'action': 'Strong rejection zone. Use for SL placement.'
                    })
        
        return bprs[-5:]
    
    def check_bpr_rejection(self, current_price: float, bprs: List[Dict]) -> Dict:
        """Check if price is at a BPR for rejection."""
        for bpr in bprs:
            if bpr['bottom'] <= current_price <= bpr['top']:
                return {
                    'at_bpr': True,
                    'bpr': bpr,
                    'action': 'Price at BPR. Expect rejection. Use tight SL.'
                }
        
        return {'at_bpr': False}
