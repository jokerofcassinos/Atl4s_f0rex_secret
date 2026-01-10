"""
Laplace Demon - Momentum Analysis Module v2.0

Implements momentum and trend analysis:
- RSI with divergence detection
- MACD with signal quality scoring
- Stochastic momentum
- Toxic Order Flow detection (Compression/Expansion)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger("Laplace-Momentum")


@dataclass
class MomentumSignal:
    """Momentum analysis result."""
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-100
    divergence: Optional[str]  # BULLISH_DIV, BEARISH_DIV, None
    overbought: bool
    oversold: bool
    recommendation: str


class MomentumAnalyzer:
    """
    Complete momentum analysis suite.
    """
    
    def __init__(self):
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_k = 14
        self.stoch_d = 3
        # AGI COMPONENT: Toxic Flow Detector
        self.flow_detector = ToxicFlowDetector()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Full momentum analysis.
        Now integrates TOXIC FLOW Perception (AGI).
        """
        try:
            if df is None or len(df) < 50:
                return {'error': 'Insufficient data'}
            
            # 1. Standard Indicators (Lagging / Confirmation)
            rsi_data = self._calculate_rsi_analysis(df)
            macd_data = self._calculate_macd_analysis(df)
            stoch_data = self._calculate_stochastic(df)
            
            # 2. Flow Perception (Leading / Context)
            compression = self.flow_detector.detect_compression(df)
            expansion = self.flow_detector.detect_expansion(df)
            vector = self.flow_detector.detect_exhaustion(df) # New exhaustion logic
            
            result = {
                'rsi': rsi_data,
                'macd': macd_data,
                'stochastic': stoch_data,
                'flow': {
                    'compression': compression,
                    'expansion': expansion,
                    'exhaustion': vector
                },
                'composite': None
            }
            
            # Calculate composite signal with Flow Weighting
            result['composite'] = self._composite_signal(result)
            
            return result
            
        except Exception as e:
            logger.error(f"MOMENTUM CRASH: {e}")
            return {
                'rsi': {'signal': 'NEUTRAL', 'value': 50, 'divergence': None},
                'macd': {'direction': 'NEUTRAL'},
                'flow': {'compression':{'detected':False}, 'expansion':{'detected':False}, 'exhaustion':{'detected':False}},
                'composite': {'direction': 'NEUTRAL', 'strength': 0, 'confidence': 0, 'agreement': False}
            }
    
    def _calculate_rsi_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI with divergence detection."""
        close = df['close']
        
        # RSI calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Divergence detection
        divergence = self._detect_rsi_divergence(df, rsi)
        
        return {
            'value': round(current_rsi, 2),
            'overbought': current_rsi > 70,
            'oversold': current_rsi < 30,
            'divergence': divergence,
            'signal': 'SELL' if current_rsi > 70 else ('BUY' if current_rsi < 30 else 'NEUTRAL')
        }
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 20) -> Optional[str]:
        """
        Detect RSI divergence.
        
        Bullish: Price makes lower low, RSI makes higher low
        Bearish: Price makes higher high, RSI makes lower high
        """
        if len(df) < lookback * 2:
            return None
        
        mid = lookback
        
        # Price swings
        price_prev_low = df['low'].iloc[-lookback*2:-lookback].min()
        price_curr_low = df['low'].iloc[-lookback:].min()
        price_prev_high = df['high'].iloc[-lookback*2:-lookback].max()
        price_curr_high = df['high'].iloc[-lookback:].max()
        
        # RSI swings
        rsi_prev_low = rsi.iloc[-lookback*2:-lookback].min()
        rsi_curr_low = rsi.iloc[-lookback:].min()
        rsi_prev_high = rsi.iloc[-lookback*2:-lookback].max()
        rsi_curr_high = rsi.iloc[-lookback:].max()
        
        # Bullish divergence
        if price_curr_low < price_prev_low and rsi_curr_low > rsi_prev_low:
            return "BULLISH_DIVERGENCE"
        
        # Bearish divergence
        if price_curr_high > price_prev_high and rsi_curr_high < rsi_prev_high:
            return "BEARISH_DIVERGENCE"
        
        return None
    
    def _calculate_macd_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD with histogram analysis."""
        close = df['close']
        
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # Determine signal
        signal = 'NEUTRAL'
        if current_macd > current_signal and current_hist > prev_hist:
            signal = 'BUY'
        elif current_macd < current_signal and current_hist < prev_hist:
            signal = 'SELL'
        
        # Cross detection
        cross = None
        if macd_line.iloc[-2] < signal_line.iloc[-2] and current_macd > current_signal:
            cross = 'BULLISH_CROSS'
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and current_macd < current_signal:
            cross = 'BEARISH_CROSS'
        
        return {
            'macd': round(current_macd, 6),
            'signal': round(current_signal, 6),
            'histogram': round(current_hist, 6),
            'histogram_growing': current_hist > prev_hist,
            'cross': cross,
            'direction': signal
        }
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Dict:
        """Calculate Stochastic oscillator."""
        high = df['high'].rolling(self.stoch_k).max()
        low = df['low'].rolling(self.stoch_k).min()
        close = df['close']
        
        k = ((close - low) / (high - low).replace(0, 1e-10)) * 100
        d = k.rolling(self.stoch_d).mean()
        
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        
        signal = 'NEUTRAL'
        if current_k < 20 and current_k > d.iloc[-2]:
            signal = 'BUY'
        elif current_k > 80 and current_k < d.iloc[-2]:
            signal = 'SELL'
        
        return {
            'k': round(current_k, 2),
            'd': round(current_d, 2),
            'overbought': current_k > 80,
            'oversold': current_k < 20,
            'signal': signal
        }
    
    def _composite_signal(self, analysis: Dict) -> Dict:
        """
        Combine all momentum signals with AGI Context Weighting.
        Flow > Indicators.
        """
        signals = []
        scores = []
        
        # 1. Flow Signals (Primary)
        flow = analysis['flow']
        
        # Compression = Potential Explosion (Warning)
        if flow['compression']['detected']:
            # If compressing, we ignore RSI overbought/sold. We wait for breakout.
            return {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'context': 'COMPRESSION_TRAP_BUILDING',
                'recommendation': 'WAIT_FOR_EXPANSION'
            }
            
        # Expansion = Momentum Injection
        if flow['expansion']['detected']:
            direction = flow['expansion']['direction'] # UP/DOWN
            signals.append('BUY' if direction == "UP" else 'SELL')
            scores.append(65) # Reduced from 90 (Needs context)
            
        # Exhaustion = Reversal
        if flow['exhaustion']['detected']:
            # Vector candle exhaustion
            rev_dir = "SELL" if flow['exhaustion']['direction'] == "UP" else "BUY"
            signals.append(rev_dir)
            scores.append(60) # Reduced from 85 (Reversals are risky)
        
        # 2. Indicator Signals (Secondary)
        # RSI signal
        if analysis['rsi']['signal'] != 'NEUTRAL':
            signals.append(analysis['rsi']['signal'])
            scores.append(70 if analysis['rsi']['divergence'] else 40) # Lower weight for raw RSI
        
        # MACD signal
        if analysis['macd']['direction'] != 'NEUTRAL':
            signals.append(analysis['macd']['direction'])
            scores.append(60 if analysis['macd']['cross'] else 40)
        
        if not signals:
            return {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'agreement': False
            }
        
        # Consensus Logic
        buy_score = sum(s for sig, s in zip(signals, scores) if sig == 'BUY')
        sell_score = sum(s for sig, s in zip(signals, scores) if sig == 'SELL')
        
        total_score = buy_score + sell_score
        
        if buy_score > sell_score:
            direction = 'BUY'
            final_conf = (buy_score / total_score) * 100 if total_score > 0 else 0
        elif sell_score > buy_score:
            direction = 'SELL'
            final_conf = (sell_score / total_score) * 100 if total_score > 0 else 0
        else:
            direction = 'NEUTRAL'
            final_conf = 0
            
        # Normalize confidence to 0-100
        final_conf = min(100, final_conf)
        
        return {
            'direction': direction,
            'strength': max(buy_score, sell_score), # Raw score
            'confidence': final_conf,
            'agreement': len(set(signals)) == 1 or (buy_score > 0 and sell_score == 0) or (sell_score > 0 and buy_score == 0)
        }


class ToxicFlowDetector:
    """
    Toxic Order Flow Detection
    
    Detects institutional compression (staircasing) that precedes explosions.
    5-7 consecutive small candles = Compression (trap building)
    """
    
    def detect_compression(self, df: pd.DataFrame) -> Dict:
        """
        Detect toxic compression (staircasing).
        
        Pattern: 5+ small consecutive candles moving slowly in one direction.
        Warning: Big players are building a trap.
        """
        if df is None or len(df) < 10:
            return {'detected': False}
        
        recent = df.iloc[-10:]
        
        # Calculate average candle size
        avg_body = abs(recent['close'] - recent['open']).mean()
        
        # Check for compression
        consecutive_bullish = 0
        consecutive_bearish = 0
        small_candles = 0
        
        for i in range(len(recent)):
            candle = recent.iloc[i]
            body = abs(candle['close'] - candle['open'])
            
            if body < avg_body * 0.5:  # Smaller than half average
                small_candles += 1
            
            if candle['close'] > candle['open']:
                consecutive_bullish += 1
                consecutive_bearish = 0
            else:
                consecutive_bearish += 1
                consecutive_bullish = 0
        
        compression_detected = small_candles >= 5
        
        if compression_detected:
            direction = "UP" if consecutive_bullish > consecutive_bearish else "DOWN"
            
            return {
                'detected': True,
                'direction': direction,
                'small_candles': small_candles,
                'warning': f'TOXIC FLOW: {small_candles} small candles {direction}. TRAP BUILDING!',
                'action': f'Prepare to FADE {direction} on expansion candle',
                'risk': 'HIGH - Expect violent move opposite to current direction'
            }
        
        return {
            'detected': False,
            'small_candles': small_candles
        }
    
    def detect_expansion(self, df: pd.DataFrame, atr: float = None) -> Dict:
        """
        Detect expansion candle (trap sprung).
        
        Large candle after compression = The real move.
        """
        if df is None or len(df) < 2:
            return {'detected': False}
        
        if atr is None:
            # Calculate ATR
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())
                )
            )
            atr = tr.iloc[-14:].mean()
        
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        
        if body > atr * 2:
            direction = "UP" if last_candle['close'] > last_candle['open'] else "DOWN"
            
            return {
                'detected': True,
                'direction': direction,
                'body_atr_ratio': body / atr,
                'action': f'EXPANSION {direction}: Trade continuation in this direction',
                'momentum': 'HIGH'
            }
            
        return {'detected': False}
        
    def detect_exhaustion(self, df: pd.DataFrame) -> Dict:
        """
        [AGI PERCEPTION] Exhaustion Vector Detection.
        
        Detects "Parabolic Moves" that are unsustainable.
        Logic: 3+ Consecutive candles of same color, increasing in size.
        """
        if df is None or len(df) < 5:
             return {'detected': False}
             
        recent = df.iloc[-5:]
        closes = recent['close'].values
        opens = recent['open'].values
        
        # Check last 3
        c1, c2, c3 = abs(closes[-3]-opens[-3]), abs(closes[-2]-opens[-2]), abs(closes[-1]-opens[-1])
        
        # Increasing volatility? (Acceleration)
        acceleration = c3 > c2 > c1
        
        # Same direction?
        dir3 = closes[-3] > opens[-3]
        dir2 = closes[-2] > opens[-2]
        dir1 = closes[-1] > opens[-1]
        
        same_dir = (dir3 == dir2 == dir1)
        
        if acceleration and same_dir:
             direction = "UP" if dir1 else "DOWN"
             return {
                 'detected': True,
                 'direction': direction,
                 'type': 'PARABOLIC_EXHAUSTION',
                 'action': f'Prepare to FADE {direction}'
             }
             
        return {'detected': False}
