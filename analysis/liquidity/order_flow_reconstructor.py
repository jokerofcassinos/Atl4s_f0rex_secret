"""
Order Flow Reconstructor - Reverse Engineering Institutional Order Flow.

Reconstructs hidden order flow from price action signatures
using causal inference and pattern synthesis.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger("OrderFlowReconstructor")


class FlowType(Enum):
    AGGRESSIVE_BUY = "AGGRESSIVE_BUY"
    AGGRESSIVE_SELL = "AGGRESSIVE_SELL"
    PASSIVE_BUY = "PASSIVE_BUY"
    PASSIVE_SELL = "PASSIVE_SELL"
    ABSORPTION = "ABSORPTION"
    EXHAUSTION = "EXHAUSTION"


@dataclass
class OrderFlowSignature:
    """Detected order flow signature."""
    flow_type: FlowType
    price: float
    volume_estimate: float
    aggression: float  # 0-1
    persistence: int  # Bars maintained
    causal_strength: float


@dataclass
class FlowReconstruction:
    """Reconstructed order flow analysis."""
    signatures: List[OrderFlowSignature]
    net_flow: float  # Positive = buying
    delta_cumulative: float
    absorption_zones: List[Tuple[float, float]]  # (price, strength)
    exhaustion_detected: bool
    
    # Causal inference
    cause_chain: List[Dict]
    predicted_continuation: str
    confidence: float


class OrderFlowReconstructor:
    """
    The Flow Archaeologist.
    
    Reconstructs institutional order flow through:
    - Price action signature analysis
    - Delta and cumulative delta tracking
    - Absorption/exhaustion detection
    - Causal inference for flow continuation
    """
    
    def __init__(self):
        self.price_history: deque = deque(maxlen=500)
        self.volume_history: deque = deque(maxlen=500)
        self.delta_history: deque = deque(maxlen=500)
        self.flow_signatures: List[OrderFlowSignature] = []
        
        # Causal model
        self.causal_priors = {
            'aggressive_buy_continues': 0.65,
            'aggressive_sell_continues': 0.65,
            'absorption_reverses': 0.70,
            'exhaustion_reverses': 0.80,
        }
        
        # Cumulative tracking
        self.cumulative_delta = 0.0
        
        logger.info("OrderFlowReconstructor initialized")
    
    def reconstruct(self, ohlcv: Dict, tick_time: Optional[datetime] = None) -> FlowReconstruction:
        """
        Reconstruct order flow from OHLCV data.
        
        Args:
            ohlcv: {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
            tick_time: Timestamp
            
        Returns:
            FlowReconstruction with order flow analysis.
        """
        if tick_time is None:
            tick_time = datetime.now(timezone.utc)
        
        o, h, l, c, v = ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        
        # Calculate delta (buying vs selling pressure)
        delta = self._calculate_delta(o, h, l, c, v)
        
        # Update histories
        self.price_history.append(c)
        self.volume_history.append(v)
        self.delta_history.append(delta)
        
        # Update cumulative delta
        self.cumulative_delta += delta
        
        # Detect flow signatures
        new_signatures = self._detect_signatures(o, h, l, c, v, delta)
        self.flow_signatures.extend(new_signatures)
        
        # Clean old signatures
        self._clean_old_signatures()
        
        # Calculate net flow
        net_flow = sum(s.volume_estimate * (1 if 'BUY' in s.flow_type.value else -1) 
                      for s in self.flow_signatures[-20:])
        
        # Detect absorption zones
        absorption = self._detect_absorption_zones(c)
        
        # Detect exhaustion
        exhaustion = self._detect_exhaustion()
        
        # Causal inference
        cause_chain, continuation, confidence = self._infer_continuation()
        
        return FlowReconstruction(
            signatures=self.flow_signatures[-10:],
            net_flow=net_flow,
            delta_cumulative=self.cumulative_delta,
            absorption_zones=absorption,
            exhaustion_detected=exhaustion,
            cause_chain=cause_chain,
            predicted_continuation=continuation,
            confidence=confidence
        )
    
    def _calculate_delta(self, o: float, h: float, l: float, c: float, v: float) -> float:
        """
        Calculate buying/selling delta using candle analysis.
        
        Uses the principle that:
        - Close near high = more buying
        - Close near low = more selling
        """
        if h == l:
            return 0.0
        
        # Position of close in range
        position = (c - l) / (h - l)
        
        # Delta = volume * (2 * position - 1)
        # -v when close at low, +v when close at high
        delta = v * (2 * position - 1)
        
        return delta
    
    def _detect_signatures(self, o: float, h: float, l: float, c: float,
                          v: float, delta: float) -> List[OrderFlowSignature]:
        """Detect order flow signatures from current bar."""
        signatures = []
        
        if len(self.delta_history) < 5:
            return signatures
        
        recent_deltas = list(self.delta_history)[-5:]
        avg_vol = np.mean(list(self.volume_history)[-20:]) if len(self.volume_history) >= 20 else v
        
        # Aggressive buying - strong positive delta with volume
        if delta > 0 and v > avg_vol * 1.5:
            signatures.append(OrderFlowSignature(
                flow_type=FlowType.AGGRESSIVE_BUY,
                price=c,
                volume_estimate=delta,
                aggression=min(1.0, v / (avg_vol * 2)),
                persistence=1,
                causal_strength=0.7
            ))
        
        # Aggressive selling - strong negative delta with volume
        elif delta < 0 and v > avg_vol * 1.5:
            signatures.append(OrderFlowSignature(
                flow_type=FlowType.AGGRESSIVE_SELL,
                price=c,
                volume_estimate=abs(delta),
                aggression=min(1.0, v / (avg_vol * 2)),
                persistence=1,
                causal_strength=0.7
            ))
        
        # Absorption - high volume but price didn't move much
        price_range = h - l
        if v > avg_vol * 2 and price_range < (h * 0.0005):  # Less than 5 pips
            flow_type = FlowType.ABSORPTION
            signatures.append(OrderFlowSignature(
                flow_type=flow_type,
                price=c,
                volume_estimate=v,
                aggression=0.3,
                persistence=1,
                causal_strength=0.8
            ))
        
        # Exhaustion - decreasing delta with decreasing volume
        if len(self.volume_history) >= 5:
            vol_trend = np.polyfit(range(5), list(self.volume_history)[-5:], 1)[0]
            delta_trend = np.polyfit(range(5), recent_deltas, 1)[0]
            
            if vol_trend < 0 and abs(delta_trend) < abs(recent_deltas[0]) * 0.3:
                signatures.append(OrderFlowSignature(
                    flow_type=FlowType.EXHAUSTION,
                    price=c,
                    volume_estimate=v,
                    aggression=0.2,
                    persistence=1,
                    causal_strength=0.75
                ))
        
        return signatures
    
    def _detect_absorption_zones(self, current_price: float) -> List[Tuple[float, float]]:
        """Detect price zones where volume was absorbed."""
        zones = []
        
        absorption_sigs = [
            s for s in self.flow_signatures
            if s.flow_type == FlowType.ABSORPTION
        ]
        
        # Group by price level
        price_levels = {}
        for sig in absorption_sigs:
            key = round(sig.price, 4)
            if key not in price_levels:
                price_levels[key] = 0
            price_levels[key] += sig.volume_estimate
        
        # Return sorted by volume
        sorted_zones = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
        return sorted_zones[:5]
    
    def _detect_exhaustion(self) -> bool:
        """Detect if current trend is exhausting."""
        exhaustion_sigs = [
            s for s in self.flow_signatures[-10:]
            if s.flow_type == FlowType.EXHAUSTION
        ]
        
        return len(exhaustion_sigs) >= 2
    
    def _clean_old_signatures(self, max_count: int = 100):
        """Keep only recent signatures."""
        if len(self.flow_signatures) > max_count:
            self.flow_signatures = self.flow_signatures[-max_count // 2:]
    
    def _infer_continuation(self) -> Tuple[List[Dict], str, float]:
        """Infer likely continuation using causal model."""
        if not self.flow_signatures:
            return [], 'NEUTRAL', 0.5
        
        recent = self.flow_signatures[-10:]
        
        # Count flow types
        flow_counts = {}
        for sig in recent:
            ft = sig.flow_type
            flow_counts[ft] = flow_counts.get(ft, 0) + 1
        
        # Build causal chain
        chain = []
        
        # Most common flow type
        if flow_counts:
            dominant = max(flow_counts, key=flow_counts.get)
            
            if dominant == FlowType.AGGRESSIVE_BUY:
                chain.append({
                    'cause': 'aggressive_buying',
                    'effect': 'price_increase',
                    'probability': self.causal_priors['aggressive_buy_continues']
                })
                continuation = 'BULLISH'
                confidence = self.causal_priors['aggressive_buy_continues']
                
            elif dominant == FlowType.AGGRESSIVE_SELL:
                chain.append({
                    'cause': 'aggressive_selling',
                    'effect': 'price_decrease',
                    'probability': self.causal_priors['aggressive_sell_continues']
                })
                continuation = 'BEARISH'
                confidence = self.causal_priors['aggressive_sell_continues']
                
            elif dominant == FlowType.ABSORPTION:
                chain.append({
                    'cause': 'volume_absorption',
                    'effect': 'trend_reversal',
                    'probability': self.causal_priors['absorption_reverses']
                })
                # Determine reversal direction from cumulative delta
                continuation = 'BULLISH' if self.cumulative_delta < 0 else 'BEARISH'
                confidence = self.causal_priors['absorption_reverses']
                
            elif dominant == FlowType.EXHAUSTION:
                chain.append({
                    'cause': 'trend_exhaustion',
                    'effect': 'reversal_imminent',
                    'probability': self.causal_priors['exhaustion_reverses']
                })
                continuation = 'REVERSAL_IMMINENT'
                confidence = self.causal_priors['exhaustion_reverses']
            else:
                continuation = 'NEUTRAL'
                confidence = 0.5
        else:
            continuation = 'NEUTRAL'
            confidence = 0.5
        
        return chain, continuation, confidence
    
    def get_flow_summary(self) -> Dict:
        """Get summary of current order flow state."""
        if not self.flow_signatures:
            return {'state': 'INITIALIZING', 'confidence': 0.0}
        
        reconstruction = FlowReconstruction(
            signatures=self.flow_signatures[-10:],
            net_flow=sum(s.volume_estimate * (1 if 'BUY' in s.flow_type.value else -1) 
                        for s in self.flow_signatures[-10:]),
            delta_cumulative=self.cumulative_delta,
            absorption_zones=[],
            exhaustion_detected=self._detect_exhaustion(),
            cause_chain=[],
            predicted_continuation='',
            confidence=0.0
        )
        
        _, continuation, confidence = self._infer_continuation()
        
        return {
            'state': continuation,
            'net_flow': reconstruction.net_flow,
            'cumulative_delta': self.cumulative_delta,
            'exhaustion': reconstruction.exhaustion_detected,
            'confidence': confidence
        }
