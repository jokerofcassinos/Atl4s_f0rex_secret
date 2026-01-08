
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("DynamicGeometry")

class DynamicGeometryEngine:
    """
    Sistema E-1: Dynamic Risk Geometry.
    Replaces fixed SL/TP with Market Structure-based points.
    Calculates "Vital Levels" based on Volatility (ATR) and Fractals.
    """
    def __init__(self):
        pass

    def calculate_geometry(self, market_state: Dict[str, Any], signal_side: str, current_price: float) -> Tuple[float, float]:
        """
        Calculates distinct VSL (Stop Loss) and VTP (Take Profit) levels.
        Returns: (vsl_price, vtp_price)
        """
        if isinstance(market_state, dict) and 'metrics' in market_state and hasattr(market_state['metrics'], 'get'):
             atr = market_state.get('metrics', {}).get('atr_value', 0.0005)
        else:
             # Handle Object
             metrics_obj = market_state.get('metrics') if isinstance(market_state, dict) else getattr(market_state, 'metrics', None)
             atr = getattr(metrics_obj, 'atr_value', 0.0005) if metrics_obj else 0.0005
        # Fallback if ATR is 0
        if atr <= 0: atr = 0.0005
        
        # 1. Structural Stop (VSL)
        # Place SL behind noise (3.0 * ATR is standard "Safe Zone")
        # In a real fractal system, we would look for the last low/high. 
        # Here we simulate finding the 'Liquidity Void' via ATR extension.
        stop_distance = atr * 3.0
        
        # SAFETY: Minimum Distance Enforcement
        # Prevent tight stops (< 5 pips) that get killed by spread.
        # 0.0015 = 0.15% (Approx 15 pips on major pairs)
        min_dist = current_price * 0.0015
        if stop_distance < min_dist:
             stop_distance = min_dist
        
        # 2. Dynamic Target (VTP)
        # Adapt reward based on session volatility.
        # If volatile, expand target. If quiet, contract.
        if isinstance(market_state, dict) and 'metrics' in market_state and hasattr(market_state['metrics'], 'get'):
             vol_val = market_state['metrics'].get('volatility', 0.001)
        else:
             metrics_obj = market_state.get('metrics') if isinstance(market_state, dict) else getattr(market_state, 'metrics', None)
             vol_val = getattr(metrics_obj, 'volatility', 0.001) if metrics_obj else 0.001
             
        vol_score = vol_val / 0.001
        reward_ratio = max(1.5, min(4.0, vol_score * 0.5 + 1.5))
        
        target_distance = stop_distance * reward_ratio
        
        if signal_side == "BUY":
            vsl = current_price - stop_distance
            vtp = current_price + target_distance
        else:
            vsl = current_price + stop_distance
            vtp = current_price - target_distance
            
        return vsl, vtp

    def get_risk_reward_ratio(self, entry, sl, tp):
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk == 0: return 0.0
        return reward / risk
