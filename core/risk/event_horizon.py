
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("EventHorizonRisk")

class EventHorizonRisk:
    """
    Phase 116: The Event Horizon (Parabolic Trailing Stop).
    
    Addresses User Feedback: "Spread eats profits / waiting too long to trail."
    
    Logic:
    1. Spread Awareness: Activates as soon as Price clears Spread * RewardRatio.
    2. Gravity Well: As Price Acceleration increases (Explosion), the Stop Distance decreases.
       Dist = BaseDist / (1 + k * Acceleration)
    """
    
    def __init__(self, base_risk_pips: float = 20.0):
        self.base_risk_pips = base_risk_pips
        # k factor: Sensitivity to acceleration. 
        # Reduced from 5.0 to 2.0 to give more breathing room (User Feedback)
        self.k_factor = 2.0 
        
        # State tracking per symbol
        # {symbol: {'last_price': float, 'velocity': float, 'acceleration': float}}
        self.kinematics = {} 
        
    def update_kinematics(self, symbol: str, current_price: float):
        """
        Updates the physics model of the price action.
        """
        if symbol not in self.kinematics:
            self.kinematics[symbol] = {
                'last_price': current_price,
                'velocity': 0.0,
                'acceleration': 0.0
            }
            return
            
        state = self.kinematics[symbol]
        delta_p = current_price - state['last_price']
        
        # Simple Physics
        new_velocity = delta_p
        acceleration = new_velocity - state['velocity']
        
        # Update State
        state['last_price'] = current_price
        state['velocity'] = new_velocity
        state['acceleration'] = abs(acceleration) # Magnitude matters
        
    def calculate_dynamic_stop(self, symbol: str, entry_price: str, current_price: float, side: str, spread: float) -> Optional[float]:
        """
        Calculates the Event Horizon Stop Level.
        Returns None if not yet active (profit < spread requirement).
        """
        self.update_kinematics(symbol, current_price)
        state = self.kinematics.get(symbol)
        accel = state['acceleration'] if state else 0.0
        
        # 1. Spread Check (User Feedback Fix)
        # We need to cover the spread before trailing.
        # Increased to 5.0x Spread to prevent "choking" the trade early.
        profit = 0.0
        if side == "BUY":
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price
            
        if profit <= (spread * 5.0):
            return None # Not profitable enough to trail yet correctly.
            
        # 2. Event Horizon Equation
        # We normalize accel to some meaningful scale (e.g. pips^2)
        # For simplicity, we use raw price delta. XAUUSD 0.10 move is big.
        # Let's dampen it.
        
        # Dynamic Distance
        # As Accel -> Infinity, Dist -> 0
        # Base Distance (e.g. 500 points / $5)
        # We need a dynamic base based on volatility, but for now fixed.
        # Let's use the spread as a unit!
        base_dist = spread * 20.0 # Start loose (20x spread, up from 15x)
        if base_dist < self.base_risk_pips: 
             base_dist = self.base_risk_pips * 0.01 
        
        # Formula
        dynamic_dist = base_dist / (1.0 + (self.k_factor * 0.5 * accel))
        
        # Minimum breathing room (Spread * 2.0)
        min_dist = spread * 2.0
        if dynamic_dist < min_dist:
            dynamic_dist = min_dist
            
        # Logging for Debug
        if accel > 0.1: # Significant move
             logger.info(f"EVENT HORIZON: {symbol} Accel={accel:.4f} -> Tightening Stop to {dynamic_dist:.2f} (Base: {base_dist:.2f})")
             
        # Calculate Absolute Price Level
        if side == "BUY":
            stop_level = current_price - dynamic_dist
            # Ratchet: Never move DOWN. (Handled by caller usually, but we assume current calc is target)
        else:
            stop_level = current_price + dynamic_dist
            
        return stop_level
