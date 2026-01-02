import logging
import time
import config
import numpy as np

logger = logging.getLogger("Atl4s-SecondEye")

class SecondEye:
    """
    Relentless Sniper.
    Characteristics:
    - 1 Trade per Candle (Max).
    - Dynamic Lot Sizing based on Confidence.
    - High Filter Thresholds.
    """
    def __init__(self):
        self.last_candle_time = None
        self.trade_executed_this_candle = False
        self.risk_factor = 0.5 # Default Risk Multiplier (can be config)
        
    def process_tick(self, tick, df_m5, market_state={}, **kwargs):
        """
        Decides if the Sniper should fire.
        Returns: (Action, Lots, Reason) -> ORDER CHANGED to match main.py expectation [Action, Lots, Reason]
        """
        # 1. Candle Check
        if df_m5.empty: return None, 0, None
        
        latest_candle_time = df_m5.index[-1]
        if self.last_candle_time != latest_candle_time:
            self.last_candle_time = latest_candle_time
            self.trade_executed_this_candle = False
            
        if self.trade_executed_this_candle:
            return None, 0, "Candle Limit"
            
        # 2. Extract Intelligence from Holographic State
        # Overlord = Deep Brain (Alpha)
        # Swarm = Technical (Tech)
        # Micro = Physics (Energy)
        
        overlord = market_state.get('Overlord', {})
        micro = market_state.get('Micro', {})
        
        # Normalize Alpha Score (-100 to 100 -> -1.0 to 1.0)
        raw_alpha = overlord.get('score', 0)
        alpha_score = raw_alpha / 100.0 
        
        # Tech Score (Base Swarm)
        # tech_score = market_state.get('Swarm', {}).get('score', 0)
        
        # Physics
        orbit_energy = micro.get('energy', 0)
        
        action = None
        reason = None
        confidence = 0.0
        
        # Mode: Smart Predator
        # Threshold: High Confidence (> 0.60 -> > 60 Score)
        if abs(alpha_score) > 0.60: 
             # Check Confluence with Physics
             if orbit_energy > 2.0 or abs(raw_alpha) > 80:
                 action = "BUY" if alpha_score > 0 else "SELL"
                 confidence = abs(alpha_score)
                 reason = f"Sniper Shot (Conf: {confidence:.2f})"
        
        if action:
            # 3. Dynamic Sizing
            # Base Lots = 0.01
            # Multiplier = Confidence * 10 (0.6 -> 0.06, 0.9 -> 0.09)
            dynamic_lots = round(confidence * 10 * 0.01, 2)
            if dynamic_lots < 0.01: dynamic_lots = 0.01
            if dynamic_lots > 0.10: dynamic_lots = 0.10
            
            self.trade_executed_this_candle = True
            return action, dynamic_lots, reason # Returns (Action, Lots, Reason)
            
        return None, 0, None
            
        return None, None, 0
