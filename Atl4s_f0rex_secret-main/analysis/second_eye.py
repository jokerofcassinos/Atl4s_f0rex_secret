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
        
    def process_tick(self, tick, df_m5, alpha_score, tech_score, orbit_energy):
        """
        Decides if the Sniper should fire.
        Returns: (Action, Reason, DynamicLots)
        """
        # 1. Candle Check
        if df_m5.empty: return None, None, 0
        
        latest_candle_time = df_m5.index[-1]
        if self.last_candle_time != latest_candle_time:
            self.last_candle_time = latest_candle_time
            self.trade_executed_this_candle = False
            
        if self.trade_executed_this_candle:
            return None, "Candle Limit", 0
            
        # 2. Sniper Logic (Stricter than Swarm)
        # We require Deep Brain (Alpha) to be VERY sure.
        
        action = None
        reason = None
        confidence = 0.0
        
        # Mode: Smart Predator (Only) - The Second Eye doesn't Bait. It Hunts.
        # It waits for the Brain to identify a clear imbalance.
        
        if abs(alpha_score) > 0.60: # High Confidence Threshold
             # Check Confluence with Physics (Energy)
             # If High Energy, we trust the Brain more.
             if orbit_energy > 2.0 or abs(tech_score) > 10.0:
                 action = "BUY" if alpha_score > 0 else "SELL"
                 confidence = abs(alpha_score)
                 reason = f"Sniper Shot (Conf: {confidence:.2f})"
        
        if action:
            # 3. Dynamic Sizing
            # Base Lots = 0.01
            # Multiplier = Confidence * 10 
            # 0.60 -> 6x -> 0.06 lots
            # 0.90 -> 9x -> 0.09 lots
            # Max Cap at 0.10 for safety
            
            dynamic_lots = round(confidence * 10 * 0.01, 2)
            if dynamic_lots < 0.01: dynamic_lots = 0.01
            if dynamic_lots > 0.10: dynamic_lots = 0.10
            
            self.trade_executed_this_candle = True
            return action, reason, dynamic_lots
            
        return None, None, 0
