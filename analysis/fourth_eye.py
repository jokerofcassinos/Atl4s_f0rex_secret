import logging
import time
import config
import numpy as np

logger = logging.getLogger("Atl4s-FourthEye")

class FourthEye:
    """
    The Whale (Consensus Commander).
    Characteristics:
    - Triggers only on EXTREME Consensus (Score > 33).
    - Represents "Institutional Agreement".
    - 1 Trade per Candle.
    - Dynamic Sizing: Scales with Score Excess.
    """
    def __init__(self):
        self.last_candle_time = None
        self.trade_executed_this_candle = False
        self.threshold = 33.0
        self.last_trade_time = 0
        self.cooldown_seconds = 300 # 5 Minutes Hard Cooldown
        
    def process_tick(self, tick, df_m5, consensus_score):
        """
        Calculates Whale Entry.
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
            
        # Hard Cooldown Check (Backup for Candle Logic)
        if time.time() - self.last_trade_time < self.cooldown_seconds:
             return None, "Cooldown", 0
            
        # 2. Whale Logic
        # Consensus Score > 33 means multiple engines strongly agree.
        # This is rare and powerful.
        
        action = None
        reason = None
        
        if consensus_score > self.threshold:
            action = "BUY"
            reason = f"Whale Surge (Score: {consensus_score:.1f})"
        elif consensus_score < -self.threshold:
            action = "SELL"
            reason = f"Whale Dump (Score: {consensus_score:.1f})"
            
        if action:
            # 3. Dynamic Sizing
            # Base Lots = 0.01
            # Add 0.01 for every 5 points above threshold?
            # 33 -> 0.01
            # 38 -> 0.02
            # 43 -> 0.03
            # Limit to 0.10 max for safety unless user wants "All In" (not wise yet)
            
            excess = abs(consensus_score) - self.threshold
            extra_lots = (excess // 5) * 0.01
            dynamic_lots = round(0.01 + extra_lots, 2)
            
            if dynamic_lots > 0.15: dynamic_lots = 0.15 # Hard Cap for Whale
            
            self.trade_executed_this_candle = True
            self.last_trade_time = time.time()
            return action, reason, dynamic_lots
            
        return None, None, 0
