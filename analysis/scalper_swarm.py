import logging
import time
import config
import numpy as np

logger = logging.getLogger("Atl4s-Swarm")

class ScalpSwarm:
    def __init__(self):
        self.max_trades = config.SWARM_MAX_TRADES
        self.trade_count = 0
        self.last_candle_time = None
        self.last_trade_time = 0
        self.cooldown = config.SWARM_COOLDOWN
        
    def process_tick(self, tick, df_m5, alpha_score, tech_score, phy_score, signal_dir):
        """
        Analyzes the tick and context to decide if one of the 'Eyes' should fire.
        Returns: (Action: str, Reason: str, Price: float) or (None, None, None)
        """
        current_time = time.time()
        
        # 1. New Candle Reset
        # Assuming df_m5 index is datetime, we check if the last candle timestamp changed
        if not df_m5.empty:
            latest_candle_time = df_m5.index[-1]
            if self.last_candle_time != latest_candle_time:
                self.trade_count = 0
                self.last_candle_time = latest_candle_time
                # logger.info("Swarm: New Candle. Trade Limit Reset.")

        # 2. Constraints Check
        if self.trade_count >= self.max_trades:
            return None, "Max Trades Reached", 0
            
        if current_time - self.last_trade_time < self.cooldown:
            return None, "Cooldown", 0
            
        if signal_dir not in ["BUY", "SELL"]:
            return None, "No Direction", 0

        # 3. Strategy Logic (The Eyes)
        price = tick['last']
        action = None
        reason = None
        
        # --- Eye 1: The Alpha Strike (High Conviction) ---
        # If Deep Cognition is very strong (>0.7) and Tech agrees
        if abs(alpha_score) > 0.7 and np.sign(alpha_score) == np.sign(tech_score):
             action = signal_dir
             reason = "Eye 1: Alpha Strike"
             
        # --- Eye 2: Pullback Sniper (Better Price) ---
        # If signal is BUY but price is below Open (Discount)
        elif not df_m5.empty:
            candle_open = df_m5.iloc[-1]['open']
            if signal_dir == "BUY" and price < candle_open - 0.5: # 50 points dip
                if alpha_score > 0.4: # Decent support
                    action = "BUY"
                    reason = "Eye 2: Pullback Sniper"
            elif signal_dir == "SELL" and price > candle_open + 0.5:
                if alpha_score < -0.4:
                    action = "SELL"
                    reason = "Eye 2: Pullback Sniper"
                    
        # --- Eye 3: Momentum Burst (Physics) ---
        # If Physics implies High Energy (>3.0) and direction aligns
        if not action and abs(phy_score) > 3.0: # High Energy
             # Check if phy direction aligns with signal
             # Usually positive kinematics means acceleration UP? Need to verify kinematics output.
             # Assuming phy_score sign matches direction (it usually does in simple kinematics)
             if np.sign(phy_score) == (1 if signal_dir == "BUY" else -1):
                 action = signal_dir
                 reason = "Eye 3: Momentum Burst"

        # 4. Execution Update
        if action:
            self.trade_count += 1
            self.last_trade_time = current_time
            logger.info(f"SWARM EXECUTION ({self.trade_count}/{self.max_trades}): {reason} @ {price}")
            return action, reason, price
            
        return None, None, 0
