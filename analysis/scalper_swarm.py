import logging
import time
import config
import numpy as np

logger = logging.getLogger("Atl4s-Swarm")

class ScalpSwarm:
    """
    HFT-tier Swarm (1st Eye).
    Implements a multi-agent voting system (Swarm Intelligence).
    Consists of 5 specialized 'Eyes' that analyze micro-structure in real-time.
    """
    def __init__(self):
        self.max_trades = config.SWARM_MAX_TRADES
        self.trade_count = 0
        self.last_candle_time = None
        self.last_trade_time = 0
        self.cooldown = config.SWARM_COOLDOWN
        self.weights = config.SWARM_EYE_WEIGHTS
        self.threshold = config.SWARM_THRESHOLD
        
    def process_tick(self, tick, df_m5, alpha_score, tech_score, phy_score, micro_stats):
        """
        Main HFT entry point. Uses a weighted consensus of 5 'Eyes'.
        Returns: (Action, Reason, Price)
        """
        current_time = time.time()
        
        # 1. Housekeeping & Constraints
        if not df_m5.empty:
            latest_candle_time = df_m5.index[-1]
            if self.last_candle_time != latest_candle_time:
                self.trade_count = 0
                self.last_candle_time = latest_candle_time

        if self.trade_count >= self.max_trades: return None, "Max Trades", 0
        if current_time - self.last_trade_time < self.cooldown: return None, "Cooldown", 0
        
        # 2. Micro-Entropy Gate
        entropy = micro_stats.get('entropy', 1.0)
        e_min, e_max = config.SWARM_ENTROPY_LIMITS
        if entropy < e_min or entropy > e_max:
            # logger.debug(f"Swarm: Entropy Gate Active ({entropy:.2f})")
            return None, "Entropy Gate", 0
            
        # 3. The 5 Eyes Consensus Engine
        votes = {
            'hybrid': 0,        # Eye 1
            'pullback': 0,      # Eye 2
            'momentum': 0,      # Eye 3
            'ofi': 0,           # Eye 4 (NEW)
            'hurst_climax': 0   # Eye 5 (NEW)
        }
        
        price = tick['last']
        is_chaotic = phy_score > 2.5
        
        # --- Eye 1: Hybrid Mode ---
        if is_chaotic:
            if abs(alpha_score) > 0.4: votes['hybrid'] = 1 if alpha_score > 0 else -1
        else:
            if abs(tech_score) > 5: votes['hybrid'] = 1 if tech_score > 0 else -1
            
        # --- Eye 2: Pullback Sniper ---
        if not df_m5.empty:
            open_p = df_m5.iloc[-1]['open']
            if price < open_p - 0.5: votes['pullback'] = 1 # Discount
            elif price > open_p + 0.5: votes['pullback'] = -1 # Premium
            
        # --- Eye 3: Momentum Burst ---
        if abs(phy_score) > 3.0:
            votes['momentum'] = 1 if phy_score > 0 else -1
            
        # --- Eye 4: Order Flow Shadow (OFI) ---
        ofi = micro_stats.get('ofi', 0)
        if abs(ofi) >= 5: # Threshold for consensus
            votes['ofi'] = 1 if ofi > 0 else -1
            
        # --- Eye 5: Fractal Micro-Climax (Hurst) ---
        h = micro_stats.get('micro_hurst', 0.5)
        if h < 0.3: # Strong Mean Reversion (Exhaustion)
            # Fade the current velocity
            vel = micro_stats.get('velocity', 0)
            if abs(vel) > 0.1:
                votes['hurst_climax'] = -1 if vel > 0 else 1
                
        # 4. Weighted Voting Calculation
        final_vector = 0
        active_eyes = []
        for eye, weight in self.weights.items():
            if votes[eye] != 0:
                final_vector += votes[eye] * weight
                active_eyes.append(eye)
                
        # 5. Final Decision
        action = None
        if final_vector >= self.threshold: action = "BUY"
        elif final_vector <= -self.threshold: action = "SELL"
        
        if action:
            # Velocity Guard Check
            velocity = micro_stats.get('velocity', 0)
            if (action == "BUY" and velocity < -0.3) or (action == "SELL" and velocity > 0.3):
                return None, "Velocity Guard", 0
                
            self.trade_count += 1
            self.last_trade_time = current_time
            reason = f"Swarm Consensus ({final_vector:.2f}) | Eyes: {','.join(active_eyes)}"
            logger.info(f"SWARM EXECUTION ({self.trade_count}/{self.max_trades}): {reason} @ {price}")
            return action, reason, price
            
        return None, None, 0
