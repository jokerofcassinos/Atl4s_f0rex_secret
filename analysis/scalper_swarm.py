import logging
import time
import config
import numpy as np

logger = logging.getLogger("Atl4s-Swarm")

class ScalpSwarm:
    """
    Swarm Intelligence 2.0: The Unified Field.
    
    Models market price as a particle moving through a fluid field of Order Flow.
    Integrates:
    1. Kinematic Velocity (Particle Speed)
    2. Field Pressure (Order Flow/Alpha Potential)
    3. Strange Attractors (Chaos/Lyapunov)
    4. Entropy-Dynamic Weighting
    """
    def __init__(self):
        self.max_trades = config.SWARM_MAX_TRADES
        self.trade_count = 0
        self.last_candle_time = None
        self.last_trade_time = 0
        self.cooldown = config.SWARM_COOLDOWN
        self.threshold = config.SWARM_THRESHOLD # Usually 0.5
        
    def process_tick(self, tick, df_m5, alpha_score, tech_score, phy_score, micro_stats):
        """
        Calculates Unified Field Vector.
        Returns: (Action, Reason, Price)
        """
        current_time = time.time()
        
        # 1. Housekeeping
        if not df_m5.empty:
            latest_candle_time = df_m5.index[-1]
            if self.last_candle_time != latest_candle_time:
                self.trade_count = 0
                self.last_candle_time = latest_candle_time

        if self.trade_count >= self.max_trades: return None, "Max Trades", 0
        if current_time - self.last_trade_time < self.cooldown: return None, "Cooldown", 0
        
        price = tick['last']
        
        # --- UNIFIED FIELD CALCULATION ---
        
        # 1. Particle Velocity (v)
        # Normalized velocity from micro-structure (-1.0 to 1.0)
        # We assume micro_stats['velocity'] is effectively slope. 
        # We boost it to ensure it spans -1 to 1.
        raw_vel = micro_stats.get('velocity', 0)
        v = np.clip(raw_vel * 5, -1.0, 1.0) 
        
        # 2. Field Pressure (P)
        # Driven by Order Flow Imbalance (OFI) and Alpha Score
        ofi = micro_stats.get('ofi', 0)
        norm_ofi = np.clip(ofi / 10.0, -1.0, 1.0) # Assume OFI > 10 is strong
        norm_alpha = np.clip(alpha_score, -1.0, 1.0)
        
        # Pressure is the average of OFI (Immediate) and Alpha (Slightly longer term)
        P = (norm_ofi * 0.6) + (norm_alpha * 0.4)
        
        # 3. Strange Attractor (A)
        # Chaos Score (phy_score) usually indicates "Energy".
        # If High Energy + High Entropy -> Pull towards Mean (Reversion)
        # If High Energy + Low Entropy -> Laminar Flow (Trend)
        entropy = micro_stats.get('entropy', 0.5)
        hurst = micro_stats.get('micro_hurst', 0.5)
        
        # We derive the "Attractor Force".
        # If Hurst < 0.5 (Mean Reversion), Attractor opposes Velocity.
        # If Hurst > 0.5 (Trend), Attractor aligns with Velocity.
        
        # This is tricky: phy_score is scalar energy. We need direction.
        # We use the 'tech_score' sign as the "baseload" direction of the attractor.
        attractor_dir = np.sign(tech_score) if tech_score != 0 else 0
        
        A = 0
        if hurst < 0.4:
            # Mean Reversion: Attractor pulls OPPOSITE to current movement if overextended
            # Or pulls towards 'tech_score' (Consensus Mean)
            A = attractor_dir * min(abs(phy_score)/5.0, 1.0)
        else:
            # Trend: Attractor pushes WITH momentum
            A = v * min(abs(phy_score)/5.0, 1.0)
            
        # 4. Entropy-Dynamic Weighting
        # w_v (Velocity Weight), w_p (Pressure Weight), w_a (Attractor Weight)
        
        if entropy < 0.3:
            # Low Entropy (Ordered/Laminar) -> Trust Velocity & Pressure
            w_v = 0.4
            w_p = 0.4
            w_a = 0.2
            regime = "LAMINAR"
        elif entropy > 0.7:
             # High Entropy (Chaotic/Turbulent) -> Trust Attractor & Pressure (OFI), ignore Velocity (Noise)
            w_v = 0.1
            w_p = 0.5
            w_a = 0.4
            regime = "TURBULENT"
        else:
            # Transition
            w_v = 0.33
            w_p = 0.33
            w_a = 0.33
            regime = "TRANSITION"
            
        # 5. The Unified Vector (S)
        S = (w_v * v) + (w_p * P) + (w_a * A)
        
        # Boost S if all components align
        if np.sign(v) == np.sign(P) == np.sign(A) and abs(S) > 0.1:
            S *= 1.2 # Constructive Interference
            
        # Clip
        S = np.clip(S, -1.0, 1.0)
        
        # 6. Decision Logic
        action = None
        
        # Dynamic Threshold based on Regime
        # Turbulent markets need higher conviction (Safety)
        # Laminar markets need LOWER conviction (Speed/Frequency)
        needed_threshold = self.threshold
        
        if regime == "TURBULENT": 
            needed_threshold += 0.15
        elif regime == "LAMINAR":
            needed_threshold -= 0.15 # Aggressive Boost for Trend Following
        
        # Base sensitivity boost
        needed_threshold = max(0.35, needed_threshold) # Hard floor
        
        if S > needed_threshold: action = "BUY"
        elif S < -needed_threshold: action = "SELL"
        
        if action:
            self.trade_count += 1
            self.last_trade_time = current_time
            reason = f"Unified Field {regime} (Vec:{S:.2f} > {needed_threshold:.2f} | v:{v:.2f} P:{P:.2f} A:{A:.2f})"
            logger.info(f"SWARM EXECUTION ({self.trade_count}/{self.max_trades}): {reason} @ {price}")
            return action, reason, price
            
        # Log Vector state
        if abs(S) > 0.2:
             logger.info(f"Swarm Field: {S:.2f} (Req: {needed_threshold:.2f}) | {regime} | v:{v:.2f} P:{P:.2f} A:{A:.2f}")

        return None, None, 0
