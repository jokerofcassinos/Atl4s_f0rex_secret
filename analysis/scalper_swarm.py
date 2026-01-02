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
        
    def check_resonance(self, df_m5, df_m1):
        """
        Fractal Resonance Verification.
        Checks if M5 and M1 are singing the same note (Trend Alignment).
        Returns: (is_resonant, direction)
        """
        if df_m5.empty or df_m1.empty: return False, 0
        
        # M5 Trend (EMA 20 vs EMA 50)
        # We need to calculate if not present, but usually we assume trend_architect handles big trend.
        # Let's do a quick slope check on Close
        m5_slope = df_m5['close'].iloc[-1] - df_m5['close'].iloc[-5]
        m1_slope = df_m1['close'].iloc[-1] - df_m1['close'].iloc[-5]
        
        # Resonance = Both Slopes Strong and Same Sign
        if abs(m5_slope) > 0.5 and abs(m1_slope) > 0.1: # Thresholds for "Slope"
             if np.sign(m5_slope) == np.sign(m1_slope):
                 return True, np.sign(m5_slope)
                 
        return False, 0

    def process_tick(self, tick, df_m5, df_m1, alpha_score, tech_score, phy_score, micro_stats, forced_lots=None):
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
        
        # --- 13th EYE OVERRIDE (Project Tachyon) ---
        # If Alpha Brain screams (100.0), we force the Vector to match.
        # This prevents "Calculated Indecision" from stopping the Sniper Shot.
        if alpha_score >= 99.0:
            S = 1.0
            S_override = True
        elif alpha_score <= -99.0:
            S = -1.0
            S_override = True
        else:
            S_override = False
        
        # 6. Decision Logic
        action = None
        
        # Dynamic Threshold based on Regime
        # Turbulent markets need higher conviction (Safety)
        # Laminar markets need LOWER conviction (Speed/Frequency)
        needed_threshold = self.threshold
        
        if regime == "TURBULENT": 
            needed_threshold += 0.05 # Lowered penalty to keep Swarm active
        elif regime == "LAMINAR":
            needed_threshold -= 0.15 # Aggressive Boost for Trend Following
        
        # FRACTAL RESONANCE (PROJECT TACHYON)
        # If M5 and M1 align perfectly, we drop threshold to near zero (0.15) for immediate entry
        is_resonant, res_dir = self.check_resonance(df_m5, df_m1)
        if is_resonant:
             # Check if Resonance Direction matches Vector S
             if np.sign(S) == res_dir:
                 needed_threshold = 0.15 # TACHYON MODE
                 regime += "_RESONANCE"
        
        # Base sensitivity boost (User Request: "Execute something!")
        needed_threshold = max(0.15, needed_threshold) # Hard floor lowered from 0.25 for Tachyon
        
        if S > needed_threshold: 
            # Check for 13th Eye Override (Alpha Score extreme)
            is_override = abs(alpha_score) >= 99.0

            # MOMENTUM SAFETY: Don't BUY if we are actively crashing
            # UNLESS it is an Override (Knife Catch)
            if v > -0.1 or is_override: 
                # FOMO GUARD (Avg Reversion)
                is_safe_entry = True
                
                if is_override:
                     # Skip Wick/Density checks
                     pass
                else: 
                    # 1. Wick Rejection Logic (Buying into resistance?)
                    # If High - Close > Body * 1.5 -> Large Upper Wick (Rejection)
                    if not df_m1.empty:
                        last_candle = df_m1.iloc[-1]
                        c_open = last_candle['open']
                        c_close = last_candle['close']
                        c_high = last_candle['high']
                        c_low = last_candle['low']
                        body = abs(c_close - c_open)
                        rng = c_high - c_low
                        upper_wick = c_high - max(c_open, c_close)
                          
                        # Wick Rejection (Relaxed for M1)
                        # Allow larger wicks if Momentum is very strong (v > 0.5)
                        wick_ratio = 2.0 if v > 0.5 else 1.5
                        if upper_wick > (body * wick_ratio) and upper_wick > 0.15: # Raised min wick size
                            logger.warning(f"Swarm BUY Vetoed: Wick Rejection (Wick {upper_wick:.2f})")
                            is_safe_entry = False

                        # Body Density Logic (Relaxed)
                        density = body / rng if rng > 0 else 0
                        if density < 0.15 and rng > 0.30: # Only veto huge dojis
                            logger.warning(f"Swarm BUY Vetoed: Low Body Density ({density:.2f})")
                            is_safe_entry = False

                if is_safe_entry:
                    action = "BUY"
            else:
                 logger.debug(f"Swarm BUY Vetoed: Negative Velocity ({v:.2f})")

        elif S < -needed_threshold: 
            # Check for 13th Eye Override (Alpha Score extreme)
            is_override = abs(alpha_score) >= 99.0
            
            # MOMENTUM SAFETY: Don't SELL if we are actively rocketing
            # UNLESS it is an Override (Selling the Top)
            if v < 0.1 or is_override: 
                # FOMO GUARD (Avg Reversion)
                is_safe_entry = True
                
                if is_override:
                     # Skip Wick/Density checks
                     pass
                else: 
                    # 1. Wick Rejection Logic (Selling into support?)
                    if not df_m1.empty:
                        last_candle = df_m1.iloc[-1]
                        c_open = last_candle['open']
                        c_close = last_candle['close']
                        c_low = last_candle['low']
                        c_high = last_candle['high']
                        body = abs(c_close - c_open)
                        rng = c_high - c_low
                        lower_wick = min(c_open, c_close) - c_low
                          
                        # Wick Rejection
                        if lower_wick > (body * 1.5) and lower_wick > 0.10: # Significant rejection
                            logger.warning(f"Swarm SELL Vetoed: Wick Rejection (Wick {lower_wick:.2f} > Body {body:.2f} * 1.5)")
                            is_safe_entry = False
                               
                        # Body Density Logic (Conviction Check)
                        density = body / rng if rng > 0 else 0
                        if density < 0.25 and rng > 0.20:
                            logger.warning(f"Swarm SELL Vetoed: Low Body Density ({density:.2f}). Indecision.")
                            is_safe_entry = False

                if is_safe_entry:
                    action = "SELL"
            else:
                 logger.debug(f"Swarm SELL Vetoed: Positive Velocity ({v:.2f})")
        
        # 7. Trend Alignment (Laminar Mode Only)
        # In Laminar (Easy) mode, we should NOT fight the consensus trend.
        if action and regime == "LAMINAR":
            # tech_score > 0 means General Bullish
            if action == "BUY" and tech_score < -10:
                action = None
                logger.debug(f"Swarm BUY Vetoed: Fighting Laminar Bear Trend (Tech:{tech_score})")
            elif action == "SELL" and tech_score > 10:
                action = None
                logger.debug(f"Swarm SELL Vetoed: Fighting Laminar Bull Trend (Tech:{tech_score})")

        # 8. HYPER-SCALP EXTENSION (Distance from M1 EMA)
        # If buying, price should not be > 2.0 SD from M1 EMA (Overbought)
        # We can approximate SD if not available via simple percent distance
        if action:
             if not df_m1.empty:
                 # Check last 5 candles volatility to estimate SD
                 # This is computationally cheap
                 pass # Placeholder for v2.1 (Micro-Bollinger)
        
        if action:
            self.trade_count += 1
            self.last_trade_time = current_time
            reason = f"Unified Field {regime} (Vec:{S:.2f} > {needed_threshold:.2f} | v:{v:.2f} P:{P:.2f} A:{A:.2f})"
            logger.info(f"SWARM EXECUTION ({self.trade_count}/{self.max_trades}): {reason} @ {price}")
            return action, reason, price
            
        # Log Vector state
        if abs(S) > 0.2:
             # Only log if action is None (prevent double logging)
             if not action:
                 logger.info(f"Swarm Field: {S:.2f} (Req: {needed_threshold:.2f}) | {regime} | v:{v:.2f} P:{P:.2f} A:{A:.2f}")

        return None, None, 0
