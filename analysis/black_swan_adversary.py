import logging
import numpy as np

logger = logging.getLogger("Atl4s-Adversary")

class BlackSwanAdversary:
    """
    The Abyss (Stress System).
    Simulates extreme market scenarios for a given signal.
    - Fat-Tail Jump Simulation
    - SL Fragility Test
    - Survival Probability Calculation
    """
    def __init__(self):
        pass

    def scenario_stress_test(self, decision, entry_price, sl_price, atr):
        """
        Simulates 1000 extreme paths (Black Swans).
        If the probability of survival is low, the signal is vetoed.
        """
        if decision == "WAIT": return True, 1.0 # Safe to wait
        
        num_sims = 500
        steps = 5
        
        # SL Distance
        sl_dist = abs(entry_price - sl_price)
        
        # Black Swan Params: Heavy tailed noise (Cauchy-like)
        # We use a mix of Normal and high-amplitude sporadic jumps.
        survivals = 0
        
        for _ in range(num_sims):
            # Normal Volatility
            path_vol = np.random.normal(0, atr * 0.5, steps).sum()
            
            # Black Swan Jump (10% chance of a 2.0 * ATR jump against us)
            jump = 0
            if np.random.random() < 0.1:
                jump_dir = -1 if decision == "BUY" else 1
                jump = jump_dir * (atr * 2.5)
                
            total_dev = path_vol + jump
            
            # If the deviation hits our SL
            if decision == "BUY":
                if entry_price + total_dev <= sl_price: continue
            else:
                if entry_price + total_dev >= sl_price: continue
                
            survivals += 1
            
        survival_prob = survivals / num_sims
        
        # Safety Threshold: Must survive 28% of 'reasonable' Black Swans (Phase 1 Relaxed)
        # NOTE: Reduced from 0.85 (was 0.95) to 0.28.
        # This acts as a "Catastrophe Guard" rather than a strict filter.
        is_safe = survival_prob > 0.28
        
        return is_safe, survival_prob

    def audit_trade(self, decision, price, sl, atr):
        """
        Public audit interface.
        """
        is_safe, prob = self.scenario_stress_test(decision, price, sl, atr)
        
        if not is_safe:
             logger.warning(f"ADVERSARY VETO: Trade failed Black-Swan stress test (Surv Prob: {prob:.2f})")
             
        return is_safe, prob
