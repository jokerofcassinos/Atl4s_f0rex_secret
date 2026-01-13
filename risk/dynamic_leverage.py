
import logging
import math

logger = logging.getLogger("DynamicLeverage")

class DynamicLeverage:
    """
    The Throttle (Restored Quantum Logic).
    Adjusts exposure using Power Laws, Entropy, and Sigmoidal Caps.
    """
    def __init__(self, max_leverage=3000):
        self.max_leverage = max_leverage
        self.base_leverage = 500

    def calculate_lots(self, equity: float, confidence: float, market_volatility: float) -> float:
        """
        Hyper-Dynamic Risk Engine (Exness Tiers).
        
        Tiers:
        1. Equity < $1000: Leverage 1:Unlimited (Simulated 1:1B)
           -> Aggressive Power Law (Equity^0.75) for rapid compounding.
        2. Equity >= $1000: Leverage 1:2000
           -> Standard Power Law (Equity^0.70) with margin cap.
        """
        # Safety for tiny accounts
        if equity < 10: return 0.01

        # EXNESS LEVERAGE TIERS
        if equity < 1000:
            # TIER 1: UNLIMITED (1:1,000,000,000)
            # High aggression for small account growth
            # Ex: $100 -> ~0.15 lots | $500 -> ~0.50 lots
            lev_tier = "UNLIMITED"
            # Power law 0.75 is steeper than 0.65
            base_lots = (pow(equity, 0.75)) / 200.0 
            MAX_CEILING = 50.0 # Effectively no cap logic
        else:
            # TIER 2: HIGH LEVERAGE (1:2000)
            # Stabilized growth for larger capital
            # Ex: $2000 -> ~0.70 lots | $10000 -> ~2.5 lots
            lev_tier = "1:2000"
            base_lots = (pow(equity, 0.70)) / 300.0
            # Margin Ceiling: (Equity * 2000) / 100000 = Max Lots allowed by broker
            # We take 50% of that to leave room for Drawdown
            MAX_CEILING = ((equity * 2000) / 100000) * 0.5

        # 2. Confidence Multiplier (Brain)
        # Score 0 -> 1.0x
        # Score 100 -> 1.5x
        # Score 50 -> 1.25x
        # Ensure confidence is positive for magnitude scalling
        conf_factor = 1.0 + (abs(confidence) / 200.0) # Max 1.5x at 100 score
        
        # 3. Entropy Damper (Chaos/Volatility)
        # Volatility usually 0-100.
        # If Vol = 0 -> 1.2x (Clean market)
        # If Vol = 100 -> 0.8x (Chaos)
        # Normalize volatility to 0-1 range
        norm_entropy = min(1.0, max(0.0, market_volatility / 100.0))
        entropy_factor = 1.2 - (norm_entropy * 0.4)
        
        raw_lots = base_lots * conf_factor * entropy_factor
        
        # logger.info(f"Risk Tier {lev_tier}: Eq ${equity:.0f} -> Base {base_lots:.2f} -> Raw {raw_lots:.2f}")

        # 4. Sigmoid Soft Cap (Asymptote at MAX_CEILING)
        # Prevents "Infinite Lots" on huge accounts, forces diversification.
        
        # Formula: L = Max * tanh(raw / Max)
        final_lots = MAX_CEILING * math.tanh(raw_lots / MAX_CEILING)
        
        # 5. Floors and Rounding
        if final_lots < 0.01: final_lots = 0.01
        final_lots = round(final_lots, 2)
        
        # logger.info(f"Quantum Risk: Eq ${equity:.0f} -> Base {base_lots:.2f} * Conf {conf_factor:.2f} * Ent {entropy_factor:.2f} = {raw_lots:.2f} -> Sigmoid {final_lots:.2f}")
        
        return final_lots
