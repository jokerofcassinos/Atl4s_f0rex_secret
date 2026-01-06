
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
        Hyper-Dynamic Risk Engine (Quantum Sigmoid Scaling).
        Restored from Backup 'risk_manager.py'.
        
        Logic:
        1. Base Power Law: Equity^0.65 (Diminishing returns but infinite growth).
        2. Brain Factor: Confidence Score (-100 to 100) boosts/cuts size.
        3. Entropy Factor: High Chaos (Volatility) -> Lower Size.
        4. Sigmoid Cap: Smooth limit approaching Max Ceiling.
        """
        # Safety for tiny accounts
        if equity < 10: return 0.01

        # 1. Base Power Law Scaling
        # Example: $5000 ^ 0.65 = 253.
        # Divisor = 350. -> Base = 0.72 Lots.
        # Tuned for Cent Accounts or Standard?
        # Assuming Standard: $30 -> 0.02 lots. 
        # 30^0.65 = 9.1. 9.1 / 350 = 0.026. MATCHES LEGACY.
        base_lots = (pow(equity, 0.65)) / 350.0
        
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
        
        # 4. Sigmoid Soft Cap (Asymptote at MAX_CEILING)
        # Prevents "Infinite Lots" on huge accounts, forces diversification.
        MAX_CEILING = 5.0 # Increased from 2.5 per user "4.92 lots" comment
        
        # Formula: L = Max * tanh(raw / Max)
        final_lots = MAX_CEILING * math.tanh(raw_lots / MAX_CEILING)
        
        # 5. Floors and Rounding
        if final_lots < 0.01: final_lots = 0.01
        final_lots = round(final_lots, 2)
        
        # logger.info(f"Quantum Risk: Eq ${equity:.0f} -> Base {base_lots:.2f} * Conf {conf_factor:.2f} * Ent {entropy_factor:.2f} = {raw_lots:.2f} -> Sigmoid {final_lots:.2f}")
        
        return final_lots
