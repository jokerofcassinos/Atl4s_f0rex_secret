
import logging

logger = logging.getLogger("DynamicLeverage")

class DynamicLeverage:
    """
    The Throttle.
    Adjusts exposure from 1:3000 (Sniper) to 1:30 (Shield) dynamically.
    """
    def __init__(self, max_leverage=3000):
        self.max_leverage = max_leverage
        self.base_leverage = 500 # Default

    def calculate_lots(self, equity: float, confidence: float, market_volatility: float) -> float:
        """
        Determines the precise lot size.
        """
        # 1. Base on Equity (aggressive growth for $5)
        # Standard: 0.01 lots per $100
        # Aggressive (1:3000): 0.10 lots per $100 -> 0.01 lots per $10?
        
        # For $5, we need minimum lots (0.01).
        # At 1:3000, 0.01 lots on Gold requires ~$0.60 margin.
        # So $5 can theoretically open ~0.08 lots.
        
        effective_leverage = self.base_leverage
        
        # Boost Leverage if Confidence is High
        if confidence > 90.0:
            effective_leverage = 2000
        if confidence > 95.0:
            effective_leverage = 3000
            
        # Reduce if Volatility is Extreme (Flash Crash risk)
        if market_volatility > 80.0:
            effective_leverage = max(30, effective_leverage / 4)
            
        # Calculation:
        # Lot = (Equity * Leverage) / ContractSize
        # Gold Contract Size = 100
        # Price ~2600
        
        # Formula: Margin = (Lots * Contract * Price) / Leverage
        # Lots = (Margin * Leverage) / (Contract * Price)
        # Margin = Equity * RiskRatio (e.g. use 50% of equity)
        
        risk_equity = equity * 0.90 # Use 90% of equity (Maximum Aggression)
        
        current_price = 2650 # Hardcoded Approx if not passed, but caller should pass it
        
        # Simplified lot calculation logic based on leverage ratio
        # Lots = (Equity * Lever) / 265000
        
        raw_lots = (risk_equity * effective_leverage) / (100 * 2650)
        
        # Clamp Logic
        final_lots = round(raw_lots, 2)
        final_lots = max(0.01, final_lots)
        
        if final_lots > 5.0: final_lots = 5.0 # Absolute Cap
        
        # Specific Adjustment for $5 account:
        # If we have only $5, we can likely only do 0.01 or 0.02 safely-ish.
        # 0.01 lots Gold = $1 per point. 500 point move kills it. 
        # Actually 0.01 lots = $0.01 per point (tick value).
        # No, 1 Lot XAUUSD = 100oz. 1 pip (0.01) = $1.
        # 0.01 Lot = $0.01 per 0.01 move (1 cent per point).
        
        # Validating math:
        # 1.00 move in Gold (2650 -> 2651)
        # 0.01 Lot * 100 * 1 = $1 profit/loss.
        # So $5 account survives 5 points move (approx 500 pips).
        
        return final_lots
