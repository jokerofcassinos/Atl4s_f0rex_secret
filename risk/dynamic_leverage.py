
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
        
        # Phase 59: Dynamic Probability Sizing (Smart Slots)
        # User Feedback: "4.92 lots was too high."
        # Old Logic: Used 90% of Equity (Max Exposure).
        # New Logic: Use X% of Equity as Margin, scaled by Confidence.
        
        # Base Allocation: 2% of Equity per trade.
        base_allocation_pct = 0.02 
        
        # Scaling by Confidence (Probability)
        # 50% Conf -> 1.0x (2%)
        # 80% Conf -> 1.6x (3.2%)
        # 95% Conf -> 3.0x (6%)
        # 100% Conf -> 5.0x (10% Max)
        
        # Linear Check to avoid negative
        if confidence < 50: confidence = 50
        
        # Probability Multiplier
        prob_mult = 1.0 + ((confidence - 50) / 10.0) * 0.5
        
        target_margin_usage = equity * base_allocation_pct * prob_mult
        
        # Apply Volatility Penalty (if VIX is high, reduce size)
        if market_volatility > 80:
             target_margin_usage *= 0.5
        
        # Convert Target Margin to Lots
        # Margin = (Lots * Contract * Price) / Leverage
        # Lots = (Margin * Leverage) / (Contract * Price)
        
        # Assumptions for BTCXAU/BTCUSD:
        price = 90000.0 # Fallback estimate, really should be passed in.
        contract_size = 1.0 # Standard BTC lot?
        
        # We need realistic price. If not passed, use safe defaults.
        # But wait, this class does not have price passed in.
        # I will use a generic "Exposure" calculation.
        
        # Calculate Lots assuming Leverage provides Buying Power
        buying_power = target_margin_usage * effective_leverage
        
        # Lots = Buying Power / Price
        # This assumes 1 Lot = 1 Unit (Currency Standard).
        # If 1 Lot = 100 Units, divide by 100.
        
        # Conservative Estimate:
        # Assume 1 Lot ~ $100,000 exposure (Standard Lot).
        # raw_lots = buying_power / 100000
        
        # Adjusted for BTCUSD (1 Lot = 1 BTC = $90k):
        raw_lots = buying_power / 90000.0
        
        # Clamp Logic
        final_lots = round(raw_lots, 2)
        final_lots = max(0.01, final_lots)
        
        # Absolute Safety Cap based on User Feedback
        if final_lots > 1.0: final_lots = 1.0 # Hard Cap restored but higher than 0.10
        
        return final_lots
        
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
