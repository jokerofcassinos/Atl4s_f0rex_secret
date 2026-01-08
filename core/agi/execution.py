
import logging
import random
from typing import Dict, Any, List

logger = logging.getLogger("HyperExecution")

class AdaptiveExecutionMatrix:
    """
    System 21: Adaptive Execution Matrix.
    Determines optimal order type and execution strategy based on microstructure.
    """
    def resolve_order_type(self, decision: str, market_data: Dict[str, Any]) -> str:
        """
        Returns: 'MARKET', 'LIMIT', 'ICEBERG', 'TWAP'
        """
        if decision == "WAIT": return "NONE"
        
        spread = market_data.get('spread', 0.0)
        volatility = market_data.get('atr', 0.0)
        
        # 1. High Spread -> Limit Orders (Don't pay the spread)
        if spread > 1.5: 
            # Unless Volatility is Extreme (Panic) -> Market get out
            if volatility > 50:
                 return "MARKET" # Panic Sell/Buy
            return "LIMIT"
            
        # 2. Low Spread -> Market (Speed)
        if spread < 0.5:
             return "MARKET"
             
        # 3. Large Size -> Iceberg (Hidden)
        # Assuming standard size for now, but if we had 'lot_size' arg:
        # if lot_size > 10.0: return "ICEBERG"
        
        return "MARKET" # Default

class GameTheoreticArbiter:
    """
    System 22: Game Theoretic Arbiter.
    Models other participants to find Nash Equilibrium.
    """
    def check_market_saturation(self, sentiment_score: float, order_flow_imbalance: float) -> str:
        """
        Detects Crowded Trades.
        """
        # If Sentiment is Extreme Bullish (>0.8) AND OrderFlow is balanced
        # It means everyone who wants to buy has bought. = SATURATION
        
        if sentiment_score > 0.8 and abs(order_flow_imbalance) < 0.1:
            return "CROWDED_LONG"
            
        if sentiment_score < -0.8 and abs(order_flow_imbalance) < 0.1:
            return "CROWDED_SHORT"
            
        return "NEUTRAL"
        
    def suggest_nash_exit(self, saturation_state: str) -> bool:
        """
        If market is crowded, Nash Equilibrium suggests exciting BEFORE the crowd.
        Returns: True (Exit Now)
        """
        if "CROWDED" in saturation_state:
            # First Mover Advantage: Exit before the cascade.
            return True
        return False
