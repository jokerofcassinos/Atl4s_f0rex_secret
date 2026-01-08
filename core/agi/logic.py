
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("Logic")

class SymbolicReasoningModule:
    """
    System 3: Symbolic Reasoning & Logic Axioms.
    Hybrid Neuro-Symbolic Logic.
    Enforces "Hard Rules" that Neural Nets might ignore.
    """
    def __init__(self):
        self.axioms = [
            self._axiom_spread_rationality,
            self._axiom_trend_consistency
        ]
        
    def validate_logic(self, decision: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Runs the decision through the Axiom Engine.
        Returns (True, "Valid") or (False, "Violation Reason").
        """
        if decision == "WAIT": return True, "Logic holds (Inaction is safe)"
        
        for axiom in self.axioms:
            is_valid, reason = axiom(decision, context)
            if not is_valid:
                return False, reason
                
        return True, "Logically Consistent"
        
    def _axiom_spread_rationality(self, decision: str, context: Dict) -> Tuple[bool, str]:
        """
        Axiom: Transaction Cost (Spread) must not exceed 20% of Expected Move (volatility).
        """
        spread = context.get('spread', 0)
        volatility = context.get('volatility', 1.0) # ATR
        
        if volatility == 0: return True, "No Volatility data"
        
        if spread > (volatility * 0.5): # 50% of ATR is huge spread
             return False, f"Logical Violation: Spread ({spread:.2f}) > 50% of Volatility ({volatility:.2f})"
             
        return True, "Spread Rational"

    def _axiom_trend_consistency(self, decision: str, context: Dict) -> Tuple[bool, str]:
        """
        Axiom: Do not trade against a Super-Trend unless in 'Reversal' mode.
        """
        trend_score = context.get('trend_score', 0) # -1 to 1
        
        if decision == "BUY" and trend_score < -0.8:
            return False, "Logical Violation: Buying into Strong Downtrend (-0.8)"
        
        if decision == "SELL" and trend_score > 0.8:
            return False, "Logical Violation: Selling into Strong Uptrend (+0.8)"
            
        return True, "Trend Consistent"
