
import logging
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger("FuzzyLogic")

class FuzzyLogicEngine:
    """
    Phase 119: Fuzzy Logic Engine.
    Handles ambiguous states like "Roughly Overbought" or "Slightly Bearish".
    """
    def __init__(self):
        # Membership functions (Trapezoidal)
        pass
        
    def fuzzify(self, value: float, concept: str) -> Dict[str, float]:
        """
        Converts crisp value to fuzzy membership.
        Concept: RSI
        Returns: {'LOW': 0.0, 'MID': 0.2, 'HIGH': 0.8}
        """
        memberships = {}
        if concept == "RSI":
            memberships['OVERSOLD'] = self._trapezoid(value, -10, 0, 30, 40)
            memberships['NEUTRAL'] = self._trapezoid(value, 30, 40, 60, 70)
            memberships['OVERBOUGHT'] = self._trapezoid(value, 60, 70, 100, 110)
        return memberships
        
    def _trapezoid(self, x, a, b, c, d):
        return max(0, min((x - a)/(b - a + 1e-9), 1, (d - x)/(d - c + 1e-9)))

    def infer_rules(self, fuzzy_inputs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Infers output based on fuzzy rules.
        """
        # Rule 1: If RSI is OVERBOUGHT Then Bias is BEARISH
        rsi = fuzzy_inputs.get('RSI', {})
        bearish_score = rsi.get('OVERBOUGHT', 0.0)
        bullish_score = rsi.get('OVERSOLD', 0.0)
        
        return {'BEARISH': bearish_score, 'BULLISH': bullish_score}
