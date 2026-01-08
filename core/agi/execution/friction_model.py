
import logging
from typing import Dict, Any

logger = logging.getLogger("FrictionEstimator")

class FrictionEstimator:
    """
    Sistema D-7: Friction Estimator
    Calcula o 'Custo de Fricção' (Spread + Slippage + Swap).
    Verifica se a volatilidade da sessão paga a conta.
    """
    def __init__(self):
        self.avg_spread = 0.0
        self.spread_history = []
        
    def calculate_friction(self, market_state: Dict[str, Any]) -> float:
        """
        Retorna o custo estimado em Pip Value (normalizado).
        """
        current_spread = market_state.get('spread', 0.00010) # Default 1 pip
        
        # 1. Spread update
        self.spread_history.append(current_spread)
        if len(self.spread_history) > 50: self.spread_history.pop(0)
        self.avg_spread = sum(self.spread_history) / len(self.spread_history)
        
        # 2. Slippage Model
        # High volatility = High slippage risk
        volatility = market_state.get('metrics', {}).get('volatility', 0.0)
        est_slippage = current_spread * 0.5 if volatility > 0.002 else 0.0
        
        total_friction = current_spread + est_slippage
        return total_friction
        
    def is_profitable_in_session(self, market_state: Dict[str, Any], target_return: float) -> bool:
        """
        Verifica se o lucro alvo é viável dada a fricção atual.
        Regra: Reward > Friction * 2
        """
        friction = self.calculate_friction(market_state)
        
        # Se friction is excessively high relative to target, abort.
        if friction * 2 > target_return:
            logger.warning(f"PROFITABILITY CHECK FAIL: Friction {friction:.5f} too high for Target {target_return:.5f}")
            return False
            
        return True
