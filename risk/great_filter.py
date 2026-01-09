
import logging
from typing import Dict, Any, Optional

from core.agi.decision_memory import ModuleDecisionMemory

logger = logging.getLogger("GreatFilter")


class GreatFilter:
    """
    The Guardian Gate.
    Phase 10: Risk AGI - Cada entrada passa por memória de decisões.
    """

    def __init__(self, account_balance: float = 5.0):
        self.account_balance = account_balance
        self.max_daily_loss = 0.50  # For $5 account, strict!

        # Ruin Probability Matrix
        self.ruin_threshold = 0.05  # 5% Risk of Ruin allowed per trade (Aggressive)

        # Memória de decisões do filtro (AGI Risk Core)
        self.decision_memory = ModuleDecisionMemory("GreatFilter", max_memory=2000)
        
        # Phase 12: Friction & Profitability
        from core.agi.execution.friction_model import FrictionEstimator
        self.friction_estimator = FrictionEstimator()

    def validate_entry(self, signal: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final Check before Opening.
        Now includes Phase 12 Profitability Gate.
        """
        reasons = []
        allowed = True

        # 1. Reject Signals during Crash (unless Signal is SELL)
        if market_state.get("is_crash", False) and signal.get("type") == "BUY":
            reasons.append("Blocking BUY during crash phase")
            allowed = False

        # 2. Reject Low Confidence Scalps (lowered from 75 to 50 for HYDRA mode)
        confidence = float(signal.get("confidence", 0.0))
        if confidence < 50.0:
            reasons.append(f"Confidence {confidence:.1f} too low")
            
        # 3. Phase 12: Predictive Profitability Gate
        # "Can we afford the trip?"
        target_profit = market_state.get('metrics', {}).get('atr_value', 0.0005) * 1.5 # Estimating potential reward
        friction_cost = self.friction_estimator.calculate_friction(market_state)
        
        # Expectancy Logic: Reward must cover Friction heavily
        if not self.friction_estimator.is_profitable_in_session(market_state, target_profit):
            reasons.append(f"UNPROFITABLE: Friction ({friction_cost:.5f}) eats Reward ({target_profit:.5f})")
            allowed = False
            
        # 3. Spread Check (Block if spread is excessive)
        spread = market_state.get("spread", 0.0)
        atr_value = market_state.get('metrics', {}).get('atr_value', 0.0)
        candle_size = market_state.get('candle_size', 0.0) # Need to ensure this is passed
        
        # A. ATR Based Guard (Best)
        if atr_value > 0:
             max_spread = atr_value * 0.35 # Max 35% of ATR (Stricter)
             if spread > max_spread:
                  reasons.append(f"Spread {spread:.5f} > 35% ATR ({max_spread:.5f}) - Impossible Scalp")
                  allowed = False
        
        # B. Candle Context Guard (User Innovation)
        # "Observe where the candle is... spread is 3x above"
        # If Spread > Candle Body * 0.8 -> We are paying more in spread than the candle moved!
        if candle_size > 0:
             if spread > candle_size * 0.9: # Almost equal to the entire candle movement
                  reasons.append(f"Spread {spread:.5f} consumes Candle ({candle_size:.5f}) - Ghost Spread")
                  allowed = False
                  
        # C. Hard Cap (Fallback)
        typical_price = market_state.get("typical_price", 1.0)
        max_spread_ratio = 0.0005  # 0.05%
        max_spread_hard = typical_price * max_spread_ratio
        if max_spread_hard < 0.00020: max_spread_hard = 0.00020
        
        if spread > max_spread_hard and spread > 0.0008:
            reasons.append(f"Spread {spread:.5f} too wide (Hard Limit)")
            allowed = False

        if not reasons:
            reasons.append("Risk checks passed")

        reason_str = " | ".join(reasons)
        logger.info("GREAT FILTER: %s (allowed=%s)", reason_str, allowed)

        decision_label = "ALLOW" if allowed else "BLOCK"
        decision_score = confidence if allowed else -confidence

        decision_id = self.decision_memory.record_decision(
            decision=decision_label,
            score=decision_score,
            context={
                "signal": signal,
                "market_state": market_state,
            },
            reasoning=reason_str,
            confidence=abs(decision_score) / 100.0 if decision_score != 0 else 0.0,
        )

        return {
            "allowed": allowed,
            "decision_id": decision_id,
            "reason": reason_str,
        }

    def update_entry_result(self, decision_id: str, result: str, pnl: Optional[float] = None) -> None:
        """
        Fase 10: Integra resultado real de uma entrada filtrada.
        """
        self.decision_memory.record_result(decision_id=decision_id, result=result, pnl=pnl)

    def check_micro_anti_loss(self, trade_ticket: int, current_pnl: float, duration_ms: float) -> bool:
        """
        The 'Instant Regret' Button.
        If trade is strictly negative after 500ms and momentum is against us -> KILL.
        """
        # For HFT, if we are not winning in 3 seconds, we failed.
        if duration_ms > 3000 and current_pnl < -0.10: # -10 cents
            logger.warning(f"MICRO ANTI-LOSS: Killing Ticket {trade_ticket} PnL: {current_pnl}")
            return True # Signal to Close
            
        return False
