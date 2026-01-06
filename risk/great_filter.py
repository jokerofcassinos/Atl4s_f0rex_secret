
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

    def validate_entry(self, signal: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final Check before Opening.
        Args:
            signal: {'type': 'BUY', 'confidence': 85.0}
            market_state: {'volatility': 50, 'is_crash': False, 'spread': float}

        Returns:
            dict com:
                'allowed': bool
                'decision_id': str
                'reason': str
        """
        reasons = []
        allowed = True

        # 1. Reject Signals during Crash (unless Signal is SELL)
        if market_state.get("is_crash", False) and signal.get("type") == "BUY":
            reasons.append("Blocking BUY during crash phase")
            allowed = False

        # 2. Reject Low Confidence Scalps
        confidence = float(signal.get("confidence", 0.0))
        if confidence < 75.0:
            reasons.append(f"Confidence {confidence:.1f} too low")
            allowed = False

        # 3. Spread Check (If High Spread, Scalp is Ruined)
        spread = market_state.get("spread", 0.0)
        if spread > 50:  # 50 points = 5 pips (High for Scalp)
            reasons.append(f"Spread {spread} too wide for scalp")
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
