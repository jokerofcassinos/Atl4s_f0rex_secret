
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
            
        return {
            'allowed': allowed,
            'decision_id': "GF-" + str(int(market_state.get('tick_time', 0))),
            'reason': "; ".join(reasons)
        }

        # 3. Spread Check (Block if spread is excessive)
        # For Forex: spread is in price units (0.0003 = 3 pips for GBPUSD)
        # For Gold: spread is in price units (0.50 = 50 cents)
        spread = market_state.get("spread", 0.0)
        
        # Use relative threshold: block if spread > 0.05% of price (covers both Forex and Gold)
        # For GBPUSD ~1.34: 0.05% = 0.00067 (~6.7 pips) 
        # For USDJPY ~156: 0.05% = 0.078 (~7.8 pips)
        # For Gold ~2600: 0.05% = 1.30 ($1.30)
        max_spread_ratio = 0.0005  # 0.05%
        typical_price = market_state.get("typical_price", 1.0)  # Fallback
        max_spread = typical_price * max_spread_ratio if typical_price > 10 else max_spread_ratio
        
        if spread > max_spread and spread > 0.0008:  # Also hard floor of 8 pips
            reasons.append(f"Spread {spread:.5f} too wide (max: {max_spread:.5f})")
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
