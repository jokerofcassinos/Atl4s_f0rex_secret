
import logging
from typing import Dict, Any

logger = logging.getLogger("GreatFilter")

class GreatFilter:
    """
    The Guardian Gate.
    Calculates Ruin Probability and executes Micro Anti-Loss.
    """
    def __init__(self, account_balance: float = 5.0):
        self.account_balance = account_balance
        self.max_daily_loss = 0.50 # For $5 account, strict!
        
        # Ruin Probability Matrix
        self.ruin_threshold = 0.05 # 5% Risk of Ruin allowed per trade (Aggressive)

    def validate_entry(self, signal: Dict, market_state: Dict) -> bool:
        """
        Final Check before Opening.
        Args:
            signal: {'type': 'BUY', 'confidence': 85.0}
            market_state: {'volatility': 50, 'is_crash': False}
        """
        # 1. Reject Signals during Crash (unless Signal is SELL)
        if market_state.get('is_crash', False):
            if signal['type'] == 'BUY':
                logger.warning("GREAT FILTER: Blocking Buy during Crash Phase.")
                return False

        # 2. Reject Low Confidence Scalps
        if signal['confidence'] < 75.0:
            logger.info(f"GREAT FILTER: Confidence {signal['confidence']} too low.")
            return False

        # 3. Spread Check (If High Spread, Scalp is Ruined)
        spread = market_state.get('spread', 0)
        # Assuming Gold points (Digits=2)
        if spread > 50: # 50 points = 5 pips (High for Scalp)
            logger.warning(f"GREAT FILTER: Spread {spread} too wide for Scalp.")
            return False
            
        return True

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
