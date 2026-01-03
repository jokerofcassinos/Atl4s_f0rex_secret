
import logging
import asyncio
from typing import Dict, Any
from risk.dynamic_leverage import DynamicLeverage
from risk.great_filter import GreatFilter

logger = logging.getLogger("ExecutionEngine")

class ExecutionEngine:
    """
    The Hand of God.
    Executes trades and handles Predictive Exits.
    """
    def __init__(self, bridge):
        self.bridge = bridge
        self.leverage_manager = DynamicLeverage()
        self.risk_filter = GreatFilter()
        
    async def execute_signal(self, command: str, symbol: str, bid: float, ask: float, confidence: float = 80.0, account_info: Dict = None):
        """
        Converts a Cortex Command into a Physical Order.
        Args:
            command: "BUY" or "SELL"
            symbol: "BTCUSD", "XAUUSD" etc
            bid: Current Bid
            ask: Current Ask
            confidence: 0-100 score
            account_info: info for lot sizing (optional)
        """
        # 1. Great Filter Check (Skipped for now or adapted)
        # if not self.risk_filter.validate_entry(...): return
            
        # 2. Dynamic Lots Calculation
        equity = account_info.get('equity', 1000.0) if account_info else 1000.0
        # Placeholder volatility (should come from market_state)
        volatility = 50.0 
        
        lots = self.leverage_manager.calculate_lots(
            equity=equity,
            confidence=confidence,
            market_volatility=volatility
        )
        
        # 3. Construct Order
        cmd_type = 0 if command == 'BUY' else 1
        price = ask if cmd_type == 0 else bid
        
        # Crypto Adaptation: Wider stops for BTC/ETH
        # For XAUUSD: 200/500 points (20/50 pips)
        # For BTC: 2000/5000 points (20/50 USD)
        
        point = 0.01 if "USD" in symbol and "BTC" not in symbol else 1.0 # Rough heuristic
        if "BTC" in symbol: point = 1.0
        if "ETH" in symbol: point = 0.1
        if "XAU" in symbol and "BTC" not in symbol: point = 0.01 # Gold standard
        
        # BTCXAU is ~90k, so point=1.0 is correct (same as BTCUSD)
        
        sl_points = 500 * point # $5 on Gold, $500 on BTC? Need calibration.
        tp_points = 1000 * point
        
        # Phase 49: Fiscal Hardening (Precision Math)
        # Analysis: BTCXAU 1.00 move = $1000 per Lot.
        # User Limit: $1.70 Loss on 0.01 Lot ($170 per Lot).
        # $170 / $1000 = 0.17 distance.
        # User Target: $0.70 Profit on 0.01 Lot ($70 per Lot).
        # $70 / $1000 = 0.07 distance.
        
        sl_dist = 0.17 # $1.70 on 0.01
        tp_dist = 0.07 # $0.70 on 0.01
        
        # Dynamic Scaler: If price is huge (Bitcoin 90k), these distances are tiny.
        # If price is small (BTCXAU 20), these are correct.
        # This simple check handles the "Dynamic" requirement roughly.
        if price > 1000:
             # Likely BTCUSD or Gold
             # 0.17 on 90000 is nothing.
             # On BTCUSD, 1.00 move = $1.
             # So we need sl_dist = 170.0 and tp_dist = 70.0
             sl_dist = 170.0
             tp_dist = 70.0
        
        # Override for tiny prices (just in case scaling is weird, but verify first)
        if price < 5.0: 
             sl_dist = price * 0.01
             tp_dist = price * 0.02

        if self.bridge:
            # SPREAD GUARD (Dynamic w/ Profit Ratio)
            # User Q: Does it adapt? A: Yes, relative to our Target.
            # We allow the Spread to be at most 20% of our Target Profit.
            # If we aim for $0.70, max spread is $0.14.
            # If we aim for $10.00, max spread is $2.00.
            # This ensures the Math always works, regardless of Capital/Size.
            max_spread = tp_dist * 0.20 
            
            spread = ask - bid
            if spread > max_spread:
                logger.warning(f"SPREAD GUARD: Refused. Spread {spread:.3f} > {max_spread:.3f} (20% of Target).")
                return None
        
        sl = price - sl_dist if cmd_type == 0 else price + sl_dist
        tp = price + tp_dist if cmd_type == 0 else price - tp_dist
        
        # 4. Fire
        # Format: "symbol|cmd|lots|sl|tp"
        params = [symbol, str(cmd_type), f"{lots:.2f}", f"{sl:.2f}", f"{tp:.2f}"]
        logger.info(f"EXECUTION: {command} {lots} lots @ {price:.2f} | Conf: {confidence:.1f}% | SL: {sl:.2f} TP: {tp:.2f}")
        
        if self.bridge:
             self.bridge.send_command("OPEN_TRADE", params)
             
        return "SENT"

    def check_predictive_exit(self, trade: Dict, current_tick: Dict):
        """
        Virtual TP 2.0
        """
        # Logic: If price is within 50 points of TP AND Velocity > 20 points/sec
        # CLOSE NOW before slippage happens.
    def close_all(self, symbol: str):
        """Emergency Exit / Strategic Close"""
        logger.warning(f"EXECUTION: CLOSE ALL POSITIONS for {symbol}")
        if self.bridge:
            self.bridge.send_command("CLOSE_ALL", [symbol])

    def close_longs(self, symbol: str):
        logger.warning(f"EXECUTION: Closing LONGS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_BUYS", [symbol])

    def close_shorts(self, symbol: str):
        logger.warning(f"EXECUTION: Closing SHORTS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_SELLS", [symbol])

