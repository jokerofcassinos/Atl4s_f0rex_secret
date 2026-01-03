
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
        
    async def execute_signal(self, command: str, symbol: str, bid: float, ask: float, confidence: float = 80.0, account_info: Dict = None, spread_tolerance: float = None):
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
        
        if price < 50: # BTCXAU
            tp_dist = 0.05 # 5 cents (~$0.50 profit)
            sl_dist = 0.12 # 12 cents (~$1.20 risk)
        else: # BTCUSD (Price ~90,000)
             tp_dist = price * 0.001 # 0.1% (~$90)
             sl_dist = price * 0.002 # 0.2% (~$180)
             
        # Override for tiny prices 
        if price < 5.0: 
             sl_dist = price * 0.01
             tp_dist = price * 0.02

        if self.bridge:
            # SPREAD GUARD (Dynamic w/ Profit Ratio)
            # User Q: Does it adapt? A: Yes, relative to our Target.
            # We allow the Spread to be at most 50% of our Target Profit.
            # If spread_tolerance is provided (Wolf Pack Mode), we use that instead.
            max_spread = spread_tolerance if spread_tolerance else tp_dist * 0.50 
            
            spread = ask - bid
            if spread > max_spread:
                logger.warning(f"SPREAD GUARD: Refused. Spread {spread:.3f} > {max_spread:.3f} (50% of Target).")
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
        Virtual TP 2.0 (The Magnet)
        Checks if we should close BEFORE the hard TP to secure bag.
        """
        symbol = trade.get('symbol')
        ticket = trade.get('ticket')
        profit = trade.get('profit', 0.0)
        tp = trade.get('tp', 0.0)
        open_price = trade.get('open_price', 0.0)
        current_price = current_tick.get('bid') if trade.get('type') == 0 else current_tick.get('ask') # approx
        
        if tp == 0 or open_price == 0: return # No TP set
        
        # 1. Calculation of Progress
        total_dist = abs(tp - open_price)
        current_dist = abs(current_price - open_price)
        
        if total_dist == 0: return

        progress = current_dist / total_dist
        
        # 2. Virtual Hit Logic
        # If we are 80% of the way there, and profit is decent, CLOSE.
        # User Feedback: "Impossible to reach". So we make it easier.
        
        if progress > 0.80:
             logger.info(f"VIRTUAL TP: {symbol} at 80% progress. Securing ${profit:.2f}.")
             self.close_trade(ticket, symbol)
             return True
             
        # 3. Time Decay / Stalling (Simple check)
        # If profit is > $5.00 and we are stalling (would require history, skipping for now)
        
        return False

    def close_trade(self, ticket: int, symbol: str):
        logger.info(f"EXECUTION: Closing Trade {ticket} ({symbol})")
        if self.bridge:
            # FIX: Must include symbol for ZmqBridge routing
            self.bridge.send_command("CLOSE_TRADE", [str(ticket), symbol])
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

