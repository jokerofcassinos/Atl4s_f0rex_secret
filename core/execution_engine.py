
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
    def __init__(self, bridge=None):
        self.bridge = bridge
        self.risk_filter = None # Add logic
        self.leverage_manager = DynamicLeverage()
        self.config = {"spread_limit": 0.0005} # Default

    def set_config(self, config: Dict):
        self.config = config
        
    async def execute_signal(self, command: str, symbol: str, bid: float, ask: float, confidence: float = 80.0, account_info: Dict = None, spread_tolerance: float = None):
        """
        Converts a Cortex Command into a Physical Order.
        Args:
            command: "BUY" or "SELL", "XAUUSD" etc
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
        
        if price < 50: # LOW PRICE ASSETS (Silver? Cheap Crypto?)
            # Keep hardcoded for now or adapt if needed
            tp_dist = 0.05 # 5 cents (~$0.50 profit)
            sl_dist = 0.12 # 12 cents (~$1.20 risk)
        else: 
             # DYNAMIC LIMITS (Forex / Gold / High Crypto)
             # Default to 0.1%/0.2% if not in config
             tp_pct = self.config.get('phys_tp_pct', 0.001)
             sl_pct = self.config.get('phys_sl_pct', 0.002)
             
             tp_dist = price * tp_pct
             sl_dist = price * sl_pct
             
        # Override for tiny prices 
        if price < 5.0: 
             sl_dist = price * 0.01
             tp_dist = price * 0.02

        if self.bridge:
            # 3. Dynamic Spread Guard
            # Adaptive Threshold: MAX_SPREAD = Config % of Price
            spread_limit = self.config.get('spread_limit', 0.0005)
            max_spread_allowed = price * spread_limit
            
            # Hard cap minimum for Forex
            if max_spread_allowed < 0.00050: max_spread_allowed = 0.00050
            
            # Override for User Settings if needed
            # But this adaptive check is safer.
            
            spread = ask - bid
            if spread > max_spread_allowed:
                 logger.warning(f"SPREAD GUARD: Refused. Spread {spread:.3f} > {max_spread_allowed:.3f} ({spread_limit*100:.2f}% of Price).")
                 return None # Changed from False to None to match original function's return type on refusal
        
        sl = price - sl_dist if cmd_type == 0 else price + sl_dist
        tp = price + tp_dist if cmd_type == 0 else price - tp_dist
        
        # 4. Fire
        # Format: "symbol|cmd|lots|sl|tp"
        
        # Phase 106: Dynamic Precision
        params_sl = f"{sl:.2f}"
        params_tp = f"{tp:.2f}"
        
        if price < 50:
             params_sl = f"{sl:.5f}"
             params_tp = f"{tp:.5f}"
        
        params = [symbol, str(cmd_type), f"{lots:.2f}", params_sl, params_tp]
        logger.info(f"EXECUTION: {command} {lots} lots @ {price:.5f} | Conf: {confidence:.1f}% | SL: {params_sl} TP: {params_tp}")
        
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

    def prune_losers(self, symbol: str):
        """
        Surgically removes only losing positions for a symbol.
        Preserves winning hedges.
        """
        logger.warning(f"EXECUTION: PRUNING LOSERS for {symbol}")
        if self.bridge:
            self.bridge.send_command("PRUNE_LOSERS", [symbol])

    def harvest_winners(self, symbol: str):
        """
        Surgically removes only winning positions (taking profit).
        """
        logger.warning(f"EXECUTION: HARVESTING WINNERS for {symbol}")
        if self.bridge:
            self.bridge.send_command("HARVEST_WINNERS", [symbol])

    def close_longs(self, symbol: str):
        logger.warning(f"EXECUTION: Closing LONGS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_BUYS", [symbol])

    def close_shorts(self, symbol: str):
        logger.warning(f"EXECUTION: Closing SHORTS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_SELLS", [symbol])

