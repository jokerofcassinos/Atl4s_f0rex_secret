
import logging
import asyncio
from typing import Dict, Any, Optional

from risk.dynamic_leverage import DynamicLeverage
from risk.great_filter import GreatFilter
from core.risk.event_horizon import EventHorizonRisk

logger = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """
    The Hand of God.
    Executes trades and handles Predictive Exits.
    """
    def __init__(self, bridge=None):
        self.bridge = bridge
        # Phase 10: Risk Core AGI-aware filter
        self.risk_filter = GreatFilter()
        self.leverage_manager = DynamicLeverage()
        self.event_horizon = EventHorizonRisk()  # Phase 116
        self.config = {"spread_limit": 0.0005}
        self.last_stop_update = {}  # {ticket: timestamp}

    def set_config(self, config: Dict):
        self.config = config
        
    async def manage_dynamic_stops(self, tick: Dict):
        """
        Phase 116: Event Horizon Loop.
        Iterates through open trades and applies Parabolic Trailing Stops.
        """
        if not self.bridge: return
        
        trades = tick.get('trades_json', [])
        # If string, parse it? ZmqBridge usually handles it.
        # Assuming list of dicts: [{'ticket':1, 'symbol':'XAUUSD', 'type':0, 'open':2000, 'sl':1990, 'profit':5.0}]
        
        if not trades: return
        
        current_price = tick.get('bid') # Default to Bid
        if not current_price: return
        
        symbol_spread = tick.get('ask', 0) - tick.get('bid', 0)
        
        for trade in trades:
            symbol = trade.get('symbol')
            if symbol != tick.get('symbol'): continue # Only manage current symbol tick matches
            
            ticket = trade.get('ticket')
            type_int = trade.get('type') # 0=Buy, 1=Sell
            open_price = trade.get('open_price')
            current_sl = trade.get('sl')
            
            side = "BUY" if type_int == 0 else "SELL"
            trade_price = tick.get('bid') if side == "BUY" else tick.get('ask')
            
            # Event Horizon Calculation
            new_stop_level = self.event_horizon.calculate_dynamic_stop(
                symbol, open_price, trade_price, side, symbol_spread
            )
            
            if new_stop_level:
                # Ratchet Logic: Only move closer to price
                update_needed = False
                
                if side == "BUY":
                    # For BUY, SL must move UP (Higher than current SL)
                    if new_stop_level > current_sl and new_stop_level < trade_price:
                        update_needed = True
                else:
                    # For SELL, SL must move DOWN (Lower than current SL)
                    # (Remember SL is above price for Sell)
                    if (current_sl == 0 or new_stop_level < current_sl) and new_stop_level > trade_price:
                        update_needed = True
                
                # Check for Min Distance (Don't spam updates for 0.01 change)
                if update_needed:
                     diff = abs(new_stop_level - current_sl)
                     if diff > symbol_spread * 0.2: # Material change
                         logger.info(f"EVENT HORIZON: Updating SL for Ticket {ticket} -> {new_stop_level:.2f}")
                         # Command: MODIFY_TRADE|ticket|sl|tp
                         # Keep existing TP
                         tp = trade.get('tp', 0.0)
                         self.bridge.send_command("MODIFY_TRADE", [str(ticket), f"{new_stop_level:.2f}", f"{tp:.2f}"])
                         self.last_stop_update[ticket] = tick.get('time_msc', 0)

    async def execute_signal(
        self,
        command: str,
        symbol: str,
        bid: float,
        ask: float,
        confidence: float = 80.0,
        account_info: Optional[Dict] = None,
        spread_tolerance: Optional[float] = None,
    ):
        """
        Converts a Cortex Command into a Physical Order.
        Args:
            command: "BUY" or "SELL", "XAUUSD" etc
            bid: Current Bid
            ask: Current Ask
            confidence: 0-100 score
            account_info: info for lot sizing (optional)
        """
        # 1. Great Filter Check (Phase 10 - AGI Risk Core)
        if self.risk_filter:
            risk_signal = {"type": command, "confidence": confidence}
            # spread in raw price units; for Gold it's "points"
            market_state = {
                "spread": (ask - bid),
                "is_crash": False,  # placeholder, could be wired to macro modules
            }
            verdict = self.risk_filter.validate_entry(risk_signal, market_state)
            if not verdict.get("allowed", False):
                logger.info("EXECUTION BLOCKED by GreatFilter: %s", verdict.get("reason"))
                return None

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
            spread_limit = spread_tolerance if spread_tolerance is not None else self.config.get('spread_limit', 0.0005)
            max_spread_allowed = price * spread_limit

            # Hard cap minimum for Forex
            if max_spread_allowed < 0.00050:
                max_spread_allowed = 0.00050

            spread = ask - bid
            if spread > max_spread_allowed:
                logger.warning(
                    "SPREAD GUARD: Refused. Spread %.3f > %.3f (%.2f%% of Price).",
                    spread,
                    max_spread_allowed,
                    spread_limit * 100.0,
                )
                return None  # Changed from False to None to match original function's return type on refusal
        
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

    def check_individual_guards(self, trades: list, v_tp: float, v_sl: float):
        """
        Phase 122: Granular Guard.
        Checks each trade individually against Virtual TP and Virtual SL.
        """
        if not trades: return

        for trade in trades:
            ticket = trade.get('ticket')
            profit = trade.get('profit', 0.0)
            symbol = trade.get('symbol')
            
            # 1. Individual Take Profit (The Snipe)
            if profit >= v_tp:
                logger.info(f"INDIVIDUAL GUARD: Harvesting Ticket {ticket} ({symbol}) | Profit ${profit:.2f} >= ${v_tp:.2f}")
                self.close_trade(ticket, symbol)
                continue # One action per trade

            # 2. Individual Stop Loss (The Shield)
            # v_sl is usually positive in config (e.g. $40), so we check against -40
            limit = -abs(v_sl) 
            if profit <= limit:
                logger.info(f"INDIVIDUAL GUARD: Pruning Ticket {ticket} ({symbol}) | Profit ${profit:.2f} <= ${limit:.2f}")
                self.close_trade(ticket, symbol)
                continue

    def close_longs(self, symbol: str):
        logger.warning(f"EXECUTION: Closing LONGS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_BUYS", [symbol])

    def close_shorts(self, symbol: str):
        logger.warning(f"EXECUTION: Closing SHORTS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_SELLS", [symbol])

