
import logging
import asyncio
import time
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
        self.close_attempts = {}    # {ticket: timestamp_msc} - Retry mechanism
        self.trade_start_times = {} # {ticket: local_timestamp_sec} - For accurate age tracking
        self.closed_tickets_cache = set() # Phase 98: Anti-Ghosting Cache


    def set_config(self, config: Dict):
        self.config = config

    def _get_current_time_msc(self):
        return int(time.time_ns() / 1_000_000)
        
        
    async def monitor_positions(self, tick: Dict):
        """
        Orchestrates all position monitoring checks (VTP, VSL, Predictive).
        """
        if not self.bridge: return
        
        trades = tick.get('trades_json', [])
        if not trades: return
        
        # 1. Config Thresholds
        v_tp = self.config.get('virtual_tp', 2.0)
        v_sl = self.config.get('virtual_sl', 10.0)
        
        # 2. Check Individual Guards (VTP / VSL)
        self.check_individual_guards(trades, v_tp, v_sl)
        
        # 3. Check Predictive Exits (The Magnet)
        for trade in trades:
             if not isinstance(trade, dict): continue
             self.check_predictive_exit(trade, tick)
        
        
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
            if not isinstance(trade, dict): continue
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
        multiplier: float = 1.0,
        atr_value: Optional[float] = None # Phase 3: AGI ATR Override
    ):
        """
        Converts a Cortex Command into a Physical Order.
        Args:
            command: "BUY" or "SELL", "XAUUSD" etc
            confidence: 0-100 score
            atr_value: (Optional) Real ATR from AGI Profiler for dynamic stops.
        """
        # 1. Great Filter Check (Phase 10 - AGI Risk Core)
        if self.risk_filter:
            risk_signal = {"type": command, "confidence": confidence}
            # spread in raw price units; for Gold it's "points"
            market_state = {
                "spread": (ask - bid),
                "typical_price": (ask + bid) / 2,  # For relative spread calculation
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
        
        # Apply Logic Multiplier (Sovereign/Singularity)
        lots = round(lots * multiplier, 2)
        
        # 3. Dynamic Geometry (Phase 14)
        # Replacing 0.0/0.0 with intelligent stops
        from core.agi.risk.dynamic_geometry import DynamicGeometryEngine
        # Usually we'd instantiate this in __init__, but for hot-patching ease we do it here or assume singleton
        # Ideally, move to __init__. I will assume self.geometry_engine exists or create it.
        # For safety in this patch, I'll instantiate locally. Cost is negligible.
        geometry_engine = DynamicGeometryEngine()
        
        entry_price = ask if command == 'BUY' else bid
        # Mock market state for geometry (needs spread and ATR)
        geo_market_state = {
            'metrics': {'atr_value': atr_value if atr_value else 0.0005, 'volatility': 0.002}
        }
        
        vsl, vtp = geometry_engine.calculate_geometry(geo_market_state, command, entry_price)
        
        # 4. Construct Order
        cmd_type = 0 if command == 'BUY' else 1
        price = ask if cmd_type == 0 else bid
        
        # Create params for bridge command
        # Format: symbol, cmd_type, volume, sl, tp, confidence
        # EA expects: parts[1]=symbol, parts[2]=cmd, parts[3]=vol, parts[4]=sl, parts[5]=tp, parts[6]=conf
        
        # FIXED: Disable MQL5/Broker SL (Send 0.0) to prevent premature Price Closures.
        # We rely 100% on Python's Dollar Guard (check_individual_guards) which is now Volume-Scaled.
        # We KEEP VTP to allow Predictive Exits (The Magnet) to function.
        params = [symbol, cmd_type, lots, "0.00000", f"{vtp:.5f}", confidence]

        if self.bridge:
            logger.info(f"TRANSMITTING ORDER: {command} {symbol} @ {price:.5f} | Lots: {lots} | Conf: {confidence:.1f}%")
            logger.info(f"DYNAMIC GEOMETRY (INTERNAL): VSL={vsl:.5f} (DISABLED in MQL5), VTP={vtp:.5f}")
            self.bridge.send_command("OPEN_TRADE", params)
            logger.info(f"ORDER SENT TO MT5: {params}")
        else:
            logger.warning("NO BRIDGE - Order not sent!")
             
        return "SENT"

    async def execute_hydra_burst(
        self,
        command: str,
        symbol: str,
        bid: float,
        ask: float,
        confidence: float,
        account_info: Optional[Dict] = None,
        volatility: float = 50.0,
        entropy: float = 0.5,
        infinite_depth: int = 0
    ):
        """
        PHASE 11: HYDRA PROTOCOL (AGI Multi-Vector Execution).
        Dynamically multiplies execution vectors based on conviction (Confidence + Depth).
        """
        logger.info(f"HYDRA PROTOCOL: Initiating Burst Sequence for {symbol} ({command})")
        
        # 1. Determine Hydra Heads (Number of Orders)
        # Base: 1 order.
        # Conviction Bonus: +1 per 10% above 70% confidence.
        # Depth Bonus: +1 per 1000 InfiniteWhy branches (simulated via depth arg).
        
        heads = 1
        if confidence > 70: heads += 1
        if confidence > 80: heads += 1
        if confidence > 90: heads += 1
        if confidence > 95: heads += 1 # Max 5 from confidence
        
        # AGI Depth Bonus (The "Ontological Nuance" Factor)
        if infinite_depth >= 10: heads += 1
        if infinite_depth >= 50: heads += 1 # Rare deep thought
        
        # Entropy Damper (If chaos is too high, reduce heads)
        if entropy > 0.8:
            heads = max(1, heads - 2)
            logger.info(f"HYDRA: High Entropy ({entropy:.2f}) pruned heads to {heads}")
            
        # Hard Cap
        heads = min(heads, 8)
        
        logger.info(f"HYDRA: Spawning {heads} Heads (Conf={confidence}%, Depth={infinite_depth})")
        
        # 2. Execute Burst
        # We fire them sequentially with slight delays to avoid broker flood rejection,
        # but logically they are a single "attack".
        
        for i in range(heads):
            # Dynamic variance per head?
            # Maybe slight limit offsets? For now, pure market swarm.
            
            # Recalculate price each time? No, rapid fire.
            
            # Logic: We invoke execute_signal for each head.
            # We scale the individual lot size down? Or multiply total risk?
            # User said "mutiplica o numero de ordens". Implies scaling UP total volume.
            # We use standard sizing per order. Aggressive!
            
            await self.execute_signal(
                command=command,
                symbol=symbol,
                bid=bid,
                ask=ask,
                confidence=confidence, # Pass original confidence
                account_info=account_info,
                multiplier=1.0 # Each head strikes with full force
            )
            
            # Micro-sleep to prevent sequence errors in MT5
            await asyncio.sleep(0.2)
            
        return f"HYDRA_SENT_{heads}"
        # Extract context
        used_slots = account_info.get('positions', 0) if account_info else 0
        max_slots = account_info.get('max_slots', 10) if account_info else 10
        equity = account_info.get('equity', 1000.0) if account_info else 1000.0
        
        # Calculate Dynamic Exits
        tp_dist, sl_dist = self.calculate_smart_exits(
            price=price,
            equity=equity,
            used_slots=used_slots,
            max_slots=max_slots,
            volatility=volatility,
            atr_value=atr_value
        )
        
        # Override for tiny prices (Safety)
        if price < 5.0: 
             # Forex/Altcoin Safety
             if sl_dist < price * 0.005: sl_dist = price * 0.01
             if tp_dist < price * 0.01: tp_dist = price * 0.02
        
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
        
        # Include confidence as 6th parameter for MQL5 adaptive executor
        params = [symbol, str(cmd_type), f"{lots:.2f}", params_sl, params_tp, f"{confidence:.1f}"]
        logger.info(f"EXECUTION: {command} {lots} lots @ {price:.5f} | Conf: {confidence:.1f}% | SL: {params_sl} TP: {params_tp} (ATR: {atr_value})")
        
        if self.bridge:
             self.bridge.send_command("OPEN_TRADE", params)
             
        return "SENT"

    def calculate_smart_exits(self, price: float, equity: float, used_slots: int, max_slots: int, volatility: float, atr_value: float = None) -> tuple:
        """
        AGI-Driven Physical Exits.
        Adjusts TP/SL based on ATR or 'Capital Saturation'.
        """
        # Base Settings (Gold) - Fallback
        base_tp_dist = 2.0  # $2.00
        base_sl_dist = 1.5  # $1.50
        
        # Phase 3: AGI ATR Override
        if atr_value and atr_value > 0:
             # Logic: Stop Loss = 1.5x ATR (To avoid noise)
             # Logic: Take Profit = 2.0x ATR (2:1 Ratio attempt, or 1.5:1)
             base_sl_dist = atr_value * 1.5
             base_tp_dist = atr_value * 2.5 # Ambition
             
             # If ATR is crazy (News), clamp it
             if base_sl_dist > 20.0: base_sl_dist = 20.0
             
             return base_tp_dist, base_sl_dist
             
        # 1. Saturation Multiplier
        # Saturation 0.0 to 1.0
        saturation = used_slots / max(1, max_slots)
        
        if saturation > 0.8: # Heavy Load (>80% capacity)
            # Defensive Mode: Tighten everything to clear exposure
            sat_mult = 0.7 
        elif saturation < 0.3: # Light Load
            # Expansion Mode: Go for home runs
            sat_mult = 1.5 
        else:
            sat_mult = 1.0
            
        # 2. Volatility Multiplier
        # Volatility usually 0-100
        # High Vol -> Widen SL to breathe, Widen TP to catch spikes
        vol_mult = 1.0 + (volatility - 50.0) / 100.0 # e.g., Vol 80 -> 1.3x
        
        # Combined
        final_tp = base_tp_dist * sat_mult * vol_mult
        final_sl = base_sl_dist * sat_mult * vol_mult # SL also widens in Volatility to avoid whipsaw
        
        # Hard Safety Guard (Don't let SL get too tight or too wide)
        final_sl = max(0.50, min(final_sl, 5.00))
        final_tp = max(0.50, min(final_tp, 10.00))
        
        return final_tp, final_sl

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
             # FIX: Ensure we are actually in profit!
             # The distance logic 'abs(current - open)' triggers on LOSSES too if price moves away.
             if profit <= 0:
                 return False
            
             # Mode-based Threshold
             # SNIPER: 95% (Secure it)
             # HYDRA/WOLF: 98% (Let it run)
             mode = self.config.get('mode', 'SNIPER')
             threshold = 0.98 if mode in ["HYDRA", "WOLF_PACK"] else 0.95
             
             if progress < threshold:
                 return False
                 
             logger.info(f"VIRTUAL TP: {symbol} at {progress:.1%} progress (>{threshold:.1%}). Securing ${profit:.2f}.")
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
            self.closed_tickets_cache.add(ticket) # Mark as dead to ExecutionEngine

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

    def check_individual_guards(self, trades: list, v_tp: float, v_sl: float, market_bias: str = "NEUTRAL"):
        """
        Phase 122: Granular Guard.
        Checks each trade individually against Virtual TP and Virtual SL.
        """
        if not trades: return

        for trade in trades:
            if not isinstance(trade, dict): continue
            
            ticket = trade.get('ticket')
            trade_type = trade.get('type') # 0=Buy, 1=Sell
            
            # Retry Mechanism (Phase 135 Fix):
            # Instead of ignoring forever, we respect a 5-second cooldown.
            current_time = self._get_current_time_msc()
            # Anti-Ghosting Check
            if ticket in self.closed_tickets_cache:
                continue

            # Instead of ignoring forever, we respect a 5-second cooldown via self.close_attempts
            # But the detailed logs usually come from close_trade() which we now gate.
            
            if ticket in self.close_attempts:
                last_attempt = self.close_attempts[ticket]
                if (current_time - last_attempt) < 5000: # 5 seconds
                    continue


                
            try:
                profit = float(trade.get('profit', 0.0))
            except:
                profit = 0.0
                
            symbol = trade.get('symbol')
            
            # SCALING FIX: Adjust Dollar Limits by Volume
            # Standard Config ($10) is based on 0.01 lots.
            # If we trade 1.0 lots, $10 is 1 pip (Instant Death).
            # We must scale limit by (volume / 0.01).
            min_scaling = 1.0
            try:
                # DEBUG: Check if volume exists
                vol_raw = trade.get('volume')
                if vol_raw is None:
                     # logger.warning(f"VSL WARNING: Ticket {ticket} has NO VOLUME key. Trade keys: {list(trade.keys())}")
                     vol = 1.0 # Default to 1.0 (Safe Mode - $1000 limit)
                else:
                     vol = float(vol_raw)
                
                if vol <= 0: vol = 0.01
                scaling_factor = vol / 0.01
            except Exception as e:
                logger.error(f"VSL SCALING ERROR: {e}")
                scaling_factor = 100.0 # Default to 1.0 lot equivalent
                
            # Logarithmic Damping? No, linear is fair for Pips.
            # limit = -abs(v_sl) * scaling_factor
            
            real_v_tp = v_tp * scaling_factor
            real_v_sl = -abs(v_sl) * scaling_factor
            
            # Debug: Show profit vs thresholds (only for positive logic or close calls)
            if profit < -5.0:
                 pass
                 # logger.debug(f"VSL MONITOR: Ticket {ticket} | Vol {vol:.2f} | Scale {scaling_factor:.1f}x | Profit ${profit:.2f} vs Limit ${real_v_sl:.2f}")

            if profit > 0:
                logger.debug(f"VTP CHECK: Ticket {ticket} | Profit ${profit:.2f} | VTP ${real_v_tp:.2f} | Trigger: {profit >= real_v_tp}")
            
            if profit >= real_v_tp:
                logger.info(f"INDIVIDUAL GUARD: Harvesting Ticket {ticket} ({symbol}) | Profit ${profit:.2f} >= ${real_v_tp:.2f}")
                self.close_trade(ticket, symbol)
                self.close_attempts[ticket] = current_time # Mark attempt
                self.closed_tickets_cache.add(ticket)
                continue # One action per trade

            # 2. Individual Stop Loss (The Shield)
            # v_sl is usually positive in config (e.g. $40), so we check against -40
            limit = real_v_sl 
            # 2. Individual Stop Loss (The Shield) - with BREATHING ROOM
            # Tracker Cleanup & Init
            # We track local start time to avoid Timezone issues with Broker Server Time
            if ticket not in self.trade_start_times:
                 self.trade_start_times[ticket] = current_time / 1000.0
            
            local_start_time = self.trade_start_times[ticket]
            age_seconds = (current_time / 1000.0) - local_start_time
            
            # BREATHING ROOM: First 5 Minutes (300s)
            # We relax the VSL to allow trade to develop (survive spread/initial drawdown)
            if age_seconds < 300: 
                limit = limit * 3.0 # e.g. -$10 becomes -$30

            # 3. PREDICTIVE HOLD (The Oracle)
            if profit <= limit:
                # Calculate Panic Limit (Hard Floor)
                panic_limit = limit * 1.5 # e.g. -10 -> -15 (or -30 -> -45 if breathing)
                
                prediction_saved_us = False
                
                if profit > panic_limit:
                    # We are in the "Grey Zone" (Between Stop and Panic)
                    # Use Forecast to decide.
                    if trade_type == 0: # BUY
                        if market_bias in ["BUY", "BULLISH", "STRONG"]:
                            prediction_saved_us = True
                    elif trade_type == 1: # SELL
                        if market_bias in ["SELL", "BEARISH", "STRONG"]:
                            prediction_saved_us = True

                if prediction_saved_us:
                     # logger.info(f"PREDICTIVE HOLD: Ticket {ticket} survived VSL check via Forecast ({market_bias})")
                     continue # HOLD THE LINE

                logger.info(f"INDIVIDUAL GUARD: Pruning Ticket {ticket} ({symbol}) | Profit ${profit:.2f} <= ${limit:.2f} (Age: {age_seconds:.0f}s)")
                self.close_trade(ticket, symbol)
                self.close_attempts[ticket] = current_time  # Mark attempt
                continue

    def close_longs(self, symbol: str):
        logger.warning(f"EXECUTION: Closing LONGS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_BUYS", [symbol])

    def close_shorts(self, symbol: str):
        logger.warning(f"EXECUTION: Closing SHORTS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_SELLS", [symbol])

