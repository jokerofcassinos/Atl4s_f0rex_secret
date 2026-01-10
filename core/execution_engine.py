
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from collections import deque

from risk.dynamic_leverage import DynamicLeverage
from risk.great_filter import GreatFilter
from core.risk.event_horizon import EventHorizonRisk
from core.utils.logger_structured import trade_logger

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
        self.history_engine = None # Phase 4: Feedback Loop
        
        # Phase 4.2: Win Rate per Module
        self.trade_sources = {} # {ticket: source_id}
        self.pending_sources = deque(maxlen=50) # Queue of {'symbol', 'time', 'source'}

    def register_learning_engine(self, engine):
        """Connects the History/Learning Engine for outcome tracking."""
        self.history_engine = engine
        logger.info("ExecutionEngine connected to HistoryLearningEngine.")


    def set_config(self, config: Dict):
        self.config = config

    def _get_current_time_msc(self):
        return int(time.time_ns() / 1_000_000)
        
        
    async def monitor_positions(self, tick: Dict, agi_context: Dict = None):
        """
        Orchestrates all position monitoring checks (VTP, VSL, Predictive, Stalemate).
        """
        if not self.bridge: return
        
        trades = tick.get('trades_json', [])
        if not trades: return
        
        # Phase 4.2: Map new trades to pending sources
        for trade in trades:
            ticket = trade.get('ticket')
            symbol = trade.get('symbol')
            if ticket not in self.trade_sources and ticket not in self.closed_tickets_cache:
                # New Trade Detected - Find Source
                found_data = {'source': "UNKNOWN", 'confidence': 50.0}
                
                # Check pending queue (FIFO)
                now = time.time()
                matcher = None
                
                for i, p in enumerate(self.pending_sources):
                    if p['symbol'] == symbol and (now - p['time'] < 30): # 30s match window
                        matcher = i
                        found_data = {'source': p['source'], 'confidence': p.get('confidence', 50.0)}
                        break
                
                if matcher is not None:
                    # Remove from pending (consumed)
                    del self.pending_sources[matcher]
                    
                self.trade_sources[ticket] = found_data
                # logger.info(f"TRADE SOURCE MAPPED: Ticket {ticket} -> {found_source}")
        
        # 1. Config Thresholds
        v_tp = self.config.get('virtual_tp', 2.0)
        v_sl = self.config.get('virtual_sl', 10.0)
        
        # 2. Check Individual Guards (VTP / VSL)
        self.check_individual_guards(trades, v_tp, v_sl)
        
        # 3. Pack Synchronization (Wolf Pack Logic)
        # If leader secured bag, stragglers must move to safety.
        self.apply_pack_synchronization(trades, tick)
        
        # 4. Check Predictive Exits (The Magnet)
        for trade in trades:
             if not isinstance(trade, dict): continue
             
             # A. Predictive Exit (VTP)
             closed = self.check_predictive_exit(trade, tick, agi_context)
             if closed: continue
             
             # B. Stalemate Check (The Broom)
             self.check_stalemate(trade, tick, agi_context)
        
        
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
                # SAFETY: Enforce 5 Minute Breathing Room for Physical SL too
                # Unless we are protecting significant profit (> $20)
                if ticket not in self.trade_start_times:
                    self.trade_start_times[ticket] = self._get_current_time_msc() / 1000.0
                
                age_seconds = (self._get_current_time_msc() / 1000.0) - self.trade_start_times[ticket]
                
                current_profit = trade.get('profit', 0.0)
                
                if age_seconds < 300 and current_profit < 20.0:
                     # logger.debug(f"EVENT HORIZON: Holding Fire on Ticket {ticket} (Age {age_seconds:.0f}s < 300s)")
                     new_stop_level = 0.0 # Cancel update
                
                # Ratchet Logic: Only move closer to price
                update_needed = False
                
            if new_stop_level:
                # SAFETY: Enforce 5 Minute Breathing Room for Physical SL too
                # Unless we are protecting significant profit (> $20)
                if ticket not in self.trade_start_times:
                    self.trade_start_times[ticket] = self._get_current_time_msc() / 1000.0
                
                age_seconds = (self._get_current_time_msc() / 1000.0) - self.trade_start_times[ticket]
                
                current_profit = trade.get('profit', 0.0)
                
                if age_seconds < 300 and current_profit < 20.0:
                     new_stop_level = 0.0 # Cancel update
                
                # Ratchet Logic: Only move closer to price
                update_needed = False
                
                if new_stop_level > 0:
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
        atr_value: Optional[float] = None, # Phase 3: AGI ATR Override
        hydra_multiplier: float = 1.0, # Phase 11: Hydra Burst Scaling
        volatility: float = 50.0, # Phase 11: Swarm Volatility
        range_data: Optional[Dict] = None, # Phase 17: Range Scanner Context
        source: str = "UNKNOWN" # Phase 4.2: Source Attribution
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
            risk_signal = {"type": command, "confidence": confidence, "symbol": symbol}
            # spread in raw price units; for Gold it's "points"
            market_state = {
                "spread": (ask - bid),
                "typical_price": (ask + bid) / 2,
                "is_crash": False,
                "range_status": range_data.get('status', 'TRENDING') if range_data else 'TRENDING',
                "range_proximity": range_data.get('proximity', 'MID') if range_data else 'MID',
            }
            verdict = self.risk_filter.validate_entry(risk_signal, market_state)
            if not verdict.get("allowed", False):
                logger.info("EXECUTION BLOCKED by GreatFilter: %s", verdict.get("reason"))
                return None
            
            # Ping-Pong Mode: If GreatFilter inverted the signal, use the new command
            if verdict.get("inverted_action"):
                 original_command = command
                 command = verdict["inverted_action"]
                 logger.info(f"PING-PONG INVERSION: Changing {original_command} -> {command} (Lateral Market)")

        # 2. Dynamic Lots Calculation
        equity = account_info.get('equity', 1000.0) if account_info else 1000.0
        # Volatility passed from Swarm/Hydra (or default 50.0)
        # volatility = 50.0 (Removed hardcoding) 
        
        lots = self.leverage_manager.calculate_lots(
            equity=equity,
            confidence=confidence,
            market_volatility=volatility
        )
        
        # Apply Logic Multiplier (Sovereign/Singularity)
        lots = round(lots * multiplier * hydra_multiplier, 2)
        
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
        
        # We KEEP VTP to allow Predictive Exits (The Magnet) to function.
        params = [symbol, cmd_type, lots, "0.00000", f"{vtp:.5f}", confidence]

        # Phase 4.2: Track Source for when trade appears
        self.pending_sources.append({
             'symbol': symbol,
             'time': time.time(),
             'source': source,
             'confidence': confidence # Phase 4.3
        })

        if self.bridge:
            logger.info(f"TRANSMITTING ORDER: {command} {symbol} @ {price:.5f} | Lots: {lots} | Conf: {confidence:.1f}%")
            logger.info(f"DYNAMIC GEOMETRY (INTERNAL): VSL={vsl:.5f} (DISABLED in MQL5), VTP={vtp:.5f}")
            self.bridge.send_command("OPEN_TRADE", params)
            logger.info(f"ORDER SENT TO MT5: {params}")
            
            # Structured Log
            trade_logger.log_trade_event("TRADE_OPEN", {
                "symbol": symbol,
                "type": command,
                "lots": lots,
                "price": price,
                "vtp": vtp,
                "confidence": confidence
            })
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
        infinite_depth: int = 0,
        atr_value: float = None, # Added for GreatFilter Spread Guard
        range_data: Dict = None, # Phase 17: Range Scanner Context
        source: str = "HYDRA" # Default source
    ):
        """
        PHASE 11: HYDRA PROTOCOL (AGI Multi-Vector Execution).
        Dynamically multiplies execution vectors based on conviction (Confidence + Depth).
        """
        logger.info(f"HYDRA PROTOCOL: Initiating Burst Sequence for {symbol} ({command}) [Source: {source}]")
        
        # 1. Determine Hydra Heads (Number of Orders)
        # Base: 1 order.
        # Conviction Bonus: +1 per 10% above 70% confidence.
        # User Specific Mapping (Aggressive Mode)
        heads = 1
        if confidence >= 54:
            heads = 10
        elif confidence >= 50:
            heads = 6
        elif confidence >= 47:
            heads = 3
        
        # AGI Depth Bonus (The "Ontological Nuance" Factor)
        # Add bonus ONLY if we aren't already maxed out
        if infinite_depth >= 50 and heads < 10: 
             heads += 1 
        
        # Entropy Damper (If chaos is too high, reduce slightly)
        if entropy > 0.8 and heads > 1:
            heads = max(1, heads - 1)
            # logger.info(f"HYDRA: High Entropy ({entropy:.2f}) pruned heads to {heads}")
            
        # Hard Cap (Safety)
        heads = min(heads, 10)
        
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
            
            # Determine Scaling Factor to prevent Margin Blowout
            # 10 Heads * 1.0 Lots = 10x Risk (Too much).
            # We want "Aggressive" but "Possible".
            # 10 Heads -> 0.3x each (Total 3.0x Risk)
            # 6 Heads -> 0.4x each (Total 2.4x Risk)
            # 3 Heads -> 0.6x each (Total 1.8x Risk)
            
            scaling_factor = 1.0
            if heads >= 8: scaling_factor = 0.3
            elif heads >= 5: scaling_factor = 0.4
            elif heads >= 2: scaling_factor = 0.6
            
            await self.execute_signal(
                command=command,
                symbol=symbol,
                bid=bid,
                ask=ask,
                confidence=confidence,
                account_info=account_info,
                volatility=volatility,
                hydra_multiplier=scaling_factor, # Pass scaling
                atr_value=atr_value, # Pass ATR for Spread Guard
                range_data=range_data, # Pass Range Context for Lateral Veto
                source=f"{source}_HEAD_{i+1}" # Unique sub-source
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

    def check_predictive_exit(self, trade: Dict, current_tick: Dict, agi_context: Dict = None):
        """
        Virtual TP 2.0 (The Magnet) with Dynamic Assessment.
        Checks if we should close BEFORE the hard TP to secure bag.
        Analyzes "Difficulty" of reaching the target using AGI Context.
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

        # Anti-Ghosting
        if ticket in self.closed_tickets_cache: return

        progress = current_dist / total_dist
        
        # 2. Virtual Hit Logic - SIMPLIFIED
        # Close immediately when profit >= $2.00 (Reference)
        if profit >= 2.0:
             logger.info(f"VTP: {symbol} reached ${profit:.2f}. Closing immediately.")
             self.close_trade(ticket, symbol, profit=profit, reason="VTP_HIT")
             return True
             
        # 3. Dynamic "Smart Greed" (Difficulty Analysis) (User Request)
        # Scan 0.00 -> 2.00 Range.
        # If we are profitable (>0.50) but facing "Heavy Resistance", take the money.
        
        if profit > 0.50:
             difficulty_score = 0.0
             reason = []
             
             # A. Agi Context Analysis
             if agi_context:
                  # 1. Volatility Drop (Dead Market)
                  vol = agi_context.get('volatility_score', 50.0)
                  if vol < 35.0: # Dying momentum
                       difficulty_score += 30.0
                       reason.append("Low Vol")
                       
                  # 2. Counter-Trend Bias
                  chronos = agi_context.get('chronos_narrative', {})
                  trend_bias = chronos.get('trend_bias', 'NEUTRAL')
                  trade_type = trade.get('type') # 0=Buy
                  
                  if trade_type == 0 and "BEAR" in trend_bias:
                       difficulty_score += 40.0 # Fighting the tide
                       reason.append("Trend Mismatch")
                  elif trade_type == 1 and "BULL" in trend_bias:
                       difficulty_score += 40.0
                       reason.append("Trend Mismatch")
                       
             # B. Time Decay (Stagnation)
             current_time = self._get_current_time_msc() / 1000.0
             if ticket in self.trade_start_times:
                  age = current_time - self.trade_start_times[ticket]
                  if age > 900: # 15 mins
                       difficulty_score += 20.0
                       reason.append("Stagnation")
             
             # C. Threshold Logic
             # High Profit (> 1.20) -> Low Difficulty needed to close (Secure 60% of target)
             if profit > 1.20 and difficulty_score >= 20.0:
                   logger.info(f"SMART GREED: Closing {ticket} at ${profit:.2f}. Difficulty: {difficulty_score} ({reason})")
                   self.close_trade(ticket, symbol, profit=profit, reason=f"SMART_GREED:{reason}")
                   return True
                  
             # Medium Profit (> 0.50) -> High Difficulty needed to close (Bail out)
             if profit > 0.50 and difficulty_score >= 50.0:
                   logger.info(f"SMART BAILOUT: Closing {ticket} at ${profit:.2f}. High Difficulty: {difficulty_score} ({reason})")
                   self.close_trade(ticket, symbol, profit=profit, reason=f"SMART_BAILOUT:{reason}")
                   return True
        
        return False

    def close_trade(self, ticket: int, symbol: str, profit: float = 0.0, reason: str = "MANUAL"):
        logger.info(f"EXECUTION: Closing Trade {ticket} ({symbol}) Reason: {reason}")
        if self.bridge:
            # FIX: Must include symbol for ZmqBridge routing
            self.bridge.send_command("CLOSE_TRADE", [str(ticket), symbol])
            self.closed_tickets_cache.add(ticket) # Mark as dead to ExecutionEngine
            
            # Structured Log
            trade_logger.log_trade_event("TRADE_CLOSE", {
                "ticket": ticket,
                "symbol": symbol,
                "reason": reason,
                "profit": profit
            })
            
            # Phase 4: Feedback Loop
            perf_data = self.trade_sources.get(ticket, {'source': "UNKNOWN", 'confidence': 50.0})
            if isinstance(perf_data, str): perf_data = {'source': perf_data, 'confidence': 50.0} # Legacy safety
            
            if self.history_engine:
                 self.history_engine.notify_trade_close(ticket, symbol, profit, reason, 
                                                        source=perf_data['source'], 
                                                        confidence=perf_data.get('confidence', 50.0))
            
            # Cleanup
            if ticket in self.trade_sources:
                del self.trade_sources[ticket]

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
            
            # CRITICAL FIX: Enforce Minimum Drawdown Space per Lot
            # User reported closing at -$10 on 1.3 lots (< 1 pip).
            # We strictly enforce that VSL cannot be tighter than -$100 per 1.0 lot.
            # 1.3 lots -> -$130 minimum space.
            min_drawdown = -100.0 * vol
            if real_v_sl > min_drawdown:
                 real_v_sl = min_drawdown
                 # logger.debug(f"VSL ADJUST: Enforcing Min Drawdown ${min_drawdown:.2f} (Was ${real_v_sl:.2f})")
            
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
                self.close_trade(ticket, symbol, profit=profit, reason="INDIVIDUAL_GUARD_VSL")
                self.close_attempts[ticket] = current_time  # Mark attempt
                continue

    def apply_pack_synchronization(self, trades: List[Dict], tick: Dict):
        """
        Wolf Pack Synchronization.
        If any trade in the pack has secured significant profit (e.g. > $2.0 or hit TP1),
        all other trades in that direction MUST move to Break-Even immediately.
        We do not allow a winning prediction to turn into a losing position for the stragglers.
        """
        if not trades: return
        
        # 1. Group by Direction
        buys = [t for t in trades if t.get('type') == 0 and t.get('symbol') == tick.get('symbol')]
        sells = [t for t in trades if t.get('type') == 1 and t.get('symbol') == tick.get('symbol')]
        
        self.sync_direction_pack(buys, "BUY", tick)
        self.sync_direction_pack(sells, "SELL", tick)

    def sync_direction_pack(self, pack: List[Dict], side: str, tick: Dict):
        if not pack: return
        
        # 2. Check for Leaders (Winners)
        # A Leader is a trade that has > $2.0 profit OR has already been closed (can't check closed here easily, so we rely on Open pnl)
        # OR if we detect a High Water Mark? No, keep it simple.
        
        max_profit = -999.0
        leader_ticket = None
        
        for t in pack:
             p = float(t.get('profit', 0.0))
             if p > max_profit:
                 max_profit = p
                 leader_ticket = t.get('ticket')
                 
        # 3. Decision Logic
        # User Issue: "Fechou +2 e outros ficaram 0.00 e desceram".
        # If Max Profit > 1.50 (slightly less than 2.0 to catch early), we trigger "Safety Mode".
        
        if max_profit > 1.50:
             # logger.debug(f"PACK SYNC ({side}): Leader {leader_ticket} reached ${max_profit:.2f}. Securing Stragglers.")
             
             for t in pack:
                 ticket = t.get('ticket')
                 if ticket == leader_ticket: continue # Leader is managed by VTP
                 
                 profit = float(t.get('profit', 0.0))
                 open_price = float(t.get('open_price', 0.0))
                 current_sl = float(t.get('sl', 0.0))
                 
                 # 4. Check if Straggler is unsafe (Negative or 0.00 SL)
                 # We want to force SL to Break-Even + Spread
                 
                 # Calculate Break-Even Level
                 # For BUY: Open Price + Spread
                 # For SELL: Open Price - Spread
                 
                 current_price = tick.get('bid')
                 spread = tick.get('ask', 0) - tick.get('bid', 0)
                 if spread <= 0: spread = 0.0001
                 
                 be_level = 0.0
                 update_needed = False
                 
                 if side == "BUY":
                      target_sl = open_price + (spread * 1.5) # Slight profit to cover comms
                      if current_sl < target_sl:
                          # Only update if Price is currently ABOVE target (otherwise we close immediately? No, we set SL and let market modify)
                          # Actually, bridge handles modify.
                          # But MT5 rejects SL if price is too close.
                          if current_price > target_sl + spread:
                               be_level = target_sl
                               update_needed = True
                 else: # SELL
                      target_sl = open_price - (spread * 1.5)
                      # Sell SL is ABOVE price. We want to move it DOWN to Entry.
                      # If Current SL is 0 (no SL) or > Target SL (Riskier), we tighten.
                      if (current_sl == 0 or current_sl > target_sl):
                           if current_price < target_sl - spread:
                                be_level = target_sl
                                update_needed = True
                                
                 if update_needed:
                      # logger.info(f"PACK SAFETY: Dragging Straggler {ticket} to BE ({be_level:.5f}) because Leader is up ${max_profit:.2f}")
                      if self.bridge:
                           tp = t.get('tp', 0.0)
                           self.bridge.send_command("MODIFY_TRADE", [str(ticket), f"{be_level:.5f}", f"{tp:.2f}"])
                           # Mark as updated to avoid spam? 
                           # Implementation detail: MT5 bridge might reject spam, but here we only send if condition met.
                           # We relying on next tick to update 'sl' in json, so we won't spam infinitely if update works.

    def check_stalemate(self, trade: Dict, current_tick: Dict, agi_context: Dict = None):
        """
        Lateral Market Decay (The Broom).
        Clears out old, stagnant positions via AGI-Aware Decay.
        """
        ticket = trade.get('ticket')
        if ticket in self.closed_tickets_cache: return
        
        # Calculate Age
        current_time_sec = self._get_current_time_msc() / 1000.0
        
        # Ensure start time is tracked
        if ticket not in self.trade_start_times:
            self.trade_start_times[ticket] = current_time_sec
            return

        local_start_time = self.trade_start_times[ticket]
        age_seconds = current_time_sec - local_start_time
        profit = float(trade.get('profit', 0.0))
        symbol = trade.get('symbol')
        trade_type = trade.get('type') # 0=Buy, 1=Sell
        
        # --- AGI CONTEXT AWARENESS ---
        base_threshold = 2700 # 45 minutes
        volatility_score = 50.0
        trend_bias = "NEUTRAL"
        
        if agi_context:
            # 1. Volatility Scaling
            # If Volatility is Low (<30), market is dead. Exit faster.
            # If Volatility is High (>70), give more room (noise).
            volatility_score = agi_context.get('volatility_score', 50.0)
            
            if volatility_score < 30:
                base_threshold = 1800 # 30 minutes (Dead Market)
            elif volatility_score > 70:
                base_threshold = 3600 # 60 minutes (High Noise)
                
            # 2. Trend Alignment Bonus
            # If I am Long and Trend is Bullish -> Be Patient (+30m)
            chronos = agi_context.get('chronos_narrative', {})
            trend_bias = chronos.get('trend_bias', 'NEUTRAL')
            
            aligned = False
            if trade_type == 0 and trend_bias in ["BULLISH", "STRONG_BULL"]: aligned = True
            if trade_type == 1 and trend_bias in ["BEARISH", "STRONG_BEAR"]: aligned = True
            
            if aligned:
                base_threshold += 1800 # +30m Patience
                # logger.debug(f"STALEMATE: Ticket {ticket} aligned with {trend_bias}. Extended patience to {base_threshold/60:.0f}m")

        # Logic 1: Stagnation (Smart Threshold)
        if age_seconds > base_threshold: 
             if profit >= 0.50:
                   logger.info(f"STALEMATE: Ticket {ticket} ({symbol}) old ({age_seconds/60:.1f}m > {base_threshold/60:.0f}m) & green (${profit:.2f}). Bias: {trend_bias}. Closing.")
                   self.close_trade(ticket, symbol, profit=profit, reason="STALEMATE_GREEN")
                   return True
                  
        # Logic 2: Ancient Decay (Hard Cleanup - 120m cap or 90m default)
        # If aligned, we might wait up to 120m. Unaligned 90m.
        final_cap = 7200 if (agi_context and trend_bias != "NEUTRAL") else 5400
        
        if age_seconds > final_cap: 
             if profit >= 0.10: # Just get out green
                   logger.info(f"DECAY: Ticket {ticket} is ancient ({age_seconds/60:.1f}m). Closing at ${profit:.2f}.")
                   self.close_trade(ticket, symbol, profit=profit, reason="ANCIENT_DECAY")
                   return True
                  
        return False

    def close_longs(self, symbol: str):
        logger.warning(f"EXECUTION: Closing LONGS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_BUYS", [symbol])

    def close_shorts(self, symbol: str):
        logger.warning(f"EXECUTION: Closing SHORTS for {symbol}")
        if self.bridge: self.bridge.send_command("CLOSE_SELLS", [symbol])

