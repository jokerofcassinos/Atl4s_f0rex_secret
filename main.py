
import asyncio
import logging
import pandas as pd
from core.zmq_bridge import ZmqBridge
from core.swarm_orchestrator import SwarmOrchestrator
from core.execution_engine import ExecutionEngine
from core.opportunity_flow import OpportunityFlowManager
from data_loader import DataLoader
import json

# Setup Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("omega_logs.log"),
        logging.StreamHandler()
    ]
)
# Filter noisy 3rd party logs
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING) # Disabled per user request
# logging.getLogger("urllib3").setLevel(logging.WARNING) # Enabled per user request

logger = logging.getLogger("OmegaProtocol")

class OmegaSystem:
    def __init__(self, zmq_port=5557):
        self.bridge = ZmqBridge(port=zmq_port)
        self.cortex = SwarmOrchestrator()
        self.executor = ExecutionEngine(self.bridge)
        self.flow_manager = OpportunityFlowManager()
        self.data_loader = DataLoader()
        self.symbol = "XAUUSD" # Default
        self.last_trade_times = {} # Cooldown tracking
        self.burst_tracker = {} # Burst Execution Manager
        
    async def boot_sequence(self):
        logger.info("Initializing Omega Protocol...")
        await self.cortex.initialize_swarm()
        logger.info("Cortex Online.")
        logger.info("System Ready. Waiting for Market Data...")

    async def run(self):
        await self.boot_sequence()
        
        while True:
            try:
                # 1. Get Live Tick (Blocking or optimized async)
                # In real prod, this should be non-blocking. 
                # For this prototype, we poll.
                tick = self.bridge.get_tick()
                
                if tick:
                    self.symbol = tick.get('symbol', 'XAUUSD')
                    
                    if 'last' not in tick:
                        tick['last'] = (tick['bid'] + tick['ask']) / 2

                    # --- 0. EMERGENCY GUARDIAN (Latency Bypass) ---
                    # CRITICAL: If profit is secured, skipping ALL heavy calculation/IO to close immediately.
                    current_profit_guard = tick.get('profit', 0.0)
                    best_profit_guard = tick.get('best_profit', -999.0)
                    positions_guard = tick.get('positions', 0)
                    
                    if positions_guard > 0:
                        # 1. SURGICAL PRIORITY: Close the big winner first.
                        # This avoids closing small nascent trades (e.g. $1 profit) when a big one hits ($30).
                        if best_profit_guard > 3.0:
                             best_ticket = tick.get('best_ticket')
                             logger.critical(f"[!] GUARDIAN INTERVENTION: SURGICAL PROFIT ${best_profit_guard:.2f} > $3.00. CLOSING TICKET {best_ticket}.")
                             self.executor.close_trade(best_ticket, self.symbol)
                             await asyncio.sleep(0.2) 
                             continue # Continue loop to refresh tick and check next best

                        # 2. GLOBAL SAFETY: Only Close All if total profit is massive or protecting a large basket.
                        # Raised from 3.0 to 15.0 to prevent premature closure of small baskets.
                        if current_profit_guard > 15.0:
                            logger.critical(f"[!] GUARDIAN INTERVENTION: GLOBAL PROFIT ${current_profit_guard:.2f} > $15.00. BASKET EXIT.")
                            self.executor.close_all(self.symbol)
                            await asyncio.sleep(0.5)
                            continue

                    # 2. Fetch/Update Context (Dataframes)
                    # Ideally, data_loader handles live updates via tick injection
                    # For now, we reload/resample.
                    # Fetches M1, M5, M15, M30, H1, H4, D1, W1
                    data_map = self.data_loader.get_data(self.symbol) 
                    
                    # 3. Cortex Thinking
                    # Returns (decision, confidence, metadata)
                    decision_tuple = await self.cortex.process_tick(tick, data_map)
                    
                    decision = decision_tuple[0]
                    confidence = decision_tuple[1]
                    metadata = decision_tuple[2]
                    
                    # Phase 30: Apex Routing
                    if decision == "ROUTING":
                         new_symbol = metadata.get('best_asset')
                         if new_symbol and new_symbol != self.symbol:
                             logger.warning(f"APEX ROUTING: Switching Focus from {self.symbol} to {new_symbol}")
                             self.symbol = new_symbol
                             # Clean memory slightly to avoid pollution
                             # self.cortex.short_term_memory = {} # Too aggressive?
                             # Just let it flow. Next loop will fetch new data.
                             
                             # Synthetic tick update to prevent stale price logic
                             # We don't have new price yet, but next loop data_loader will get it.
                             # We continue to let loop refresh.
                             continue 
                    
                    # 4. Execution
                    current_time = pd.Timestamp.now()
                    last_fire = self.last_trade_times.get(self.symbol, None)
                    
                    cooldown_seconds = 45 # Safety: 45s cooldown to prevent order stacking
                    
                    if decision in ["BUY", "SELL"]:
                        # --- CONTRARIAN MODE (OFF) ---
                        # Flip the signal: Buy -> Sell, Sell -> Buy
                        # original_decision = decision
                        # if decision == "BUY": decision = "SELL"
                        # elif decision == "SELL": decision = "BUY"
                        # logger.warning(f"CONTRARIAN MODE: Inverting {original_decision} to {decision}")
                        # ---------------------------------------
                        # --- DYNAMIC SLOT MANAGEMENT (Phase 28) ---
                        # 1. Get State
                        current_equity = tick.get('equity', 1000.0)
                        current_positions = tick.get('positions', 0)
                        
                        # 2. Base Slots (1 per $500)
                        base_slots = max(1, int(current_equity / 500))
                        
                        # 3. Regime Modifier (Architect)
                        # We need access to Architect's directive. It's in the signals meta_data.
                        # For now, we assume simple logic or extract if available.
                        # Let's use a simpler heuristic for now:
                        # If HIGH CONFIDENCE (Architect aligned), multiplier = 1.5
                        
                        slot_multiplier = 1.0
                        if confidence >= 90.0: slot_multiplier = 1.5 
                        
                        # 4. Final Calculation
                        max_slots = int(base_slots * slot_multiplier)
                        
                        # 5. Cap (Safety)
                        max_slots = min(max_slots, 20) 
                        
                        logger.info(f"SLOT MANAGER: {current_positions}/{max_slots} slots used. (Eq: {current_equity:.0f})")

                        # --- DYNAMIC BURST EXECUTION (Scale-In) ---
                        # Logic: High Confidence = Machine Gun Mode. Low Confidence = Single Shot.
                        # Window: 45 seconds.
                        
                        if self.symbol not in self.burst_tracker:
                            self.burst_tracker[self.symbol] = {'start': current_time, 'count': 0}
                            
                        tracker = self.burst_tracker[self.symbol]
                        
                        # 1. Check/Reset Window
                        if (current_time - tracker['start']).total_seconds() > 45:
                            tracker['start'] = current_time
                            tracker['count'] = 0
                            
                        max_burst = 1
                        if confidence >= 95.0: max_burst = 5
                        elif confidence >= 90.0: max_burst = 3
                        elif confidence >= 80.0: max_burst = 1
                        
                        # 3. Fire Check
                        logger.info(f"DEBUG BURST: Conf={confidence:.1f} MaxBurst={max_burst} Count={tracker['count']} Pos={current_positions} MaxSlots={max_slots}")

                        # GATE: Only fire if we have slots AND haven't exceeded burst
                        if tracker['count'] < max_burst and current_positions < max_slots:
                             await self.executor.execute_signal(decision, self.symbol, 
                                                                tick.get('bid'), tick.get('ask'), 
                                                                confidence=confidence,
                                                                account_info={'equity': current_equity}) 
                             
                             tracker['count'] += 1
                             logger.info(f"FIRE! Burst {tracker['count']}/{max_burst} | Slots {current_positions + 1}/{max_slots}")
                        else:
                             if current_positions >= max_slots:
                                 logger.info("Signal Blocked: Max Slots Reached.")
                             
                    elif decision == "EXIT_LONG":
                        self.executor.close_longs(self.symbol)
                        logger.warning("SMART EXIT: Dropping LONGS.")
                        
                    elif decision == "EXIT_SHORT":
                        self.executor.close_shorts(self.symbol)
                        logger.warning("SMART EXIT: Dropping SHORTS.")

                    elif decision == "EXIT_ALL":
                        # Immediate Priority Override
                        self.executor.close_all(self.symbol)
                        logger.warning("STRATEGIC EXIT TRIGGERED.")
                        
                    elif decision == "EXIT_SPECIFIC":
                        # Surgical Close of Winner
                        ticket = meta_data.get('ticket')
                        if ticket:
                             self.executor.close_trade(ticket, self.symbol)
                             logger.warning(f"SURGICAL EXIT: Closing Ticket {ticket}. Reason: {meta_data.get('reason')}")


                    state = {
                        'swarm_state': self.cortex.state,
                        'neuro_weights': self.cortex.neuroplasticity.get_dynamic_weights(),
                        'last_decision': decision
                    }
                    # self.bridge.send_dashboard(state)
                    
                await asyncio.sleep(0.1) 
                
            except Exception as e:
                logger.error(f"Omega Loop Error: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    system = OmegaSystem()
    asyncio.run(system.run())
