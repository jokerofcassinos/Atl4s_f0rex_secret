
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
        self.symbol = "ETHUSD" # Default for Sunday Crypto
        self.last_trade_times = {} # Cooldown tracking
        self.burst_tracker = {} # Burst Execution Manager
        
        # User Configuration (Defaults)
        # These are overwritten by Profile Selection
        self.config = {
            "virtual_sl": 40.0,
            "virtual_tp": 3.0,
            "mode": "SNIPER",
            "spread_limit": 0.05
        }
        
    def interactive_startup(self):
        print("\n" + "="*50)
        print("   OMEGA PROTOCOL v4.0 - SINGULARITY EDITION   ")
        print("="*50)
        
        # --- PROFILE SELECTION ---
        print("\nSelect Operational Profile:")
        print("1. CRYPTO (Weekend/Volatile) - High SL/TP, Wide Spreads")
        print("2. FOREX/GOLD (Weekday/Normal) - Tight SL/TP, Low Spreads")
        
        try:
            profile_sel = input("Selection [1/2]: ").strip()
        except:
            profile_sel = "1"
            
        if profile_sel == "2":
            print(">> PROFILE: FOREX/GOLD ACTIVATED.")
            self.symbol = "XAUUSD" # Reset to Gold
            self.config["virtual_sl"] = 10.0  # Tight SL on Gold ($10)
            self.config["virtual_tp"] = 2.0   # Quick Scalp ($2)
            self.config["spread_limit"] = 0.02 # 0.02% limit (~0.40 on 2000)
            self.config["phys_sl_pct"] = 0.005 # 0.5% Hard Stop
            self.config["phys_tp_pct"] = 0.010 # 1.0% Hard Target
            
            # Update Opportunity Flow
            self.flow_manager.active_symbols = ["XAUUSD", "XAUAUD", "XAUEUR"]
            
        else:
            print(">> PROFILE: CRYPTO ACTIVATED.")
            self.symbol = "ETHUSD" # Default Crypto
            self.config["virtual_sl"] = 40.0 # Wide SL for Crypto
            self.config["virtual_tp"] = 5.0  # Bigger moves
            self.config["spread_limit"] = 0.05 # 0.05% limit
            self.config["phys_sl_pct"] = 0.020 # 2.0% Hard Stop
            self.config["phys_tp_pct"] = 0.050 # 5.0% Hard Target
            
            # Update Opportunity Flow
            self.flow_manager.active_symbols = ["ETHUSD", "BTCUSD"]
            
        
        print(f"Virtual SL ($) [Default: {self.config['virtual_sl']}]:")
        try:
            vsl_in = input().strip()
            if vsl_in: self.config['virtual_sl'] = float(vsl_in)
        except: pass
            
        print(f"Virtual TP ($) [Default: {self.config['virtual_tp']}]:")
        try:
            vtp_in = input().strip()
            if vtp_in: self.config['virtual_tp'] = float(vtp_in)
        except: pass

        print("\nSelect Operational Mode:")
        print("1. SNIPER (Precision, 1 Order per Signal)")
        print("2. WOLF PACK (Aggressive, Scaled Burst Execution)")
        try:
            sel = input("Selection [1/2]: ").strip()
            if sel == "2": self.config['mode'] = "WOLF_PACK"
            else: self.config['mode'] = "SNIPER"
        except: pass
        
        if self.config['mode'] == "WOLF_PACK":
             print(">> WOLF PACK MODE ENGAGED. UNLEASH THE HOUNDS.")
        else:
             print(">> SNIPER MODE ENGAGED. ONE SHOT, ONE KILL.")

        print(f"\nConfiguration Loaded: Profile={self.symbol} | VSL=${self.config['virtual_sl']} | VTP=${self.config['virtual_tp']} | Mode={self.config['mode']}")
        print("="*50 + "\n")

    async def boot_sequence(self):
        logger.info("Initializing Omega Protocol...")
        
        # Phase 102: Inject Configuration into Executor
        self.executor.set_config(self.config)
        
        await self.cortex.initialize_swarm()
        logger.info("Cortex Online.")
        logger.info("System Ready. Waiting for Market Data...")

    async def run(self):
        self.interactive_startup()
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
                        # DEBUG: What does Guardian see?
                        # logger.info(f"GUARDIAN SCAN: BestProfit=${best_profit_guard:.2f} (Target=${self.config['virtual_tp']}) TotalProfit=${current_profit_guard:.2f}")

                        # 1. SURGICAL PRIORITY: Close the big winner first.
                        # Uses Dynamic VTP from Config
                        if best_profit_guard > self.config['virtual_tp']:
                             best_ticket = tick.get('best_ticket')
                             logger.critical(f"[!] GUARDIAN INTERVENTION: SURGICAL PROFIT ${best_profit_guard:.2f} > ${self.config['virtual_tp']}. CLOSING TICKET {best_ticket}.")
                             
                             # Dynamic Closure Logic
                             # Try primary symbol first
                             self.executor.close_trade(best_ticket, self.symbol)
                             
                             # Try all other active symbols (Just in case the ticket belongs to them)
                             for sec_symbol in self.flow_manager.active_symbols:
                                 if sec_symbol != self.symbol:
                                     self.executor.close_trade(best_ticket, sec_symbol)
                                 
                             await asyncio.sleep(0.2) 
                             continue 
                             
                        # 2. GUARDIAN VIRTUAL SL (Phase 94 - Emergency Layer)
                        # Uses Dynamic VSL from Config
                        # If we bleed too much, cut the basket blindly.
                        limit = -abs(self.config['virtual_sl']) # Ensure negative
                        if current_profit_guard < limit:
                             logger.critical(f"[!] GUARDIAN INTERVENTION: VIRTUAL SL ${current_profit_guard:.2f} < ${limit}. EMERGENCY BASKET EXIT.")
                             self.executor.close_all(self.symbol) # Try primary
                             
                             # Dynamic Basket Closure
                             for sec_symbol in self.flow_manager.active_symbols:
                                 if sec_symbol != self.symbol:
                                     self.executor.close_all(sec_symbol)

                             await asyncio.sleep(0.5)
                             continue

                        # 3. GLOBAL SAFETY Profit (Hardcoded Basket Target)
                        if current_profit_guard > 15.0:
                            logger.critical(f"[!] GUARDIAN INTERVENTION: GLOBAL PROFIT ${current_profit_guard:.2f} > $15.00. BASKET EXIT.")
                            self.executor.close_all(self.symbol)
                            
                            # Dynamic Basket Closure
                            for sec_symbol in self.flow_manager.active_symbols:
                                if sec_symbol != self.symbol:
                                    self.executor.close_all(sec_symbol)

                            await asyncio.sleep(0.5)
                            continue

                    # 2. Fetch/Update Context (Dataframes)
                    # Ideally, data_loader handles live updates via tick injection
                    # For now, we reload/resample.
                    # Fetches M1, M5, M15, M30, H1, H4, D1, W1
                    data_map = self.data_loader.get_data(self.symbol) 
                    
                    # 3. Cortex Thinking
                    # Returns (decision, confidence, metadata)
                    decision, confidence, metadata = await self.cortex.process_tick(tick, data_map)
                    
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
                    
                    # 4. Neural Execution 
                    # 4. Neural Execution 
                    if decision == "BUY" or decision == "SELL":
                        cmd = 0 if decision == "BUY" else 1
                        
                        # Phase 95: Wolf Pack Logic (Dynamic Burst)
                        max_burst = 1 # Default Sniper
                        
                        if self.config['mode'] == "WOLF_PACK":
                            if confidence >= 95.0: max_burst = 5 # FULL PACK
                            elif confidence >= 90.0: max_burst = 4
                            elif confidence >= 85.0: max_burst = 3
                            elif confidence >= 80.0: max_burst = 2
                            else: max_burst = 1
                            
                        # Existing Burst Logic (Simulates HFT volley)
                        # We use the 'max_burst' determined above.
                        
                        current_positions = tick.get('positions', 0)
                        max_slots = 10 # Allow more slots in Wolf Mode?
                        if self.config['mode'] == "WOLF_PACK": max_slots = 15

                        tracker = self.burst_tracker.get(self.symbol, {'timestamp': 0, 'count': 0})
                        
                        # Reset burst if 1 minute passed
                        if tick.get('time_msc', 0) - tracker['timestamp'] > 60000:
                             tracker = {'timestamp': tick.get('time_msc', 0), 'count': 0}

                        if current_positions < max_slots and tracker['count'] < max_burst:
                             logger.info(f"FIRE! Burst {tracker['count']+1}/{max_burst} | Slots {current_positions+1}/{max_slots}")
                             
                             # Fix: Method is execute_signal, and takes 'decision' string ("BUY"/"SELL"), not 'cmd' int.
                             # Wolf Pack Mode: We allow lenient spreads (0.05) to ensure burst execution.
                             spread_tol = 0.05 if self.config['mode'] == "WOLF_PACK" else None
                             
                             await self.executor.execute_signal(decision, self.symbol, 
                                                                tick.get('bid'), tick.get('ask'), 
                                                                confidence=confidence,
                                                                account_info={'equity': tick.get('equity', 1000)},
                                                                spread_tolerance=spread_tol) 
                             
                             tracker['count'] += 1
                             self.burst_tracker[self.symbol] = tracker
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
