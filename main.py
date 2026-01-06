
import asyncio
import logging
import pandas as pd
import datetime
from core.zmq_bridge import ZmqBridge
from core.swarm_orchestrator import SwarmOrchestrator
from core.execution_engine import ExecutionEngine
from core.opportunity_flow import OpportunityFlowManager
from data_loader import DataLoader
import json

# Sub-Engines
from core.consciousness_bus import ConsciousnessBus
from core.genetics import EvolutionEngine
from core.neuroplasticity import NeuroPlasticityEngine
from core.transformer_lite import TransformerLite
from core.mcts_planner import MCTSPlanner
from core.agi.omega_agi_core import OmegaAGICore # Project Awakening

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
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# === REASONING LOGS - REDUCED SPAM ===
# Set spammy initialization logs to WARNING to reduce noise
logging.getLogger("MCTS_Planner").setLevel(logging.INFO)
logging.getLogger("SwarmOrchestrator").setLevel(logging.INFO)
logging.getLogger("InfiniteWhyEngine").setLevel(logging.WARNING)  # Spammy initialization
logging.getLogger("ThoughtTree").setLevel(logging.WARNING)
logging.getLogger("HolographicMemory").setLevel(logging.WARNING)  # Spammy initialization
logging.getLogger("LaplaceSwarm").setLevel(logging.INFO)
logging.getLogger("RiemannSwarm").setLevel(logging.INFO)
logging.getLogger("OmegaProtocol").setLevel(logging.INFO)
logging.getLogger("ExecutionEngine").setLevel(logging.DEBUG)  # Debug VSL checks

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn") 

logger = logging.getLogger("OmegaProtocol")

class OmegaSystem:
    def __init__(self, zmq_port=5557):
        self.bridge = ZmqBridge(port=zmq_port)
        
        # Initialize Core Cognitive Engines
        self.bus = ConsciousnessBus()
        self.evolution = EvolutionEngine()
        self.neuroplasticity = NeuroPlasticityEngine()
        self.attention = TransformerLite(embed_dim=64, head_dim=64) # Simple init stats
        self.grandmaster = MCTSPlanner()
        self.agi = OmegaAGICore() # The Brain
        
        

        
        # Inject into Cortex
        self.cortex = SwarmOrchestrator(
            bus=self.bus,
            evolution=self.evolution,
            neuroplasticity=self.neuroplasticity,
            attention=self.attention,
            grandmaster=self.grandmaster
        )
        
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

    def smart_startup(self):
        """
        AGI Logic for System Startup. ("The Interview")
        """
        print("\n" + "="*50)
        print("   OMEGA PROTOCOL v5.0 - AWAKENED CORE   ")
        print("="*50)
        
        # --- AGI PRE-FLIGHT CHECK ---
        print("\n[AGI]: Connecting to Market Matrix for Analysis...")
        
        # 1. Quick Data Fetch
        try:
            # Try to get a live tick to confirm connection
            tick = self.bridge.get_tick()
            if tick and 'symbol' in tick:
                self.symbol = tick['symbol']
                print(f"[AGI]: Connection Established. Target: {self.symbol}")
            else:
                print("[AGI]: No live tick from Bridge.")
                sym_in = input(f"Enter Target Symbol [Default: {self.symbol}]: ").strip().upper()
                if sym_in: self.symbol = sym_in

            # 2. Populate DataLoader
            print("[AGI]: Downloading Quantum History...")
            self.data_loader.get_data(self.symbol) # Forces download/update
            
        except Exception as e:
            print(f"[AGI WARNING]: Data fetch failed ({e}). Proceeding blind.")

        from core.agi.profiler import AGIProfiler
        profiler = AGIProfiler(self.data_loader)
        rec = profiler.analyze_market_conditions()
        
        print(f"\n[AGI ANALYSIS]: {rec['reason']}")
        print(f"RECOMMENDATION: Mode={rec['mode']} | Risk={rec['risk_profile']}")
        
        try:
            # Non-blocking check for user override
            print("\nAccept AGI Recommendation? [Y/n] (Auto-accept in 5s)")
            use_rec = 'y' # Default
            # In a real GUI/Terminal this would have a timeout input.
            # For now we assume User wants AGI unless they type 'n' fast? 
            # Actually, standard input() blocks. We'll use input() for now.
            use_rec_in = input(f"Confirm {rec['mode']}? [Y/n]: ").strip().lower()
            if use_rec_in: use_rec = use_rec_in
        except:
             use_rec = 'y'

        if use_rec == 'n':
            self.interactive_startup() # Fallback to manual
        else:
            self.config['mode'] = rec['mode']
            self.symbol = "XAUUSD" if rec['risk_profile'] == "STANDARD" else "BTCUSD" # Simple fallback
            if self.symbol == "XAUUSD":
                # Apply Gold Standard
                self.config["virtual_sl"] = 10.0
                self.config["virtual_tp"] = 2.0
                self.config["spread_limit"] = 0.02
                self.config["phys_sl_pct"] = 0.005
                self.config["phys_tp_pct"] = 0.010
            print(f">> Applying {rec['mode']} Protocol.")
            print(f"Configuration Loaded: Profile={self.symbol} | Mode={self.config['mode']}")
            print("="*50 + "\n")
            
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
            self.config["phys_tp_pct"] = 0.010 # 1.0% Hard Target
            
            # Update Opportunity Flow
            # Update Opportunity Flow
            # Removed AUDCAD and AUDJPY per user request (High fees/Bad performance)
            # Update Opportunity Flow
            # User Request: Majors for tight spreads
            self.flow_manager.active_symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"]
            
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
        print("3. HYBRID (Balanced, Adaptive Execution)")
        print("4. AGI MAPPER (Full AGI Control, Self-Optimizing)")
        try:
            sel = input("Selection [1/2/3/4]: ").strip()
            if sel == "2": 
                self.config['mode'] = "WOLF_PACK"
            elif sel == "3": 
                self.config['mode'] = "HYBRID"
            elif sel == "4": 
                self.config['mode'] = "AGI_MAPPER"
            else: 
                self.config['mode'] = "SNIPER"
        except: pass
        
        mode_messages = {
            "WOLF_PACK": ">> WOLF PACK MODE ENGAGED. UNLEASH THE HOUNDS.",
            "SNIPER": ">> SNIPER MODE ENGAGED. ONE SHOT, ONE KILL.",
            "HYBRID": ">> HYBRID MODE ENGAGED. ADAPTIVE PRECISION + AGGRESSION.",
            "AGI_MAPPER": ">> AGI MAPPER MODE ENGAGED. FULL AUTONOMOUS CONTROL ACTIVE."
        }
        print(mode_messages.get(self.config['mode'], ">> UNKNOWN MODE"))

        print(f"\nConfiguration Loaded: Profile={self.symbol} | VSL=${self.config['virtual_sl']} | VTP=${self.config['virtual_tp']} | Mode={self.config['mode']}")
        print("="*50 + "\n")

    async def boot_sequence(self):
        logger.info("Initializing Omega Protocol...")
        
        # Phase 102: Inject Configuration into Executor
        self.executor.set_config(self.config)
        
        await self.cortex.initialize_swarm()
        self.cortex.inject_bridge(self.bridge) # Link Visuals to Swarm (must be after init)
        logger.info("Cortex Online.")
        logger.info("System Ready. Waiting for Market Data...")

    async def run(self):
        self.smart_startup()
        await self.boot_sequence()
        
        tick_count = 0  # Heartbeat counter
        last_heartbeat = datetime.datetime.now()
        
        while True:
            try:
                # 0. AGI Schedule Enforcement
                should_trade, reason = self.agi.should_trade_now(self.symbol)
                if not should_trade:
                    logger.info(f"AGI PAUSE: {reason}. Sleeping...")
                    await asyncio.sleep(60)
                    continue
                
                # Heartbeat Log (every 30 seconds)
                now = datetime.datetime.now()
                if (now - last_heartbeat).total_seconds() >= 30:
                    logger.info(f"[HEARTBEAT] {tick_count} ticks | Mode: {self.config.get('mode', 'N/A')} | Status: ALIVE")
                    last_heartbeat = now
                    tick_count = 0

                # 1. Get Live Tick (Blocking or optimized async)
                # In real prod, this should be non-blocking. 
                # For this prototype, we poll.
                tick = self.bridge.get_tick()
                
                if tick:
                    tick_count += 1
                    self.symbol = tick.get('symbol', 'XAUUSD')
                    
                    # --- PROJECT AWAKENING: AGI PRE-TICK ---
                    # The Brain reasons about the market before the Body moves.
                    agi_adjustments = self.agi.pre_tick(tick, self.config)
                    if agi_adjustments:
                         # Apply self-healing or optimization adjustments
                         if 'switch_mode' in agi_adjustments:
                             # self.config['mode'] = agi_adjustments['switch_mode'] # Optional auto-switch
                             pass 
                         logger.info(f"AGI CORE: Adjustments Generated: {agi_adjustments}")
                    
                    if 'last' not in tick:
                        tick['last'] = (tick['bid'] + tick['ask']) / 2

                    # --- 0. INDIVIDUAL GUARDIAN (Latency Bypass) ---
                    # Phase 122: Per-Order Management (No more "Stop All")
                    
                    trades_snapshot = tick.get('trades_json', [])
                    if trades_snapshot:
                        # self.config['virtual_tp'] and ['virtual_sl'] are per-trade targets now
                         self.executor.check_individual_guards(
                             trades_snapshot, 
                             self.config['virtual_tp'], 
                             self.config['virtual_sl']
                         )
                         
                         # Keep Global Catastrophe Guard (Optional, but good for safety)
                         # Only triggered if Total Profit is absurdly negative (e.g. 5x SL)
                         total_profit = tick.get('profit', 0.0)
                         catastrophe_limit = -abs(self.config['virtual_sl']) * 5 
                         
                         if total_profit < catastrophe_limit:
                             logger.critical(f"CATASTROPHE GUARD: Global Equity Dropped to ${total_profit:.2f}. EMERGENCY EJECT.")
                             self.executor.close_all(self.symbol)
                             await asyncio.sleep(1.0)
                             continue # Reboot loop

                    # 2. Fetch/Update Context (Dataframes)
                    # Ideally, data_loader handles live updates via tick injection
                    # For now, we reload/resample.
                    # Fetches M1, M5, M15, M30, H1, H4, D1, W1
                    # Fetches M1, M5, M15, M30, H1, H4, D1, W1
                    data_map = self.data_loader.get_data(self.symbol) 
                    
                    if self.flow_manager and self.flow_manager.active_symbols:
                        basket_data = self.data_loader.get_basket_data(self.flow_manager.active_symbols)
                        data_map['basket_data'] = basket_data
                        
                    # Phase 116: Event Horizon (Parabolic Exits)
                    # 1. Fetch Open Trades if we have positions but no details
                    # Or refresh every 5s to keep stops valid
                    current_positions_count = tick.get('positions', 0)
                    now_msc = tick.get('time_msc', 0)
                    last_trade_fetch = self.last_trade_times.get('fetch', 0)
                    
                    if current_positions_count > 0:
                        if now_msc - last_trade_fetch > 2000: # Every 2s
                             self.bridge.send_command("GET_OPEN_TRADES", [self.symbol])
                             self.last_trade_times['fetch'] = now_msc
                             
                    # 2. Run Dynamic Stop Manager
                    # This relies on ZmqBridge merging TRADES_JSON into the tick
                    await self.executor.manage_dynamic_stops(tick)

                    # 3. Cortex Thinking
                    # Returns (decision, confidence, metadata)
                    decision, confidence, metadata = await self.cortex.process_tick(tick, data_map, self.config)
                    
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
                        
                        # --- SOVEREIGN MULTIPLIER (Restored) ---
                        lot_multiplier = 1.0
                        sov_state = metadata.get('sovereign_state', 'NEUTRAL')
                        
                        if sov_state == "SINGULARITY":
                            lot_multiplier = 3.0
                            logger.info(">>> !!! SINGULARITY STRIKE !!!: Multiplier 3.0x Active <<<")
                        elif sov_state == "STRONG":
                            lot_multiplier = 2.0
                            logger.info(">>> SOVEREIGN ASSERTIVENESS: Multiplier 2.0x Active <<<")
                        
                        # Phase 95: Wolf Pack Logic (Dynamic Burst)
                        max_burst = 1 # Default Sniper
                        
                        if self.config['mode'] == "WOLF_PACK":
                            if confidence >= 95.0: max_burst = 5 # FULL PACK
                            elif confidence >= 90.0: max_burst = 4
                            elif confidence >= 85.0: max_burst = 3
                            elif confidence >= 80.0: max_burst = 2
                            else: max_burst = 1
                        elif self.config['mode'] == "HYBRID":
                            # Balanced approach: 2-3 orders based on confidence
                            if confidence >= 90.0: max_burst = 3
                            elif confidence >= 80.0: max_burst = 2
                            else: max_burst = 1
                        elif self.config['mode'] == "AGI_MAPPER":
                            # AGI Full Control - Dynamic scaling based on MCTS recommendation
                            mcts_action = self.grandmaster.search({
                                'price': tick.get('bid', 0),
                                'entry': tick.get('bid', 0),
                                'side': decision,
                                'pnl': 0,
                                'volatility': 1.0
                            }, trend_bias=confidence/100.0)
                            
                            # AGI can scale from 1-6 based on its decision
                            if mcts_action in ["HOLD", "CLOSE"]:
                                max_burst = 1  # Conservative
                            elif mcts_action == "TRAIL":
                                max_burst = 3  # Moderate
                            elif mcts_action == "PARTIAL_TP":
                                max_burst = 4  # Aggressive
                            else:
                                max_burst = min(6, int(confidence / 15))  # Dynamic
                            
                        # Existing Burst Logic (Simulates HFT volley)
                        # We use the 'max_burst' determined above.
                        
                        current_positions = tick.get('positions', 0)
                        
                        # --- AGI DYNAMIC SLOTS ---
                        # Use Micro-Stats (Entropy) and Confidence (Trend Strength) to decide slot capacity on the fly.
                        # This replaces the hardcoded limits.
                        micro_stats = metadata.get('micro_stats', {})
                        volatility_idx = micro_stats.get('volatility', 50.0) # Default to mid
                        # Or use entropy * 100 if volatility is not directly available
                        if 'entropy' in micro_stats:
                             volatility_idx = micro_stats['entropy'] * 100.0
                             
                        max_slots = self.cortex.calculate_dynamic_slots(
                            volatility=volatility_idx,
                            trend_strength=confidence, # Use Cortex Confidence as Trend Strength Proxy
                            mode=self.config['mode']
                        )
                        
                        # logger.info(f"AGI SLOTS: {max_slots} (Vol: {volatility_idx:.1f}, Conf: {confidence:.1f})")

                        tracker = self.burst_tracker.get(self.symbol, {'timestamp': 0, 'count': 0})
                        
                        # Reset burst if 1 minute passed
                        if tick.get('time_msc', 0) - tracker['timestamp'] > 60000:
                             tracker = {'timestamp': tick.get('time_msc', 0), 'count': 0}

                        if current_positions < max_slots and tracker['count'] < max_burst:
                             logger.info(f"FIRE! Burst {tracker['count']+1}/{max_burst} | Slots {current_positions+1}/{max_slots}")
                             
                             # Mode-based spread tolerance
                             if self.config['mode'] == "WOLF_PACK": 
                                 spread_tol = 0.05  # Lenient
                             elif self.config['mode'] == "HYBRID":
                                 spread_tol = 0.03  # Moderate
                             elif self.config['mode'] == "AGI_MAPPER":
                                 spread_tol = 0.02  # Tight - AGI knows best
                             else:
                                 spread_tol = None  # Default (SNIPER)
                             
                             await self.executor.execute_signal(decision, self.symbol, 
                                                                tick.get('bid'), tick.get('ask'), 
                                                                confidence=confidence,
                                                                account_info={'equity': tick.get('equity', 1000)},
                                                                spread_tolerance=spread_tol,
                                                                multiplier=lot_multiplier) 
                             
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


                    # --- PROJECT AWAKENING: AGI POST-TICK ---
                    # The Brain learns from the outcome of the Tick.
                    result_feedback = {
                        'decision': decision,
                        'trade_executed': decision in ["BUY", "SELL"],
                        'profit': tick.get('profit', 0.0), # Current PnL snapshot
                        'success': tick.get('profit', 0.0) > 0 # Simple heuristic for now
                    }
                    self.agi.post_tick(decision, result_feedback)
                    
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
