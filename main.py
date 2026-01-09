import asyncio
import logging
import pandas as pd
import datetime
import time
from core.zmq_bridge import ZmqBridge
from core.system_guard import SystemGuard # Zombie Defense
# from core.api_server import APIServer # Next.js Dashboard Link (Future)
from core.agi.omega_agi_core import OmegaAGICore
from core.swarm_orchestrator import SwarmOrchestrator
from core.agi.omni_cortex import OmniCortex # New Hybrid AGI
from core.execution_engine import ExecutionEngine
from data_loader import DataLoader
from core.opportunity_flow import OpportunityFlowManager
from core.genetics import EvolutionEngine
from core.neuroplasticity import NeuroPlasticityEngine
from core.transformer_lite import TransformerLite
from core.consciousness_bus import ConsciousnessBus
from core.grandmaster import GrandMaster
from core.genetics import EvolutionEngine
from core.neuroplasticity import NeuroPlasticityEngine
from core.transformer_lite import TransformerLite
from core.consciousness_bus import ConsciousnessBus
from core.grandmaster import GrandMaster
from core.agi.profiler import AGIProfiler # Phase 3: Real Analysis
from core.agi.symbiosis import UserIntentModeler, ExplanabilityGenerator # Phase 5.2
from core.agi.learning import HistoryLearningEngine # Phase 6

# ============================================================================
# CONFIGURATION
# ============================================================================

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
logging.getLogger("ThoughtTree").setLevel(logging.INFO) # Reasoning Visible!
logging.getLogger("HolographicMemory").setLevel(logging.WARNING)  # Spammy initialization
logging.getLogger("LaplaceSwarm").setLevel(logging.INFO)
logging.getLogger("RiemannSwarm").setLevel(logging.INFO)
logging.getLogger("OmegaProtocol").setLevel(logging.INFO)
logging.getLogger("ExecutionEngine").setLevel(logging.INFO)  # Execution is key
logging.getLogger("GrandMaster").setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn") 

logger = logging.getLogger("OmegaProtocol")

class OmegaSystem:
    def __init__(self, zmq_port=5558):
        self.bridge = ZmqBridge(port=zmq_port)
        
        # Initialize Core Cognitive Engines
        self.data_loader = DataLoader()
        self.flow_manager = OpportunityFlowManager()
        self.bus = ConsciousnessBus()
        self.evolution = EvolutionEngine()
        self.neuroplasticity = NeuroPlasticityEngine()
        self.attention = TransformerLite(embed_dim=64, head_dim=64) # Simple init stats
        self.grandmaster = GrandMaster() # The Apex DECISION ENGINE
        
        # --- AGI Dependencies ---
        from core.agi.infinite_why_engine import InfiniteWhyEngine
        from core.agi.simulation_system_agi import SimulationSystemAGI
        self.infinite_why = InfiniteWhyEngine()
        self.sim_system = SimulationSystemAGI()
        
        self.agi = OmegaAGICore(self.infinite_why, self.sim_system) # The Brain
        self.profiler = AGIProfiler(self.data_loader) # Phase 3: The Interviewer
        self.agi_metrics = {'atr': 0.0, 'entropy': 0.5, 'volScore': 50.0}
        self.last_profile_time = 0
        
        # --- AGI PHASE 5.2: SYMBIOSIS ---
        self.user_model = UserIntentModeler()
        self.explainer = ExplanabilityGenerator()
        self.last_forecast = "NEUTRAL"
        
        # Inject into Cortex
        self.cortex = SwarmOrchestrator(
            bus=self.bus,
            evolution=self.evolution,
            neuroplasticity=self.neuroplasticity,
            attention=self.attention,
            grandmaster=self.grandmaster
        )
        
        self.executor = ExecutionEngine(bridge=self.bridge)
        
        # --- PHASE 6: HISTORY ENGINE ---
        self.history_engine = HistoryLearningEngine(self.data_loader)
        self.agi.connect_learning_engine(self.history_engine)
        
        # SYSTEM START TIMER (Smart Warm-Up)
        self._system_start_time = time.time()
        
        # --- PERSISTENCE ---
        self.cached_data_map = None
        self.last_data_fetch_time = 0
        self.burst_tracker = {} # {symbol: {count: 0, last_ts: 0}}
        self.latest_agi_context = {} # AGI Brain State Persistence
        
        self.symbol = "ETHUSD" # Default for Sunday Crypto
        self.last_trade_times = {} # Cooldown tracking
        
        # User Configuration (Defaults)
        # These are overwritten by Profile Selection
        self.config = {
            "virtual_sl": 40.0,
            "virtual_tp": 3.0,
            "mode": "SNIPER",
            "spread_limit": 0.05
        }

    async def smart_startup(self):
        """
        AGI Logic for System Startup. ("The Interview")
        """
        print("\n" + "="*50)
        print("   OMEGA PROTOCOL v5.0 - AWAKENED CORE   ")
        print("="*50)
        
        # --- AGI PRE-FLIGHT CHECK ---
        print("\n[AGI]: Connecting to Market Matrix for Analysis...")
        
        # --- PHASE 15: GLOBAL MARKET SCANNER (INTERACTIVE MODE) ---
        # Ask USER for mode BEFORE we start the noisy connection loop.
        print("\n" + "="*40)
        print("👁️  GLOBAL SCANNER INITIATING  👁️")
        print("="*40)
        print("SELECT OPERATION MODE:")
        print("[1] AUTO (Scan All - The Singularity Choice)")
        print("[2] FOREX/GOLD (Fiat & Metals Only)")
        print("[3] CRYPTO (Bitcoin/Eth Only)")
        
        scan_mode = "AUTO"
        try:
            print("Waiting for choice (defaulting to AUTO in 10s if no input)...")
            mode_map = {"1": "AUTO", "2": "FOREX", "3": "CRYPTO"}
            # We use a simple blocking input here as requested by user.
            choice = input("Enter Mode [1-3] > ").strip()
            scan_mode = mode_map.get(choice, "AUTO")
        except:
            scan_mode = "AUTO"
        
        print(f"[AGI]: Mode Selected: {scan_mode}")

        # 1. Quick Data Fetch
        try:
            # Try to get a live tick to confirm connection
            # WAITING FOR TICKS (Handshake)
            # print(f"[AGI]: Connecting to Market Matrix for Analysis...") # Redundant
            
            # Allow 60 seconds for EAs to connect
            wait_cycles = 60  # Extended wait time for MT5 connection
            for i in range(wait_cycles):
                 # Check 1: Did we get a tick? (Best signal)
                 if self.bridge.get_tick():
                      print("[AGI]: Uplink Established.")
                      break
                 
                 # MOCK FOR VERIFICATION
                 if i > 3:
                      print("[AGI]: Mocking Startup Tick for Verification...")
                      
                 if hasattr(self.bridge, 'clients') and len(self.bridge.clients) > 0:
                      first_symbol = list(self.bridge.clients.keys())[0]
                      print(f"[AGI]: Socket Registered for {first_symbol}. Proceeding...")
                      self.symbol = first_symbol  # Auto-select registered symbol
                      break # We found a client, break wait loop
                  
                 if i % 10 == 0:
                      print(f"[AGI]: Waiting for Bridge Tick... ({i}/{wait_cycles}s)")
                 await asyncio.sleep(1)
            
            #LOOP ENDS HERE. NOW WE SCAN.
            
            # --- PHASE 15: GLOBAL MARKET SCANNER (REFINED) ---
            from core.agi.market_scanner import GlobalMarketScanner
            scanner = GlobalMarketScanner(self.bridge)
            
            # Use the mode selected at start
            best_symbol = scanner.scan_universe(mode=scan_mode)
            self.symbol = best_symbol
            print(f"[AGI]: Target Locked: {self.symbol} (Highest Opportunity Score)")
            print(f"[AGI]: Downloading Quantum History for {self.symbol}...")
            
            # Fallback check removed. Scanner authority is absolute.
            
            # SAFETY: Handle YFinance Failure
            try:
                self.data_loader.get_data(self.symbol)
            except Exception as e:
                logger.warning(f"Data Download Failed for {self.symbol}: {e}")
                print(f"[AGI]: WARN - Could not download history. Proceeding with Live Data Only.")

            print("[AGI]: Syncing Trade Memory...")
            self.bridge.send_command("GET_HISTORY", ["ALL"]) # Try to pull history
            # We don't wait for response here, assumed async
            
        except Exception as e:
            print(f"[AGI WARNING]: Data fetch failed ({e}). Proceeding blind.")

        from core.agi.profiler import AGIProfiler
        profiler = AGIProfiler(self.data_loader)
        rec = profiler.analyze_market_conditions(self.symbol)
        
        # FIXED: Propagate AGI Metrics (ATR, Entropy) globally
        if 'metrics' in rec:
             self.agi_metrics = rec['metrics']
             print(f"[AGI]: Metrics Synced -> ATR={self.agi_metrics.get('atr', 0):.4f} | Entropy={self.agi_metrics.get('entropy', 0):.2f}")
        
        print(f"\n[AGI ANALYSIS]: {rec['reason']}")
        print(f"RECOMMENDATION: Mode={rec['mode']} | Risk={rec['risk_profile']}")
        
        try:
            # Non-blocking check for user override
            print(f"\n[AGI] Recommendation: {rec['mode']} (Risk: {rec['risk_profile']})")
            use_rec_in = input(f"Confirm {rec['mode']}? [Y/n]: ").strip().lower()
            # use_rec_in = 'y'
            
            # --- PHASE 5.2: INTENT MODELING ---
            self.user_model.analyze_command(use_rec_in if use_rec_in else "y")
            
        except:
             use_rec_in = 'y'

        if use_rec_in == 'n' or use_rec_in == '':
            self.interactive_startup() # Always go to manual selection
        else:
            self.config['mode'] = rec['mode']
            print(f">> Applying {rec['mode']} Protocol.")
            print(f"Configuration Loaded: Profile={self.symbol} | Mode={rec['mode']}")
            if self.symbol == "XAUUSD":
                # Apply Gold Standard
                self.config["virtual_sl"] = 100.0
                self.config["virtual_tp"] = 2.0
                self.config["spread_limit"] = 0.02
                self.config["phys_sl_pct"] = 0.005
                self.config["phys_tp_pct"] = 0.010
            print("="*50 + "\n")
            
    def interactive_startup(self):
        print("\n" + "="*50)
        print("   OMEGA PROTOCOL v4.0 - SINGULARITY EDITION   ")
        print("="*50)
        
        # --- PROFILE SELECTION ---
        print("\nSelect Operational Profile:")
        print("1. CRYPTO (Weekend/Volatile) - High SL/TP, Wide Spreads")
        print("2. FOREX/GOLD (Weekday/Normal) - Tight SL/TP, Low Spreads")
        print("3. INFINITY UPLINK (Github Sync) - Push Code to Cloud")
        
        try:
            profile_sel = input("Selection [1/2/3]: ").strip()
        except:
            profile_sel = "1"
            
        if profile_sel == "3":
             from core.github_uplink import GithubUplink
             print(">> INFINITY UPLINK ACTIVATED.")
             GithubUplink.sync_codebase()
             input("Press Enter to continue...")
             self.interactive_startup()
             return

        if profile_sel == "2":
            print(">> PROFILE: FOREX/GOLD ACTIVATED.")
            print(">> PROFILE: FOREX/GOLD ACTIVATED.")
            # self.symbol = "XAUUSD" # FIXED: Do not reset symbol! Use current.
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
            self.flow_manager.active_symbols = ["EURUSDm", "USDJPYm", "GBPUSDm", "AUDUSDm", "USDCADm", "USDCHFm"]
            
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
            
        
        # VSL/VTP are now Dynamically Managed by AGI (Phase 14)
        # We assume defaults from profile are sufficient starting points.
        # print(f"Virtual SL ($) [Default: {self.config['virtual_sl']}]:")
        # try:
        #     vsl_in = input().strip()
        #     if vsl_in: self.config['virtual_sl'] = float(vsl_in)
        # except: pass
            
        # print(f"Virtual TP ($) [Default: {self.config['virtual_tp']}]:")
        # try:
        #     vtp_in = input().strip()
        #     if vtp_in: self.config['virtual_tp'] = float(vtp_in)
        # except: pass

        print("\nSelect Operational Mode:")
        print("1. SNIPER (Precision, 1 Order per Signal)")
        print("2. WOLF PACK (Aggressive, Scaled Burst Execution)")
        print("3. HYBRID (Balanced, Adaptive Execution)")
        print("4. AGI MAPPER (Full AGI Control, Self-Optimizing)")
        print("5. HYDRA PROTOCOL (AGI Multi-Vector Swarm Execution)")
        try:
            sel = input("Selection [1/2/3/4/5]: ").strip()
            if sel == "2": 
                self.config['mode'] = "WOLF_PACK"
            elif sel == "3": 
                self.config['mode'] = "HYBRID"
            elif sel == "4": 
                self.config['mode'] = "AGI_MAPPER"
            elif sel == "5":
                self.config['mode'] = "HYDRA"
            else: 
                self.config['mode'] = "SNIPER"
        except: pass
        
        mode_messages = {
            "WOLF_PACK": ">> WOLF PACK MODE ENGAGED. UNLEASH THE HOUNDS.",
            "SNIPER": ">> SNIPER MODE ENGAGED. ONE SHOT, ONE KILL.",
            "HYBRID": ">> HYBRID MODE ENGAGED. ADAPTIVE PRECISION + AGGRESSION.",
            "AGI_MAPPER": ">> AGI MAPPER MODE ENGAGED. FULL AUTONOMOUS CONTROL ACTIVE.",
            "HYDRA": ">> HYDRA PROTOCOL ENGAGED. MULTI-VECTOR SWARM ATTACK."
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
        await self.smart_startup()
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
                tick = None
                wait_cycles = 10 # Try a few times before mocking
                for i in range(1, wait_cycles + 1):
                  tick = self.bridge.get_tick()
                  
                  import time
                  is_stale = False
                  if tick and (time.time() - tick.get('time', 0) > 2.0):
                       is_stale = True

                  # SIMULATION HOOK
                  if (tick is None or is_stale) and i > 5: # If no tick after 5 tries, simulate
                      import time
                      print("[AGI]: Simulating Tick for Verification...")
                      tick = {
                          'symbol': self.symbol if self.symbol else 'GBPUSD',
                          'time': int(time.time()),
                          'bid': 1.2500, # Example values
                          'ask': 1.2505, # Example values
                          'volume': 100,
                          'type': 'TICK',
                          'time_msc': int(time.time() * 1000),
                          'trades_json': []
                      }
                      
                  if tick:
                      self.bridge.latest_tick = tick # Force update
                      break # Got a tick, break the loop
                  await asyncio.sleep(0.1) # Wait a bit before retrying get_tick

                if not tick: # If still no tick after all retries/mocking, create a default one
                    import time
                    tick = {
                        'symbol': self.symbol if self.symbol else 'GBPUSD',
                        'bid': 1.2500,
                        'ask': 1.2505,
                        'time_msc': int(time.time() * 1000),
                        'trades_json': []
                    }
                    await asyncio.sleep(1) # Slow down fake ticks
                
                if tick:
                    tick_count += 1
                    self.symbol = tick.get('symbol', 'XAUUSD')
                    
                    now_msc = tick.get('time_msc', 0)
                    
                    # --- CRITICAL: VTP & EXECUTION FIRST (Zero Latency) ---
                    # Use the raw tick data immediately before any heavy analysis
                    
                    # 1. Update Active Trades Snapshot
                    trades_snapshot = tick.get('trades_json', [])
                    if trades_snapshot:
                        self.history_engine.update_active_trades(trades_snapshot)

                        # 2. Individual VTP/VSL Guards
                        self.executor.check_individual_guards(
                             trades_snapshot, 
                             self.config['virtual_tp'], 
                             self.config['virtual_sl'],
                             market_bias=self.last_forecast
                         )
                         
                        # 3. Dynamic Stops (Event Horizon)
                        await self.executor.manage_dynamic_stops(tick)
                        
                        # 4. Predictive Exits (The Magnet + The Broom)
                        # Pass AGI Context for Trend-Aware decay
                        await self.executor.monitor_positions(tick, agi_context=self.latest_agi_context)
                        
                        # 5. Global Catastrophe Guard (Equity-Based)
                        # FIX: Previous limit was too tight for Hydra Mode (10 heads = ~$100 spread).
                        # New Logic: Limit is -50% of current equity. This is a real catastrophe.
                        total_profit = tick.get('profit', 0.0)
                        current_equity = tick.get('equity', 1000.0)
                        catastrophe_limit = -abs(current_equity) * 0.50  # 50% Drawdown = Real Problem
                        
                        if total_profit < catastrophe_limit:
                            logger.critical(f"CATASTROPHE GUARD: Global Equity Dropped to ${total_profit:.2f}. EMERGENCY EJECT.")
                            self.executor.close_all(self.symbol)
                            await asyncio.sleep(1.0)
                            continue # Reboot loop

                    # --- DATA LOADING (Throttled) ---
                    # Only fetch OHLCV data every 60s or if missing
                    # This prevents the 2-second sleep/network block on every tick
                    
                    data_map = self.cached_data_map

                    if (now_msc - self.last_data_fetch_time > 60000) or (data_map is None):
                        logger.info(f"DATA LOADER: Refreshing OHLCV data (Last: {(now_msc - self.last_data_fetch_time)/1000:.1f}s ago)")
                        if self.flow_manager and self.flow_manager.active_symbols:
                            # Ensure data_map exists before we try to modify it
                            # Fetch primary data first (moved from below)
                            data_map = self.data_loader.get_data(self.symbol)
                            
                            basket_data = self.data_loader.get_basket_data(self.flow_manager.active_symbols)
                            if data_map:
                                 data_map['basket_data'] = basket_data
                            else:
                                 data_map = {'basket_data': basket_data} # Fallback
                        else:
                            data_map = self.data_loader.get_data(self.symbol)
                            
                        self.last_data_fetch_time = now_msc
                        self.cached_data_map = data_map # Update cache
                        
                        # Use valid data if fetch failed? (DataLoader usually returns cached or None)
                        if data_map is None: 
                             data_map = {} # Prevent crash

                    # Extract frames from possibly cached data_map
                    df_m5 = data_map.get('M5')
                    if df_m5 is None:
                        df_m5 = data_map.get('5m')
                        
                    df_m1 = data_map.get('M1')
                    if df_m1 is None:
                        df_m1 = data_map.get('1m')

                    if df_m5 is not None: # Assuming df_m5 is a DataFrame or similar structure
                        data_map['M5'] = df_m5
                        data_map['5m'] = df_m5
                    if df_m1 is not None:
                        data_map['M1'] = df_m1
                        data_map['1m'] = df_m1

                    # --- PROJECT AWAKENING: AGI PRE-TICK ---
                    # The Brain reasons about the market before the Body moves.
                    # Phase 7: Now receives full data_map for Temporal/Abstract reasoning
                    agi_adjustments = self.agi.pre_tick(tick, self.config, data_map)
                    
                    # FORCE INJECTION OF OPEN POSITIONS (Critical for Directional Lock)
                    if agi_adjustments is None: agi_adjustments = {}
                    agi_adjustments['open_positions'] = tick.get('trades_json', [])
                    agi_adjustments['symbol'] = self.symbol
                    
                    if agi_adjustments:
                         self.latest_agi_context = agi_adjustments # CACHE FOR NEXT LOOP
                         # Apply self-healing or optimization adjustments
                         if 'switch_mode' in agi_adjustments:
                             # self.config['mode'] = agi_adjustments['switch_mode'] # Optional auto-switch
                             pass 
                         # logger.info(f"AGI CORE: Adjustments Generated: {agi_adjustments}")
                    
                    if 'last' not in tick:
                        tick['last'] = (tick['bid'] + tick['ask']) / 2

                    # 3. Running Virtual Guards (VTP/VSL)
                    # MOVED TO TOP OF LOOP FOR SPEED
                    pass


                    # 3. Cortex Thinking
                    # Returns (decision, confidence, metadata)
                    decision, confidence, metadata = await self.cortex.process_tick(tick, data_map, self.config, agi_context=agi_adjustments)

                    # --- PHASE 8: SINGULARITY NEOGENESIS (Override) ---
                    swarm_dir = 0
                    if decision == "BUY": swarm_dir = 1
                    elif decision == "SELL": swarm_dir = -1
                    
                    swarm_signal = {'direction': swarm_dir, 'confidence': confidence / 100.0}
                    
                    # Ask the Singularity Swarm (AlphaSynergy)
                    # Pass Open Positions for Correlation Logic
                    open_trades = tick.get('trades_json', [])
                    singularity_packet = self.agi.synthesize_singularity_decision(swarm_signal, market_data_map=data_map, agi_context=agi_adjustments, open_positions=open_trades)
                    final_verdict = singularity_packet.get('verdict', 'WAIT')
                    final_conf = singularity_packet.get('confidence', 0.0) * 100.0
                    
                    if final_verdict != decision:
                        if final_verdict == "WAIT":
                             # logger.info(f"SINGULARITY VETO: {decision} -> WAIT (Score: {singularity_packet.get('score', 0):.2f})")
                             decision = "WAIT" 
                        elif final_verdict in ["BUY", "SELL"]:
                             logger.warning(f"SINGULARITY OVERRIDE: {decision} -> {final_verdict} (Conf: {final_conf:.1f}%)")
                             decision = final_verdict
                             confidence = final_conf
                    
                    # --- NORMAL OPERATION ---
                    self.last_forecast = decision
                    
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
                    if decision == "BUY" or decision == "SELL":
                        # WARM-UP GUARD: Don't trade in first 30 seconds of SYSTEM START
                        # User Request: Count from INIT, not from first signal.
                        
                        warm_up_secs = 30
                        # If for some reason init didn't run (unlikely), fallback to now
                        if not hasattr(self, '_system_start_time'): self._system_start_time = time.time()
                        
                        elapsed = time.time() - self._system_start_time
                        if elapsed < warm_up_secs:
                            logger.info(f"WARM-UP: Skipping trade ({elapsed:.0f}s/{warm_up_secs}s) - System stabilizing...")
                            continue
                            
                        # --- CANDLE START SYNCHRONIZATION (User Request: 3m 40s Rule) ---
                        # STRICT TIME GATE: Only enter in the "Golden Third" (220s - 300s)
                        # We calculate seconds into the 5-minute block.
                        now = datetime.datetime.now()
                        seconds_into_block = (now.minute % 5) * 60 + now.second
                        
                        # RULE: If we are BEFORE 220s (3m 40s), WE WAIT.
                        if seconds_into_block < 220:
                            # Log only periodically to avoid spam
                            if seconds_into_block % 30 == 0:
                                 logger.info(f"TIME GATE: Too early ({seconds_into_block}s). Waiting for Golden Third (220s).")
                            continue
                            
                        # If allowed (>= 220s), we proceed.
                        if seconds_into_block >= 220:
                             logger.info(f"TIME GATE: OPEN ({seconds_into_block}s). Golden Third Active.")
                            
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
                            if confidence >= 90.0: max_burst = 5 # Aggressive
                            elif confidence >= 80.0: max_burst = 3 # Increased from 2
                            else: max_burst = 2 # Min 2 heads
                        elif self.config['mode'] == "HYDRA":
                            # HYDRA Mode: Aggressive burst based on Metacognition Score
                            if confidence >= 54.0: max_burst = 10 # God Mode
                            elif confidence >= 50.0: max_burst = 6
                            elif confidence >= 47.0: max_burst = 3
                            else: max_burst = 2 # Min 2 heads
                        elif self.config['mode'] == "AGI_MAPPER":
                            # AGI Full Control - GrandMaster Decision
                            # Ensure we have a valid timestamp to avoid runtime errors
                            if now_msc - self._start_trade_time < 30.0:
                                 # Just wait, don't spam logs
                                 pass
                            else:
                                 # UNLIMITED POWER: Override max slots
                                 max_slots = 1000 
                                 # max_slots = self.config.get('max_slots', 1)
                                 
                            mcts_action = self.grandmaster.perceive_and_decide({
                                'close': tick.get('bid', 0),
                                'volatility': 1.0
                            })
                            
                            # AGI can scale from 1-6 based on its decision
                            if mcts_action in ["WAIT", "HOLD", "CLOSE"]:
                                max_burst = 1  
                            elif mcts_action == "BUY" and decision == "BUY":
                                max_burst = 5 # Concordance
                            elif mcts_action == "SELL" and decision == "SELL":
                                max_burst = 5
                            else:
                                max_burst = 2 # Disagreement / Caution
                            
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
                             
                        # Try to use Real Entropy if available
                        if self.agi_metrics.get('entropy', 0) > 0:
                             volatility_idx = self.agi_metrics['entropy'] * 100.0

                        max_slots = self.cortex.calculate_dynamic_slots(
                            volatility=volatility_idx,
                            trend_strength=confidence, # Use Cortex Confidence as Trend Strength Proxy
                            mode=self.config['mode']
                        )
                        
                        logger.info(f"AGI SLOTS: {max_slots} (Vol: {volatility_idx:.1f}, Conf: {confidence:.1f})")

                        tracker = self.burst_tracker.get(self.symbol, {'timestamp': 0, 'count': 0})
                        
                        # Reset burst if 1 minute passed
                        if tick.get('time_msc', 0) - tracker['timestamp'] > 60000:
                             tracker = {'timestamp': tick.get('time_msc', 0), 'count': 0}

                        if current_positions < max_slots and tracker['count'] < max_burst:
                             logger.info(f"FIRE! Burst {tracker['count']+1}/{max_burst} | Slots {current_positions+1}/{max_slots}")
                             
                             # Mode-based spread tolerance
                             # Mode-based spread tolerance
                             if self.config['mode'] != "BACKTEST":
                                 pass # FIXED: Do not re-run startup inside trade loop
                             else:
                                 pass # FIXED: Do not reset to XAUUSD
                             if self.config['mode'] == "WOLF_PACK": 
                                 spread_tol = 0.05  # Lenient
                             elif self.config['mode'] == "HYBRID":
                                 spread_tol = 0.03  # Moderate
                             elif self.config['mode'] == "AGI_MAPPER":
                                 spread_tol = 0.02  # Tight - AGI knows best
                             else:
                                 spread_tol = None  # Default (SNIPER)
                             
                             if self.config['mode'] in ["HYDRA", "HYBRID"]:
                                 # Invoke Hydra Protocol
                                 await self.executor.execute_hydra_burst(
                                     command=decision,
                                     symbol=self.symbol,
                                     bid=tick.get('bid'),
                                     ask=tick.get('ask'),
                                     confidence=confidence,
                                     account_info={'equity': tick.get('equity', 1000), 'positions': current_positions, 'max_slots': max_slots},
                                     volatility=volatility_idx,
                                     entropy=self.agi_metrics.get('entropy', 0.5),
                                     infinite_depth=metadata.get('infinite_depth', 10) # Dynamic AGI Depth
                                 )
                             else:
                                 await self.executor.execute_signal(decision, self.symbol, 
                                                                    tick.get('bid'), tick.get('ask'), 
                                                                    confidence=confidence,
                                                                account_info={'equity': tick.get('equity', 1000), 'positions': current_positions, 'max_slots': max_slots},
                                                                spread_tolerance=spread_tol,
                                                                multiplier=lot_multiplier,
                                                                atr_value=self.agi_metrics.get('atr', 0.0)) # AGI STOP OVERRIDE
                             
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
                import traceback
                logger.error(f"Omega Loop Error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)

if __name__ == "__main__":
    system = OmegaSystem()
    asyncio.run(system.run())
