
import time
import logging
import pandas as pd
from datetime import datetime
import pytz
import config
import json
import zmq
import threading
import MetaTrader5 as mt5

# Core Modules
from data_loader import DataLoader
from bridge import ZmqBridge
from report_generator import ReportGenerator

# Old Analysis (Legacy Support)
from analysis.consensus import ConsensusEngine
from analysis.consensus import ConsensusEngine
from risk_manager import RiskManager # Using New Quantum Engine

# NEW Advanced Modules
from src.notifications import NotificationManager
from src.mt5_monitor import MT5Monitor
from analysis.smart_money import SmartMoneyEngine
from analysis.deep_cognition import DeepCognition
from analysis.hyper_dimension import HyperDimension
from analysis.scalper_swarm import ScalpSwarm
from analysis.second_eye import SecondEye
from analysis.fourth_eye import FourthEye
from analysis.tenth_eye import TenthEye
from analysis.eleventh_eye import EleventhEye
from analysis.trade_manager import TradeManager

# Setup Logging
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG for Swarm Analysis
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("atl4s.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Atl4s-Main")

def check_market_status():
    """Checks if the market is likely closed (Weekends)."""
    now = datetime.now()
    # 5 = Saturday, 6 = Sunday
    if now.weekday() >= 5:
        return False
    return True

def main():
    logger.info("--- Atl4s-Forex System 2.0: Deep Awakening ---")
    bot_start_time = time.time() # Startup Timer for Warmup Protocol
    
    if not check_market_status():
        print("\n" + "!"*50)
        print("WARNING: MARKET IS CLOSED (WEEKEND)")
        print("The bot will wait for ticks, but no new data is expected.")
        print("To test parameters, run the simulation system:")
        print(">>> python simulation_system.py --days 15")
        print("!"*50 + "\n")
        time.sleep(5)
    
    # --- v2.0 INTERACTIVE STARTUP ---
    print("\n" + "="*50)
    print("      Atl4s-Forex v2.0 | SYSTEM STARTUP")
    print("="*50)
    
    # 1. Capital Confirmation
    print("\n[RISK MANAGER] Dynamic Lot Sizing is ACTIVE.")
    print("Base Logic: $30 Equity -> 0.02 Lots (Linear Scaling)")
    ui_capital = input("Enter INITIAL CAPITAL to validate logic (or Press Enter to Auto-Detect): ")
    if ui_capital.strip():
        try:
            val = float(ui_capital)
            print(f"Verified: ${val:.2f} -> {val/30*0.02:.2f} Lots (Approx)")
        except:
            print("Invalid input. Using Auto-Detect.")
            
    # 2. Port Configuration
    ui_port = input("\nEnter NETWORK PORT (Default: 5555): ")
    zmq_port = 5555
    if ui_port.strip():
        try:
            zmq_port = int(ui_port)
        except:
            print("Invalid Port. Using Default 5555.")
    
    print(f"\n>>> SYSTEM INITIALIZING ON PORT {zmq_port}...")
    print("="*50 + "\n")
    time.sleep(2)

    # 1. Initialize Infrastructure
    try:
        bridge = ZmqBridge(port=zmq_port)
    except Exception as e:
        logger.critical(f"Failed to start Bridge Server: {e}")
        return

    # Initialize Managers
    notif_manager = NotificationManager(cooldown_minutes=5)
    # Initialize MT5 Monitor
    mt5_monitor = MT5Monitor()
    
    # Get Dynamic Symbol Info (Digits/Point)
    symbol_info = mt5_monitor.get_symbol_info(config.SYMBOL)
    if not symbol_info:
        logger.error(f"CRITICAL: Could not fetch info for {config.SYMBOL}. Using defaults (Digits=2).")
        symbol_digits = 2
        symbol_point = 0.01
    else:
        symbol_digits = symbol_info['digits']
        symbol_point = symbol_info['point']
        symbol_stops_level = symbol_info['stops_level']
        logger.info(f"Symbol {config.SYMBOL}: Digits={symbol_digits}, Point={symbol_point}, StopsLevel={symbol_stops_level}")

    def normalize_price(price):
        """Rounds price to the correct number of digits."""
        return f"{round(price, symbol_digits):.{symbol_digits}f}"

    def validate_sl_tp(current_bid, current_ask, sl, tp, cmd_type):
        """
        Enforces minimum StopsLevel distance relative to the correct price anchor (Bid/Ask).
        Returns (valid_sl, valid_tp) adjusted if necessary.
        """
        # Hard Minimum Fallback: 200 points ($2.00 on Gold) - Safer for scalping but valid
        # Error 10016 = Stops too close. 
        effective_stops_level = max(symbol_stops_level, 200) 
        min_dist = effective_stops_level * symbol_point
        
        # Buffer: Add 50 points ($0.50) breathing room
        safe_dist = min_dist + (50 * symbol_point) 
        
        # LOGGING FOR DEBUGGING
        # logger.debug(f"Validate SL/TP: Bid={current_bid}, Ask={current_ask}, SL={sl}, TP={tp}, Cmd={cmd_type}, SafeDist={safe_dist}")

        if cmd_type == "BUY":
            # BUY Order:
            # SL must be below Bid: (Bid - SL) >= safe_dist  =>  SL <= Bid - safe_dist
            # TP must be above Ask (Exec at Ask, Target Price > Ask) -> TP >= Ask + safe_dist
            
            # 1. Validate SL (Anchor: BID)
            max_sl = current_bid - safe_dist
            if sl > max_sl:
                # logger.warning(f"BUY SL too close to Bid ({sl} > {max_sl}). Pushing down.")
                sl = max_sl
                
            # 2. Validate TP (Anchor: ASK)
            min_tp = current_ask + safe_dist
            if tp < min_tp:
                # logger.warning(f"BUY TP too close to Ask ({tp} < {min_tp}). Pushing up.")
                tp = min_tp

        else: # SELL
            # SELL Order:
            # SL must be above Ask: (SL - Ask) >= safe_dist  =>  SL >= Ask + safe_dist
            # TP must be below Bid (Exec at Bid, Target Price < Bid) -> TP <= Bid - safe_dist
            
            # 1. Validate SL (Anchor: ASK)
            min_sl = current_ask + safe_dist
            if sl < min_sl:
                # logger.warning(f"SELL SL too close to Ask ({sl} < {min_sl}). Pushing up.")
                sl = min_sl
                
            # 2. Validate TP (Anchor: BID)
            max_tp = current_bid - safe_dist
            if tp > max_tp:
                # logger.warning(f"SELL TP too close to Bid ({tp} > {max_tp}). Pushing down.")
                tp = max_tp
                 
        return sl, tp

    # Load Historical Data
    logger.info("Loading Historical Data...")
    data_loader = DataLoader() # Historical Data
    reporter = ReportGenerator()
    
    # Initialize Minds (Analysis Engines)
    consensus = ConsensusEngine() # Old Logic
    risk_manager = RiskManager() # Quantum Sigmoid Risk Engine
    
    # New Engines
    smc_engine = SmartMoneyEngine()
    deep_brain = DeepCognition()
    third_eye = HyperDimension()
    swarm = ScalpSwarm()
    sniper = SecondEye()
    whale = FourthEye() # The Consensus Commander
    tenth_eye = TenthEye() # The Holographic Architect
    eleventh_eye = EleventhEye() # The Gap Exploiter
    trade_manager = TradeManager() # For Active Management

    # Dashboard Publisher
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5558")
    
    # Load Initial Data
    data_map = data_loader.get_data()
    df_m5 = data_map.get('M5')
    if df_m5 is None:
         logger.warning("No M5 data found. Waiting for live ticks...")
         df_m5 = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    logger.info("Systems Online. Monitoring 4th Dimension...")
    
    # Initialize variables
    last_candle_minute = -1
    last_log_print = 0
    last_analysis_time = 0
    last_candle_minute = -1
    last_log_print = 0
    last_analysis_time = 0
    analysis_cooldown = 0.1 # FAST: 10x faster reaction time (User complained about latency)
    
    # Pre-initialize metrics for Swarm safety
    final_cortex_decision = 0
    base_score = 0
    original_base_score = 0
    orbit_energy = 0
    micro_stats = {}
    details = {}
    phy_state = "INITIALIZING"
    future_prob = 0.5
    lot_multiplier = 1.0
    
    sp_tz = pytz.timezone('America/Sao_Paulo')
    
    # Configure Notification Manager (Zero cooldown, we control timing here)
    notif_manager.cooldown = 0 
    
    def refresh_data_async(loader, data_container):
        """Helper to load data in background without blocking execution"""
        try:
             new_data = loader.get_data()
             if new_data:
                 # CRITICAL FIX: Do NOT overwrite 'M5' with YFinance data.
                 # YFinance is delayed (15m+). Our local 'M5' built from ticks is real-time.
                 # Overwriting it would lobotomize the bot.
                 for key, df in new_data.items():
                     if key == 'M5': continue # Skip M5
                     data_container[key] = df
                     
                 logger.info("High-Timeframe data refreshed (Background Thread). M5 preserved.")
        except Exception as e:
             logger.error(f"Async Refresh Error: {e}")

    try:
        while True:
            # 1. Connectivity & Data Ingestion
            if not bridge.conn:
                if time.time() - last_log_print > 5:
                    logger.info("Waiting for MT5 Connection via ZMQ...")
                    last_log_print = time.time()
                time.sleep(1)
                continue
                
            live_tick = bridge.get_tick()

            if live_tick is None:
                 time.sleep(0.01)
                 continue # Skip loop if no tick data
                 
            # --- ACTIVE POSITION MANAGEMENT (High Priority: <10ms latency) ---
            # Moves 0: Monitor & Close BEFORE Thinking or Analyzing
            
            # --- STARTUP WARMUP PROTOCOL ---
            is_warming_up = False
            if time.time() - bot_start_time < 120: # 2 Minute Warmup
                 is_warming_up = True
            
            # This ensures we hit Virtual TPs immediately upon tick arrival.
            open_positions = mt5_monitor.get_open_positions() 
            if open_positions:
                 # Ensure we have data for trailing
                 atr = df_m5.iloc[-1]['ATR'] if not df_m5.empty and 'ATR' in df_m5.columns else 1.0
                 struc_low = df_m5.iloc[-20:]['low'].min() if not df_m5.empty else 0
                 struc_high = df_m5.iloc[-20:]['high'].max() if not df_m5.empty else 0

                 for pos in open_positions:
                     pos_dict = pos._asdict() if hasattr(pos, '_asdict') else pos
                     
                     # 1. Hard Exit (Virtual TP/SL) - PRIORITY 1: CLOSE INSTANTLY
                     # Logic: Check PnL -> Close via Direct API -> Fallback to Bridge
                     exit_signal = trade_manager.check_hard_exit(pos_dict, config.SCALP_TP, config.SCALP_SL)
                     if exit_signal and exit_signal['action'] == "CLOSE_FULL":
                         ticket = exit_signal['ticket']
                         reason = exit_signal['reason']
                         logger.info(f"[!] INSTANT EXIT: {reason} | Closing Ticket {ticket}...")
                         
                         # PRIMARY: Direct Python API Close (Fastest: Bypass Socket/Bridge)
                         if mt5_monitor.close_position(ticket):
                             logger.info(f"Direct Python Close confirmed for Ticket {ticket}.")
                         else:
                             # FALLBACK: Send Bridge Command (Slower)
                             logger.warning(f"Direct Python Close failed. Sending Bridge Command fallback...")
                             bridge.send_command("CLOSE_TRADE", [str(ticket)])
                             
                         notif_manager.send_notification("INSTANT PROFIT", f"{reason}", "PROFIT")
                         continue # Skip other checks for this position if closed

                     # 2. Micro-Stop (Instant Regret) - NEW
                     if config.ENABLE_FIRST_EYE: # Assuming we use Deep Brain score which is mapped to 'final_cortex_decision'
                          abort_signal = trade_manager.check_early_abort(
                              pos_dict, 
                              final_cortex_decision, 
                              micro_stats.get('velocity', 0), 
                              time.time()
                          )
                          if abort_signal:
                               ticket = abort_signal['ticket']
                               reason = abort_signal['reason']
                               logger.warning(f"[!] MICRO-STOP: {reason} | Closing Ticket {ticket}...")
                               mt5_monitor.close_position(ticket)
                               notif_manager.send_notification("MICRO-STOP", f"{reason}", "PROTECT")
                               continue
                     
                     # 3. Trailing Stop
                     new_sl = trade_manager.check_trailing_stop(pos_dict, live_tick['last'], struc_low, struc_high)
                     if new_sl:
                         bridge.send_command("MODIFY_TRADE", [str(pos_dict['ticket']), normalize_price(new_sl), str(pos_dict['tp'])])
                     
                     # 3. Partial TP
                     partial_action = trade_manager.check_partial_tp(pos_dict, live_tick['last'])
                     if partial_action:
                         bridge.send_command("CLOSE_PARTIAL", [str(pos_dict['ticket']), str(partial_action['volume'])])
                         notif_manager.send_notification("PARTIAL PROFIT", f"Ticket {pos_dict['ticket']} closed {partial_action['volume']} lots", "PROFIT")
                 
            # Smart Time Check (Sao Paulo)
            now_sp = datetime.now(sp_tz)
            current_minute = now_sp.minute
            
            # Check for 5-Minute Trigger
            is_report_time = False
            if current_minute % 5 == 0 and current_minute != last_candle_minute:
                is_report_time = True
                last_candle_minute = current_minute
                logger.info(f"--- 5-MINUTE CYCLE TRIGGER: {now_sp.strftime('%H:%M:%S')} ---")
                
                try:
                    # Async Refresh (Non-Blocking) allows Whale to see the new candle immediately
                    # instead of waiting 2-3s for YFinance.
                    t = threading.Thread(target=refresh_data_async, args=(data_loader, data_map))
                    t.daemon = True # Ensure it doesn't hang the script on exit
                    t.start()
                    
                    # Note: We do NOT update df_m5 here anymore. M5 is local-only.
                    
                    
                except Exception as e:
                    logger.error(f"Failed to trigger async refresh: {e}")

            if live_tick.get('last', 0) > 0:
                # Update Data Logic
                if not df_m5.empty:
                    last_time = df_m5.index[-1]
                else:
                    last_time = pd.Timestamp.now(tz=config.TIMEZONE)

                tick_time = pd.to_datetime(live_tick['time'], unit='ms')
                
                # Timezone Handling
                if not df_m5.empty and df_m5.index.tz is not None and tick_time.tz is None:
                    tick_time = tick_time.tz_localize(df_m5.index.tz)
                elif not df_m5.empty and df_m5.index.tz is None and tick_time.tz is not None:
                    tick_time = tick_time.tz_localize(None)

                # Candle Management (Standard M5 construction)
                if df_m5.empty or tick_time >= (last_time + pd.Timedelta(minutes=5)):
                    new_row = pd.DataFrame({
                        'open': [live_tick['last']],
                        'high': [live_tick['last']],
                        'low': [live_tick['last']],
                        'close': [live_tick['last']],
                        'volume': [live_tick['volume']]
                    }, index=[tick_time])
                    df_m5 = pd.concat([df_m5, new_row])
                    data_map['M5'] = df_m5
                    # NOTE: We rely on the Clock Trigger (is_report_time) for notifications, 
                    # not just the new candle event, to ensure alignment even if no ticks come immediately.
                else:
                    df_m5.iloc[-1, df_m5.columns.get_loc('close')] = live_tick['last']
                    df_m5.iloc[-1, df_m5.columns.get_loc('high')] = max(df_m5.iloc[-1]['high'], live_tick['last'])
                    df_m5.iloc[-1, df_m5.columns.get_loc('low')] = min(df_m5.iloc[-1]['low'], live_tick['last'])
                    df_m5.iloc[-1, df_m5.columns.get_loc('volume')] += live_tick['volume']

            else:
                # If we are waiting for ticks but it's report time, we proceed with last known data
                if not is_report_time:
                    time.sleep(0.1)
                    continue

            # 2. Advanced Multi-Dimensional Analysis (Rate Limited)
            # CRITICAL: Feed MicroStructure EVERY TICK for accurate HFT metrics
            if live_tick:
                deep_brain.micro.on_tick(live_tick)

            current_time = time.time()

            # Initialize Analysis Variables (Safe Defaults)
            final_cortex_decision = 0
            original_base_score = 0
            base_score = 0
            smc_score = 0
            orbit_energy = 0
            micro_stats = {}
            reality_state = "NEUTRAL"
            details = {}

            if len(df_m5) >= 50 and (current_time - last_analysis_time >= analysis_cooldown or is_report_time):
                try:
                    # 1. Consensus Deliberation (Standard Eyes)
                    base_decision, base_score, details = consensus.deliberate(data_map, verbose=False)
                    
                    # --- INVERSE MODE LOGIC ---
                    original_base_score = base_score 
                    tech_label_suffix = ""
                    if config.INVERT_TECHNICALS:
                        base_score = -base_score 
                        tech_label_suffix = " (Inv)"
                    
                    # 2. Advanced Analysis
                    reality_score, reality_state = third_eye.analyze_reality(df_m5)
                    smc_score = smc_engine.analyze(df_m5)

                    # 3. Deep Brain (Physics & Subconscious)
                    final_cortex_decision, phy_state, future_prob, orbit_energy, micro_stats = deep_brain.consult_subconscious(
                        trend_score=base_score,
                        volatility_score=details.get('Volatility', {}).get('score', 0),
                        pattern_score=reality_score,
                        smc_score=smc_score,
                        df_m5=df_m5,
                        live_tick=live_tick,
                        details=details
                    )
                    
                    # 4. Tenth Eye (The Holographic Architect)
                    # Needs Market State (Hurst, Lyapunov, Volatility, RIVER)
                    market_state = {
                        'hurst': micro_stats.get('micro_hurst', 0.5),
                        'lyapunov': micro_stats.get('lyapunov', 0.0),
                        'volatility': details.get('Volatility', {}).get('score', 0),
                        'river': details.get('Trend', {}).get('river', 0), # Inject H1 Trend for Reality Check
                        'regime': details.get('Trend', {}).get('regime', 'RANGING')
                    }
                    
                    architect_res = tenth_eye.deliberate(details, market_state)
                    details['Architect'] = architect_res # Inject back into details
                    
                    # 5. Eleventh Eye (The Gap Exploiter)
                    # Scans for divergence between Architect (Strategy) and Consensus (Body)
                    gap_res = eleventh_eye.scan_for_divergence(
                        architect_data=architect_res, 
                        consensus_score=base_score, 
                        market_state=market_state
                    )
                    
                    if gap_res['action'] != "WAIT":
                        # OVERRIDE: Trust the Gap Exploiter in Chaos
                        final_cortex_decision = gap_res['score']
                        logger.info(f"[!] ELEVENTH EYE OVERRIDE: {gap_res['action']} (Score: {gap_res['score']:.1f})")
                        details['EleventhEye'] = gap_res
                    
                    # Update Weights based on Architect
                    # (Optional: Modify 'details' weights if ConsensusEngine supported dynamic weights per tick)
                    
                    last_analysis_time = current_time
                except Exception as e:
                    logger.error(f"Analysis Block Error: {e}")
                    # Keep using defaults or previous values
            
            
            # --- RISK MANAGEMENT v2.0 ---
            # Update Account Info for Dynamic Lots
            acc_stats = mt5_monitor.get_account_summary()
            current_equity = config.INITIAL_CAPITAL
            current_margin_free = 0 # Safety initial
            
            if acc_stats:
                current_equity = acc_stats.get('equity', config.INITIAL_CAPITAL)
                current_margin_free = acc_stats.get('margin_free', 0)
                
            # Check Margin Survival BEFORE Thinking about trading
            if is_warming_up:
                logger.info(f"WARMUP: Calibrating Neural Pathways... ({int(120 - (time.time() - bot_start_time))}s left)")
                final_cortex_decision = 0
            
            elif not risk_manager.check_margin_survival(acc_stats):
                # Force "Wait" state if margin is critical
                logger.warning("SKIPPING TRADES: Low Margin Protection Active.")
                final_cortex_decision = 0 
                
            # --- QUANTUM VERTICALITY PROTOCOL (Twelfth Eye) ---
            # User Request: "Complex Reasoning for Vertical Markets".
            # If Velocity is Extreme, we DISABLE Mean Reversion and FORCE Momentum.
            
            micro_velocity = micro_stats.get('velocity', 0)
            is_vertical_crash = micro_velocity < -0.8
            is_vertical_rocket = micro_velocity > 0.8
            
            if is_vertical_crash:
                 # If Brain wants to BUY (Reversion) during a Crash, we FORCE alignment with Gravity.
                 if final_cortex_decision > 0.1: 
                     logger.warning(f"[!] VERTICAL CRASH (v={micro_velocity:.2f}): Overriding Reversion BUY -> Momentum SELL.")
                     final_cortex_decision = -abs(final_cortex_decision) # Force Sell
                 # If Brain already wants to Sell, we Boost it.
                 elif final_cortex_decision < -0.1:
                     final_cortex_decision *= 1.5 # Boost Conviction
                 
            elif is_vertical_rocket:
                 # If Brain wants to SELL (Reversion) during a Rocket, we FORCE alignment with Thrust.
                 if final_cortex_decision < -0.1:
                     logger.warning(f"[!] VERTICAL ROCKET (v={micro_velocity:.2f}): Overriding Reversion SELL -> Momentum BUY.")
                     final_cortex_decision = abs(final_cortex_decision) # Force Buy
                 elif final_cortex_decision > 0.1:
                     final_cortex_decision *= 1.5 # Boost Conviction 
                
            # --- TREND ALIGNMENT PROTOCOL (Smart Inversion) ---
            # User Request: "Reverse the wrong signal" using advanced reasoning.
            # Logic: If Market is TRENDING, Reversion Signals (Sniper) are mostly wrong (Counter-Trend).
            # We FLIP them to align with the River.
            
            trend_data = details.get('Trend', {})
            h1_river = trend_data.get('river', 0)
            regime = trend_data.get('regime', "RANGING")
            
            # FORCE INVERSION if River is Active (Ignore ADX Lag)
            # User Feedback: "Wrong orders" in crash start (Low ADX).
            # If River says UP/DOWN, we align with it.
            # EXCEPTION: If Market is Vertical, we ignore Trend (Physics Overrides History)
            if h1_river != 0 and not is_vertical_crash and not is_vertical_rocket:
                quant_z = details.get('Quant', {}).get('z_score', 0)
                
                # 1. Bearish Trend + Buy Signal -> FLIP TO SELL
                # 1. Bearish Trend + Buy Signal -> BLOCK (VETO)
                if h1_river == -1 and final_cortex_decision > 0:
                    logger.info(f"[SMART] VETO: Blocking BUY ({final_cortex_decision:.1f}) due to BEARISH RIVER. Waiting for Trend Alignment.")
                    final_cortex_decision = 0 # NEUTRAL (Do not Invert)
                    
                # 2. Bullish Trend + Sell Signal -> BLOCK (VETO)
                elif h1_river == 1 and final_cortex_decision < 0:
                    logger.info(f"[SMART] VETO: Blocking SELL ({final_cortex_decision:.1f}) due to BULLISH RIVER. Waiting for Trend Alignment.")
                    final_cortex_decision = 0 # NEUTRAL (Do not Invert)

            # --- CRASH GUARD (Falling Knife Protection) ---
            # If the last closed candle was a huge crash (> 3x ATR), forbid BUYs for safety.
            # (If Smart Inversion already flipped it to Sell, this won't trigger, which is GOOD).
            try:
                last_open = df_m5.iloc[-2]['open']
                last_close = df_m5.iloc[-2]['close']
                last_atr = df_m5.iloc[-2]['ATR'] if 'ATR' in df_m5.columns else 1.0
                
                crash_size = last_open - last_close
                if crash_size > (3 * last_atr): # Massive Red Candle
                    if final_cortex_decision > 0: # Still Trying to Buy?
                        logger.critical(f"CRASH GUARD: Blocking BUY Signal (Score {final_cortex_decision}) during Market Crash Phase.")
                        final_cortex_decision = 0
            except: pass 
                
            # Calculate Dynamic Lots (QUANTUM SIGMOID LOGIC)
            # Scaling with Equity, Brain Confidence, and Chaos
            dynamic_base_lots = risk_manager.calculate_quantum_lots(
                current_equity, 
                confidence_score=final_cortex_decision,
                entropy=micro_stats.get('entropy', 0.5)
            )
            
            # --- MARGIN SAFETY GUARD ---
            # Calculate absolute max lots allowed by margin (90% usage cap of FREE MARGIN)
            current_price = live_tick['last']
            # We track 'live' margin free as we execute trades to avoid "Double Spend" in same tick
            simulated_margin_free = current_margin_free
            
            max_safe_lots = risk_manager.calculate_safe_margin_lots(current_equity, simulated_margin_free, current_price, config.LEVERAGE)
            
            if dynamic_base_lots > max_safe_lots:
                # logger.warning(f"Margin Safety: Capping Base Lots {dynamic_base_lots} -> {max_safe_lots}")
                dynamic_base_lots = max_safe_lots
            
            micro_velocity = micro_stats.get('velocity', 0)
            
            # --- SOVEREIGN ASSERTIVENESS (Dynamic Scaling) ---
            overlord_res = details.get('Overlord', {})
            sovereign_res = details.get('Sovereign', {})
            
            lot_multiplier = 1.0
            is_sovereign = False
            is_singularity_strike = False
            
            # --- THE HIERARCHY OF ASSERTIVENESS ---
            if details.get('Singularity', {}).get('decision') == "SINGULARITY_REACHED":
                lot_multiplier = 3.0
                is_singularity_strike = True
                logger.info(">>> !!! SINGULARITY STRIKE !!!: Multiplier 3.0x Active <<<")
            elif "STRONG" in sovereign_res.get('decision', ''):
                lot_multiplier = 2.0
                is_sovereign = True
                logger.info(">>> SOVEREIGN ASSERTIVENESS: Multiplier 2.0x Active <<<")
            elif abs(overlord_res.get('score', 0)) > 80:
                lot_multiplier = 1.5
                logger.info(">>> OVERLORD CONFIDENCE: Multiplier 1.5x Active <<<")
            
            # Final Margin Safety Check on Multiplied Lots
            final_raw_lots = dynamic_base_lots * lot_multiplier
            
            # HARD CAP ENFORCEMENT (User Request: "3.54 is too risky")
            if final_raw_lots > config.MAX_LOTS_PER_TRADE:
                logger.warning(f"Safety Cap: Reducing Lots {final_raw_lots:.2f} -> {config.MAX_LOTS_PER_TRADE} (Max Limit).")
                final_raw_lots = config.MAX_LOTS_PER_TRADE

            if final_raw_lots > max_safe_lots:
                 # Reduce multiplier if it breaks bank
                 logger.warning(f"Margin Safety: Capping Lots to {max_safe_lots:.2f} due to Free Margin.")
                 final_raw_lots = max_safe_lots
            
            # --- FALLING KNIFE / ROCKET GUARD (Physics Protection) ---
            # User Report: "Wrong purchase in downtrend" (Catching a knife).
            # If Velocity is extreme, we BAN counter-trend entries.
            # v < -0.5: Crashing -> NO BUYS.
            # v > 0.5: Rocketing -> NO SELLS.
            block_buys = False
            block_sells = False
            
            if micro_velocity < -0.6:
                logger.warning(f"CRASH DETECTED (v={micro_velocity:.2f}): BLOCKING ALL BUYS.")
                block_buys = True
            elif micro_velocity > 0.6:
                logger.warning(f"ROCKET DETECTED (v={micro_velocity:.2f}): BLOCKING ALL SELLS.")
                block_sells = True
            
            # --- SCALP SWARM (High Frequency Tick Execution) ---
            # User Request: "Inverse of Inverse" -> Use Original Technical Score (The 'Retail' Score)
            if config.ENABLE_FIRST_EYE:
                # Calculate Global Direction based on the Backup Score (Original)
                # If Retail Tech says BUY -> Swarm should BUY (Normal Logic)
                swarm_tech_dir = "WAIT"
                if original_base_score > 5: swarm_tech_dir = "BUY"
                elif original_base_score < -5: swarm_tech_dir = "SELL"
                
                # APPLY PHYSICS GUARD
                if block_buys and swarm_tech_dir == "BUY":
                    swarm_tech_dir = "WAIT" 
                if block_sells and swarm_tech_dir == "SELL":
                    swarm_tech_dir = "WAIT"
                
                swarm_action, swarm_reason, swarm_price = swarm.process_tick(
                    tick=live_tick,
                    df_m5=df_m5,
                    alpha_score=final_cortex_decision, # SmartBrain
                    tech_score=original_base_score,    # Retail Technical
                    phy_score=orbit_energy,            # Physics Energy
                    micro_stats=micro_stats,           # NEW: Full HFT Metrics
                    forced_lots=final_raw_lots         # Pass capped lots
                )
                
                if swarm_action:
                     # Execute Trade
                    cmd_type = swarm_action # "BUY" or "SELL"
                    
                    # Fix: Use ASK for BUY, BID for SELL to account for Spread
                    if cmd_type == "BUY":
                        base_price = live_tick['ask'] # Entry at Ask
                        sl_price = live_tick['bid'] - config.SCALP_SL # SL below Bid
                        tp_price = base_price + config.SCALP_TP
                        mt5_type = 0 # OP_BUY
                    else:
                        base_price = live_tick['bid'] # Entry at Bid
                        sl_price = live_tick['ask'] + config.SCALP_SL # SL above Ask
                        tp_price = base_price - config.SCALP_TP
                        mt5_type = 1 # OP_SELL
                        
                    # Validate Stops
                    sl_price, tp_price = validate_sl_tp(live_tick['bid'], live_tick['ask'], sl_price, tp_price, cmd_type)
                        
                    # PROTOCOL: OPEN_TRADE|SYMBOL|TYPE(Int)|VOLUME|SL|TP
                    final_volume = round(dynamic_base_lots * lot_multiplier, 2)
                    
                    if final_volume < 0.01:
                        logger.warning(f"Trade Aborted: Calculated Volume {final_volume} is too small (Margin Restriction?).")
                        continue # Skip execution
                    
                    params = [
                        config.SYMBOL, 
                        str(mt5_type), 
                        f"{final_volume:.2f}", 
                        normalize_price(sl_price),
                        normalize_price(tp_price)
                    ]
                    
                    # Log attempt
                    logger.info(f"SWARM SIGNAL: {swarm_reason} | {cmd_type} | Attempting Execution...")
                    
                    # Send correct command: OPEN_TRADE
                    resp = bridge.send_command("OPEN_TRADE", params)
                    if resp == "SENT":
                        logger.info(f">>> SWARM SENT: {swarm_reason} | {cmd_type} (Int: {mt5_type}) @ {base_price:.2f}")
                        notif_manager.send_notification("SWARM EXECUTION", f"{swarm_reason} | {cmd_type}", "TRADE")
                        
                        # MARGIN TRACKING UPDATE
                        used_margin = (current_price * 100 * (dynamic_base_lots * lot_multiplier)) / config.LEVERAGE
                        simulated_margin_free -= used_margin
                        # Re-calc safe max lots for next agent
                        max_safe_lots = risk_manager.calculate_safe_margin_lots(current_equity, simulated_margin_free, current_price, config.LEVERAGE)
                        
            # --- SECOND EYE (The Sniper) ---
            if config.ENABLE_SECOND_EYE:
                sniper_action, sniper_lots, sniper_reason = sniper.process_tick(
                    tick=live_tick,
                    df_m5=df_m5,
                    market_state=details
                )
                
                # APPLY PHYSICS GUARD
                if block_buys and sniper_action == "BUY":
                    sniper_action = "WAIT"
                if block_sells and sniper_action == "SELL":
                    sniper_action = "WAIT"
                
                # USER REQUEST: INVERT SNIPER SIGNAL REMOVED
                # if sniper_action == "BUY": ...
                
                if sniper_action:
                    # Execute Sniper Trade
                    # Protocol: OPEN_TRADE|SYMBOL|TYPE(Int)|VOLUME|SL|TP
                    mt5_type = 0 if sniper_action == "BUY" else 1
                    curr_price = live_tick['last']
                    
                    # Sniper uses same Stop Distance as Swarm effectively, or maybe slightly wider?
                    # Let's use Config SCALP_SL/TP for now but potentially wider eventually.
                    # Using same hard stops for safety.
                    
                    if sniper_action == "BUY":
                        base_price = live_tick['ask']
                        sl_price = live_tick['bid'] - config.SCALP_SL
                        tp_price = base_price + config.SCALP_TP
                    else:
                        base_price = live_tick['bid']
                        sl_price = live_tick['ask'] + config.SCALP_SL
                        tp_price = base_price - config.SCALP_TP
                        
                    # Validate Stops Distance (StopsLevel Check)
                    sl_price, tp_price = validate_sl_tp(live_tick['bid'], live_tick['ask'], sl_price, tp_price, sniper_action)
                        
                    params = [
                        config.SYMBOL, 
                        str(mt5_type), 
                        f"{(dynamic_base_lots * lot_multiplier):.2f}", # Dynamic + Sovereign Multiplier
                        normalize_price(sl_price),
                        normalize_price(tp_price)
                    ]
                    
                    logger.info(f"SNIPER SIGNAL: {sniper_reason} | {sniper_action} | Lots: {sniper_lots} | Executing...")
                    resp = bridge.send_command("OPEN_TRADE", params)
                    if resp == "SENT":
                        logger.info(f">>> SNIPER FIRED: {sniper_reason} | {sniper_action} @ {normalize_price(curr_price)}")
                        notif_manager.send_notification("SNIPER EXECUTION", f"{sniper_reason} | {sniper_lots} Lots", "TRADE")

                        # MARGIN TRACKING UPDATE
                        # MARGIN TRACKING UPDATE
                        used_margin = (current_price * 100 * (dynamic_base_lots * lot_multiplier)) / config.LEVERAGE
                        simulated_margin_free -= used_margin
                        max_safe_lots = risk_manager.calculate_safe_margin_lots(current_equity, simulated_margin_free, current_price, config.LEVERAGE)

            # --- FOURTH EYE (The Whale) ---
            if config.ENABLE_FOURTH_EYE:
                # Prepare Quantum Data
                vol_score = details.get('Volatility', {}).get('score', 0)
                
                whale_action, whale_reason, whale_lots = whale.process_tick(
                     tick=live_tick,
                     df_m5=df_m5,
                     consensus_score=original_base_score, 
                     smc_score=smc_score,
                     reality_state=reality_state if 'reality_state' in locals() else "NEUTRAL",
                     volatility_score=vol_score,
                     base_lots=dynamic_base_lots # Pass Capital-Based Lots
                )
                
                # APPLY PHYSICS GUARD (Fix for "Ultra Wrong Buy")
                if block_buys and whale_action == "BUY": 
                     whale_action = "WAIT"
                     logger.warning(f"Whale BLOCKED by Crash Guard (v={micro_velocity:.2f})")
                if block_sells and whale_action == "SELL": 
                     whale_action = "WAIT"
                     logger.warning(f"Whale BLOCKED by Rocket Guard (v={micro_velocity:.2f})")
                
                if whale_action:
                    # Execute Whale Trade
                    # Standard Logic: Follow the Sign
                    mt5_type = 0 if whale_action == "BUY" else 1
                    
                    if whale_action == "BUY":
                        base_price = live_tick['ask']
                        sl_price = live_tick['bid'] - config.SCALP_SL
                        tp_price = base_price + config.SCALP_TP
                    else:
                        base_price = live_tick['bid']
                        sl_price = live_tick['ask'] + config.SCALP_SL
                        tp_price = base_price - config.SCALP_TP
                        
                    # Validate Stops
                    sl_price, tp_price = validate_sl_tp(live_tick['bid'], live_tick['ask'], sl_price, tp_price, whale_action)
                    
                    # --- FINAL WHALE MARGIN CLAMP ---
                    # HARD CAP ENFORCEMENT
                    if whale_lots > config.MAX_LOTS_PER_TRADE:
                         logger.warning(f"Safety Cap: Reducing Whale Lots {whale_lots:.2f} -> {config.MAX_LOTS_PER_TRADE}")
                         whale_lots = config.MAX_LOTS_PER_TRADE

                # 3.2 OVERSOLD/OVERBOUGHT PROTECTION (The "Don't Sell Low" Guard)
                # If we are flipping to SELL, check if we are already Oversold
                if whale_action == "SELL": # We are about to SELL
                    z_score = micro_stats.get('z_score', 0)
                    if z_score < -2.0: # TIGHTENED to 2.0 (User feedback: "Wrong Decision")
                         logger.warning(f"[SMART] OVERSOLD PROTECT: Blocking SELL (Z:{z_score:.2f}). Waiting for bounce.")
                         whale_action = "WAIT" # ABORT

                elif whale_action == "BUY": # We are about to BUY
                    z_score = micro_stats.get('z_score', 0)
                    if z_score > 2.0: # TIGHTENED to 2.0
                         logger.warning(f"[SMART] OVERBOUGHT PROTECT: Blocking BUY (Z:{z_score:.2f}). Waiting for dip.")
                         whale_action = "WAIT" # ABORT

                    if whale_lots > max_safe_lots:
                         logger.warning(f"Margin Safety: Capping Whale Lots {whale_lots} -> {max_safe_lots}")
                         whale_lots = max_safe_lots

                    if whale_lots < 0.01:
                        logger.warning(f"Whale Trade Aborted: Calculated Volume {whale_lots} is too small.")
                        continue
                        
                    params = [
                        config.SYMBOL, 
                        str(mt5_type), 
                        f"{whale_lots:.2f}", # Use Whale Calculated Lots (Dynamic Scaling)
                        normalize_price(sl_price),
                        normalize_price(tp_price)
                    ]
                    
                    logger.info(f"WHALE SIGNAL: {whale_reason} | Action: {whale_action} | Lots: {whale_lots}")
                    resp = bridge.send_command("OPEN_TRADE", params)
                    if resp == "SENT":
                        logger.info(f">>> WHALE SURFACED: {whale_reason} | {whale_action} @ {normalize_price(base_price)}")
                        notif_manager.send_notification("WHALE EXECUTION", f"{whale_reason} | {whale_action} | {whale_lots} Lots", "TRADE")

                        # MARGIN TRACKING UPDATE
                        used_margin = (current_price * 100 * whale_lots) / config.LEVERAGE
                        simulated_margin_free -= used_margin
                        max_safe_lots = risk_manager.calculate_safe_margin_lots(current_equity, simulated_margin_free, current_price, config.LEVERAGE)

            # --- GAP EXPLOITER (Eleventh Eye) ---
            # Re-using the analysis from earlier (lines 427+) to execution
            # gap_res was stored in details['EleventhEye']
            
            gap_res = details.get('EleventhEye', {'action': 'WAIT', 'score': 0, 'reason': ''})
            
            if gap_res['action'] != "WAIT":
                 gap_action = gap_res['action']
                 gap_reason = gap_res['reason']
                 
                 # APPLY PHYSICS GUARD
                 if block_buys and gap_action == "BUY": 
                     gap_action = "WAIT"
                     logger.warning(f"Gap Exploiter BLOCKED by Crash Guard (v={micro_velocity:.2f})")
                 if block_sells and gap_action == "SELL": 
                     gap_action = "WAIT"
                     logger.warning(f"Gap Exploiter BLOCKED by Rocket Guard (v={micro_velocity:.2f})")
                     
                 if gap_action == "WAIT": continue # Abort safely
                 
                 # USER REQUEST: INVERT GAP SIGNAL REMOVED
                 # if gap_action == "BUY": ...
                 
                 # Gap Exploiter Logic
                 # High conviction -> 1.5x Base Size
                 gap_lots = dynamic_base_lots * 1.5 
                 
                 # --- FINAL GAP MARGIN CLAMP ---
                 # HARD CAP ENFORCEMENT
                 if gap_lots > config.MAX_LOTS_PER_TRADE:
                      logger.warning(f"Safety Cap: Reducing Gap Lots {gap_lots:.2f} -> {config.MAX_LOTS_PER_TRADE}")
                      gap_lots = config.MAX_LOTS_PER_TRADE

                 if gap_lots > max_safe_lots:
                      logger.warning(f"Margin Safety: Capping Gap Lots {gap_lots} -> {max_safe_lots}")
                      gap_lots = max_safe_lots
                 
                 if gap_lots < 0.01:
                      logger.warning(f"Gap Trade Aborted: Calculated Volume {gap_lots} is too small.")
                      continue

                 mt5_type = 0 if gap_action == "BUY" else 1
                 
                 if gap_action == "BUY":
                     base_price = live_tick['ask']
                     sl_price = live_tick['bid'] - config.SCALP_SL
                     tp_price = base_price + config.SCALP_TP
                 else:
                     base_price = live_tick['bid']
                     sl_price = live_tick['ask'] + config.SCALP_SL
                     tp_price = base_price - config.SCALP_TP
                     
                 sl_price, tp_price = validate_sl_tp(live_tick['bid'], live_tick['ask'], sl_price, tp_price, gap_action)
                 
                 params = [
                     config.SYMBOL, 
                     str(mt5_type), 
                     f"{gap_lots:.2f}", 
                     normalize_price(sl_price),
                     normalize_price(tp_price)
                 ]
                 
                 # Only fire if not cooling down (check gap_res metadata if possible or rely on internal eye cooldown)
                 # For now, we trust the Eye returned "BUY"/"SELL" meaning it wants to trade NOW.
                 
                 logger.info(f"GAP SIGNAL: {gap_reason} | Action: {gap_action} | Lots: {gap_lots}")
                 resp = bridge.send_command("OPEN_TRADE", params)
                 if resp == "SENT":
                     logger.info(f">>> GAP SNIPER FIRED: {gap_reason} | {gap_action} @ {normalize_price(base_price)}")
                     notif_manager.send_notification("GAP EXECUTION", f"{gap_reason} | {gap_lots} Lots", "TRADE")

                     # MARGIN TRACKING UPDATE
                     used_margin = (current_price * 100 * gap_lots) / config.LEVERAGE
                     simulated_margin_free -= used_margin
            
            # 3. Account Awareness (MT5 Check)
            # Run every minute or on new candle to avoid spamming API
            if time.time() - last_log_print > 10: # Check every 10s for PnL visibility
                acc_stats = mt5_monitor.get_account_summary()
                # perf_stats = mt5_monitor.analyze_manual_performance() # Too heavy/slow for 10s loop?
                
                if acc_stats:
                    # Floating PnL is in acc_stats['profit']
                    pnl = acc_stats['profit']
                    pnl_emoji = "(+)" if pnl > 0 else "(-)" if pnl < 0 else "(=)"
                    logger.info(f"MONITOR: {pnl_emoji} Floating PnL: ${pnl:.2f} | Eq: {acc_stats['equity']:.2f}")
                    
                last_log_print = time.time()

            # Heartbeat Log (if everything is quiet but running)
            if time.time() - last_log_print > 60:
                logger.debug(f"HEARTBEAT: System Active | Tick: {live_tick.get('last') if live_tick else 'N/A'}")
                last_log_print = time.time()

            # 4. Smart Notification Logic (Clock Aligned)
            if is_report_time:
                # Prepare Message Data
                threshold_buy = 0.50 
                threshold_sell = -0.50
                
                direction = "EQUILIBRIUM"
                if final_cortex_decision > threshold_buy: direction = "BUY"
                elif final_cortex_decision < threshold_sell: direction = "SELL"
                elif abs(final_cortex_decision) > 0.2: direction = "WAIT" # Monitoring
                
                # Always notify on the 5-minute mark
                # Calculate Lots
                acc_bal = config.INITIAL_CAPITAL
                if 'acc_stats' in locals() and acc_stats: acc_bal = acc_stats['balance']
                
                atr = df_m5.iloc[-1]['ATR'] if 'ATR' in df_m5.columns else 1.0
                sl_dist = atr * 1.5 
                risk_amt = acc_bal * 0.01 
                rec_lots = round(risk_amt / (sl_dist * 100), 2) if sl_dist > 0 else 0.01
                if rec_lots < 0.01: rec_lots = 0.01
                
                future_pct = int(future_prob * 100) if direction == "BUY" else int((1-future_prob)*100)
                
                # Determine Tech Signal Label
                tech_signal = "WAIT"
                if base_score > 5: tech_signal = "BUY"
                elif base_score < -5: tech_signal = "SELL"
                
                # Determine Alpha Signal Label
                alpha_signal = direction # This is already BUY/SELL/EQUILIBRIUM/WAIT
                
                # Title: Tech (Inv): BUY | Alpha: WAIT
                emoji = "" # Emojis removed to prevent UnicodeEncodeError
                title = f"10-EYE ALIGN: {tech_signal} ({base_score:.1f}) | State: {alpha_signal} ({final_cortex_decision:.2f})"
                body = f"Size: {rec_lots} | Future: {future_pct}% | Phys: {phy_state} | Time: {now_sp.strftime('%H:%M')}"
                
                notif_manager.send_notification(title, body, direction)
                logger.info(f">>> REPORT SENT ({now_sp.strftime('%H:%M')}): {title} | {body}")
                
                # --- FIRST EYE / SCALP SWARM (Auto-Scalper) ---
                # NOTE: Swarm is processed on EVERY TICK (see above).
                # This block was redundant and empty.


            # Dashboard Pub (Live)
            try:
                pub_data = {
                    'price': live_tick['last'] if live_tick else 0,
                    'score': final_cortex_decision,
                    'reality': reality_state if 'reality_state' in locals() else phy_state
                }
                pub_socket.send_string(f"LIVE {json.dumps(pub_data)}")
            except:
                pass
                
    
            # Reduce Latency: Sleep significantly less to catch ticks
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Shutdown Signal.")
    finally:
        bridge.close()
        logger.info("System Offline.")

if __name__ == "__main__":
    main()