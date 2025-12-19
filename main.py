import time
import logging
import pandas as pd
from datetime import datetime
import pytz
import config
import json
import zmq

# Core Modules
from data_loader import DataLoader
from bridge import ZmqBridge
from report_generator import ReportGenerator

# Old Analysis (Legacy Support)
from analysis.consensus import ConsensusEngine
from analysis.risk_neural import NeuralRiskManager

# NEW Advanced Modules
from src.notifications import NotificationManager
from src.mt5_monitor import MT5Monitor
from analysis.smart_money import SmartMoneyEngine
from analysis.deep_cognition import DeepCognition
from analysis.hyper_dimension import HyperDimension
from analysis.scalper_swarm import ScalpSwarm
from analysis.scalper_swarm import ScalpSwarm
from analysis.second_eye import SecondEye
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

def main():
    logger.info("--- Atl4s-Forex System 2.0: Deep Awakening ---")
    
    # 1. Initialize Infrastructure
    try:
        bridge = ZmqBridge()
    except Exception as e:
        logger.critical(f"Failed to start Bridge Server: {e}")
        return

    # Initialize Managers
    notif_manager = NotificationManager(cooldown_minutes=5)
    mt5_monitor = MT5Monitor()
    
    data_loader = DataLoader() # Historical Data
    reporter = ReportGenerator()
    
    # Initialize Minds (Analysis Engines)
    consensus = ConsensusEngine() # Old Logic
    risk_manager = NeuralRiskManager()
    
    # New Engines
    smc_engine = SmartMoneyEngine()
    deep_brain = DeepCognition()
    third_eye = HyperDimension()
    swarm = ScalpSwarm()
    sniper = SecondEye()
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
    sp_tz = pytz.timezone('America/Sao_Paulo')
    
    # Configure Notification Manager (Zero cooldown, we control timing here)
    notif_manager.cooldown = 0 

    try:
        while True:
            # 1. Connectivity & Data Ingestion
            if not bridge.conn:
                time.sleep(1)
                continue
                
            live_tick = bridge.get_tick()
            
            # Smart Time Check (Sao Paulo)
            now_sp = datetime.now(sp_tz)
            current_minute = now_sp.minute
            
            # Check for 5-Minute Trigger (00, 05, 10, 15...)
            # We trigger once when the minute changes to a multiple of 5
            is_report_time = False
            if current_minute % 5 == 0 and current_minute != last_candle_minute:
                is_report_time = True
                last_candle_minute = current_minute
                logger.info(f"--- 5-MINUTE CYCLE TRIGGER: {now_sp.strftime('%H:%M:%S')} ---")
            
            if live_tick and live_tick['last'] > 0:
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

            # 2. Advanced Multi-Dimensional Analysis
            # Run this ONLY on report time (every 5 mins) OR if debugging
            # To be responsive, we can run analysis every loop but only NOTIFY on schedule.
            # Let's run analysis every loop to update Dashboard live.
            
            if len(df_m5) < 50: continue 
            
            try:
                base_decision, base_score, details = consensus.deliberate(data_map, verbose=False)
            except:
                base_score = 0
            logging.getLogger("Atl4s-Consensus").setLevel(logging.INFO)
            
            # --- INVERSE MODE LOGIC ---
            original_base_score = base_score # Backup for Scalper (Inverse of Inverse)
            tech_label_suffix = ""
            if config.INVERT_TECHNICALS:
                base_score = -base_score # FLIP THE SIGNAL
                tech_label_suffix = " (Inv)"
            
            # Scalp Swarm Logic removed from here (will be re-inserted after Deep Cognition)
            
            smc_score = smc_engine.analyze(df_m5)
            reality_score, reality_state = third_eye.analyze_reality(df_m5)
            
            final_cortex_decision, phy_state, future_prob, orbit_energy, micro_velocity = deep_brain.consult_subconscious(
                trend_score=base_score,
                volatility_score=details.get('Vol', {}).get('score', 0) if 'details' in locals() else 0,
                pattern_score=reality_score,
                smc_score=smc_score,
                df_m5=df_m5,
                live_tick=live_tick
            )
            
            # --- ACTIVE PROFIT TAKER (Virtual TP) ---
            # Checks every tick if any trade has hit the target ($0.70)
            # This combats broker latency/spread issues.
            open_positions = mt5_monitor.get_open_positions() 
            if open_positions:
                 for pos in open_positions:
                     # Check Hard Exit
                     exit_signal = trade_manager.check_hard_exit(pos, config.SCALP_TP, config.SCALP_SL)
                     if exit_signal and exit_signal['action'] == "CLOSE_FULL":
                         # Execute Immediate Close
                         ticket = exit_signal['ticket']
                         reason = exit_signal['reason']
                         logger.info(f"⚡ INSTANT EXIT: {reason} | Closing Ticket {ticket}...")
                         bridge.send_command("CLOSE_TRADE", [str(ticket)])
                         notif_manager.send_notification("INSTANT PROFIT", f"{reason}", "PROFIT")
            
            # --- SCALP SWARM (High Frequency Tick Execution) ---
            # User Request: "Inverse of Inverse" -> Use Original Technical Score (The 'Retail' Score)
            if config.ENABLE_FIRST_EYE:
                # Calculate Global Direction based on the Backup Score (Original)
                # If Retail Tech says BUY -> Swarm should BUY (Normal Logic)
                swarm_tech_dir = "WAIT"
                if original_base_score > 5: swarm_tech_dir = "BUY"
                elif original_base_score < -5: swarm_tech_dir = "SELL"
                
                swarm_action, swarm_reason, swarm_price = swarm.process_tick(
                    tick=live_tick,
                    df_m5=df_m5,
                    alpha_score=final_cortex_decision, # SmartBrain (Inverted if active)
                    tech_score=original_base_score,    # Retail Technical (Non-Inverted)
                    phy_score=orbit_energy,            # Physics Energy for Hybrid Switching
                    velocity=micro_velocity,           # NEW: Micro Velocity for Guard
                    signal_dir=swarm_tech_dir # (Unused in new logic but kept for sig info)
                )
                
                if swarm_action:
                     # Execute Trade
                    cmd_type = swarm_action # "BUY" or "SELL"
                    curr_price = swarm_price
                    if cmd_type == "BUY":
                        sl_price = curr_price - config.SCALP_SL
                        tp_price = curr_price + config.SCALP_TP
                        mt5_type = 0 # OP_BUY
                    else:
                        sl_price = curr_price + config.SCALP_SL
                        tp_price = curr_price - config.SCALP_TP
                        mt5_type = 1 # OP_SELL
                        
                    # Protocol: OPEN_TRADE|SYMBOL|TYPE(Int)|VOLUME|SL|TP
                    # Note: Volume is index 3 in EA, SL is 4, TP is 5.
                    params = [
                        config.SYMBOL, 
                        str(mt5_type), 
                        str(config.SCALP_LOTS), 
                        f"{sl_price:.2f}", 
                        f"{tp_price:.2f}"
                    ]
                    
                    # Log attempt
                    logger.info(f"SWARM SIGNAL: {swarm_reason} | {cmd_type} | Attempting Execution...")
                    
                    # Send correct command: OPEN_TRADE
                    resp = bridge.send_command("OPEN_TRADE", params)
                    if resp == "SENT":
                        logger.info(f">>> SWARM SENT: {swarm_reason} | {cmd_type} (Int: {mt5_type}) @ {curr_price:.2f}")
                        notif_manager.send_notification("SWARM EXECUTION", f"{swarm_reason} | {cmd_type}", "TRADE")
                        
            # --- SECOND EYE (The Sniper) ---
            if config.ENABLE_SECOND_EYE:
                sniper_action, sniper_reason, sniper_lots = sniper.process_tick(
                    tick=live_tick,
                    df_m5=df_m5,
                    alpha_score=final_cortex_decision,
                    tech_score=original_base_score,
                    orbit_energy=orbit_energy
                )
                
                if sniper_action:
                    # Execute Sniper Trade
                    # Protocol: OPEN_TRADE|SYMBOL|TYPE(Int)|VOLUME|SL|TP
                    mt5_type = 0 if sniper_action == "BUY" else 1
                    curr_price = live_tick['last']
                    
                    # Sniper uses same Stop Distance as Swarm effectively, or maybe slightly wider?
                    # Let's use Config SCALP_SL/TP for now but potentially wider eventually.
                    # Using same hard stops for safety.
                    
                    if sniper_action == "BUY":
                        sl_price = curr_price - config.SCALP_SL
                        tp_price = curr_price + config.SCALP_TP
                    else:
                        sl_price = curr_price + config.SCALP_SL
                        tp_price = curr_price - config.SCALP_TP
                        
                    params = [
                        config.SYMBOL, 
                        str(mt5_type), 
                        str(sniper_lots), # Dynamic Lots
                        f"{sl_price:.2f}", 
                        f"{tp_price:.2f}"
                    ]
                    
                    logger.info(f"SNIPER SIGNAL: {sniper_reason} | {sniper_action} | Lots: {sniper_lots} | Executing...")
                    resp = bridge.send_command("OPEN_TRADE", params)
                    if resp == "SENT":
                        logger.info(f">>> SNIPER FIRED: {sniper_reason} | {sniper_action} @ {curr_price:.2f}")
                        notif_manager.send_notification("SNIPER EXECUTION", f"{sniper_reason} | {sniper_lots} Lots", "TRADE")
            
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
                emoji = ""
                if "[SUPER]" in phy_state: 
                    emoji = " 🔥"
                
                title = f"Tech{tech_label_suffix}: {tech_signal} ({base_score:.1f}) | Alpha: {alpha_signal} ({final_cortex_decision:.2f}){emoji}"
                body = f"Size: {rec_lots} | Future: {future_pct}% | Phys: {phy_state} | Time: {now_sp.strftime('%H:%M')}"
                
                notif_manager.send_notification(title, body, direction)
                logger.info(f">>> REPORT SENT ({now_sp.strftime('%H:%M')}): {title} | {body}")
                
                # --- FIRST EYE / SCALP SWARM (Auto-Scalper) ---
                # NOTE: Swarm is processed on EVERY TICK (see above), but we log here for visibility if needed.
                # Actually, the user wants Swarm to run continuously.
                # If we put it only here, it runs once every 5 mins.
                # Ideally, we should remove this block and let the tick-level logic handle it.
                # However, to be safe and ensure at least one check per 5 mins, we can leave a manual check or rely on the loop above.
                # WAIT! I haven't inserted the loop above yet. I am replacing THIS block with the Tick Loop block?
                # No, I should insert the tick loop block earlier in the code (around line 170) and REMOVE this block.
                # But I cannot insert earlier without a separate edit.
                # Strategy:
                # 1. I will effectively DELETE this "Report Time" execution block.
                # 2. I'll insert the real execution logic higher up in the next step.
                pass # Swarm Logic moved to main loop for High Frequency Execution

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
