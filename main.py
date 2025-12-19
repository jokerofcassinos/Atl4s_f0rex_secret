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

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
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
            
            # A. Standard Matrix (Old Consensus)
            # Suppress logs for this part unless needed
            logging.getLogger("Atl4s-Consensus").setLevel(logging.WARNING) 
            try:
                base_decision, base_score, details = consensus.deliberate(data_map, verbose=False)
            except:
                base_score = 0
            logging.getLogger("Atl4s-Consensus").setLevel(logging.INFO)
            
            smc_score = smc_engine.analyze(df_m5)
            reality_score, reality_state = third_eye.analyze_reality(df_m5)
            
            final_cortex_decision, phy_state, future_prob = deep_brain.consult_subconscious(
                trend_score=base_score,
                volatility_score=details.get('Vol', {}).get('score', 0) if 'details' in locals() else 0,
                pattern_score=reality_score,
                smc_score=smc_score,
                df_m5=df_m5,
                live_tick=live_tick
            )
            
            # 3. Account Awareness (MT5 Check)
            if time.time() - last_log_print > 60:
                acc_stats = mt5_monitor.get_account_summary()
                perf_stats = mt5_monitor.analyze_manual_performance()
                if acc_stats:
                    logger.info(f"STATS | Bal: {acc_stats['balance']} | Man.WinRate: {perf_stats['accuracy_label']} | Net: {perf_stats['net']}")
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
                
                title = f"{direction} SIGNAL (Alpha: {final_cortex_decision:.2f})"
                body = f"Size: {rec_lots} | FutureProb: {future_pct}% | State: {phy_state} | Time: {now_sp.strftime('%H:%M')}"
                
                notif_manager.send_notification(title, body, direction)
                logger.info(f">>> REPORT SENT ({now_sp.strftime('%H:%M')}): {title} | {body}")

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
                
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Shutdown Signal.")
    finally:
        bridge.close()
        logger.info("System Offline.")

if __name__ == "__main__":
    main()
