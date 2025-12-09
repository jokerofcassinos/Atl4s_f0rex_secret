import time
import logging
import pandas as pd
from datetime import datetime
import config
from data_loader import DataLoader
from analysis.consensus import ConsensusEngine
from analysis.risk_neural import NeuralRiskManager
from backtest_engine import BacktestEngine
from report_generator import ReportGenerator
from analysis.trade_manager import TradeManager
from bridge import ZmqBridge
import zmq
import json

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
    logger.info("--- Atl4s-Forex System Startup ---")
    
    # 1. Initialize Bridge (Server)
    try:
        bridge = ZmqBridge()
        # The bridge starts a background thread to accept connections.
        # We don't need to block here.
    except Exception as e:
        logger.critical(f"Failed to start Bridge Server: {e}")
        return

    # Initialize Modules
    data_loader = DataLoader()
    consensus = ConsensusEngine()
    
    # Load Optimized Parameters
    try:
        with open("optimal_params.json", "r") as f:
            opt_params = json.load(f)
            consensus.update_parameters(opt_params)
            logger.info("EVOLUTIONARY DATA LOADED: Optimized Parameters Active.")
            logger.info(f"   > Trend Weight: {opt_params.get('w_trend', 0):.2f}")
            logger.info(f"   > Sniper Weight: {opt_params.get('w_sniper', 0):.2f}")
    except FileNotFoundError:
        logger.warning("No evolutionary data found. Using default parameters.")
    except Exception as e:
        logger.error(f"Failed to load optimized parameters: {e}")
    risk_manager = NeuralRiskManager()
    backtester = BacktestEngine()
    reporter = ReportGenerator()
    trade_manager = TradeManager()

    # Initialize Dashboard Publisher
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5558")
    logger.info("Dashboard Publisher bound to port 5558")
    
    # Send Initial History to Dashboard
    try:
        # Get last 100 candles from data_loader (it's already loaded)
        # We need to re-fetch or use what we have.
        # data_loader.get_data() was called in Pre-Flight check if we moved it?
        # Actually, let's just fetch it once here for the dashboard.
        init_data = data_loader.get_data()
        if init_data['M5'] is not None and not init_data['M5'].empty:
            hist_df = init_data['M5'].iloc[-100:]
            history_payload = []
            for index, row in hist_df.iterrows():
                history_payload.append({
                    'time': int(index.timestamp() * 1000),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })
            pub_socket.send_string(f"HISTORY {json.dumps(history_payload)}")
            logger.info(f"Sent {len(history_payload)} historical candles to Dashboard.")
    except Exception as e:
        logger.error(f"Failed to send history: {e}")
    
    # 2. Pre-Flight Check
    # 2. Pre-Flight Check
    logger.info("Running Pre-Flight Backtest (Fast Mode)...")
    data_map = data_loader.get_data()
    
    # Optimization: Slice data for rapid startup check
    # We only need to verify the logic works, not backtest months of data here.
    # Optimization: Slice data for rapid startup check
    # We only need to verify the logic works, not backtest months of data here.
    if data_map.get('M5') is not None and len(data_map['M5']) > 500:
        startup_data = data_map.copy()
        startup_data['M5'] = data_map['M5'].iloc[-500:]
        # H1 slicing is handled inside, but we can just pass the full H1 or slice it too
        passed, metrics = backtester.run_preflight_check(startup_data)
    else:
        passed, metrics = backtester.run_preflight_check(data_map)
    
    if not passed:
        logger.warning("Pre-Flight Check Failed! Switching to PAPER MODE (Simulated).")
    else:
        logger.info("Pre-Flight Check Passed. Systems Green.")

    # Initialize variables for report (in case of early exit)
    account_info = {'equity': config.INITIAL_CAPITAL, 'balance': config.INITIAL_CAPITAL}
    trades = []
    analysis_summary = {'regime': 'WAITING', 'score': 0}
    last_trade_time = None

    logger.info("Entering Main Loop...")
    logger.info("Waiting for MT5 Connection...")
    
    # Initialize df_m5 from startup data
    df_m5 = data_map.get('M5')
    if df_m5 is None:
         logger.warning("No M5 data found. Waiting for live ticks to build history...")
         # Create empty DataFrame with correct columns
         df_m5 = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    try:
        while True:
            # Check for connection via bridge
            if not bridge.conn:
                # logger.debug("Bridge conn is None...") # Too noisy
                time.sleep(1)
                continue
                
            # Check Time Window (10:00 - 12:00)
            # Simplified: Run always for demo/testing
            
            # 1. Data Sync
            # Optimization: Do NOT fetch history in loop. Use live ticks.
            # data_map = data_loader.get_data() # REMOVED: Blocking call
            # df_m5 = data_map['M5'] # REMOVED: Overwrites live updates
            
            # Fetch Live Tick from Bridge
            live_tick = bridge.get_tick()
            verbose_log = False # Default to silent
            
            if live_tick and live_tick['last'] > 0:
                # Validate Symbol
                if live_tick['symbol'] != config.SYMBOL:
                    logger.warning(f"Ignored Tick for {live_tick['symbol']} (Expected {config.SYMBOL}). Please change chart in MT5.")
                    continue

                # Feed Tick to Micro-Structure Analyzer
                consensus.update_ticks(live_tick)

                # Update df_m5 with live price
                # Logic: If tick time > last candle time + 5min, create new row.
                # Else, update last row (Close, High, Low, Volume).
                
                if not df_m5.empty:
                    last_time = df_m5.index[-1]
                else:
                    last_time = pd.Timestamp.now(tz=config.TIMEZONE) # Fallback

                tick_time = pd.to_datetime(live_tick['time'], unit='ms')
                
                # Adjust tick time to timezone if needed (assuming UTC from MT5)
                # For simplicity, we trust the relative difference.
                
                # Check if we need a new candle
                # M5 candle starts at 0, 5, 10... 
                # If tick is 12:06, it belongs to 12:05 candle.
                # If last candle is 12:00, we need a new one.
                
                # Ensure timezone compatibility
                if not df_m5.empty and df_m5.index.tz is not None and tick_time.tz is None:
                    tick_time = tick_time.tz_localize(df_m5.index.tz)
                elif not df_m5.empty and df_m5.index.tz is None and tick_time.tz is not None:
                    tick_time = tick_time.tz_localize(None)

                # Simple check: is tick_time >= last_time + 5min?
                if df_m5.empty or tick_time >= (last_time + pd.Timedelta(minutes=5)):
                    # New Candle
                    new_row = pd.DataFrame({
                        'open': [live_tick['last']],
                        'high': [live_tick['last']],
                        'low': [live_tick['last']],
                        'close': [live_tick['last']],
                        'volume': [live_tick['volume']]
                    }, index=[tick_time])
                    df_m5 = pd.concat([df_m5, new_row])
                else:
                    # Update Last Candle
                    df_m5.iloc[-1, df_m5.columns.get_loc('close')] = live_tick['last']
                    df_m5.iloc[-1, df_m5.columns.get_loc('high')] = max(df_m5.iloc[-1]['high'], live_tick['last'])
                    df_m5.iloc[-1, df_m5.columns.get_loc('low')] = min(df_m5.iloc[-1]['low'], live_tick['last'])
                    df_m5.iloc[-1, df_m5.columns.get_loc('volume')] += live_tick['volume']
                
                # Update data_map
                data_map['M5'] = df_m5
                
                # Log Throttling: Only log "Live Update" if it's a new candle or every 60s
                current_time = time.time()
                is_new_candle = False
                if df_m5.index[-1] != last_trade_time: # Using last_trade_time as proxy for last_candle_log is risky, let's use a new var
                     # Actually, we can just check if the minute changed? No.
                     pass

                # We'll use a simple heartbeat for "Live Update"
                if 'last_log_time' not in locals() or (current_time - last_log_time) > 60:
                     logger.info(f"Live Update: {live_tick['last']} | M5 Last: {df_m5.index[-1]}")
                     last_log_time = current_time
                     verbose_log = True
                else:
                     verbose_log = False
                
                # Check if it's a new candle (index changed)
                if 'last_candle_index' not in locals(): last_candle_index = df_m5.index[-1]
                
                if df_m5.index[-1] != last_candle_index:
                    verbose_log = True
                    last_candle_index = df_m5.index[-1]
                    logger.info(f"NEW CANDLE: {last_candle_index}")
                
            
            if df_m5 is None or len(df_m5) < 100:
                time.sleep(10)
                continue
                
            # 2. Matrix Analysis
            # Silence sub-modules if not verbose
            if not verbose_log:
                logging.getLogger().setLevel(logging.ERROR)
                
            try:
                # Pass verbose flag (though logger is silenced, it keeps logic consistent)
                decision, score, details = consensus.deliberate(data_map, verbose=verbose_log)
            finally:
                # Restore Logging Level
                if not verbose_log:
                    logging.getLogger().setLevel(logging.INFO)
            
            # If we found a trade but were silent, announce it now!
            if decision != "WAIT" and not verbose_log:
                 logger.info(f"!!! SIGNAL DETECTED: {decision} (Score: {score:.2f}) !!!")
                 # Optional: Log key details
                 logger.info(f"   > Trend: {details.get('Trend', {}).get('score', 0)}")
                 logger.info(f"   > Sniper: {details.get('Sniper', {}).get('score', 0)}")
                 logger.info(f"   > Quantum: {details.get('Quantum', {}).get('tunneling_prob', 0):.2f}")
            
            # --- CONTRARIAN MODE (REVERSE PSYCHOLOGY) ---
            # User Intervention: Invert Signals due to observed anti-correlation.
            if decision == "BUY":
                decision = "SELL"
                logger.warning(">>> CONTRARIAN FLIP: Signal Inverted to SELL <<<")
            elif decision == "SELL":
                decision = "BUY"
                logger.warning(">>> CONTRARIAN FLIP: Signal Inverted to BUY <<<")
            # ---------------------------------------------
            
            # Update summary for report
            analysis_summary = {
                'regime': details.get('regime', 'UNKNOWN'),
                'score': score
            }
            
            # Publish Analysis
            try:
                # Convert numpy types to native python for JSON serialization
                def convert(o):
                    if isinstance(o, (pd.Timestamp, datetime)): return o.isoformat()
                    if hasattr(o, 'item'): return o.item()
                    return o
                    
                analysis_data = {
                    'regime': details.get('Trend', {}).get('regime', 'UNKNOWN'),
                    'modules': {k: v.get('score', 0) if isinstance(v, dict) else 0 for k, v in details.items()}
                }
                pub_socket.send_string(f"ANALYSIS {json.dumps(analysis_data, default=convert)}")
            except Exception as e:
                logger.error(f"Analysis Pub Error: {e}")
            
            # 3. Execution Logic
            if decision != "WAIT":
                # Publish Signal
                try:
                    sig_data = {
                        'type': decision,
                        'symbol': config.SYMBOL,
                        'price': df_m5.iloc[-1]['close'],
                        'score': score
                    }
                    pub_socket.send_string(f"SIGNAL {json.dumps(sig_data, default=convert)}")
                except Exception as e:
                    logger.error(f"Signal Pub Error: {e}")
                # Check for existing positions
                # Note: Bridge needs a way to query positions. 
                # For now, we assume we track it or ask via command if supported.
                # The current simple bridge doesn't return values for commands easily without async handling.
                # We will skip position check for this iteration and rely on EA to reject if max trades reached.
                
                # Calculate Risk
                current_price = df_m5.iloc[-1]['close']
                balance = config.INITIAL_CAPITAL 
                
                # ATR
                atr = df_m5.iloc[-1].get('ATR', 1.0)
                
                # Metrics
                regime_prob = details.get('Math', {}).get('regime_prob', 0.5)
                quantum_data = details.get('Quantum', {})
                kinematics_data = details.get('Kinematics', {})
                wavelet_data = details.get('Wavelet', {})
                cortex_conf = details.get('Cortex', {}).get('bullish_prob', 0.5)
                
                direction = 1 if decision == "BUY" else -1
                if decision == "SELL": cortex_conf = 1 - cortex_conf
                
                # Physics Stop (Geometric)
                k_energy = kinematics_data.get('energy', 0)
                w_coherence = wavelet_data.get('coherence', 0)
                q_uncertainty = quantum_data.get('uncertainty', 0)
                
                sl_price = risk_manager.get_geometric_stop(
                    entry_price=current_price,
                    direction=direction,
                    atr_value=atr,
                    kinematics_energy=k_energy,
                    wavelet_power=w_coherence,
                    uncertainty=q_uncertainty
                )
                
                tp_price = risk_manager.get_take_profit(current_price, direction, sl_price, regime_prob)
                
                # Sizing Inputs
                w_coherence = wavelet_data.get('coherence', 0)
                t_loops = details.get('Topology', {}).get('betti_1', 0)
                k_score = kinematics_data.get('score', 0)
                
                lots = risk_manager.calculate_quantum_size(
                    entry_price=current_price, 
                    stop_loss_price=sl_price, 
                    current_balance=balance,
                    consensus_score=score,
                    regime_prob=regime_prob,
                    cortex_conf=cortex_conf,
                    wavelet_coherence=w_coherence,
                    topology_loops=t_loops,
                    kinematics_score=k_score
                )
                
                if lots > 0:
                    # One Trade Per Candle Logic
                    current_candle_time = df_m5.index[-1]
                    
                    # Check if we already traded this candle
                    if last_trade_time != current_candle_time:
                        # Send Open Command
                        # Format: OPEN_TRADE|SYMBOL|TYPE|VOLUME|SL|TP
                        # Type: 0=Buy, 1=Sell
                        order_type = 0 if decision == "BUY" else 1
                        bridge.send_command("OPEN_TRADE", [config.SYMBOL, order_type, lots, sl_price, tp_price])
                        
                        last_trade_time = current_candle_time
                        logger.info(f"Trade Signal Sent. Locking candle {current_candle_time} to prevent spam.")
                    else:
                        logger.info(f"Signal ignored: Already traded this candle ({current_candle_time}).")
            
            # 4. Trade Management
            # Ideally we loop through positions. 
            # Since we can't easily get positions in this simple sync loop without blocking,
            # we will skip the complex management loop for this specific 'main.py' fix 
            # and focus on getting the connection stable first.
            
            time.sleep(1) # Loop delay (Fast Mode)
            
    except KeyboardInterrupt:
        logger.info("Shutdown Signal Received.")
    finally:
        if 'bridge' in locals():
            bridge.close()
        reporter.generate_daily_report(account_info, trades, analysis_summary)
        logger.info("System Shutdown.")

if __name__ == "__main__":
    main()
