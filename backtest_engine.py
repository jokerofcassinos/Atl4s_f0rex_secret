import pandas as pd
import logging
from analysis.consensus import ConsensusEngine
from analysis.smart_money import SmartMoneyEngine
from analysis.deep_cognition import DeepCognition
from analysis.hyper_dimension import HyperDimension
from analysis.scalper_swarm import ScalpSwarm
from analysis.second_eye import SecondEye
from analysis.fourth_eye import FourthEye
from analysis.trade_manager import TradeManager
import config

logger = logging.getLogger("Atl4s-Backtest")

class BacktestEngine:
    def __init__(self):
        self.consensus = ConsensusEngine()
        self.smc_engine = SmartMoneyEngine()
        self.deep_brain = DeepCognition()
        self.third_eye = HyperDimension()
        self.swarm = ScalpSwarm()
        self.sniper = SecondEye()
        self.whale = FourthEye()
        self.trade_manager = TradeManager()

    def run_preflight_check(self, data_map):
        """
        Runs a rapid backtest on the provided data.
        data_map: {'M5': df_m5, 'H1': df_h1}
        """
        results = self.run_simulation(data_map, verbose=False)
        
        passed = results['pf'] > 1.5 and results['total_trades'] > 5 and results['safety_score'] > 50
        
        metrics = {
            "pf": results['pf'], 
            "wr": results['wr'], 
            "balance": results['final_balance'],
            "safety_score": results['safety_score'],
            "risk_of_ruin": results['risk_of_ruin']
        }
        return passed, metrics

    def run_simulation(self, data_map, params=None, verbose=True):
        """
        Detailed simulation with Ultra Optimization (Pre-calculated signals).
        """
        df = data_map.get('M5')
        df_h1 = data_map.get('H1')
        
        if df is None or len(df) < 200:
            logger.error("Insufficient data for simulation.")
            return {"error": "Insufficient data"}

        if params:
            self.consensus.update_parameters(params)

        if verbose:
            logger.info(f"Preparing Advanced Simulation ({len(df)} candles)...")
        
        # --- PRE-DELIBERATION (Parallel Pre-calculation) ---
        signals_cache = {}
        smc_cache = {}
        reality_cache = {}
        
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        start_pre = time.time()
        
        def calculate_candle_signals(idx):
            current_time = df.index[idx]
            current_slice_m5 = df.iloc[:idx+1]
            current_slice_h1 = None
            if df_h1 is not None:
                current_slice_h1 = df_h1[df_h1.index <= current_time]
            
            # 1. Consensus
            try:
                base_decision, base_score, details = self.consensus.deliberate({'M5': current_slice_m5, 'H1': current_slice_h1}, verbose=False)
            except:
                base_score = 0
                details = {}
            
            # 2. SMC & Hyper
            smc_score = self.smc_engine.analyze(current_slice_m5)
            reality_score, reality_state = self.third_eye.analyze_reality(current_slice_m5)
            
            return idx, (base_score, details), smc_score, (reality_score, reality_state)

        # Only analyze candles in trading hours to save more time
        indices_to_analyze = []
        for i in range(200, len(df)):
            current_time = df.index[i]
            if config.TRADING_START_TIME <= current_time.time() <= config.TRADING_END_TIME:
                indices_to_analyze.append(i)

        if verbose:
            logger.info(f"Analyzing {len(indices_to_analyze)} trading windows in parallel...")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(calculate_candle_signals, indices_to_analyze))
            for idx, cons, smc, real in results:
                signals_cache[idx] = cons
                smc_cache[idx] = smc
                reality_cache[idx] = real

        pre_duration = time.time() - start_pre
        if verbose:
            logger.info(f"Pre-Deliberation complete in {pre_duration:.2f}s")

        # --- SIMULATION LOOP ---
        balance = config.INITIAL_CAPITAL
        trades_log = []
        trades_pnl = []
        active_trade = None
        
        start_sim = time.time()

        for i in range(200, len(df)):
            current_candle = df.iloc[i]
            current_time = current_candle.name
            
            if i not in signals_cache:
                if active_trade:
                    # Close at candle close if outside hours
                    pnl = (current_candle['close'] - active_trade['entry']) * active_trade['direction'] * active_trade['size']
                    balance += pnl
                    trades_pnl.append(pnl)
                    active_trade.update({'exit_time': current_time, 'exit_price': current_candle['close'], 'pnl': pnl, 'reason': "EndOfDay"})
                    trades_log.append(active_trade)
                    active_trade = None
                continue

            # Get pre-calculated signals
            base_score, details = signals_cache[i]
            smc_score = smc_cache[i]
            reality_score, reality_state = reality_cache[i]
            
            orig_score = base_score
            if config.INVERT_TECHNICALS:
                base_score = -base_score
            
            # --- TICK SIMULATION ---
            ms_time = int(current_time.timestamp() * 1000)
            ticks = [
                {'last': current_candle['open'], 'bid': current_candle['open'], 'ask': current_candle['open'], 'time': ms_time, 'volume': 100},
                {'last': current_candle['high'], 'bid': current_candle['high'], 'ask': current_candle['high'], 'time': ms_time, 'volume': 100},
                {'last': current_candle['low'], 'bid': current_candle['low'], 'ask': current_candle['low'], 'time': ms_time, 'volume': 100},
                {'last': current_candle['close'], 'bid': current_candle['close'], 'ask': current_candle['close'], 'time': ms_time, 'volume': 100}
            ]

            for tick in ticks:
                price = tick['last']
                
                if active_trade:
                    active_trade['profit'] = (price - active_trade['open_price']) * active_trade['direction'] * active_trade['size'] * 100
                    atr_tm = df.iloc[max(0, i-20):i+1]['close'].diff().abs().mean() or 1.0
                    struc_low = df.iloc[max(0, i-20):i+1]['low'].min()
                    struc_high = df.iloc[max(0, i-20):i+1]['high'].max()

                    new_sl = self.trade_manager.check_trailing_stop(active_trade, price, struc_low, struc_high)
                    if new_sl: active_trade['sl'] = new_sl
                    
                    exit_signal = self.trade_manager.check_hard_exit(active_trade, config.SCALP_TP, config.SCALP_SL)
                    
                    hit_sl = (active_trade['direction'] == 1 and price <= active_trade['sl']) or \
                             (active_trade['direction'] == -1 and price >= active_trade['sl'])
                    hit_tp = (active_trade['direction'] == 1 and price >= active_trade['tp']) or \
                             (active_trade['direction'] == -1 and price <= active_trade['tp'])

                    if hit_sl or (exit_signal and exit_signal['action'] == "CLOSE_FULL"):
                        exit_price = active_trade['sl'] if hit_sl else price
                        pnl = (exit_price - active_trade['entry']) * active_trade['direction'] * active_trade['size']
                        balance += pnl
                        trades_pnl.append(pnl)
                        active_trade.update({'exit_time': current_time, 'exit_price': exit_price, 'pnl': pnl, 'reason': "SL" if hit_sl else exit_signal['reason']})
                        trades_log.append(active_trade)
                        active_trade = None
                        break 

                    elif hit_tp:
                        exit_price = active_trade['tp']
                        pnl = (exit_price - active_trade['entry']) * active_trade['direction'] * active_trade['size']
                        balance += pnl
                        trades_pnl.append(pnl)
                        active_trade.update({'exit_time': current_time, 'exit_price': exit_price, 'pnl': pnl, 'reason': "TP"})
                        trades_log.append(active_trade)
                        active_trade = None
                        break

                if active_trade is None:
                    if current_time.time() > config.TRADING_LAST_ENTRY_TIME:
                        continue

                    final_decision_value, phy_state, future_prob, orbit_energy, micro_velocity = self.deep_brain.consult_subconscious(
                        trend_score=base_score,
                        volatility_score=details.get('Vol', {}).get('score', 0) if details else 0,
                        pattern_score=reality_score,
                        smc_score=smc_score,
                        df_m5=df.iloc[:i+1],
                        live_tick=tick
                    )

                    decision = "WAIT"
                    if final_decision_value > 0.5: decision = "BUY"
                    elif final_decision_value < -0.5: decision = "SELL"

                    # Fast Eye Check
                    if decision == "WAIT" and config.ENABLE_FIRST_EYE:
                         swarm_action, _, _ = self.swarm.process_tick(tick, df.iloc[:i+1], final_decision_value, orig_score, orbit_energy, micro_velocity)
                         if swarm_action: decision = swarm_action
                    
                    if decision == "WAIT" and config.ENABLE_SECOND_EYE:
                        sniper_action, _, _ = self.sniper.process_tick(tick, df.iloc[:i+1], final_decision_value, orig_score, orbit_energy)
                        if sniper_action: decision = sniper_action

                    if decision != "WAIT":
                        direction = 1 if decision == "BUY" else -1
                        risk_amt = balance * config.RISK_PER_TRADE
                        atr_val = df.iloc[max(0, i-14):i+1]['close'].diff().abs().mean() or 1.0
                        sl_dist = max(1.5 * atr_val, 0.50)
                        pos_size = risk_amt / sl_dist 
                        
                        active_trade = {
                            "ticket": len(trades_log) + 1,
                            "entry_time": current_time,
                            "entry": price,
                            "open_price": price,
                            "direction": direction,
                            "type": 0 if direction == 1 else 1,
                            "size": pos_size,
                            "sl": price - (direction * sl_dist),
                            "tp": price + (direction * sl_dist * 2.0),
                            "decision": decision,
                            "score": final_decision_value,
                            "volume": pos_size,
                            "profit": 0.0
                        }
                        break
        
        sim_duration = time.time() - start_sim
        if verbose:
            logger.info(f"Simulation loop finished in {sim_duration:.2f}s")

        # Metrics
        wins = [t for t in trades_pnl if t > 0]
        losses = [t for t in trades_pnl if t <= 0]
        total_p = sum(wins)
        total_l = abs(sum(losses))
        pf = total_p / total_l if total_l > 0 else 999
        wr = len(wins) / len(trades_pnl) if trades_pnl else 0
        
        # Monte Carlo
        from analysis.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(num_simulations=500, initial_capital=config.INITIAL_CAPITAL)
        trades_pct = [t / config.INITIAL_CAPITAL for t in trades_pnl]
        mc_results = mc.run(trades_pct)
        
        if verbose:
            logger.info(f"Simulation Complete. Trades: {len(trades_pnl)}, PF: {pf:.2f}, WR: {wr*100:.1f}%, Bal: {balance:.2f}")

        return {
            "pf": pf,
            "wr": wr,
            "final_balance": balance,
            "total_trades": len(trades_pnl),
            "trades": trades_log,
            "safety_score": mc_results['safety_score'],
            "risk_of_ruin": mc_results['risk_of_ruin']
        }
