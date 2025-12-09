import pandas as pd
import logging
from analysis.consensus import ConsensusEngine
import config

logger = logging.getLogger("Atl4s-Backtest")

class BacktestEngine:
    def __init__(self):
        self.consensus = ConsensusEngine()

    def run_preflight_check(self, data_map):
        """
        Runs a rapid backtest on the provided data.
        data_map: {'M5': df_m5, 'H1': df_h1}
        """
        df = data_map.get('M5')
        df_h1 = data_map.get('H1')
        
        if df is None or len(df) < 500:
            logger.error("Insufficient data for backtest.")
            return False, {}

        logger.info("Starting Pre-Flight Backtest...")
        
        balance = config.INITIAL_CAPITAL
        trades = []
        active_trade = None
        
        # Iterate through data
        for i in range(200, len(df)):
            current_candle = df.iloc[i]
            current_time = current_candle.name
            
            # Slice M5 Data up to current time
            current_slice_m5 = df.iloc[:i+1]
            
            # Slice H1 Data up to current time
            current_slice_h1 = None
            if df_h1 is not None:
                current_slice_h1 = df_h1[df_h1.index <= current_time]
            
            # Build Slice Map
            current_data_map = {'M5': current_slice_m5, 'H1': current_slice_h1}
            
            # Check Time Window (only trade 10-12)
            if current_candle.name.time() < config.TRADING_START_TIME or current_candle.name.time() > config.TRADING_END_TIME:
                # Close active trade if any (Day Trade)
                if active_trade:
                    pnl = (current_candle['close'] - active_trade['entry']) * active_trade['direction'] * active_trade['size']
                    balance += pnl
                    trades.append(pnl)
                    active_trade = None
                continue

            # Logic
            if active_trade is None:
                # Check Last Entry Time
                if current_candle.name.time() > config.TRADING_LAST_ENTRY_TIME:
                    continue

                # Silence sub-modules during backtest
                logging.getLogger().setLevel(logging.ERROR)
                try:
                    decision, score, _ = self.consensus.deliberate(current_data_map, verbose=False)
                finally:
                    logging.getLogger().setLevel(logging.INFO)
                
                if decision != "WAIT":
                    # Enter Trade
                    direction = 1 if decision == "BUY" else -1
                    # Simple Position Sizing for Backtest
                    risk_amt = balance * config.RISK_PER_TRADE
                    
                    # Calculate ATR for dynamic stops
                    tr_list = []
                    for k in range(max(0, i-14), i+1):
                        h = df['high'].iloc[k]
                        l = df['low'].iloc[k]
                        c_prev = df['close'].iloc[k-1] if k > 0 else l
                        tr = max(h-l, abs(h-c_prev), abs(l-c_prev))
                        tr_list.append(tr)
                    
                    atr = sum(tr_list) / len(tr_list) if tr_list else 1.0
                    
                    # Stop Loss distance (1.5 * ATR)
                    sl_dist = 1.5 * atr
                    sl_dist = max(sl_dist, 0.50)

                    pos_size = risk_amt / sl_dist 
                    
                    active_trade = {
                        "entry": current_candle['close'],
                        "direction": direction,
                        "size": pos_size,
                        "sl": current_candle['close'] - (direction * sl_dist),
                        "tp": current_candle['close'] + (direction * sl_dist * 2.0) # 1:2 RR
                    }
            else:
                # Manage Trade
                # Check SL/TP
                # Hit SL
                if (active_trade['direction'] == 1 and current_candle['low'] <= active_trade['sl']) or \
                   (active_trade['direction'] == -1 and current_candle['high'] >= active_trade['sl']):
                    loss = -1 * (active_trade['size'] * abs(active_trade['entry'] - active_trade['sl']))
                    balance += loss
                    trades.append(loss)
                    active_trade = None
                
                # Hit TP
                elif (active_trade['direction'] == 1 and current_candle['high'] >= active_trade['tp']) or \
                     (active_trade['direction'] == -1 and current_candle['low'] <= active_trade['tp']):
                    profit = active_trade['size'] * abs(active_trade['entry'] - active_trade['tp'])
                    balance += profit
                    trades.append(profit)
                    active_trade = None

        # Calculate Metrics
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        
        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 999
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        
        # --- Monte Carlo Simulation ---
        from analysis.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(num_simulations=500, initial_capital=config.INITIAL_CAPITAL)
        
        # Convert trades to PnL % relative to initial capital (Approximation)
        # Ideally we track % per trade.
        trades_pct = [t / config.INITIAL_CAPITAL for t in trades]
        
        mc_results = mc.run(trades_pct)
        safety_score = mc_results['safety_score']
        ror = mc_results['risk_of_ruin']
        
        logger.info(f"Backtest Complete. PF: {profit_factor:.2f}, WR: {win_rate*100:.1f}%, Bal: {balance:.2f}")
        logger.info(f"Monte Carlo: Safety Score {safety_score:.0f}/100, RoR {ror:.1f}%")
        
        passed = profit_factor > 1.5 and len(trades) > 5 and safety_score > 50
        
        metrics = {
            "pf": profit_factor, 
            "wr": win_rate, 
            "balance": balance,
            "safety_score": safety_score,
            "risk_of_ruin": ror
        }
        return passed, metrics
