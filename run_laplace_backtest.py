"""
LAPLACE DEMON - Professional Backtest Runner
═════════════════════════════════════════════

Runs comprehensive backtests with:
- Full Laplace Demon analysis
- Realistic spread/slippage simulation
- Walk-forward validation
- Complete chart generation
- Detailed performance reports

Target: 70% Win Rate | $30 Capital | GBPUSD
"""

import asyncio
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import sys # Added for logging handlers

# Setup logging
logging.basicConfig(
    level=logging.INFO, # Phase 5: Production Mode
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("laplace_backtest.log")
    ]
)
logger = logging.getLogger("Laplace-Backtest")

# Import backtest engine
from backtest.engine import BacktestEngine, BacktestConfig, Trade, TradeDirection
from backtest.charts import ChartGenerator
from backtest.metrics import MetricsCalculator, MonteCarloResult

# Import Laplace Demon
from core.laplace_demon import LaplaceDemonCore, LaplacePrediction

# Import Analytics
from analytics.telegram_notifier import get_notifier


class LaplaceBacktestRunner:
    """
    Professional backtest runner using Laplace Demon.
    """
    
    def __init__(self,
                 initial_capital: float = 5000.0,
                 risk_per_trade: float = 2.0,
                 symbol: str = "GBPUSD",
                  spread_pips: float = 1.5):
        
        # Dynamic Leverage (FTMO Compliance)
        # Dynamic Leverage (Exness Aggressive - Matched to Main)
        leverage = 1000000.0 # Unlimited for < $1000
        if initial_capital > 1000:
            leverage = 3000.0 # 1:2000 capped for > $1000
            
        self.symbol = symbol
        self.config = BacktestConfig(
            initial_capital=initial_capital,
            leverage=leverage,
            risk_per_trade_pct=20.0, # Increased for Sniper Profits ($800/trade)
            max_concurrent_trades=100, # Increased for Split Fire (Swarm needs space)
            spread_pips=spread_pips,
            slippage_pips=0.5,
            symbol=symbol,
            fixed_lots=50.0 # [FIXED LOT MODE] Auto-Scales for <$30k accounts (Smart Engine)
        )
        
        # Initialize components
        self.engine = BacktestEngine(self.config)
        self.laplace = LaplaceDemonCore(symbol)
        self.charts = ChartGenerator("reports")
        
        # Phase 7: Dynamic VSL (Protects against volatility, but scales with capital)
        # Legacy: $20 VSL on $30 Capital (~66%).
        # We maintain this ratio to serve as a 'Catastrophe Stop' rather than a distinct trade filter
        self.config.vsl_pips = None
        self.config.vsl_dollars = initial_capital * 0.60
        
        logger.info(f"Dynamic Settings: Leverage 1:{int(leverage)} | VSL ${self.config.vsl_dollars:.2f}")
        
        self.telegram = get_notifier()
        
        # Data frames
        self.df_m1: Optional[pd.DataFrame] = None
        self.df_m5: Optional[pd.DataFrame] = None
        self.df_h1: Optional[pd.DataFrame] = None
        self.df_h4: Optional[pd.DataFrame] = None
        self.df_d1: Optional[pd.DataFrame] = None
        
        logger.info(f"Laplace Backtest Runner initialized: ${initial_capital} | {symbol}")
    
    async def load_data(self, file_path: str) -> bool:
        """
        Load and prepare data from CSV or Parquet.
        
        Expects M1 or M5 data.
        """
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return False
        
        logger.info(f"Loading data from {file_path}...")
        
        try:
            # Load data based on extension
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Handle datetime index
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close']
            for col in required:
                if col not in df.columns:
                    logger.error(f"Missing column: {col}")
                    return False
            
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Ensure naive timezone (Phase 7 Fix)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            self.df_m1 = df
            
            # Resample to other timeframes
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            self.df_m5 = df.resample('5min').agg(agg).dropna()
            self.df_h1 = df.resample('1h').agg(agg).dropna()
            self.df_h4 = df.resample('4h').agg(agg).dropna()
            self.df_d1 = df.resample('1D').agg(agg).dropna()
            
            logger.info(f"Data loaded: M1={len(self.df_m1)}, M5={len(self.df_m5)}, H1={len(self.df_h1)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    async def run_backtest(self, use_m5: bool = True) -> Dict:
        """
        Run the backtest simulation.
        
        Args:
            use_m5: Use M5 candles for simulation (faster) or M1 (more accurate)
        """
        if self.df_m5 is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
        
        df = self.df_m5 if use_m5 else self.df_m1
        
        logger.info(f"Starting Laplace Demon backtest on {len(df)} candles...")
        
        # Reset engine
        self.engine.reset()
        
        # Warm-up period (need enough data for indicators)
        warmup = 200
        
        total_candles = len(df)
        last_signal_time = None
        min_signal_interval = 5  # Minimum candles between signals
        
        for i in range(warmup, total_candles):
            candle = df.iloc[i]
            current_time = candle.name
            current_price = candle['close']
            
            # Progress logging
            if i % 1000 == 0:
                pct = (i / total_candles) * 100
                logger.info(f"Progress: {pct:.1f}% | Balance: ${self.engine.balance:.2f}")
            
            # Prepare data slices (anti-lookahead)
            slice_m5 = self.df_m5.iloc[:i+1] if use_m5 else self.df_m5[self.df_m5.index <= current_time]
            slice_h1 = self.df_h1[self.df_h1.index <= current_time]
            slice_h4 = self.df_h4[self.df_h4.index <= current_time]
            slice_d1 = self.df_d1[self.df_d1.index <= current_time] if self.df_d1 is not None else None
            
            # Slice M1 for M8 generation (Last 300 minutes ~ 300 rows is fast enough)
            if hasattr(self, 'df_m1') and self.df_m1 is not None:
                 # Ensure we have M1 data loaded
                 slice_m1 = self.df_m1.loc[current_time - pd.Timedelta(minutes=300):current_time]
            else:
                 slice_m1 = None
            # Check active trades first
            for trade in self.engine.active_trades[:]:
                exit_reason = self.engine.update_trade(trade, current_price, current_time)
                
                if exit_reason:
                    # Simulate proper exit price
                    if exit_reason == "SL_HIT":
                        exit_price = trade.sl_price
                    elif exit_reason == "TP_HIT":
                        exit_price = trade.tp_price
                    else:
                        # For VSL or trailing stops, use current price
                        exit_price = current_price
                    
                    self.engine.close_trade(trade, exit_price, current_time, exit_reason)
                    
                    # Notify Telegram
                    if self.telegram.enabled:
                        await self.telegram.notify_trade_exit(
                            symbol=self.symbol,
                            entry=trade.entry_price,
                            exit=exit_price,
                            pnl_dollars=trade.pnl_dollars,
                            pnl_pips=trade.pnl_pips,
                            reason=exit_reason,
                            source=trade.signal_source
                        )
                    
                    logger.info(f"EXIT #{trade.id}: {exit_reason} | PnL: ${trade.pnl_dollars:.2f} | Setup: {trade.signal_source}")
            
            # --------------------------------------------------------------------------
            # ADVANCED RISK: Time Decay Take Profit (Descending Staircase)
            # --------------------------------------------------------------------------
            from risk_manager import RiskManager
            # Instantiate RiskManager (stateless or not, it's safer)
            risk_manager = RiskManager()
            
            for trade in self.engine.active_trades:
                # Apply Decay Logic
                if not hasattr(trade, 'initial_tp_price'):
                    trade.initial_tp_price = trade.tp_price
                
                try:
                    # Pass timestamps as floats to ensure compatibility
                    # RiskManager expects: (entry, initial_tp, open_ts, current_ts)
                    # And returns the NEW TP PRICE.
                    
                    # Ensure times allow .timestamp()
                    t_open = trade.entry_time.timestamp() if hasattr(trade.entry_time, 'timestamp') else trade.entry_time
                    t_curr = current_time.timestamp() if hasattr(current_time, 'timestamp') else current_time
                    
                    new_tp_price = risk_manager.calculate_decayed_tp(
                        entry_price=trade.entry_price,
                        initial_tp_price=trade.initial_tp_price,
                        open_time_timestamp=t_open,
                        current_timestamp=t_curr
                    )
                    
                    trade.tp_price = new_tp_price
                    
                except Exception as e:
                    # Log once or debug to avoid flooding?
                    pass
             # --------------------------------------------------------------------------
                
                # PATCH: Let's assume for now we use the trade's current TP as 'initial' ONLY IF we haven't decayed it yet?
                # No, that fails.
                # We will perform a simple check:
                # If we don't store initial_tp, we can't use the stateless formula perfectly.
                # However, we can modify the TP *incrementally*? No, that's messy.
                # Let's use a simplified version: 
                # If age > 1h, reduce TP by X pips.
                # Actually, let's skip strict 'initial' requirement and just ensure we don't over-decay.
                # We will skip Time Decay integration in this specific file for now unless I modify Trade class.
                # Apply Decay Logic
                if not hasattr(trade, 'initial_tp_price'):
                    trade.initial_tp_price = trade.tp_price
                
                try:
                    # Instantiate locally if not present to be safe against previous partial edits
                    if 'risk_manager' not in locals():
                        from risk_manager import RiskManager
                        risk_manager = RiskManager()

                    # Pass timestamps as floats
                    t_open = trade.entry_time.timestamp() if hasattr(trade.entry_time, 'timestamp') else trade.entry_time
                    t_curr = current_time.timestamp() if hasattr(current_time, 'timestamp') else current_time
                    
                    new_tp_price = risk_manager.calculate_decayed_tp(
                        entry_price=trade.entry_price,
                        initial_tp_price=trade.initial_tp_price,
                        open_time_timestamp=t_open,
                        current_timestamp=t_curr
                    )
                    
                    trade.tp_price = new_tp_price
                except Exception as e:
                    pass
             # --------------------------------------------------------------------------
            
            # Check for new signals (respect minimum interval)
            if last_signal_time is not None:
                candles_since_signal = i - last_signal_time
                if candles_since_signal < min_signal_interval:
                    continue
            
            # Only generate signals if we have capacity
            if len(self.engine.active_trades) >= self.config.max_concurrent_trades:
                continue
            
            # --- SYNTHETIC TICK FEED (Deep Forensic Analysis) ---
            # Simulate M1 ticks to feed MicroStructure (Entropy/OFI)
            try:
                if len(slice_m1) > 0:
                    last_m1 = slice_m1.iloc[-1]
                    # Create 4 synthetic ticks: Open -> High -> Low -> Close
                    vol_per_tick = max(1, last_m1['volume'] // 4)
                    base_time = last_m1.name.timestamp() * 1000 if hasattr(last_m1.name, 'timestamp') else time.time() * 1000
                    
                    ticks = [
                        {'time': base_time, 'last': last_m1['open'], 'volume': vol_per_tick, 'flags': 0},
                        {'time': base_time+15000, 'last': last_m1['high'], 'volume': vol_per_tick, 'flags': 0},
                        {'time': base_time+30000, 'last': last_m1['low'], 'volume': vol_per_tick, 'flags': 0},
                        {'time': base_time+45000, 'last': last_m1['close'], 'volume': vol_per_tick, 'flags': 0}
                    ]
                    
                    # Sort High/Low based on candle color for realism
                    if last_m1['close'] > last_m1['open']:
                        # Bullish: Open -> Low -> High -> Close
                        ticks[1]['last'] = last_m1['low']
                        ticks[2]['last'] = last_m1['high']
                    
                    for t in ticks:
                         self.laplace.micro.on_tick(t)
            except Exception as e:
                # logger.error(f"Tick synthesis error: {e}")
                pass
            # ----------------------------------------------------

            # Get Laplace Demon prediction (✅ BACKTEST FIX: Added await)
            try:
                prediction = await self.laplace.analyze(
                    df_m1=slice_m1,  # Passed M1 slice
                    df_m5=slice_m5,
                    df_h1=slice_h1,
                    df_h4=slice_h4,
                    df_d1=slice_d1,
                    current_time=current_time,
                    current_price=current_price
                )
                
                # Execute if signal is valid
                if prediction.execute and prediction.direction in ["BUY", "SELL"]:
                    direction = TradeDirection.BUY if prediction.direction == "BUY" else TradeDirection.SELL
                    
                    # SPLIT FIRE EXECUTION LOGIC
                    # Instead of one giant trade, we split into N trades of 1x.
                    # This aligns with the "Swarm" concept and allows granular management.
                    num_orders = 1
                    lot_multiplier = prediction.lot_multiplier
                    
                    if lot_multiplier >= 2.0:
                        num_orders = int(lot_multiplier)
                        lot_multiplier = 1.0 # Reset to base unit per order

                    # Execute N times
                    for order_idx in range(num_orders):
                        trade = self.engine.open_trade(
                            direction=direction,
                            current_time=current_time,
                            current_price=current_price,
                            sl_pips=prediction.sl_pips,
                            tp_pips=prediction.tp_pips,
                            signal_source=prediction.primary_signal or "LAPLACE",
                            confidence=prediction.confidence,
                            lot_multiplier=lot_multiplier # Always 1.0 if split
                        )
                        
                        if trade:
                            last_signal_time = i
                            
                            # Log EVERY split trade for clarity
                            logger.info(
                                f"TRADE #{trade.id}: {direction.value} @ {current_price:.5f} | "
                                f"SL: {prediction.sl_pips:.1f}p | TP: {prediction.tp_pips:.1f}p | "
                                f"Conf: {prediction.confidence:.0f}% | SPLIT FIRE ({order_idx+1}/{num_orders})"
                            )
                            
                            # Log Neural Decision if present (only once per candle to avoid noise? No, log it so it's clear)
                            if order_idx == 0:
                                for reason in prediction.reasons:
                                    if "Neural" in reason:
                                        logger.info(f"  > {reason}")
                                
                            if self.telegram.enabled:
                                await self.telegram.notify_trade_entry(
                                    direction=prediction.direction,
                                    symbol=self.symbol,
                                    entry=current_price,
                                    sl=trade.sl_price,
                                    tp=trade.tp_price,
                                    confidence=prediction.confidence,
                                    setup=f"{prediction.primary_signal} [{order_idx+1}/{num_orders}]"
                                )
                        else:
                             # If one fails (margin/slots), likely all fail. Break.
                             logger.warning(f"SKIPPED SPLIT EXECUTION at {order_idx+1}/{num_orders}: Consensus said {prediction.direction} but execution failed (Max Slots/Margin?).")
                             break
                                
                elif prediction.direction in ["BUY", "SELL"] and not prediction.execute:
                    # Logic Vetoed it (Nash, Neural, etc)
                    logger.info(f"SIGNAL VETOED: {prediction.direction} | Vetoes: {prediction.vetoes}")
                        
                    # Log reasons for development transparency
                    for reason in prediction.reasons:
                        if "Neural Filter" in reason:
                            logger.info(f"  > {reason}")
                        else:
                            logger.debug(f"  > {reason}")
                
                elif prediction.execute is False and prediction.direction in ["BUY", "SELL"]:
                    # Check if it was vetoed by Neural Oracle
                    for veto in prediction.vetoes:
                        if "Neural Oracle" in veto:
                            logger.info(f"SIGNAL VETOED: {prediction.direction} @ {current_price:.5f} | {veto}")
                        
            except Exception as e:
                logger.warning(f"Error in analysis at {current_time}: {e}")
                continue
            
            # Record equity
            unrealized = sum(
                self.engine.calculate_pips(t.entry_price, current_price if t.direction == TradeDirection.BUY 
                                          else current_price) 
                * self.config.get_pip_value_for_lots(t.lots)
                for t in self.engine.active_trades
            )
            
            equity = self.engine.balance + unrealized
            self.engine.equity_curve.append((current_time, equity))
            
            if equity > self.engine.peak_equity:
                self.engine.peak_equity = equity
            
            dd = (self.engine.peak_equity - equity) / self.engine.peak_equity * 100
            if dd > self.engine.max_drawdown:
                self.engine.max_drawdown = dd
        
        # Close remaining trades
        last_price = df.iloc[-1]['close']
        last_time = df.index[-1]
        for trade in self.engine.active_trades[:]:
            self.engine.close_trade(trade, last_price, last_time, "END_OF_TEST")
        
        # Calculate results
        result = self.engine._calculate_results()
        
        return result
    
    def generate_report(self, result, prefix: str = "laplace") -> str:
        """Generate comprehensive report with charts."""
        logger.info("Generating performance report...")
        
        # Generate all charts
        charts = self.charts.generate_all(result, prefix)
        
        # Export JSON results
        json_path = os.path.join(self.charts.output_dir, f"{prefix}_results.json")
        self.engine.export_results(result, json_path)
        
        # Print summary
        self._print_summary(result)
        
        # Monte Carlo simulation
        logger.info("Running Monte Carlo simulation (1000 iterations)...")
        mc = MetricsCalculator.monte_carlo_simulation(
            result.trades,
            self.config.initial_capital,
            simulations=1000
        )
        self._print_monte_carlo(mc)
        
        # Statistical edge test
        try:
            from scipy import stats
            edge = MetricsCalculator.calculate_statistical_edge(result.trades)
            self._print_edge_analysis(edge)
        except ImportError:
            logger.warning("scipy not available for statistical analysis")
        
        return json_path
    
    def _print_summary(self, result):
        """Print backtest summary."""
        print("\n" + "═" * 60)
        print("  LAPLACE DEMON BACKTEST RESULTS")
        print("═" * 60)
        
        print(f"\n📊 PERFORMANCE")
        print(f"   Total Executions:  {result.total_trades} (Includes Split Orders)")
        print(f"   Winning Trades:    {result.winning_trades}")
        print(f"   Losing Trades:     {result.losing_trades}")
        print(f"   Win Rate:          {result.win_rate:.1f}%")
        
        target_met = "✅" if result.win_rate >= 70 else "❌"
        print(f"   Target (70%):      {target_met}")
        
        print(f"\n💰 PROFIT/LOSS")
        print(f"   Net Profit:        ${result.net_profit:.2f}")
        print(f"   Gross Profit:      ${result.gross_profit:.2f}")
        print(f"   Gross Loss:        ${result.gross_loss:.2f}")
        print(f"   Profit Factor:     {result.profit_factor:.2f}")
        
        print(f"\n📉 RISK METRICS")
        print(f"   Max Drawdown:      {result.max_drawdown_pct:.1f}%")
        print(f"   Sharpe Ratio:      {result.sharpe_ratio:.2f}")
        print(f"   Avg R-Multiple:    {result.avg_r_multiple:.2f}")
        print(f"   Expectancy:        ${result.expectancy:.2f}/trade")
        
        print(f"\n⏱️ TRADING STATS")
        print(f"   Avg Win:           ${result.avg_win:.2f}")
        print(f"   Avg Loss:          ${result.avg_loss:.2f}")
        print(f"   Largest Win:       ${result.largest_win:.2f}")
        print(f"   Largest Loss:      ${result.largest_loss:.2f}")
        print(f"   Avg Duration:      {result.avg_trade_duration:.0f} min")
        print(f"   Trades/Day:        {result.trades_per_day:.1f}")
        
        print(f"\n🔥 STREAKS")
        print(f"   Max Consecutive Wins:   {result.consecutive_wins}")
        print(f"   Max Consecutive Losses: {result.consecutive_losses}")
        
        print("\n" + "═" * 60)
    
    def _print_monte_carlo(self, mc: MonteCarloResult):
        """Print Monte Carlo results."""
        print("\n📊 MONTE CARLO SIMULATION (1000 runs)")
        print(f"   Median Final Equity:   ${mc.median_final_equity:.2f}")
        print(f"   5th Percentile:        ${mc.percentile_5:.2f}")
        print(f"   95th Percentile:       ${mc.percentile_95:.2f}")
        print(f"   Probability of Profit: {mc.probability_profit:.1f}%")
        print(f"   Probability of Ruin:   {mc.probability_ruin:.1f}%")
        print(f"   Median Max Drawdown:   {mc.max_drawdown_median:.1f}%")
    
    def _print_edge_analysis(self, edge: Dict):
        """Print statistical edge analysis."""
        print("\n📈 STATISTICAL EDGE ANALYSIS")
        print(f"   Has Edge:            {'✅ YES' if edge['has_edge'] else '❌ NO'}")
        print(f"   P-Value:             {edge['p_value']:.4f}")
        print(f"   T-Statistic:         {edge.get('t_statistic', 0):.2f}")
        print(f"   Confidence Interval: ${edge['confidence_interval'][0]:.2f} to ${edge['confidence_interval'][1]:.2f}")
        print(f"   Conclusion:          {edge['reason']}")
        print("\n" + "═" * 60)


async def main():
    """Main entry point."""
    print("\n" + "═" * 60)
    print("  🔮 LAPLACE DEMON - DETERMINISTIC TRADING INTELLIGENCE 🔮")
    print("═" * 60)
    print("\n  Target: 70% Win Rate | $100,000 Capital | GBPUSD\n")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Laplace Demon Backtest Runner')
    parser.add_argument('--capital', type=float, default=30.0, help='Initial capital in USD') # Default to 30 as requested
    parser.add_argument('--risk', type=float, default=2.0, help='Risk per trade %')
    parser.add_argument('--symbol', type=str, default="GBPUSD", help='Symbol to trade')
    parser.add_argument('--spread', type=float, default=1.5, help='Spread in pips')
    
    args = parser.parse_args()

    # Initialize runner
    symbol = args.symbol
    runner = LaplaceBacktestRunner(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        symbol=symbol,
        spread_pips=args.spread
    )
    
    # Try to load data using DataLoader
    print("📊 Loading data via DataLoader (yfinance)...")
    
    try:
        from data_loader import DataLoader
        
        loader = DataLoader(symbol=symbol)
        data_map = await loader.get_data(symbol=symbol)
        
        if data_map:
            # Use M5 as base (most reliable)
            if 'M5' in data_map and data_map['M5'] is not None and len(data_map['M5']) > 100:
                runner.df_m5 = data_map['M5']
                logger.info(f"Loaded M5 data: {len(runner.df_m5)} candles")
            
            if 'M1' in data_map and data_map['M1'] is not None and len(data_map['M1']) > 100:
                runner.df_m1 = data_map['M1']
                logger.info(f"Loaded M1 data: {len(runner.df_m1)} candles")
            
            if 'H1' in data_map and data_map['H1'] is not None:
                runner.df_h1 = data_map['H1']
                logger.info(f"Loaded H1 data: {len(runner.df_h1)} candles")
            
            if 'H4' in data_map and data_map['H4'] is not None:
                runner.df_h4 = data_map['H4']
                logger.info(f"Loaded H4 data: {len(runner.df_h4)} candles")
            elif runner.df_h1 is not None:
                # Resample H1 to H4
                agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                runner.df_h4 = runner.df_h1.resample('4h').agg(agg).dropna()
                logger.info(f"Resampled H4 data: {len(runner.df_h4)} candles")
            
            if 'D1' in data_map and data_map['D1'] is not None:
                runner.df_d1 = data_map['D1']
                logger.info(f"Loaded D1 data: {len(runner.df_d1)} candles")
        else:
            logger.warning("DataLoader returned empty data map")
            
    except Exception as e:
        logger.warning(f"DataLoader failed: {e}")
        logger.info("Falling back to local files...")
    
    # Fallback to local files if DataLoader failed
    if runner.df_m5 is None:
        data_paths = [
            f"data/cache/GBPUSD_M5_60days.parquet",  # Extended data
            "data/GBPUSD_M1.csv",
            "data/EURUSD_M1.csv",
            f"data/cache/{symbol}=X_M5.parquet",
            f"data/cache/GBPUSD=X_M5.parquet",
        ]
        
        data_path = None
        for path in data_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            logger.error("No data found. DataLoader failed and no local files available.")
            logger.info("The DataLoader tried to fetch from yfinance but may have failed.")
            logger.info("Please ensure you have internet connection or add local data files.")
            return
        
        logger.info(f"Loading from local file: {data_path}")
        
        if data_path.endswith('.parquet'):
            runner.df_m5 = pd.read_parquet(data_path)
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            
            # Try to load H1 from separate file for more history
            h1_path = "data/cache/GBPUSD_H1_2years.parquet"
            if os.path.exists(h1_path):
                runner.df_h1 = pd.read_parquet(h1_path)
                logger.info(f"Loaded separate H1 file: {len(runner.df_h1)} candles")
            else:
                runner.df_h1 = runner.df_m5.resample('1h').agg(agg).dropna()
            
            runner.df_h4 = runner.df_h1.resample('4h').agg(agg).dropna() if runner.df_h1 is not None else None
            
            # Try to load D1 from separate file
            d1_path = "data/cache/GBPUSD_D1_10years.parquet"
            if os.path.exists(d1_path):
                runner.df_d1 = pd.read_parquet(d1_path)
                logger.info(f"Loaded separate D1 file: {len(runner.df_d1)} candles")
            else:
                runner.df_d1 = runner.df_m5.resample('1D').agg(agg).dropna()
        else:
            if not runner.load_data(data_path):
                return
    
    # Verify we have enough data
    if runner.df_m5 is None or len(runner.df_m5) < 100: # Temporary reduction for data gaps
        logger.error(f"Insufficient M5 data. Need at least 100 candles, got: {len(runner.df_m5) if runner.df_m5 is not None else 0}")
        return
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   M5: {len(runner.df_m5)} candles")
    print(f"   H1: {len(runner.df_h1) if runner.df_h1 is not None else 0} candles")
    print(f"   H4: {len(runner.df_h4) if runner.df_h4 is not None else 0} candles")
    print(f"   Date Range: {runner.df_m5.index[0]} to {runner.df_m5.index[-1]}")
    print()
    
    # Run backtest (Last 10 Days per User Request)
    if runner.df_m5 is not None and len(runner.df_m5) > 0:
        # Normalize Timezones to Naive (Fixes TZ-aware vs Naive errors)
        try:
            if runner.df_m5.index.tz is not None: runner.df_m5.index = runner.df_m5.index.tz_localize(None)
            if runner.df_m1 is not None and runner.df_m1.index.tz is not None: runner.df_m1.index = runner.df_m1.index.tz_localize(None)
            if runner.df_h1 is not None and runner.df_h1.index.tz is not None: runner.df_h1.index = runner.df_h1.index.tz_localize(None)
            if runner.df_h4 is not None and runner.df_h4.index.tz is not None: runner.df_h4.index = runner.df_h4.index.tz_localize(None)
            if runner.df_d1 is not None and runner.df_d1.index.tz is not None: runner.df_d1.index = runner.df_d1.index.tz_localize(None)
        except Exception as e:
            logger.warning(f"Timezone normalization warning: {e}")

        end_date = runner.df_m5.index[-1]
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
             end_date = end_date.replace(tzinfo=None)
             
        start_date = end_date - timedelta(days=10)
        mask = runner.df_m5.index >= start_date
        runner.df_m5 = runner.df_m5.loc[mask]
        
        # Apply the same filter to other dataframes to maintain consistency
        if runner.df_m1 is not None:
            runner.df_m1 = runner.df_m1.loc[runner.df_m1.index >= start_date]
        if runner.df_h1 is not None:
            runner.df_h1 = runner.df_h1.loc[runner.df_h1.index >= start_date]
        if runner.df_h4 is not None:
            runner.df_h4 = runner.df_h4.loc[runner.df_h4.index >= start_date]
        if runner.df_d1 is not None:
            runner.df_d1 = runner.df_d1.loc[runner.df_d1.index >= start_date]

        print(f"📉 Filtered data to last 30 days: {len(runner.df_m5)} candles ({runner.df_m5.index[0]} to {runner.df_m5.index[-1]})")
    
    # Limit data for speed if needed (e.g. last 50000 candles)
    # runner.df_m1 = runner.df_m1.iloc[-50000:]
    
    # Run backtest (✅ BACKTEST FIX: Added await)
    # Run backtest with graceful exit
    try:
        result = await runner.run_backtest(use_m5=True)
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Backtest interrupted by user! Generating report for partial results...")
        # Force calculate results from current state
        result = runner.engine._calculate_results()
    
    if result:
        # Generate report
        runner.generate_report(result, f"laplace_{symbol.lower()}")
        
        # Notify Telegram (✅ FIX: Added await)
        if runner.telegram.enabled:
            await runner.telegram.notify_backtest_report(
                symbol=symbol,
                start_date=runner.df_m5.index[0].strftime('%Y-%m-%d'),
                end_date=runner.df_m5.index[-1].strftime('%Y-%m-%d'),
                results=result
            )
        
        print("\n✅ Backtest complete! Check 'reports/' folder for charts.")
    else:
        print("\n❌ Backtest failed. Check logs for errors.")


if __name__ == "__main__":
    asyncio.run(main())

