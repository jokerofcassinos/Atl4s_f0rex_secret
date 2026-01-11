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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Laplace-Backtest")

# Import backtest engine
from backtest.engine import BacktestEngine, BacktestConfig, Trade, TradeDirection
from backtest.charts import ChartGenerator
from backtest.metrics import MetricsCalculator, MonteCarloResult

# Import Laplace Demon
from core.laplace_demon import LaplaceDemonCore, LaplacePrediction


class LaplaceBacktestRunner:
    """
    Professional backtest runner using Laplace Demon.
    """
    
    def __init__(self,
                 initial_capital: float = 30.0,
                 risk_per_trade: float = 2.0,
                 symbol: str = "GBPUSD",
                 spread_pips: float = 1.5):
        
        # Configuration
        self.config = BacktestConfig(
            initial_capital=initial_capital,
            leverage=3000.0,  # Unlimited
            risk_per_trade_pct=risk_per_trade,
            max_concurrent_trades=3,
            spread_pips=spread_pips,
            slippage_pips=0.5,
            symbol=symbol
        )
        
        # Initialize components
        self.engine = BacktestEngine(self.config)
        self.laplace = LaplaceDemonCore(symbol)
        self.charts = ChartGenerator("reports")
        
        # Data frames
        self.df_m1: Optional[pd.DataFrame] = None
        self.df_m5: Optional[pd.DataFrame] = None
        self.df_h1: Optional[pd.DataFrame] = None
        self.df_h4: Optional[pd.DataFrame] = None
        self.df_d1: Optional[pd.DataFrame] = None
        
        logger.info(f"Laplace Backtest Runner initialized: ${initial_capital} | {symbol}")
    
    def load_data(self, csv_path: str) -> bool:
        """
        Load and prepare data from CSV.
        
        Expects M1 data that will be resampled to all timeframes.
        """
        if not os.path.exists(csv_path):
            logger.error(f"Data file not found: {csv_path}")
            return False
        
        logger.info(f"Loading data from {csv_path}...")
        
        try:
            # Load M1 data
            df = pd.read_csv(csv_path)
            
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
            
            # Slice M1 for M8 generation (Last 300 minutes ~ 300 rows is fast enough)
            if not use_m5 and hasattr(self, 'df_m1'):
                 # Ensure we have M1 data loaded
                 slice_m1 = self.df_m1.loc[current_time - pd.Timedelta(minutes=300):current_time]
            else:
                 slice_m1 = None

            # Check active trades first
            for trade in self.engine.active_trades[:]:
                exit_reason = self.engine.update_trade(trade, current_price, current_time)
                if exit_reason:
                    # Simulate proper exit price
                    if trade.direction == TradeDirection.BUY:
                        exit_price = trade.sl_price if exit_reason == "SL_HIT" else trade.tp_price
                    else:
                        exit_price = trade.sl_price if exit_reason == "SL_HIT" else trade.tp_price
                    
                    self.engine.close_trade(trade, exit_price, current_time, exit_reason)
            
            # Check for new signals (respect minimum interval)
            if last_signal_time is not None:
                candles_since_signal = i - last_signal_time
                if candles_since_signal < min_signal_interval:
                    continue
            
            # Only generate signals if we have capacity
            if len(self.engine.active_trades) >= self.config.max_concurrent_trades:
                continue
            
            # Get Laplace Demon prediction (✅ BACKTEST FIX: Added await)
            try:
                prediction = await self.laplace.analyze(
                    df_m1=slice_m1,  # Passed M1 slice
                    df_m5=slice_m5,
                    df_h1=slice_h1,
                    df_h4=slice_h4,
                    df_d1=None,
                    current_time=current_time,
                    current_price=current_price
                )
                
                # Execute if signal is valid
                if prediction.execute and prediction.direction in ["BUY", "SELL"]:
                    direction = TradeDirection.BUY if prediction.direction == "BUY" else TradeDirection.SELL
                    
                    trade = self.engine.open_trade(
                        direction=direction,
                        current_time=current_time,
                        current_price=current_price,
                        sl_pips=prediction.sl_pips,
                        tp_pips=prediction.tp_pips,
                        signal_source=prediction.primary_signal or "LAPLACE",
                        confidence=prediction.confidence
                    )
                    
                    if trade:
                        last_signal_time = i
                        logger.debug(
                            f"TRADE #{trade.id}: {direction.value} @ {current_price:.5f} | "
                            f"SL: {prediction.sl_pips}p | TP: {prediction.tp_pips}p | "
                            f"Conf: {prediction.confidence:.0f}%"
                        )
                        
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
        print(f"   Total Trades:      {result.total_trades}")
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
    print("\n  Target: 70% Win Rate | $30 Capital | GBPUSD\n")
    
    # Initialize runner
    symbol = "GBPUSD"
    runner = LaplaceBacktestRunner(
        initial_capital=30.0,
        risk_per_trade=2.0,
        symbol=symbol,
        spread_pips=1.5
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
    if runner.df_m5 is None or len(runner.df_m5) < 300:
        logger.error(f"Insufficient M5 data. Need at least 300 candles, got: {len(runner.df_m5) if runner.df_m5 is not None else 0}")
        return
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   M5: {len(runner.df_m5)} candles")
    print(f"   H1: {len(runner.df_h1) if runner.df_h1 is not None else 0} candles")
    print(f"   H4: {len(runner.df_h4) if runner.df_h4 is not None else 0} candles")
    print(f"   Date Range: {runner.df_m5.index[0]} to {runner.df_m5.index[-1]}")
    print()
    
    # Filter for last 30 days for validation
    if runner.df_m5 is not None and len(runner.df_m5) > 0:
        end_date = runner.df_m5.index[-1]
        start_date = end_date - timedelta(days=30)
        mask = runner.df_m5.index >= start_date
        runner.df_m5 = runner.df_m5.loc[mask]
        print(f"📉 Filtered data to last 30 days: {len(runner.df_m5)} candles ({runner.df_m5.index[0]} to {runner.df_m5.index[-1]})")
    
    # Run backtest (✅ BACKTEST FIX: Added await)
    result = await runner.run_backtest(use_m5=True)
    
    if result:
        # Generate report
        runner.generate_report(result, f"laplace_{symbol.lower()}")
        
        print("\n✅ Backtest complete! Check 'reports/' folder for charts.")
    else:
        print("\n❌ Backtest failed. Check logs for errors.")


if __name__ == "__main__":
    asyncio.run(main())

