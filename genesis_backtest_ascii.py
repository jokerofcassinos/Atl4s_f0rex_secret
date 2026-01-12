"""
Genesis Backtest - Validation Test


Tests the unified Genesis architecture on historical data.

Target: Match or exceed LaplaceDemon's 70% Win Rate
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Genesis-Backtest")

# Import Genesis
from main_genesis import GenesisSystem, GenesisSignal

# Import backtest infrastructure
from backtest.engine import BacktestEngine, BacktestConfig, TradeDirection


async def run_genesis_backtest():
    """
    Run Genesis backtest on 1300 candles (validated sample from Phase 0).
    
    Expected: 70%+ Win Rate (match LaplaceDemon)
    """
    
    print("\n" + "=" * 60)
    print("   GENESIS SYSTEM - BACKTEST VALIDATION")
    print("=" * 60)
    print("  Target: 70%+ Win Rate (LaplaceDemon baseline)")
    print("=" * 60)
    
    # 
    # SETUP
    # 
    
    symbol = "GBPUSD"
    
    # Backtest config
    config = BacktestConfig(
        initial_capital=30.0,
        leverage=3000.0,
        risk_per_trade_pct=2.0,
        max_concurrent_trades=3,
        spread_pips=1.5,
        slippage_pips=0.5,
        symbol=symbol
    )
    
    engine = BacktestEngine(config)
    genesis = GenesisSystem(symbol=symbol, mode="backtest")
    
    # 
    # LOAD DATA
    # 
    
    logger.info("Loading data...")
    
    try:
        import yfinance as yf
        
        # Download GBPUSD data directly
        ticker = yf.Ticker("GBPUSD=X")
        df_full = ticker.history(period="60d", interval="5m")
        
        if df_full.empty:
            logger.error("Failed to download data from yfinance")
            return
        
        # Prepare dataframes
        df_m5 = df_full[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_m5.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Resample to other timeframes
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        df_h1 = df_m5.resample('1h').agg(agg).dropna()
        df_h4 = df_h1.resample('4h').agg(agg).dropna()
        
        df_m1 = None  # Not available on yfinance free tier
        
        # Filter to 1300 candles (validated sample from Phase 0)
        df_m5 = df_m5.iloc[-1300:]
        
        logger.info(f"Data loaded: {len(df_m5)} M5 candles")
        logger.info(f"Range: {df_m5.index[0]} to {df_m5.index[-1]}")
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return
    
    # 
    # BACKTEST LOOP
    # 
    
    logger.info("Starting Genesis backtest...")
    
    warmup = 200  # Need data for indicators
    total_candles = len(df_m5)
    last_signal_time = None
    min_signal_interval = 5  # Candles between signals
    
    for i in range(warmup, total_candles):
        candle = df_m5.iloc[i]
        current_time = candle.name
        current_price = candle['close']
        
        # Progress logging
        if i % 500 == 0:
            pct = (i / total_candles) * 100
            logger.info(f"Progress: {pct:.1f}% | Balance: ${engine.balance:.2f}")
        
        # Prepare data slices (anti-lookahead)
        slice_m5 = df_m5.iloc[:i+1]
        slice_h1 = df_h1[df_h1.index <= current_time] if df_h1 is not None else None
        slice_h4 = df_h4[df_h4.index <= current_time] if df_h4 is not None else None
        
        # M1 slice (last 300 minutes for M8 generation)
        if df_m1 is not None and len(df_m1) > 0:
            slice_m1 = df_m1.loc[current_time - pd.Timedelta(minutes=300):current_time]
        else:
            slice_m1 = None
        
        # Update active trades
        for trade in engine.active_trades[:]:
            exit_reason = engine.update_trade(trade, current_price, current_time)
            if exit_reason:
                if trade.direction == TradeDirection.BUY:
                    exit_price = trade.sl_price if exit_reason == "SL_HIT" else trade.tp_price
                else:
                    exit_price = trade.sl_price if exit_reason == "SL_HIT" else trade.tp_price
                
                engine.close_trade(trade, exit_price, current_time, exit_reason)
        
        # Check for new signals (respect interval)
        if last_signal_time is not None:
            if (i - last_signal_time) < min_signal_interval:
                continue
        
        # Check capacity
        if len(engine.active_trades) >= config.max_concurrent_trades:
            continue
        
        # Genesis Analysis
        try:
            signal: GenesisSignal = await genesis.analyze(
                df_m1=slice_m1,
                df_m5=slice_m5,
                df_h1=slice_h1,
                df_h4=slice_h4,
                current_time=current_time,
                current_price=current_price
            )
            
            # Execute if valid
            if signal.execute and signal.direction in ["BUY", "SELL"]:
                direction = TradeDirection.BUY if signal.direction == "BUY" else TradeDirection.SELL
                
                trade = engine.open_trade(
                    direction=direction,
                    current_time=current_time,
                    current_price=current_price,
                    sl_pips=signal.sl_pips,
                    tp_pips=signal.tp_pips,
                    signal_source=signal.primary_signal or "GENESIS",
                    confidence=signal.confidence
                )
                
                if trade:
                    last_signal_time = i
                    logger.debug(
                        f"TRADE #{trade.id}: {direction.value} @ {current_price:.5f} | "
                        f"SL: {signal.sl_pips:.1f}p | TP: {signal.tp_pips:.1f}p | "
                        f"Conf: {signal.confidence:.0f}%"
                    )
        
        except Exception as e:
            logger.warning(f"Analysis error at {current_time}: {e}")
            continue
        
        # Record equity
        unrealized = sum(
            engine.calculate_pips(t.entry_price, current_price) 
            * config.get_pip_value_for_lots(t.lots)
            for t in engine.active_trades
        )
        
        equity = engine.balance + unrealized
        engine.equity_curve.append((current_time, equity))
        
        if equity > engine.peak_equity:
            engine.peak_equity = equity
        
        dd = (engine.peak_equity - equity) / engine.peak_equity * 100 if engine.peak_equity > 0 else 0
        if dd > engine.max_drawdown:
            engine.max_drawdown = dd
    
    # Close remaining trades
    last_price = df_m5.iloc[-1]['close']
    last_time = df_m5.index[-1]
    for trade in engine.active_trades[:]:
        engine.close_trade(trade, last_price, last_time, "END_OF_TEST")
    
    # 
    # RESULTS
    # 
    
    result = engine._calculate_results()
    
    print("\n" + "=" * 60)
    print("  GENESIS BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n PERFORMANCE")
    print(f"   Total Trades:    {result.total_trades}")
    print(f"   Winning:         {result.winning_trades}")
    print(f"   Losing:          {result.losing_trades}")
    print(f"   Win Rate:        {result.win_rate:.1f}%")
    
    target_met = "" if result.win_rate >= 70 else ""
    print(f"   Target (70%):    {target_met}")
    
    print(f"\n PROFIT/LOSS")
    print(f"   Net Profit:      ${result.net_profit:.2f}")
    print(f"   Profit Factor:   {result.profit_factor:.2f}")
    print(f"   Expectancy:      ${result.expectancy:.2f}/trade")
    
    print(f"\n RISK")
    print(f"   Max Drawdown:    {result.max_drawdown_pct:.1f}%")
    print(f"   Avg Win:         ${result.avg_win:.2f}")
    print(f"   Avg Loss:        ${result.avg_loss:.2f}")
    
    print(f"\n STREAKS")
    print(f"   Consecutive Wins:   {result.consecutive_wins}")
    print(f"   Consecutive Losses: {result.consecutive_losses}")
    
    # Calculate days
    duration = (df_m5.index[-1] - df_m5.index[0]).days
    print(f"\n Period: {duration} days")
    
    print("\n" + "=" * 60)
    
    # 
    # COMPARISON WITH LAPLACE DEMON (Phase 0 Baseline)
    # 
    
    print("\n COMPARISON WITH LAPLACE DEMON (Phase 0)")
    print("=" * 60)
    print(f"Metric               | LaplaceDemon | Genesis     | Delta")
    print("-" * 60)
    print(f"Win Rate             | 70.0%        | {result.win_rate:.1f}%       | {result.win_rate - 70:.1f}%")
    print(f"Total Trades         | 10           | {result.total_trades}          | {result.total_trades - 10:+d}")
    print(f"Net Profit           | $8.00        | ${result.net_profit:.2f}      | ${result.net_profit - 8:.2f}")
    print(f"Profit Factor        | 2.33         | {result.profit_factor:.2f}       | {result.profit_factor - 2.33:+.2f}")
    print("=" * 60)
    
    if result.win_rate >= 70:
        print("\n SUCCESS: Genesis matches/exceeds LaplaceDemon performance!")
    elif result.win_rate >= 65:
        print("\n ACCEPTABLE: Genesis slightly below target (acceptable variance)")
    else:
        print("\n NEEDS WORK: Genesis below minimum threshold (65%)")
    
    print("\n" + "=" * 60)
    
    # Save report
    with open("genesis_backtest_report.txt", "w") as f:
        f.write(f"Genesis Backtest Results\n")
        f.write(f"========================\n\n")
        f.write(f"Win Rate: {result.win_rate:.1f}%\n")
        f.write(f"Trades: {result.total_trades}\n")
        f.write(f"Profit: ${result.net_profit:.2f}\n")
        f.write(f"Profit Factor: {result.profit_factor:.2f}\n")
    
    logger.info("Report saved to genesis_backtest_report.txt")


if __name__ == "__main__":
    asyncio.run(run_genesis_backtest())
