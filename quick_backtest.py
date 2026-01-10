"""
Quick backtest with limited data for faster execution.
Uses 30 days of M5 data.
"""
import asyncio
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Quick-Backtest")

from backtest.engine import BacktestEngine, BacktestConfig, TradeDirection
from core.laplace_demon import LaplaceDemonCore

async def run_quick_backtest():
    print("\n" + "="*60)
    print("  🔮 LAPLACE DEMON - QUICK BACKTEST (30 days)")
    print("="*60)
    
    # Load data
    m5_path = "data/cache/GBPUSD_M5_60days.parquet"
    if not os.path.exists(m5_path):
        print(f"❌ Data not found: {m5_path}")
        print("Run download_data.py first")
        return
    
    df_m5 = pd.read_parquet(m5_path)
    
    # Limit to last 2 weeks (~2000 candles on M5)
    df_m5 = df_m5.iloc[-2000:]
    
    # Normalize columns
    if hasattr(df_m5.columns, 'droplevel'):
        try:
            df_m5.columns = df_m5.columns.droplevel(1)
        except:
            pass
    df_m5.columns = [str(c).lower() for c in df_m5.columns]
    
    # Remove timezone if present
    if df_m5.index.tz is not None:
        df_m5.index = df_m5.index.tz_localize(None)
    
    print(f"\nData: {len(df_m5)} M5 candles")
    print(f"Range: {df_m5.index[0]} to {df_m5.index[-1]}")
    
    # Resample for H1/H4
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_h1 = df_m5.resample('1h').agg(agg).dropna()
    df_h4 = df_h1.resample('4h').agg(agg).dropna()
    
    # Initialize
    config = BacktestConfig(
        initial_capital=30.0,
        leverage=3000.0,
        risk_per_trade_pct=2.0,
        max_concurrent_trades=3,
        spread_pips=1.5,
        slippage_pips=0.5,
        symbol="GBPUSD"
    )
    
    engine = BacktestEngine(config)
    laplace = LaplaceDemonCore("GBPUSD")
    
    print(f"\nStarting backtest...")
    
    # Run simulation
    warmup = 200
    total = len(df_m5)
    last_signal_idx = None
    min_interval = 5
    
    for i in range(warmup, total):
        candle = df_m5.iloc[i]
        current_time = candle.name
        current_price = float(candle['close'])
        
        # Progress
        if i % 1000 == 0:
            pct = (i / total) * 100
            print(f"  Progress: {pct:.1f}% | Balance: ${engine.balance:.2f}")
        
        # Data slices
        slice_m5 = df_m5.iloc[:i+1]
        slice_h1 = df_h1[df_h1.index <= current_time]
        slice_h4 = df_h4[df_h4.index <= current_time]
        
        # Update trades
        for trade in engine.active_trades[:]:
            exit_reason = engine.update_trade(trade, current_price, current_time)
            if exit_reason:
                exit_price = trade.sl_price if exit_reason == "SL_HIT" else trade.tp_price
                engine.close_trade(trade, exit_price, current_time, exit_reason)
        
        # Signal interval check
        if last_signal_idx and (i - last_signal_idx) < min_interval:
            continue
        
        # Capacity check
        if len(engine.active_trades) >= config.max_concurrent_trades:
            continue
        
        # Get prediction
        try:
            prediction = laplace.analyze(
                df_m1=None,
                df_m5=slice_m5,
                df_h1=slice_h1,
                df_h4=slice_h4,
                current_time=current_time,
                current_price=current_price
            )
            
            if prediction.execute and prediction.direction in ["BUY", "SELL"]:
                direction = TradeDirection.BUY if prediction.direction == "BUY" else TradeDirection.SELL
                
                trade = engine.open_trade(
                    direction=direction,
                    current_time=current_time,
                    current_price=current_price,
                    sl_pips=prediction.sl_pips,
                    tp_pips=prediction.tp_pips,
                    signal_source="LAPLACE",
                    confidence=prediction.confidence
                )
                
                if trade:
                    last_signal_idx = i
                    
        except Exception as e:
            continue
        
        # Equity tracking
        unrealized = sum(
            engine.calculate_pips(t.entry_price, current_price if t.direction == TradeDirection.BUY else current_price) 
            * config.get_pip_value_for_lots(t.lots)
            for t in engine.active_trades
        )
        
        equity = engine.balance + unrealized
        engine.equity_curve.append((current_time, equity))
        
        if equity > engine.peak_equity:
            engine.peak_equity = equity
        
        dd = (engine.peak_equity - equity) / engine.peak_equity * 100
        if dd > engine.max_drawdown:
            engine.max_drawdown = dd
    
    # Close remaining
    last_price = float(df_m5.iloc[-1]['close'])
    last_time = df_m5.index[-1]
    for trade in engine.active_trades[:]:
        engine.close_trade(trade, last_price, last_time, "END_OF_TEST")
    
    # Results
    result = engine._calculate_results()
    
    # Print summary
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"\n📊 PERFORMANCE")
    print(f"   Total Trades:    {result.total_trades}")
    print(f"   Winning:         {result.winning_trades}")
    print(f"   Losing:          {result.losing_trades}")
    print(f"   Win Rate:        {result.win_rate:.1f}%")
    
    target = "✅" if result.win_rate >= 70 else "❌"
    print(f"   Target (70%):    {target}")
    
    print(f"\n💰 PROFIT/LOSS")
    print(f"   Net Profit:      ${result.net_profit:.2f}")
    print(f"   Profit Factor:   {result.profit_factor:.2f}")
    print(f"   Expectancy:      ${result.expectancy:.2f}/trade")
    
    print(f"\n📉 RISK")
    print(f"   Max Drawdown:    {result.max_drawdown_pct:.1f}%")
    print(f"   Avg Win:         ${result.avg_win:.2f}")
    print(f"   Avg Loss:        ${result.avg_loss:.2f}")
    
    print(f"\n🔥 STREAKS")
    print(f"   Consecutive Wins:   {result.consecutive_wins}")
    print(f"   Consecutive Losses: {result.consecutive_losses}")
    
    # Days covered
    days = (df_m5.index[-1] - df_m5.index[0]).days
    print(f"\n⏱️ Period: {days} days")
    
    print("\n" + "="*60)
    
    return result

if __name__ == "__main__":
    asyncio.run(run_quick_backtest())
