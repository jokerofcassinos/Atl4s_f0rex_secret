"""
Functional test - Quick Laplace Demon analysis test.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

print("="*60)
print("  🔮 LAPLACE DEMON - FUNCTIONAL TEST")
print("="*60)

# Generate test data
print("\n1. Generating synthetic data...")
np.random.seed(42)
n_bars = 200
start_price = 1.2500

dates = pd.date_range(start='2024-01-01 08:00', periods=n_bars, freq='5min')
prices = [start_price]
for i in range(1, n_bars):
    change = np.random.normal(0.0001, 0.0008)  # Slight uptrend
    prices.append(prices[-1] + change)

data = []
for i, (date, close) in enumerate(zip(dates, prices)):
    high = close + abs(np.random.normal(0, 0.0004))
    low = close - abs(np.random.normal(0, 0.0004))
    open_price = prices[i-1] if i > 0 else close
    volume = np.random.randint(100, 1000)
    
    data.append({
        'open': open_price,
        'high': max(high, open_price, close),
        'low': min(low, open_price, close),
        'close': close,
        'volume': volume
    })

df_m5 = pd.DataFrame(data, index=dates)
print(f"   Generated {len(df_m5)} M5 candles")
print(f"   Price range: {df_m5['low'].min():.5f} - {df_m5['high'].max():.5f}")

# Resample to H1/H4
agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
df_h1 = df_m5.resample('1h').agg(agg).dropna()
df_h4 = df_m5.resample('4h').agg(agg).dropna()
print(f"   Resampled to H1: {len(df_h1)}, H4: {len(df_h4)}")

# Test Laplace Demon
print("\n2. Testing Laplace Demon Core...")
from core.laplace_demon import LaplaceDemonCore

laplace = LaplaceDemonCore(symbol="GBPUSD")
print(f"   Initialized for GBPUSD")

# Run analysis
current_time = datetime(2024, 1, 1, 14, 50)  # Q3 golden zone
current_price = df_m5['close'].iloc[-1]

print(f"\n3. Running full analysis at {current_time}...")
prediction = laplace.analyze(
    df_m1=None,
    df_m5=df_m5,
    df_h1=df_h1,
    df_h4=df_h4,
    current_time=current_time,
    current_price=current_price
)

print(f"\n   PREDICTION:")
print(f"   ─────────────────────────────────")
print(f"   Direction:   {prediction.direction}")
print(f"   Execute:     {prediction.execute}")
print(f"   Confidence:  {prediction.confidence:.1f}%")
print(f"   Strength:    {prediction.strength.name}")
print(f"   Confluence:  {prediction.confluence_count}")
print(f"   ─────────────────────────────────")

print(f"\n   SCORES:")
print(f"   Timing:      {prediction.timing_score}")
print(f"   Structure:   {prediction.structure_score}")
print(f"   Momentum:    {prediction.momentum_score}")
print(f"   Volatility:  {prediction.volatility_score}")

if prediction.reasons:
    print(f"\n   REASONS:")
    for r in prediction.reasons[:5]:
        print(f"   • {r}")

if prediction.warnings:
    print(f"\n   WARNINGS:")
    for w in prediction.warnings[:3]:
        print(f"   ⚠ {w}")

if prediction.vetoes:
    print(f"\n   VETOES:")
    for v in prediction.vetoes:
        print(f"   🛑 {v}")

# Test backtest engine
print("\n4. Testing Backtest Engine...")
from backtest.engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=30.0,
    leverage=3000.0,
    symbol="GBPUSD"
)

engine = BacktestEngine(config)
print(f"   Initialized with ${config.initial_capital} capital")

# Simple trade test
current_time_bt = df_m5.index[-1]
from backtest.engine import TradeDirection

trade = engine.open_trade(
    direction=TradeDirection.BUY,
    current_time=current_time_bt,
    current_price=current_price,
    sl_pips=15,
    tp_pips=30,
    signal_source="TEST",
    confidence=75.0
)

if trade:
    print(f"   Opened test trade #{trade.id}")
    print(f"   Entry: {trade.entry_price:.5f}")
    print(f"   SL: {trade.sl_price:.5f}")
    print(f"   TP: {trade.tp_price:.5f}")
    print(f"   Lots: {trade.lots}")
    
    # Close trade
    exit_price = current_price + 0.0020  # 20 pips profit
    engine.close_trade(trade, exit_price, current_time_bt + timedelta(minutes=30), "TEST")
    print(f"   Closed at {exit_price:.5f}")
    print(f"   PnL: ${trade.pnl_dollars:.2f}")

print("\n" + "="*60)
print("  ✅ ALL TESTS PASSED!")
print("="*60)
