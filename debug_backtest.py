"""
Debug backtest - Check why no trades are being generated.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Load test data (use the same as backtest)
csv_path = "data/GBPUSD_M1.csv"
df = pd.read_csv(csv_path)
df.columns = [c.lower() for c in df.columns]

if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

if 'volume' not in df.columns:
    df['volume'] = 0

# Resample
agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
df_m5 = df.resample('5min').agg(agg).dropna()
df_h1 = df.resample('1h').agg(agg).dropna()
df_h4 = df.resample('4h').agg(agg).dropna()

print(f"Data loaded: M5={len(df_m5)}, H1={len(df_h1)}, H4={len(df_h4)}")
print(f"Date range: {df_m5.index[0]} to {df_m5.index[-1]}")

# Import Laplace
from core.laplace_demon import LaplaceDemonCore

laplace = LaplaceDemonCore("GBPUSD")

# Test analysis on several candles
print("\n" + "="*60)
print("ANALYZING SAMPLE CANDLES")
print("="*60)

signals_count = 0
vetoes_count = 0
wait_count = 0

# Sample candles at different times
for idx in [210, 230, 250, 280, 300, 320, 350, 380]:
    if idx >= len(df_m5):
        continue
        
    candle = df_m5.iloc[idx]
    current_time = candle.name
    current_price = candle['close']
    
    # Get slices
    slice_m5 = df_m5.iloc[:idx+1]
    slice_h1 = df_h1[df_h1.index <= current_time]
    slice_h4 = df_h4[df_h4.index <= current_time]
    
    prediction = laplace.analyze(
        df_m1=None,
        df_m5=slice_m5,
        df_h1=slice_h1,
        df_h4=slice_h4,
        current_time=current_time,
        current_price=current_price
    )
    
    print(f"\n[{idx}] {current_time}")
    print(f"   Direction: {prediction.direction}")
    print(f"   Execute: {prediction.execute}")
    print(f"   Confidence: {prediction.confidence:.1f}%")
    print(f"   Confluence: {prediction.confluence_count}")
    
    print(f"   Scores: T={prediction.timing_score} S={prediction.structure_score} M={prediction.momentum_score} V={prediction.volatility_score}")
    
    if prediction.vetoes:
        vetoes_count += 1
        print(f"   VETOES: {prediction.vetoes}")
    
    if prediction.execute:
        signals_count += 1
        print(f"   *** SIGNAL: {prediction.direction} @ {current_price:.5f}")
    else:
        wait_count += 1
        if prediction.reasons:
            print(f"   Reasons: {prediction.reasons[:3]}")

print("\n" + "="*60)
print(f"SUMMARY: {signals_count} signals, {vetoes_count} vetoes, {wait_count} waits")
print("="*60)
