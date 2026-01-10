"""
Download extended GBPUSD data for backtesting.
Fetches ~60 days of M5 data from yfinance.
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

print("="*60)
print("  📊 DOWNLOADING GBPUSD DATA FOR BACKTEST")
print("="*60)

# Configuration
symbol = "GBPUSD=X"  # Yahoo Finance ticker for GBPUSD
output_dir = "data/cache"
os.makedirs(output_dir, exist_ok=True)

# yfinance limits:
# - M1: 7 days
# - M5: 60 days  
# - H1: 730 days

print(f"\nDownloading {symbol} data...")

# Download M5 data (max 60 days)
print("\n1. Fetching M5 data (60 days)...")
df_m5 = yf.download(
    symbol,
    period="60d",
    interval="5m",
    progress=True
)

if df_m5 is not None and len(df_m5) > 0:
    # Clean columns
    df_m5.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_m5.columns]
    
    # Ensure proper column names
    if 'adj close' in df_m5.columns:
        df_m5 = df_m5.drop('adj close', axis=1)
    
    # Remove timezone if present
    if df_m5.index.tz is not None:
        df_m5.index = df_m5.index.tz_localize(None)
    
    # Save as parquet
    m5_path = os.path.join(output_dir, "GBPUSD_M5_60days.parquet")
    df_m5.to_parquet(m5_path)
    print(f"   ✅ Saved {len(df_m5)} M5 candles to {m5_path}")
    print(f"   Date range: {df_m5.index[0]} to {df_m5.index[-1]}")
else:
    print("   ❌ Failed to download M5 data")
    df_m5 = None

# Download H1 data (more history available)
print("\n2. Fetching H1 data (2 years)...")
df_h1 = yf.download(
    symbol,
    period="2y",
    interval="1h",
    progress=True
)

if df_h1 is not None and len(df_h1) > 0:
    df_h1.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_h1.columns]
    if 'adj close' in df_h1.columns:
        df_h1 = df_h1.drop('adj close', axis=1)
    if df_h1.index.tz is not None:
        df_h1.index = df_h1.index.tz_localize(None)
    
    h1_path = os.path.join(output_dir, "GBPUSD_H1_2years.parquet")
    df_h1.to_parquet(h1_path)
    print(f"   ✅ Saved {len(df_h1)} H1 candles to {h1_path}")
    print(f"   Date range: {df_h1.index[0]} to {df_h1.index[-1]}")
else:
    print("   ❌ Failed to download H1 data")
    df_h1 = None

# Download D1 data (max history)
print("\n3. Fetching D1 data (10 years)...")
df_d1 = yf.download(
    symbol,
    period="10y",
    interval="1d",
    progress=True
)

if df_d1 is not None and len(df_d1) > 0:
    df_d1.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_d1.columns]
    if 'adj close' in df_d1.columns:
        df_d1 = df_d1.drop('adj close', axis=1)
    if df_d1.index.tz is not None:
        df_d1.index = df_d1.index.tz_localize(None)
    
    d1_path = os.path.join(output_dir, "GBPUSD_D1_10years.parquet")
    df_d1.to_parquet(d1_path)
    print(f"   ✅ Saved {len(df_d1)} D1 candles to {d1_path}")
    print(f"   Date range: {df_d1.index[0]} to {df_d1.index[-1]}")
else:
    print("   ❌ Failed to download D1 data")

# Summary
print("\n" + "="*60)
print("  DOWNLOAD SUMMARY")
print("="*60)

if df_m5 is not None:
    days = (df_m5.index[-1] - df_m5.index[0]).days
    print(f"\n✅ M5: {len(df_m5)} candles ({days} days)")
    
if df_h1 is not None:
    days = (df_h1.index[-1] - df_h1.index[0]).days
    print(f"✅ H1: {len(df_h1)} candles ({days} days)")
    
if df_d1 is not None:
    days = (df_d1.index[-1] - df_d1.index[0]).days
    print(f"✅ D1: {len(df_d1)} candles ({days} days)")

print(f"\nData saved to: {output_dir}/")
print("="*60)
