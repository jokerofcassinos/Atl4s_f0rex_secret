import pandas as pd
import numpy as np
import os
import asyncio
from datetime import datetime, timedelta
from analysis.agi.akashic_core import AkashicCore, RealitySnapshot
from data_loader import DataLoader

def train_mind(data_file_path: str = None, symbol: str = "GBPUSD", timeframe: str = "1h"):
    """
    Miners the past to build the Akashic Records.
    If data_file_path is None, uses DataLoader to fetch from YFinance.
    """
    print(f"🚀 [TRAINER] Starting training session...")
    
    df = None
    
    if data_file_path and os.path.exists(data_file_path):
        print(f"📂 [TRAINER] Loading from file: {data_file_path}")
        if data_file_path.endswith('.parquet'):
            df = pd.read_parquet(data_file_path)
        elif data_file_path.endswith('.csv'):
            df = pd.read_csv(data_file_path)
    else:
        print(f"🌐 [TRAINER] connecting to Quantum Data Source (Yahoo Finance) for {symbol}...")
        loader = DataLoader()
        # Create a temporary async loop to fetch data
        try:
             # We want MAX history. DataLoader default is limited.
             # We will bypass the default get_data limits for the "Ultra Training"
             print("⚡ [TRAINER] Requesting MAX history (Ancient Scrolls)...")
             yf_ticker = loader._normalize_symbol(symbol)
             
             dfs_to_combine = []
             
             # The "Fractal Spectrum" - Intervals to mine
             # We capture the market from the perspective of an ant (5m) to a giant (1mo).
             fractal_intervals = [
                 "5m", "15m", "30m", "60m", "90m", 
                 "1h", "1d", "5d", "1wk", "1mo"
             ]
             
             print(f"🌌 [TRAINER] Initiating Fractal Data Harvest ({len(fractal_intervals)} Timeframes)...")
             
             for interval in fractal_intervals:
                 sys_msg = f"   Fetching {interval} (Fractal Layer)..."
                 print(sys_msg)
                 
                 try:
                     # Attempt to get MAX history for each layer
                     df_layer = loader._fetch_single(yf_ticker, period="max", interval=interval)
                     
                     if df_layer is not None and not df_layer.empty:
                         count = len(df_layer)
                         print(f"     -> Acquired {count} candles.")
                         
                         # normalize timezone to allow mixing
                         if df_layer.index.tz is not None:
                             df_layer.index = df_layer.index.tz_localize(None)
                             
                         dfs_to_combine.append(df_layer)
                     else:
                         print(f"     -> Void (Empty or Failed).")
                         
                 except Exception as e:
                     print(f"     -> Error mining {interval}: {e}")
            
             if not dfs_to_combine:
                 df = None
             else:
                 # Concatenate all available frames
                 # Note: Timestamps will be distinct (D1 vs H1). 
                 # Akashic Core treats them as a sequence of independent "moments".
                 df = pd.concat(dfs_to_combine)
                 df.sort_index(inplace=True)
                 # Remove duplicates if any
                 df = df[~df.index.duplicated(keep='last')]
                 print(f"   Combined Fractal Data: {len(df)} candles.")
             
        except Exception as e:
            print(f"❌ [TRAINER] Data Fetch Error: {e}")
            return

    if df is None or df.empty:
        print("❌ [TRAINER] No data available.")
        return
        
    # Ensure correct columns
    required = ['open', 'high', 'low', 'close', 'volume']
    df.columns = [c.lower() for c in df.columns] 
    
    if 'time' in df.columns: df.rename(columns={'time': 'info_date'}, inplace=True)
    if 'volume' in df.columns: df.rename(columns={'volume': 'tick_volume'}, inplace=True)
    
    # YFinance index is the date
    if 'info_date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df['info_date'] = df.index
    
    # Pre-calculate indicators (Simulated for speed)
    # In real prod, use talib or pandas_ta
    df['rsi_14'] = 50.0 
    df['atr_14'] = 0.0001
    
    # Try to calculate real RSI if possible
    try:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50)
    except:
        pass
    
    # Initialize Core
    brain = AkashicCore()
    
    # Iterate
    # We need to look into the future for outcomes
    look_forward = 10 # 10 candles future
    
    print(f"⏳ [TRAINER] Processing {len(df)} candles...")
    
    records_added = 0
    
    for i in range(50, len(df) - look_forward):
        current = df.iloc[i]
        future = df.iloc[i + look_forward]
        
        # Current State
        price = current['close']
        
        # Future State
        future_price = future['close']
        price_change = future_price - price
        
        # Calculate Max Drawdown in the window
        window = df.iloc[i+1 : i+look_forward+1]
        min_in_window = window['low'].min()
        max_in_window = window['high'].max()
        
        if price_change > 0:
            # Bullish case
            drawdown = price - min_in_window # How much it went against us
            max_profit = max_in_window - price
            outcome = "BULL_WIN" if max_profit > (drawdown * 1.5) else "NEUTRAL"
        else:
            # Bearish case
            drawdown = max_in_window - price # How much it went against us (up)
            max_profit = price - min_in_window
            outcome = "BEAR_WIN" if max_profit > (drawdown * 1.5) else "NEUTRAL"
            
        # Create Snapshot
        snapshot = RealitySnapshot(
            timestamp=current['info_date'].timestamp(),
            price_open=current['open'],
            price_high=current['high'],
            price_low=current['low'],
            price_close=current['close'],
            volume=current.get('tick_volume', 0) if 'tick_volume' in current else current.get('volume', 0),
            body_size=abs(current['close'] - current['open']),
            upper_wick=current['high'] - max(current['open'], current['close']),
            lower_wick=min(current['open'], current['close']) - current['low'],
            total_range=current['high'] - current['low'],
            rsi_14=current['rsi_14'],
            atr_14=current.get('atr_14', 0.0001),
            sma_200_dist=0.0,
            hour=current['info_date'].hour,
            minute=current['info_date'].minute,
            day_of_week=current['info_date'].weekday(),
            is_8min_cycle=(current['info_date'].minute % 8 == 0),
            outcome_label=outcome,
            future_10m_change=price_change,
            future_max_drawdown=drawdown,
            future_max_profit=max_profit
        )
        
        brain.record_moment(snapshot)
        records_added += 1
        
        if i % 1000 == 0:
            print(f"   Processed {i} candles...")

    brain.save_memory()
    print(f"✅ [TRAINER] Training Complete. Added {records_added} new memories.")

if __name__ == "__main__":
    # If run directly
    train_mind(symbol="GBPUSD")
