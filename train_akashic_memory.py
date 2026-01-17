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
            # ROBUST PARSING (Same as Historical Runner)
            try:
                # Try default
                df = pd.read_csv(data_file_path, parse_dates=False)
                if len(df.columns) == 1:
                    df = pd.read_csv(data_file_path, sep='\t', parse_dates=False)
                    if len(df.columns) == 1:
                        df = pd.read_csv(data_file_path, sep=';', parse_dates=False)
                
                # CHECK FOR HEADERLESS CSV (Ported from Historical Runner)
                first_col = str(df.columns[0])
                if first_col.startswith(('20', '19')) and len(first_col) >= 4:
                    print("⚡ [TRAINER] Detected Headerless CSV. Reloading...")
                    df = pd.read_csv(data_file_path, header=None, parse_dates=False)
                    if len(df.columns) >= 7:
                         cols = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VOL_REAL', 'SPREAD']
                         df.columns = cols[:len(df.columns)]
                
                # Normalize Columns
                df.columns = [c.upper().strip() for c in df.columns]
                rename_map = {
                    '<DATE>': 'DATE', '<TIME>': 'TIME', '<OPEN>': 'OPEN', '<HIGH>': 'HIGH', '<LOW>': 'LOW', '<CLOSE>': 'CLOSE', '<TICKVOL>': 'VOLUME', '<VOL>': 'VOLUME',
                    'DATE': 'DATE', 'TIME': 'TIME', 'OPEN': 'OPEN', 'HIGH': 'HIGH', 'LOW': 'LOW', 'CLOSE': 'CLOSE', 'VOLUME': 'VOLUME', 'VOL': 'VOLUME'
                }
                df.rename(columns=rename_map, inplace=True)
                
                # Combine Date/Time
                if 'DATE' in df.columns and 'TIME' in df.columns:
                     try:
                         df['info_date'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
                     except:
                         df['info_date'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), format='%Y.%m.%d %H:%M:%S', errors='coerce')
                     df.set_index('info_date', inplace=True, drop=False)
                elif 'DATETIME' in df.columns:
                     df['info_date'] = pd.to_datetime(df['DATETIME'])
                     df.set_index('info_date', inplace=True, drop=False)
                
                # Lowercase columns for system compatibility
                df.columns = [c.lower() for c in df.columns]
                
            except Exception as e:
                print(f"❌ [TRAINER] CSV Parse Error: {e}")
                return
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
    
    
    # Remove duplicates in columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Initialize Core
    brain = AkashicCore()
    
    # Iterate
    # We need to look into the future for outcomes
    look_forward = 10 # 10 candles future
    
    print(f"⏳ [TRAINER] Processing {len(df)} candles (Vectorized)...")
    
    # 1. VECTORIZED PRE-CALCULATION (The Speed Force)
    # Shift future close
    df['future_close'] = df['close'].shift(-look_forward)
    df['price_change'] = df['future_close'] - df['close']
    
    # Forward Rolling Window for Min/Max
    # We use a reversed rolling trick to look forward or just shift back
    # rolling(10) at index i gives statistics for i-9 to i.
    # We want statistics for i+1 to i+10.
    # So we calculate rolling(10), then shift it back by 'look_forward'.
    # Actually, we need to shift by -look_forward.
    # But wait, rolling(10) includes current candle. We want strict future?
    # Usually "future window" implies uncertainty.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=look_forward)
    df['future_max'] = df['high'].rolling(window=indexer).max()
    df['future_min'] = df['low'].rolling(window=indexer).min()
    
    # Calculate Drawdown/Profit vectors
    # Bullish Scenario (We went Long): Drawdown is (Entry - MinLow)
    # Bearish Scenario (We went Short): Drawdown is (MaxHigh - Entry)
    
    # We define outcome based on DIRECTION of change
    # If price went UP, we check if we survived the drawdown
    # If price went DOWN, we check if we survived the "drawdown" (squeeze)
    
    outcomes = []
    
    # Vectorized extraction to Numpy for raw speed
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['tick_volume'].values if 'tick_volume' in df.columns else np.zeros(len(df))
    
    future_closes = df['future_close'].values
    future_maxs = df['future_max'].values
    future_mins = df['future_min'].values
    
    # Timestamps (Vectorized conversion)
    # Force conversion to nanoseconds int64 then divide by 1e9 for float seconds
    timestamps = df.index.view(np.int64) // 10**9 
    
    # Additional features
    rsi = df['rsi_14'].fillna(50).values
    atr = df['atr_14'].fillna(0.0001).values
    
    records_added = 0
    
    # Limit loop to valid data
    limit = len(df) - look_forward
    
    # Batch processing list
    snapshots = []
    
    for i in range(50, limit):
        p_close = closes[i]
        p_future = future_closes[i]
        p_change = p_future - p_close
        
        f_max = future_maxs[i]
        f_min = future_mins[i]
        
        # Determine Outcome Label
        if p_change > 0:
            # Bullish Outcome
            drawdown = p_close - f_min
            profit = f_max - p_close
            label = "BULL_WIN" if profit > (drawdown * 1.5) else "NEUTRAL"
            max_dd = drawdown
            max_prof = profit
        else:
            # Bearish Outcome
            drawdown = f_max - p_close
            profit = p_close - f_min
            label = "BEAR_WIN" if profit > (drawdown * 1.5) else "NEUTRAL"
            max_dd = drawdown
            max_prof = profit
            
        # Create Snapshot (Fast)
        # Using raw numpy values
        snap = RealitySnapshot(
            timestamp=float(timestamps[i]),
            price_open=float(opens[i]),
            price_high=float(highs[i]),
            price_low=float(lows[i]),
            price_close=float(p_close),
            volume=float(volumes[i]),
            body_size=float(abs(p_close - opens[i])),
            upper_wick=float(highs[i] - max(opens[i], p_close)),
            lower_wick=float(min(opens[i], p_close) - lows[i]),
            total_range=float(highs[i] - lows[i]),
            rsi_14=float(rsi[i]),
            atr_14=float(atr[i]),
            sma_200_dist=0.0,
            hour=int(pd.Timestamp(timestamps[i], unit='s').hour), # Fast enough locally or pre-calc
            minute=int(pd.Timestamp(timestamps[i], unit='s').minute),
            day_of_week=int(pd.Timestamp(timestamps[i], unit='s').dayofweek),
            is_8min_cycle=(int(pd.Timestamp(timestamps[i], unit='s').minute) % 8 == 0),
            outcome_label=label,
            future_10m_change=float(p_change),
            future_max_drawdown=float(max_dd),
            future_max_profit=float(max_prof)
        )
        snapshots.append(snap)
        
        if len(snapshots) >= 10000:
            for s in snapshots:
                brain.record_moment(s)
            records_added += len(snapshots)
            snapshots = []
            print(f"   Processed {i} candles...")

    # Flush remaining
    for s in snapshots:
        brain.record_moment(s)
    records_added += len(snapshots)

    brain.save_memory()
    print(f"✅ [TRAINER] Training Complete. Added {records_added} new memories.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', type=str, help='Path to historical CSV/Parquet file')
    args = parser.parse_args()

    if args.file:
        train_mind(data_file_path=args.file)
    else:
        # Default behavior
        train_mind(symbol="GBPUSD")
