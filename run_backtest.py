
import asyncio
import logging
from backtest_engine_predator import BacktestEngine
from data_loader import DataLoader
import pandas as pd
import os

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RunBacktest")

async def main():
    # 1. Initialize Engine
    engine = BacktestEngine()
    
    # 2. Load Data
    # Depending on where the csv is.
    csv_path = "data/GBPUSD_M1.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/EURUSD_M1.csv"
        
    if not os.path.exists(csv_path):
        logger.error(f"No data found at {csv_path}")
        return

    logger.info(f"Loading data from {csv_path}...")
    
    # Simple Pandas Load for now (DataLoader might be complex)
    df_m1 = pd.read_csv(csv_path)
    
    # Standardize columns
    df_m1.columns = [c.lower() for c in df_m1.columns]
    
    # Ensure 'time' is datetime index
    if 'time' in df_m1.columns:
        df_m1['time'] = pd.to_datetime(df_m1['time'])
        df_m1.set_index('time', inplace=True)
    elif 'date' in df_m1.columns:
        df_m1['date'] = pd.to_datetime(df_m1['date'])
        df_m1.set_index('date', inplace=True)
            
    # Resample
    logger.info("Resampling data...")
    
    mapping = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    df_m5 = df_m1.resample('5min').agg(mapping).dropna()
    df_h1 = df_m1.resample('1h').agg(mapping).dropna()
    df_h4 = df_m1.resample('4h').agg(mapping).dropna()
    
    data_map = {
        'M1': df_m1,
        'M5': df_m5,
        'H1': df_h1,
        'H4': df_h4
    }
    
    # 3. Run Simulation
    symbol = "GBPUSD" if "GBP" in csv_path else "EURUSD"
    await engine.run(data_map, symbol=symbol)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Crash: {e}")
        import traceback
        traceback.print_exc()
