"""
LAPLACE DEMON - Historical Stress Test Runner
═════════════════════════════════════════════

Specialized runner for backtesting on historical CSV data (e.g., 2016, 2020, 2022).
Features:
- Loads M1 CSV data (MT5/HistData format)
- Auto-resamples to M5, H1, H4, D1
- Bypasses yfinance/live data loaders
- Optimizes for speed on large datasets
"""

import asyncio
import pandas as pd
import numpy as np
import os
import logging
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict

# Setup logging
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler("historical_backtest.log")

c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%H:%M:%S')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[c_handler, f_handler])

# Enable DEBUG for Laplace
logging.getLogger("LaplaceDemon").setLevel(logging.DEBUG)
logging.getLogger("Laplace-Backtest").setLevel(logging.DEBUG)

logger = logging.getLogger("Laplace-Historical")

# Import Core Components
from backtest.engine import BacktestEngine, BacktestConfig, Trade, TradeDirection
from backtest.charts import ChartGenerator
from backtest.metrics import MetricsCalculator
from core.laplace_demon import LaplaceDemonCore

# Import the base runner class to reuse logic
from run_laplace_backtest import LaplaceBacktestRunner

class HistoricalBacktestRunner(LaplaceBacktestRunner):
    """
    Specialized runner for Historical CSVs.
    Overrides load_data to enforce local CSV loading and resampling.
    """

    async def load_csv_data(self, file_path: str, start_date: str = None, end_date: str = None) -> bool:
        """
        Load M1 data from CSV and resample to necessary timeframes.
        Supports standard MT5/HistData formats.
        Optionally filters by start_date and end_date (YYYY-MM-DD).
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        logger.info(f"Loading historical M1 data from: {file_path}")
        if start_date: logger.info(f"   Filter Start: {start_date}")
        if end_date: logger.info(f"   Filter End:   {end_date}")
        
        try:
            # 1. Load CSV
            # Try different separators and headers
            try:
                # Common HistData/MT5 format: DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,VOL,SPREAD
                # Or <DATE>\t<TIME>...
                # We'll try reading with pandas default first
                df = pd.read_csv(file_path, parse_dates=False)
                
                # Check formatting
                if len(df.columns) == 1:
                    # Likely tab separated or semicolon
                    df = pd.read_csv(file_path, sep='\t', parse_dates=False)
                    if len(df.columns) == 1:
                        df = pd.read_csv(file_path, sep=';', parse_dates=False)
                
                # CHECK FOR HEADERLESS CSV
                # If the first column name looks like a date (starts with 20 or 19 and length >=4)
                first_col = str(df.columns[0])
                if first_col.startswith(('20', '19')) and len(first_col) >= 4:
                    logger.info("Detected Headerless CSV (First row is data). Reloading with header=None...")
                    # Reload with correct separator
                    sep = ','
                    if len(df.columns) == 1: sep = '\t' 
                    
                    df = pd.read_csv(file_path, header=None, parse_dates=False)
                    
                    # Assign default MT5 headers based on column count
                    # Standard MT5 export often: Date, Time, Open, High, Low, Close, TickVol, Vol, Spread
                    if len(df.columns) >= 7:
                         # Assume Date, Time, Open, High, Low, Close, Vol
                         cols = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VOL_REAL', 'SPREAD']
                         df.columns = cols[:len(df.columns)]
                         logger.info(f"Assigned Default Headers: {list(df.columns)}")

            except Exception as e:
                logger.error(f"Error parsing CSV: {e}")
                return False
            
            # Normalize Columns
            df.columns = [c.upper().strip() for c in df.columns]
            logger.info(f"Loaded Columns: {list(df.columns)}")
            
            rename_map = {
                '<DATE>': 'DATE', '<TIME>': 'TIME', '<OPEN>': 'OPEN', '<HIGH>': 'HIGH', '<LOW>': 'LOW', '<CLOSE>': 'CLOSE', '<TICKVOL>': 'VOLUME', '<VOL>': 'VOLUME',
                'DATE': 'DATE', 'TIME': 'TIME', 'OPEN': 'OPEN', 'HIGH': 'HIGH', 'LOW': 'LOW', 'CLOSE': 'CLOSE', 'VOLUME': 'VOLUME', 'VOL': 'VOLUME'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Combine Date and Time
            if 'DATE' in df.columns and 'TIME' in df.columns:
                logger.info("Parsing Date and Time columns...")
                # Robust parsing
                try:
                    df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
                except:
                    # Try specific formats if auto fails
                    df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), format='%Y.%m.%d %H:%M:%S', errors='coerce')
                
                df.set_index('datetime', inplace=True)
                df.drop(columns=['DATE', 'TIME'], inplace=True)
            elif 'DATETIME' in df.columns:
                df['datetime'] = pd.to_datetime(df['DATETIME'])
                df.set_index('datetime', inplace=True)
            else:
                logger.error(f"Could not find DATE/TIME columns. Available: {list(df.columns)}")
                # Try to parse first column as date if unnamed?
                pass
                
            # Rename for system compatibility (lowercase)
            df.columns = [c.lower() for c in df.columns]
            
            # Drop timezone if present (Check if it Is DatetimeIndex first)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            elif not isinstance(df.index, pd.DatetimeIndex):
                 logger.error("Index is NOT DatetimeIndex. Parsing failed.")
                 return False

            # FILTER DATE RANGE (Optimized before resampling)
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
                
            if df.empty:
                logger.error("Data empty after filtering!")
                return False

            # Store M1
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            self.df_m1 = df
            logger.info(f"M1 Data Loaded: {len(df)} rows. Range: {df.index[0]} to {df.index[-1]}")

            # 2. Resample Data (Crucial step)
            logger.info("Resampling to M5, H1, H4, D1...")
            
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            if 'volume' not in df.columns:
                df['volume'] = 1
                
            self.df_m5 = df.resample('5min').agg(agg_dict).dropna()
            self.df_h1 = df.resample('1h').agg(agg_dict).dropna()
            self.df_h4 = df.resample('4h').agg(agg_dict).dropna()
            self.df_d1 = df.resample('1D').agg(agg_dict).dropna()
            
            logger.info(f"Resampling Complete: M5={len(self.df_m5)}, H1={len(self.df_h1)}, H4={len(self.df_h4)}")
            return True

        except Exception as e:
            logger.error(f"Critical Error loading/resampling data: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    print("\n" + "═" * 70)
    print("  📜 LAPLACE DEMON - HISTORICAL FORENSIC RUNNER 📜")
    print("     Stress Testing: 2016 (Brexit), 2020 (Covid), 2022 (Inflation)")
    print("═" * 70 + "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to M1 CSV file (e.g., data/GBPUSD_2016.csv)')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=1.0, help='Risk per trade % (Conservative for Stress Test)')
    parser.add_argument('--spread', type=float, default=1.0, help='Spread in pips')
    parser.add_argument('--start', type=str, default=None, help='Start Date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End Date (YYYY-MM-DD)')
    args = parser.parse_args()

    # Init Runner
    runner = HistoricalBacktestRunner(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        symbol="GBPUSD_HIST", # Generic symbol tag
        spread_pips=args.spread
    )

    # Load Data
    success = await runner.load_csv_data(args.file, start_date=args.start, end_date=args.end)
    if not success:
        logger.error("Failed to load or process data. Exiting.")
        return

    try:
        # Run Backtest
        # Note: We assume the file contains the full range we want to test.
        # The runner.run_backtest loop iterates over whatever df_m5 has.
        
        print(f"\n🚀 Launching Simulation on {len(runner.df_m5)} M5 candles...")
        result = await runner.run_backtest(use_m5=True)
        
        # Generate Report
        runner.generate_report(result, prefix=f"history_{os.path.basename(args.file).split('.')[0]}")
        print("\n✅ Historical Stress Test Complete.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user! Generating partial report...")
        
        # FIX: Generate a proper BacktestResult object from the engine
        # runner.laplace should be initialized and have a backtest_engine
        try:
             # FIX: Access engine directly from runner
             if hasattr(runner, 'engine'):
                 partial_result = runner.engine._calculate_results()
                 prefix = f"history_{os.path.basename(args.file).split('.')[0]}_PARTIAL"
                 runner.generate_report(partial_result, prefix=prefix)
                 print(f"\n✅ Partial Report Generated successfully: {prefix}")
             else:
                 print("⚠️ Could not access backtest engine for report.")
        except Exception as e:
             print(f"❌ Failed to generate partial report: {e}")
             import traceback
             traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
