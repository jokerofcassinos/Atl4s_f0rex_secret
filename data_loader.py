import yfinance as yf
import pandas as pd
import os
import logging
import config
from datetime import datetime, timedelta

logger = logging.getLogger("Atl4s-Data")

class DataLoader:
    def __init__(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_file = os.path.join(config.CACHE_DIR, f"{symbol}_{timeframe}.parquet")

    def get_data(self):
        """
        Orchestrates the data loading process for Multi-Timeframe (M5 + H1).
        Returns a dictionary: {'M5': df_m5, 'H1': df_h1}
        """
        data_map = {}
        
        # Timeframes to fetch
        timeframes = [
            ("M1", "1m", 7), # High-Fidelity Micro-Structure (7 Days)
            ("M5", "5m", 59), 
            ("H1", "1h", 720), 
            ("D1", "1d", 3650), # 10 years for swing context
            ("MN", "1mo", 7300) # 20 years for secular context
        ]
        
        for tf_name, yf_interval, days_limit in timeframes:
            cache_file = os.path.join(config.CACHE_DIR, f"{self.symbol}_{tf_name}.parquet")
            
            # 1. Load Cache
            cached_df = self._load_cache(cache_file)
            
            # Fix: Ensure start_date is not in the future relative to NY time.
            # If last_date is today, using today as start might trigger "start > end" if it's early morning.
            # Safest bet: Go back 1 day from the calculated start just to be sure we cover the gap.
            # Overlap is handled by _clean_data / concat.
            
            if cached_df is not None and not cached_df.empty:
                last_date = cached_df.index[-1]
                # Go back 1 day from last known data to ensure continuity and avoid "future start" error
                start_date = (last_date - timedelta(days=1)).strftime('%Y-%m-%d')
                logger.debug(f"[{tf_name}] Cache found. Last: {last_date}. Fetching from {start_date}...")
            else:
                logger.info(f"[{tf_name}] No cache. Fetching last {days_limit} days...")
                start_date = (datetime.now() - timedelta(days=days_limit)).strftime('%Y-%m-%d')

            yf_ticker = "GC=F"
            
            try:
                new_df = yf.download(yf_ticker, start=start_date, interval=yf_interval, progress=False)
                
                if not new_df.empty:
                    new_df = self._clean_data(new_df)
                    
                    if cached_df is not None:
                        combined_df = pd.concat([cached_df, new_df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    else:
                        combined_df = new_df
                    
                    self._save_cache(combined_df, cache_file)
                    data_map[tf_name] = combined_df
                else:
                    logger.warning(f"[{tf_name}] No new data fetched.")
                    data_map[tf_name] = cached_df

            except Exception as e:
                logger.error(f"[{tf_name}] Error fetching data: {e}")
                data_map[tf_name] = cached_df
        
        # Derive H4 from H1
        if 'H1' in data_map and data_map['H1'] is not None:
            try:
                data_map['H4'] = self.resample_to_tf(data_map['H1'], '4h')
                logger.info(f"[H4] Derived {len(data_map['H4'])} candles from H1.")
            except Exception as e:
                logger.error(f"[H4] Error deriving data: {e}")
                data_map['H4'] = None

        # Derive W1 from D1
        if 'D1' in data_map and data_map['D1'] is not None:
            try:
                data_map['W1'] = self.resample_to_tf(data_map['D1'], 'W')
                logger.info(f"[W1] Derived {len(data_map['W1'])} candles from D1.")
            except Exception as e:
                logger.error(f"[W1] Error deriving data: {e}")
                data_map['W1'] = None
        
        return data_map

    def resample_to_tf(self, df, tf_string):
        """
        Resamples data to a specific timeframe string (e.g., '4h', 'W').
        """
        if df is None or df.empty:
            return None
            
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        return df.resample(tf_string).agg(agg_dict).dropna()

    def _load_cache(self, filepath):
        if os.path.exists(filepath):
            try:
                return pd.read_parquet(filepath)
            except Exception as e:
                logger.error(f"Corrupt cache file {filepath}: {e}")
                return None
        return None

    def _save_cache(self, df, filepath):
        try:
            df.to_parquet(filepath)
            logger.info(f"Cache updated for {filepath}. Rows: {len(df)}")
        except Exception as e:
            logger.error(f"Error saving cache {filepath}: {e}")

    def _clean_data(self, df):
        # Flatten MultiIndex columns if present (yfinance issue)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure standard columns
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        })
        
        # Ensure we have the required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        # Filter only existing columns to avoid KeyErrors if rename failed silently
        available = [c for c in required if c in df.columns]
        return df[available]

if __name__ == "__main__":
    # Test
    loader = DataLoader()
    data = loader.get_data()
    print("M5 Tail:", data['M5'].tail() if data['M5'] is not None else "None")
    print("H1 Tail:", data['H1'].tail() if data['H1'] is not None else "None")
