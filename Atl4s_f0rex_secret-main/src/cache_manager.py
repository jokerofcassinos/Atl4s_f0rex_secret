import pandas as pd
import os
from pathlib import Path
import logging

logger = logging.getLogger("CacheManager")

class CacheManager:
    def __init__(self, symbol="XAUUSD", timeframe="M5", data_dir="data"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.file_path = Path(data_dir) / f"{symbol}_{timeframe}.csv"
        
        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
    def load_cache(self) -> pd.DataFrame:
        """Loads the existing cache from CSV."""
        if self.file_path.exists():
            try:
                df = pd.read_csv(self.file_path, parse_dates=['Datetime'], index_col='Datetime')
                logger.info(f"Loaded {len(df)} rows from cache.")
                return df
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def update_cache(self, new_data: pd.DataFrame):
        """
        Updates the cache with new data. 
        Only appends data that is strictly newer than what's in the cache.
        """
        if new_data.empty:
            return

        existing_data = self.load_cache()
        
        if existing_data.empty:
            # First time save
            new_data.to_csv(self.file_path)
            logger.info("Cache initialized with new data.")
        else:
            # Incremental update
            last_timestamp = existing_data.index.max()
            
            # Filter new_data for rows after the last_timestamp
            fresh_rows = new_data[new_data.index > last_timestamp]
            
            if not fresh_rows.empty:
                # Append to file without rewriting the whole thing (mode='a')
                # Note: header=False because file already has header
                fresh_rows.to_csv(self.file_path, mode='a', header=False)
                logger.info(f"Appended {len(fresh_rows)} new candles to cache.")
            else:
                logger.info("No new data to append (Cache is up to date).")

    def get_full_history(self) -> pd.DataFrame:
        """Returns the complete dataset (Cache)."""
        return self.load_cache()
