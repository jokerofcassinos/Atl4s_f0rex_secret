
import yfinance as yf
import pandas as pd
import os
import logging
import config
import time
import random
from datetime import datetime, timedelta

logger = logging.getLogger("Atl4s-Data")

class DataLoader:
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Converts MT5 symbols (e.g., 'DOGUSD', 'BTCUSD') to Yahoo Finance Tickers (e.g., 'DOGE-USD', 'BTC-USD').
        """
        s = symbol.upper()
        
        # 1. Explicit Dictionary (The "Fixer" List)
        # Add any tricky ones here.
        mapping = {
            "DOGUSD": "DOGE-USD",
            "DOGEUSD": "DOGE-USD",
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "SOLUSD": "SOL-USD",
            "XRPUSD": "XRP-USD",
            "LTCUSD": "LTC-USD",
            "BNBUSD": "BNB-USD",
            "XAUUSD": "GC=F", # Gold Futures (Often better vol than XAUUSD=X) or "XAUUSD=X"
            "XAGUSD": "SI=F", # Silver
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "USDCHF": "USDCHF=X",
            "NZDUSD": "NZDUSD=X"
        }
        
        if s in mapping:
            return mapping[s]
            
        # 2. Generic Crypto Heuristic
        # If it ends in USD but is not in the map, try adding dash.
        # e.g. "ADAUSD" -> "ADA-USD"
        if s.endswith("USD") and len(s) > 6: 
             # Likely Crypto like "SHIBUSD" -> "SHIB-USD"
             return s.replace("USD", "-USD")
             
        # 3. Fallback for 3-letter Cryptos (BTCUSD -> BTC-USD) not in map
        if len(s) == 6 and s.endswith("USD"):
             # Could be "ADAUSD" -> "ADA-USD"
             # But checks against known forex?
             pass 
             
        return s 

    def __init__(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_file = os.path.join(config.CACHE_DIR, f"{symbol}_{timeframe}.parquet")

    def get_data(self, symbol=None, timeframe=None):
        """
        Orchestrates the data loading process.
        If symbol/timeframe provided, returns specific DataFrame.
        Otherwise returns dictionary map for default context.
        """
        raw_symbol = symbol if symbol else self.symbol
        target_symbol = self._normalize_symbol(raw_symbol)
        
        # If specific request
        if symbol and timeframe:
             # Just fetch this one
             yf_interval = "5m" if timeframe == "5m" else "1m"
             if timeframe == "1h": yf_interval = "1h"
             
             cache_file = os.path.join(config.CACHE_DIR, f"{target_symbol}_{timeframe}.parquet")
             return self._fetch_with_cache(target_symbol, yf_interval, cache_file)
             
        # Else Legacy Multi-TF Load
        data_map = {}
        
        # Timeframes to fetch
        timeframes = [
            ("M1", "1m", 7), 
            ("M5", "5m", 59), 
            ("M15", "15m", 59),
            ("M30", "30m", 59),
            ("H1", "1h", 720), 
            ("D1", "1d", 3650),
            ("MN", "1mo", 7300)
        ]
        
        for tf_name, yf_interval, days_limit in timeframes:
            cache_file = os.path.join(config.CACHE_DIR, f"{target_symbol}_{tf_name}.parquet")
            data_map[tf_name] = self._fetch_with_cache(target_symbol, yf_interval, cache_file, days_limit)
        
        # Derived
        if 'H1' in data_map and data_map['H1'] is not None:
             data_map['H4'] = self.resample_to_tf(data_map['H1'], '4h')
        if 'D1' in data_map and data_map['D1'] is not None:
             data_map['W1'] = self.resample_to_tf(data_map['D1'], 'W')
             
        # Phase 30: Apex Basket (Candidates)
        # User Request: Focus on BTCXAU (Bitcoin vs Gold). 
        # We also keep BTCUSD as reference.
        
        candidates = ["BTCXAU"]
        global_basket = {}
        
        # Helper to check if weekend (naive)
        is_weekend = False
        if pd.Timestamp.now().dayofweek >= 5: is_weekend = True
        
        for asset in candidates:
             # Skip Forex on weekends to avoid noise/stale data
             if is_weekend and asset in ["XAUUSD", "EURUSD"]: continue
             
             # Start date logic is inside fetch_single
             c_file = os.path.join(config.CACHE_DIR, f"{asset}_Apex.parquet")
             
             # Fetch H1 for robust trend analysis
             df = self._fetch_with_cache(asset, "1h", c_file, days=30)
             if df is not None and not df.empty:
                 global_basket[asset] = df
             
        data_map['global_basket'] = global_basket # Apex and Nexus both use this
        
        return data_map

    def _get_yf_ticker(self, symbol):
        ticker_map = {
            "XAUUSD": "GC=F",
            "BTCUSD": "BTC-USD",
            "BTCXAU": "BTC-USD", # PROXY: BTCXAU correlates 99% with BTCUSD on typical brokers (unless real gold denom)
            "ETHUSD": "ETH-USD",
            "EURUSD": "EURUSD=X"
        }
        return ticker_map.get(symbol, symbol)


    def _fetch_single(self, symbol, interval, cache_file, days=55):
        # YFinance Limits: 1m = 7 days max, 5m = 60 days max.
        # We set safer defaults.
        if interval == "1m": days = 5
        if interval == "5m": days = 55
        
        # 1. Load Cache
        cached_df = self._load_cache(cache_file)
        
        if cached_df is not None and not cached_df.empty:
            last_date = cached_df.index[-1]
            start_date = (last_date - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Map internal symbol using dynamic logic or map
        ticker_map = {
            "XAUUSD": "GC=F",
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "EURUSD": "EURUSD=X"
        }
        yf_ticker = ticker_map.get(symbol, symbol) # Fallback to raw symbol
        
    def _fetch_with_cache(self, symbol, interval, cache_file, days=55):
        """
        Manages the Caching Layer + Calling the Robust Fetcher.
        """
        # YFinance Limits
        if interval == "1m": days = 5
        if interval == "5m": days = 55
        
        # 1. Load Cache
        cached_df = self._load_cache(cache_file)
        
        start_date = None
        if cached_df is not None and not cached_df.empty:
            last_date = cached_df.index[-1]
            start_date = (last_date - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Map internal symbol
        ticker_map = {
            "XAUUSD": "GC=F",
            "BTCUSD": "BTC-USD",
            "BTCXAU": "BTC-USD", # PROXY
            "ETHUSD": "ETH-USD",
            "EURUSD": "EURUSD=X"
        }
        yf_ticker = ticker_map.get(symbol, symbol)
        
        # 2. Fetch New Data (Using Robust Fetcher)
        new_df = self._fetch_single(yf_ticker, start=start_date, interval=interval)
        
        # 3. Merge & Save
        if new_df is not None and not new_df.empty:
            new_df = self._clean_data(new_df)
            if cached_df is not None and not cached_df.empty:
                # Concatenate and remove duplicates
                combined_df = pd.concat([cached_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            else:
                combined_df = new_df
            
            self._save_cache(combined_df, cache_file)
            return combined_df
        
        # If fetch failed or returned no data, return cache if available
        if cached_df is not None:
             logger.warning(f"Using cached data for {symbol} due to fetch failure.")
             return cached_df
             
        return None

    def _fetch_single(self, ticker, start=None, end=None, period=None, interval="1d"):
        """
        Helper to fetch single ticker with robust retry logic.
        This method now directly wraps yf.download and handles retries and basic cleaning.
        Cache management is expected to be handled by the caller.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add random jitter to avoid synchronized stampedes
                time.sleep(random.uniform(0.5, 2.0))
                
                # Fetch data
                df = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    period=period, # 'period' can be used instead of 'start'/'end'
                    interval=interval,
                    progress=False,
                    threads=False, # Threading can cause issues with curl
                    timeout=20, # Increase timeout
                    auto_adjust=True # Silence Warning
                )
                
                if df is None or df.empty:
                    # If yf.download returns an empty DataFrame without error,
                    # it might mean no data for the period or ticker.
                    # Treat as a soft failure for retry, or return None if it's the last attempt.
                    if attempt == max_retries - 1:
                        logger.warning(f"No data returned for {ticker} after {max_retries} attempts.")
                        return None
                    else:
                        raise ValueError("Empty Data received, retrying.")
                    
                # Standardize columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df.columns = [c.lower() for c in df.columns]
                
                required = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required):
                     # This might happen if yfinance returns partial data (e.g., for indices)
                     # or if there's an issue with the data source.
                     # Filter only existing columns to avoid KeyErrors
                     available = [c for c in required if c in df.columns]
                     df = df[available]
                     if not df.empty:
                         logger.warning(f"Missing some required columns for {ticker}, using available: {available}")
                     else:
                         raise ValueError("Missing all required columns or empty after filtering.")
                     
                return df
                
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt+1}/{max_retries} failed for {ticker} (interval: {interval}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"FINAL FAIL: Could not fetch {ticker} after {max_retries} attempts.")
                    return None
                
                # Exponential Backoff
                time.sleep(2 ** attempt)
                
        return None # Should not be reached if max_retries is handled correctly

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
            if not os.path.exists(config.CACHE_DIR):
                os.makedirs(config.CACHE_DIR)
            df.to_parquet(filepath)
            # logger.info(f"Cache updated for {filepath}. Rows: {len(df)}")
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
    print("M5 Tail:", data['M5'].tail() if 'M5' in data and data['M5'] is not None else "None")
