
import yfinance as yf
import pandas as pd
import os
import logging
import config
import time
import random
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger("Atl4s-Data")

class DataLoader:
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Converts MT5 symbols (e.g., 'DOGUSD', 'BTCUSD', 'USDCADm') to Yahoo Finance Tickers (e.g., 'DOGE-USD', 'BTC-USD').
        """
        s = symbol.upper()
        
        # 0. Remove broker-specific suffixes (m, c, pro, raw, etc.)
        broker_suffixes = ['M', '.M', 'C', '.C', 'PRO', '.PRO', 'RAW', '.RAW', 'ECN', '.ECN']
        for suffix in broker_suffixes:
            if s.endswith(suffix) and len(s) > len(suffix):
                s = s[:-len(suffix)]
                break
        
        # 1. Explicit Dictionary (The "Fixer" List)
        # Add any tricky ones here.
        mapping = {
            "DOGUSD": "DOGE-USD",
            "DOGEUSD": "DOGE-USD",
            "BTCUSD": "BTC-USD",
            "BTCXAU": "BTC-USD", # PROXY: BTCXAU correlates 99% with BTCUSD on typical brokers (unless real gold denom)
            "ETHUSD": "ETH-USD",
            "SOLUSD": "SOL-USD",
            "XRPUSD": "XRP-USD",
            "LTCUSD": "LTC-USD",
            "BNBUSD": "BNB-USD",
            "XAUUSD": "GC=F", # Gold Futures (High Volume)
            "XAUEUR": "XAUEUR=X",
            "XAUGBP": "XAUGBP=X",
            "XAUAUD": "XAUAUD=X",
            "XAUCHF": "XAUCHF=X",
            "XAGUSD": "SI=F", # Silver
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "USDCHF": "USDCHF=X",
            "NZDUSD": "NZDUSD=X",
            "AUDCAD": "AUDCAD=X",
            "AUDJPY": "AUDJPY=X",
            "CADJPY": "CADJPY=X",
            "CNHJPY": "CNHJPY=X",
            "EURUSD": "EURUSD=X"
        }
        
        if "-" in s:
            return s
            
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
             # If not in mapping (which has major forex), assume crypto?
             # Forex map is extensive but not exhaustive.
             # Safe fallback: if not in forex map, try adding -?
             # For now, rely on mapping.
             pass 
             
        return s 

    def __init__(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_file = os.path.join(config.CACHE_DIR, f"{symbol}_{timeframe}.parquet")
        
        # Phase 3: Lazy Evaluation - Only recalculate on candle change
        self.last_candle_time = None
        self.cached_indicators = {}
        self.cached_data = None

    def check_candle_change(self, df: pd.DataFrame) -> bool:
        """
        Phase 3: Lazy Evaluation - Checks if candle has changed since last call.
        Returns True if new candle detected (indicators need recalculation).
        """
        if df is None or df.empty:
            return False
            
        current_candle_time = df.index[-1]
        
        if self.last_candle_time is None or current_candle_time != self.last_candle_time:
            self.last_candle_time = current_candle_time
            self.cached_indicators = {}  # Clear indicator cache on candle change
            return True
            
        return False

    async def get_data(self, symbol=None, timeframe=None):
        """
        Orchestrates the data loading process (Async & Concurrent).
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
             return await asyncio.to_thread(self._fetch_with_cache, target_symbol, yf_interval, cache_file)
             
        # Else Legacy Multi-TF Load using Concurrent Execution
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
        
        # Launch concurrent fetches
        tasks = []
        for tf_name, yf_interval, days_limit in timeframes:
            cache_file = os.path.join(config.CACHE_DIR, f"{target_symbol}_{tf_name}.parquet")
            tasks.append(
                asyncio.to_thread(self._fetch_with_cache, target_symbol, yf_interval, cache_file, days_limit)
            )
            
        # Wait for all
        results = await asyncio.gather(*tasks)
        
        # Map results back
        for i, (tf_name, _, _) in enumerate(timeframes):
            data_map[tf_name] = results[i]
        
        # Derived (CPU bound, fast enough to run in sync, or offload if needed)
        if 'H1' in data_map and data_map['H1'] is not None:
             data_map['H4'] = self.resample_to_tf(data_map['H1'], '4h')
        if 'D1' in data_map and data_map['D1'] is not None:
             data_map['W1'] = self.resample_to_tf(data_map['D1'], 'W')
             
        # Phase 30: Apex Basket (Candidates)
        # Dynamic Selection based on Asset Class
        candidates = []
        
        # Check if Crypto
        if any(c in target_symbol for c in ["BTC", "ETH", "SOL", "DOGE", "XRP"]):
             candidates = ["BTCXAU", "ETHUSD"]
        elif "XAU" in target_symbol or "GC=F" in target_symbol:
             # Gold Mode -> Compare with Silver or Majors?
             # User specifically asked NOT to see BTCXAU.
             # We can add Silver or just relevant Forex majors.
             candidates = [] # Disable Apex Routing for now in Gold mode unless requested
        elif any(c in target_symbol for c in ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]):
             # Forex Mode -> Basket of Majors
             candidates = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"]
             # Remove self from candidates
             candidates = [c for c in candidates if self._normalize_symbol(c) != target_symbol]
        
        global_basket = {}
        
        # Helper to check if weekend (naive)
        is_weekend = False
        if pd.Timestamp.now().dayofweek >= 5: is_weekend = True
        
        basket_tasks = []
        basket_symbols = []

        for asset in candidates:
             # Skip Forex on weekends to avoid noise/stale data
             if is_weekend and asset in ["XAUUSD", "EURUSD", "USDJPY", "GBPUSD"]: continue
             
             c_file = os.path.join(config.CACHE_DIR, f"{asset}_Apex.parquet")
             
             basket_symbols.append(asset)
             basket_tasks.append(
                 asyncio.to_thread(self._fetch_with_cache, asset, "1h", c_file, 30)
             )

        if basket_tasks:
            basket_results = await asyncio.gather(*basket_tasks)
            for i, df in enumerate(basket_results):
                if df is not None and not df.empty:
                    global_basket[basket_symbols[i]] = df
             
        data_map['global_basket'] = global_basket # Apex and Nexus both use this
        
        return data_map

    async def get_basket_data(self, symbols: list) -> dict:
        """
        Fetches M5 data for a list of symbols to build a Causal Basket (Async).
        Used by the Causal Nexus (Phase 105).
        """
        basket = {}
        tasks = []
        task_symbols = []
        
        for sym in symbols:
            # We assume M5 is the heartbeat timeframe
            tf_name = "M5"
            yf_interval = "5m"
            c_file = os.path.join(config.CACHE_DIR, f"{sym}_{tf_name}.parquet")
            
            task_symbols.append(sym)
            tasks.append(
                asyncio.to_thread(self._fetch_with_cache, sym, yf_interval, c_file, 5)
            )
            
        if tasks:
            results = await asyncio.gather(*tasks)
            for i, df in enumerate(results):
                if df is not None and not df.empty:
                    basket[task_symbols[i]] = df
                    
        return basket

    def _get_yf_ticker(self, symbol):
        # Strip suffixes common in MT5 (e.g., 'm', '.pro', '_i')
        clean_symbol = symbol.replace("m", "").replace(".pro", "").replace("_i", "")
        
        ticker_map = {
            "XAUUSD": "GC=F",
            "BTCUSD": "BTC-USD",
            "BTCXAU": "BTC-USD", 
            "ETHUSD": "ETH-USD",
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "USDCHF": "USDCHF=X",
            "NZDUSD": "NZDUSD=X"
        }
        return ticker_map.get(clean_symbol, f"{clean_symbol}=X") # Default to Forex format



    def _fetch_with_cache(self, symbol, interval, cache_file, days=55):
        """
        Manages the Caching Layer + Calling the Robust Fetcher.
        """
        # YFinance Limits
        if interval == "1m": days = 5
        if interval == "5m": days = 55
        
        # 1. Load Cache
        yf_ticker = self._get_yf_ticker(symbol)
        # logger.info(f"DATA LOADER: Fetching {symbol} (YF: {yf_ticker}) -> {cache_file}")
        cached_df = self._load_cache(cache_file)
        
        start_date = None
        if cached_df is not None and not cached_df.empty:
            last_date = cached_df.index[-1]
            start_date = (last_date - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Map internal symbol
        # Map internal symbol
        yf_ticker = self._normalize_symbol(symbol)
        
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
                df_raw = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                    progress=False,
                    threads=False,
                    timeout=20,
                    auto_adjust=True
                )
                
                df = df_raw
                # Handle Dict return (Multi-ticker or error format)
                if isinstance(df, dict):
                    # Sometimes yf returns a dict of DataFrames or Errors
                    if ticker in df:
                        df = df[ticker]
                    else:
                        # Attempt to find the first DataFrame value
                        found = False
                        for k, v in df.items():
                            if isinstance(v, pd.DataFrame):
                                df = v
                                found = True
                                break
                        if not found:
                             logger.warning(f"yfinance returned dict without usable DF: {df.keys()}")
                             df = None

                if not isinstance(df, pd.DataFrame) or df.empty:
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
