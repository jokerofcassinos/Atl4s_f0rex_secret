import yfinance as yf
import pandas as pd
import logging
import MetaTrader5 as mt5
from .cache_manager import CacheManager

logger = logging.getLogger("DataLoader")

class DataLoader:
    def __init__(self, symbol="XAUUSD"):
        # Yahoo Finance symbol for Gold might differ, e.g., "GC=F" or "XAUUSD=X"
        # Often "GC=F" is Gold Futures, "XAUUSD=X" is Spot Gold. Using Spot.
        self.yf_symbol = "GC=F" # Fallback/Alternative or "XAUUSD=X"
        self.cache = CacheManager(symbol=symbol)
        self.mt5_initialized = False

    def init_mt5(self):
        """Initializes connection to MT5 terminal."""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed, error code = {mt5.last_error()}")
            return False
        else:
            logger.info(f"MT5 Initialized. Terminal: {mt5.terminal_info()}")
            self.mt5_initialized = True
            return True

    def fetch_yahoo_data(self) -> pd.DataFrame:
        """
        Fetches the last ±60 days of 5m data from Yahoo Finance 
        (limit of yfinance for intraday).
        """
        try:
            # Fetch 5m data for the last 5 days (safe buffer)
            # We rely on cache for older history.
            df = yf.download(self.yf_symbol, period="5d", interval="5m", progress=False)
            
            if df.empty:
                logger.warning(f"Yahoo Finance returned empty data for {self.yf_symbol}")
                return pd.DataFrame()
                
            # Clean up multi-index if present (yfinance update often does this)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Rename columns to standard Format
            df = df.rename(columns={
                "Open": "open", 
                "High": "high", 
                "Low": "low", 
                "Close": "close", 
                "Volume": "volume"
            })
            
            # Ensure timezone is UTC then convert to BRT if needed? 
            # For now keeping it naive or source TZ, but cache manager parses index.
            # Good practice: standardize usage of UTC.
            if df.index.tz is None:
                 df.index = df.index.tz_localize('UTC') # Assumption
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Data: {e}")
            return pd.DataFrame()

    def sync_data(self):
        """
        Fetches latest data and updates the cache.
        Returns the full combined history.
        """
        logger.info("Syncing data...")
        new_data = self.fetch_yahoo_data()
        
        if not new_data.empty:
            self.cache.update_cache(new_data)
            
        return self.cache.get_full_history()

    def get_account_info(self):
        """Gets account info from MT5 (Equity, Balance, etc)."""
        if not self.mt5_initialized:
            if not self.init_mt5():
                return None
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to retreive account info")
            return None
        return account_info._asdict()
