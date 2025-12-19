import os
from datetime import time

# --- Global Configuration ---
PROJECT_NAME = "Atl4s-Forex"
VERSION = "1.0.0"

# --- Timezone Settings ---
TIMEZONE = "America/Sao_Paulo"  # Brasilia Time
TRADING_START_TIME = time(0, 0)
TRADING_LAST_ENTRY_TIME = time(23, 59) # Don't enter new trades after this
TRADING_END_TIME = time(23, 59)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MT5_PATH = r"C:\Users\pichau\AppData\Roaming\MetaQuotes\Terminal\776D2ACDFA4F66FAF3C8985F75FA9FF6"

# --- Trading Parameters ---
SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
INITIAL_CAPITAL = 30.0
RISK_PER_TRADE = 0.05  # 5% risk per trade (Reduced from 10% for survivability)
LEVERAGE = 500  # Assumed leverage, adjust as needed
INVERT_TECHNICALS = True # Set to True to fade the retail trend (Buy becomes Sell)

# --- First Eye (Auto-Scalper) Settings ---
ENABLE_FIRST_EYE = True
ENABLE_SECOND_EYE = True # The Sniper
ENABLE_FOURTH_EYE = True # The Whale (Consensus > 33)
SCALP_SL = 1.0 # Stop Loss ($1.00 Risk per 0.01)
SCALP_TP = 0.7 # Take Profit ($0.70 Gain per 0.01)
SCALP_LOTS = 0.01 # Fixed Lot Size

# --- Swarm Settings ---
SWARM_MAX_TRADES = 5
SWARM_COOLDOWN = 10 # Seconds

# --- ZeroMQ Settings ---
# --- ZeroMQ Settings ---
ZMQ_REQ_PORT = 5557  # Request/Reply port (Changed to avoid conflict)
ZMQ_SUB_PORT = 5558  # Publish/Subscribe port

# --- Ensure Directories Exist ---
for path in [DATA_DIR, CACHE_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)
