
# ══════════════════════════════════════════════════════════════════════════════
# LAPLACE DEMON CONFIGURATION v2.0
# "The Deterministic Trading Intelligence"
# ══════════════════════════════════════════════════════════════════════════════

import os

# --- SYSTEM MODE ---
# LAPLACE = Full Laplace Demon system (recommended)
# LEGACY = Old swarm/AGI system 
TRADING_MODE = "LAPLACE"

# --- BROKER SETTINGS ---
SYMBOLS = [
    "GBPUSD",   # Primary - Best for esoteric theories
    "EURUSD",   # Major - High liquidity, tight spread
    "USDJPY",   # Major - Strong trends
    "USDCAD",   # Commodity-linked
    "USDCHF",   # Safe haven correlation
]
PRIMARY_SYMBOL = "GBPUSD"  # Laplace Demon optimized for GBP
SYMBOL = PRIMARY_SYMBOL  # Alias for legacy compatibility

TIMEFRAME = 5  # M5 - Primary decision timeframe
MAGIC_NUMBER = 123456
LEVERAGE = 3000  # 1:3000 (Unlimited)

# --- SPREAD LIMITS (pips) ---
# Used to filter out high-spread periods
SPREAD_LIMITS = {
    'GBPUSD': 0.00030,  # 3 pips max
    'EURUSD': 0.00020,  # 2 pips max
    'USDJPY': 0.00030,  # 3 pips max
    'USDCAD': 0.00035,  # 3.5 pips max
    'USDCHF': 0.00030,  # 3 pips max
    'XAUUSD': 0.50,     # 50 cents max for gold
}

# --- RISK MANAGEMENT ---
INITIAL_CAPITAL = 30.0  # Starting capital
RISK_PER_TRADE = 2.0    # 2% risk per trade
MAX_LOTS_PER_TRADE = 5.0
DYNAMIC_LOT_SCALING = True

# --- LAPLACE DEMON PARAMETERS ---
LAPLACE = {
    'min_confidence': 60,        # Minimum confidence to trade
    'min_confluence': 2,         # Minimum number of aligned signals
    'sl_atr_multiplier': 1.5,    # SL = ATR * this value
    'tp_rr_ratio': 2.0,          # TP = SL * this value (2:1 R:R)
    'max_concurrent_trades': 3,
    'signal_cooldown_seconds': 60,
}

# --- OPERATION WINDOWS (KILLZONES) ---
KILLZONES = {
    'LONDON_OPEN': {'start': 8, 'end': 11, 'priority': 1},
    'NY_OVERLAP': {'start': 13, 'end': 16, 'priority': 1},  # Best
    'NY_CLOSE': {'start': 19, 'end': 21, 'priority': 2},
    'ASIAN_OPEN': {'start': 0, 'end': 3, 'priority': 3},
}

# Session-Symbol recommendations
SESSION_PAIRS = {
    'LONDON_OPEN': ['GBPUSD', 'EURUSD'],
    'NY_OVERLAP': ['GBPUSD', 'EURUSD', 'USDCAD'],
    'NY_CLOSE': ['USDCAD'],
    'ASIAN_OPEN': ['USDJPY'],
}

# --- VIRTUAL STOPS (Broker-invisible) ---
VIRTUAL_SL_DOLLARS = 20.0  # Close if loss exceeds $20
VIRTUAL_TP_DOLLARS = 40.0  # Close if profit exceeds $40

# --- TRAILING STOPS ---
TRAILING_STOP = True
TRAILING_START = 1.00  # Start after $1 profit
TRAILING_STEP = 0.50

# --- DATA PATHS ---
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
REPORTS_DIR = "reports"

# --- NETWORK ---
ZMQ_PORT = 5558

# --- LEGACY MODULE TOGGLES (for old system) ---
ENABLE_FIRST_EYE = True   # Scalp Swarm
ENABLE_SECOND_EYE = True  # The Sniper
ENABLE_FOURTH_EYE = True  # The Whale

# --- LEGACY SWARM PARAMETERS ---
SWARM_MAX_TRADES = 50
SWARM_THRESHOLD = 0.15
SWARM_COOLDOWN = 3

# --- LEGACY GRID PARAMETERS ---
GRID_LAYERS = 3
GRID_SPACING = 30

# --- INTERMARKET SYMBOLS ---
INTERMARKET_SYMBOLS = {
    'DXY': 'DX-Y.NYB',
    'US10Y': '^TNX',
    'SPX': '^GSPC',
    'OIL': 'CL=F',
    'EUR_INDEX': 'EURUSD=X',  # For SMT divergence
}
