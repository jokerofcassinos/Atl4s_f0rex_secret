
# Configuration for Atl4s-Forex Bot V2.0
# "Deep Awakening"

import os

# --- BROKER SETTINGS ---
SYMBOL = "XAUUSD" # Gold
TIMEFRAME = 5 # M5
MAGIC_NUMBER = 123456
LEVERAGE = 3000 # UPGRADED: 1:3000 (Ultra High Leverage)

# --- RISK MANAGEMENT ---
INITIAL_CAPITAL = 30.0 # Default fallback
RISK_PER_TRADE = 0.02 # 2% per trade (Standard)
MAX_LOTS_PER_TRADE = 5.0 # Increased absolute Cap
DYNAMIC_LOT_SCALING = True # Use Quantum Lot Sizing

# --- OPERATION WINDOW ---
START_HOUR = 10 # 10:00 (London/NY Overlap)
END_HOUR = 16 # 16:00 (End of active session)

# --- MODULE TOGGLES ---
ENABLE_FIRST_EYE = True # Scalp Swarm (HFT)
ENABLE_SECOND_EYE = True # The Sniper (FVG)
ENABLE_FOURTH_EYE = True # The Whale (Consensus)

# --- SCALPING PARAMETERS (XAUUSD) ---
SCALP_TP = 2.00 # $2.00 Gold Move (200 points)
SCALP_SL = 1.50 # $1.50 Gold Move (150 points) - Tighter Stop
TRAILING_STOP = True
TRAILING_START = 1.00 # Start trailing after $1 profit
TRAILING_STEP = 0.50

# --- DATA PATHS ---
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# --- NETWORK ---
ZMQ_PORT = 5555

# --- SWARM & SCALPING ---
SWARM_MAX_TRADES = 50       # Restored: Max active swarm trades
SWARM_THRESHOLD = 0.15      # Reduced from 0.30 (Tachyon Mode: Hyper-Sensitive)
SWARM_COOLDOWN = 3          # Reduced from 10s (Machine Gun Mode)

# --- 13th EYE (QUANTUM GRID) ---
GRID_LAYERS = 3             # 3 Layers for "Spray"
GRID_SPACING = 30           # 30 pts spacing (Tight Grid)

# --- LEGACY / ADVANCED ANALYSIS ---
INVERT_TECHNICALS = False # For Contrarian Testing
INTERMARKET_SYMBOLS = {
    'DXY': 'DX-Y.NYB',
    'US10Y': '^TNX',
    'SPX': '^GSPC',
    'OIL': 'CL=F'
}
