
"""
THE STRATEGY GENOME
Library of 60 High-Probability Concepts for the Swarm.
"""

# --- 30 CANDLE PATTERNS (Micro-Structure) ---

CANDLE_PATTERNS = {
    # Reversal
    "CP_01": "Micro Hammer (Rejection of Lows)",
    "CP_02": "Shooting Star (Rejection of Highs)",
    "CP_03": "Engulfing Flow (Total dominance of previous candle)",
    "CP_04": "Doji Spring (Indecision followed by violent break)",
    "CP_05": "Tweezer Bottom (Precision Support)",
    "CP_06": "Tweezer Top (Precision Resistance)",
    "CP_07": "Morning Star Fractal (3-candle turn)",
    "CP_08": "Evening Star Fractal (3-candle turn)",
    "CP_09": "Inside Bar Compression (Coiling Spring)",
    "CP_10": "Outside Bar Shakeout (Liquidity Grab)",
    
    # Continuation
    "CP_11": "Rising Three Methods (Trend Pause)",
    "CP_12": "Falling Three Methods (Trend Pause)",
    "CP_13": "Marubozu Breakout (High Momentum)",
    "CP_14": "Spinning Top Grid (Accumulation)",
    "CP_15": "Wick Fill Rejection (Failed fill)",
    
    # Volume/Flow
    "CP_16": "Volume Climax (Exhaustion)",
    "CP_17": "Volume Dry-Up (Pullback)",
    "CP_18": "Delta Divergence (Price Up, Delta Down)",
    "CP_19": "Absorption Block (High Vol, No Move)",
    "CP_20": "Iceberg Wick (Hidden orders)",
    
    # Institutional
    "CP_21": "Order Block Return (Mitigation)",
    "CP_22": "Fair Value Gap (FVG) Tap",
    "CP_23": "Breaker Block Retest",
    "CP_24": "Liquidity Void Fill",
    "CP_25": "Stop Hunt Wick (The Judas Swing)",
    
    # Exotic
    "CP_26": "Fractal Dimension Spike (Chaos)",
    "CP_27": "Entropy Collapse (Order from Chaos)",
    "CP_28": "Golden Ratio Wick (61.8% Retrace in wick)",
    "CP_29": "Time-Price Distortion (Fast move, slow correction)",
    "CP_30": "Quantum Leap (Gap logic)"
}

# --- 30 EXECUTION STRATEGIES (Timing/Entry) ---

EXECUTION_MS = {
    # Breakout
    "EX_01": "Volatility Breakout (Bollinger Squeeze)",
    "EX_02": "Box Breakout (Asian Range)",
    "EX_03": "Fractal High Break",
    "EX_04": "Trendline Shatter",
    "EX_05": "News Spike Fade",
    
    # Reversion
    "EX_06": "VWAP Reversion (Elastic Band)",
    "EX_07": "Bollinger Bounce (Mean Reversion)",
    "EX_08": "RSI Divergence Sniper",
    "EX_09": "Stochastics Cross",
    "EX_10": "MFI Overbought Fade",
    
    # Flow/Tape
    "EX_11": "Bid Stacking Front-Run",
    "EX_12": "Ask Clearing Momentum",
    "EX_13": "Spread Scalp (Inside Spread)",
    "EX_14": "Tick Velocity Trigger",
    "EX_15": "Order Flow Absorption",
    
    # HFT Concepts
    "EX_16": "Latency Arbitrage Simulation",
    "EX_17": "Ping Pong (Range Scalp)",
    "EX_18": "Micro-Grid (Cost Averaging - Risky)",
    "EX_19": "Nanosecond React",
    "EX_20": "Market Maker Shadow",
    
    # Deep Causality
    "EX_21": "Trend Alignment 3-Timeframe",
    "EX_22": "Support/Resist Flip",
    "EX_23": "Fibonacci Golden Zone Entry",
    "EX_24": "Harmonic Pattern Completion (Gartley)",
    "EX_25": "Elliott Wave 3 Impulse",
    
    # Risk-Managed
    "EX_26": "Pyramid Building (Adding to Winners)",
    "EX_27": "Scaling Out (Partial TP)",
    "EX_28": "Breakeven Trailer",
    "EX_29": "Trailing Stop Tightener",
    "EX_30": "News Event Straddle"
}
