import pandas as pd
import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.scalper_swarm import ScalpSwarm

def test_swarm_deepening():
    print("--- Testing Swarm Brain Evolution (1st Eye) ---")
    swarm = ScalpSwarm()
    
    # Mock Data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='5min')
    df_m5 = pd.DataFrame({'open': [2000]*10, 'close': [2000.1]*10}, index=dates)
    
    # Case 1: BULLISH CONSENSUS
    # High OFI, Bullish Alpha, Pullback active
    tick = {'last': 1999.0, 'time': time.time()*1000, 'volume': 10} # 1.0 below open
    micro_stats = {
        'ofi': 20,              # Strong Buy Flow
        'micro_hurst': 0.6,     # Trending
        'entropy': 1.0,         # Normal
        'velocity': 0.05        # Rising slowly
    }
    
    action, reason, price = swarm.process_tick(
        tick=tick,
        df_m5=df_m5,
        alpha_score=0.8,
        tech_score=10,
        phy_score=1.0,
        micro_stats=micro_stats
    )
    print(f"Test Case 1 (Bullish Consensus): {action} | Reason: {reason}")
    
    # Reset for next test
    swarm.last_trade_time = 0
    
    # Case 2: BEARISH EXHAUSTION (Hurst Climax)
    # Price rising fast, but Hurst < 0.3 (Exhaustion)
    tick_climax = {'last': 2002.0, 'time': time.time()*1000 + 1000, 'volume': 5}
    micro_climax = {
        'ofi': -20,             # Strong Sell Flow (Alignment with exhaustion)
        'micro_hurst': 0.25,    # CLIMAX / Exhaustion
        'entropy': 1.0,
        'velocity': 0.5         # Rocket up
    }
    
    action, reason, price = swarm.process_tick(
        tick=tick_climax,
        df_m5=df_m5,
        alpha_score=-0.2,
        tech_score=2,
        phy_score=1.0,
        micro_stats=micro_climax
    )
    print(f"Test Case 2 (Bearish Exhaustion): {action} | Reason: {reason}")
    
    # Reset for next test
    swarm.last_trade_time = 0
    
    # Case 3: ENTROPY GATE
    micro_entropy = { 'entropy': 0.05, 'ofi': 100 } # DEAD ZONE
    action, reason, price = swarm.process_tick(
        tick=tick,
        df_m5=df_m5,
        alpha_score=1.0,
        tech_score=20,
        phy_score=1.0,
        micro_stats=micro_entropy
    )
    print(f"Test Case 3 (Entropy Gate): {action} | Reason: {reason}")

if __name__ == "__main__":
    test_swarm_deepening()
