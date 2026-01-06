
import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from core.agi.omni_cortex import OmniCortex
from analysis.swarm.red_team_swarm import RedTeamSwarm

def test_omni_cortex():
    print("=== Testing Omni-Cortex ===")
    cortex = OmniCortex()
    
    # Simulate a crash curve with high jaggedness (High Fisher Info expected)
    prices = []
    import math
    for i in range(100):
        # Sine wave + Crash + Noise
        p = 100.0 + math.sin(i * 0.2) * 5
        if i > 80: p -= (i - 80) * 2 # Hard crash
        # Add high freq noise for Fisher
        if i % 2 == 0: p += 2.0
        else: p -= 2.0
        prices.append(p)
    
    context = {'prices': prices}
    cortex.perceive(context)
    
    print(f"Fisher Metric: {cortex.fisher_metric}")
    print(f"Regime: {cortex.current_regime}")
    
    thought = cortex.run_deep_thought({'current_price': 100.0, 'volatility': 0.5})
    if thought:
        print("Deep Thought Triggered:")
        print(thought)
    else:
        print("No Deep Thought (Stable Market)")
        
    print("\n=== Testing RedTeamSwarm ===")
    red = RedTeamSwarm()
    # Mock async
    import asyncio
    
    # Context with prices
    ctx = {
        'history': prices,
        'volatility': 0.5,
        'micro_stats': {'entropy': 0.9}
    }
    
    loop = asyncio.new_event_loop()
    signal = loop.run_until_complete(red.process(ctx))
    if signal:
        print(f"Red Team Signal: {signal.signal_type} | Confidence: {signal.confidence}")
        print(f"Reason: {signal.meta_data.get('reason')}")
    else:
        print("Red Team: HOLD (No Danger)")

if __name__ == "__main__":
    test_omni_cortex()
