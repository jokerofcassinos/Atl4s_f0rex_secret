import sys
import os
import time
import numpy as np

sys.path.append(os.getcwd())

from analysis.swarm.dream_swarm import DreamSwarm
from core.hyper_dimensional import HyperDimensionalEngine
# Mock Context
context = {
    'history': [1.1200 + i*0.0001 for i in range(50)], # Very stable uptrend implies low Fisher
    'volatility': 0.001
}

async def test_dreamer():
    print("=== Testing Dream-Weaver AGI ===")
    
    # 1. Test DreamSwarm (Active Imagination)
    dreamer = DreamSwarm()
    # Force last dream time to be old so it dreams now
    dreamer.last_dream_time = 0 
    
    print("DreamSwarm Initialized. Processing...")
    # It communicates with OmniCortex which uses C++ Fisher to check regime
    # If regime is STABLE, it dreams.
    
    try:
        signal = await dreamer.process(context)
        if signal:
            print(f"DREAM SIGNAL: {signal.signal_type}")
            print(f"Scenario: {signal.meta_data.get('scenario')}")
            print(f"Expected Value: {signal.meta_data.get('expected_value')}")
            if signal.meta_data.get('type') == 'PREMONITION':
                print("SUCCESS: Premonition Generated!")
        else:
            print("No Dream Signal (Maybe market too chaotic or dream yielded low value?)")
    except Exception as e:
        print(f"ERROR in Dreamer: {e}")

    # 2. Test Holographic Memory (HDC)
    print("\n=== Testing Holographic Memory ===")
    hdc = HyperDimensionalEngine()
    state = {'close_pct': 80, 'vol_pct': 20, 'rsi': 65}
    
    try:
        hv = hdc.encode(state)
        print("State Encoded into HyperVector.")
        print(f"Dimension: {len(hv.values)}")
        print(f"First 10 bits: {hv.values[:10]}")
        
        # Test Similarity
        hv2 = hdc.encode(state) # Same state
        sim = hv.similarity(hv2)
        print(f"Self-Similarity (Should be ~1.0): {sim:.4f}")
        
        state_diff = {'close_pct': 10, 'vol_pct': 90, 'rsi': 20}
        hv_diff = hdc.encode(state_diff)
        sim_diff = hv.similarity(hv_diff)
        print(f"Diff-Similarity (Should be low): {sim_diff:.4f}")
        
        if sim > 0.95 and sim_diff < 0.5:
            print("SUCCESS: HDC Engine Functional!")
        else:
            print("WARNING: HDC Similarity logic might be off.")

    except Exception as e:
         print(f"ERROR in HDC: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dreamer())
