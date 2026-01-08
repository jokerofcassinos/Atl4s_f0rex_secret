import logging
import asyncio
import numpy as np
from core.agi.omni_cortex import OmniCortex
from core.grandmaster import GrandMaster
from core.memory.holographic import HolographicMemory

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase8")

# Mock Bridge
class MockBridge:
    def __init__(self):
        self.physics = None
        self.mcts = None

import cpp_core.agi_bridge
cpp_core.agi_bridge.get_agi_bridge = lambda: MockBridge()

def test_symbiote_logic():
    print("\n--- TEST: THE SYMBIOTE (METACOGNITION + EMPATHY) ---")
    
    # 1. Test Recursive Self-Correction (Introspection)
    print("\n[Testing Introspection...]")
    memory = HolographicMemory(dimensions=1024)
    cortex = OmniCortex(memory=memory)
    
    # Simulate a wrong prediction
    # We predicted UP (1.0), Reality was DOWN (-1.0). Limit: -1 to 1.
    ctx = {'last_price': 100.0, 'regime': 'STABLE'}
    prediction = 0.8
    reality = -0.9 # Huge miss
    
    cortex.introspect(ctx, prediction, reality)
    
    # Verify memory has a correction
    # We should search for specific "correction" category or check intuition change
    # Since 'retrieve_intuition' averages everything, let's just cheat and check the memory impulses
    # Accessing internals for test
    long_term = memory.plate.temporal_levels['long_term']
    impulses = long_term.impulses
    
    found_correction = False
    for imp in impulses:
        if imp.category == "correction" and imp.outcome == reality:
            found_correction = True
            break
            
    if found_correction:
        print("PASS: Introspection stored a Correction Impulse.")
    else:
        print("FAIL: No Correction Impulse found.")
        
    # 2. Test Empathic Resonance
    print("\n[Testing Empathic Resonance...]")
    gm = GrandMaster()
    
    # Normal State
    print(f"Normal Risk Mod: {gm.risk_modifier} | Mindset: {gm.current_mindset}")
    if gm.risk_modifier == 1.0:
        print("PASS: Normal State Verified.")
    else:
        print("FAIL: Normal State Wrong.")
        
    # High Anxiety
    gm.update_user_anxiety(0.9) # User is panicking
    print(f"Panic Risk Mod: {gm.risk_modifier} | Mindset: {gm.current_mindset}")
    
    if gm.risk_modifier == 0.5 and gm.current_mindset == "DEFENSIVE":
        print("PASS: Empathic Resonance Switched to DEFENSIVE.")
    else:
        print(f"FAIL: Did not switch to Defensive. Mod={gm.risk_modifier}")

if __name__ == "__main__":
    test_symbiote_logic()
