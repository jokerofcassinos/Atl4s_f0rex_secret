import logging
import asyncio
import numpy as np
from core.agi.omni_cortex import OmniCortex
from core.memory.holographic import HolographicMemory
from cpp_core.agi_bridge import AGIBridge

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase5")

# Mock Bridge
class MockPhysics:
    def calculate_fisher(self, arr, window): return 2.5 # Chaotic

class MockMCTS:
    def run_guided_mcts(self, bias_direction=0, **kwargs):
        return {
            'expected_value': 2.0 * bias_direction, # Confirm bias works
            'visit_count': 1000
        }

class MockBridge(AGIBridge):
    def __init__(self):
        self.physics = MockPhysics()
        self.mcts = MockMCTS()

# Mock dependencies
import cpp_core.agi_bridge
cpp_core.agi_bridge.get_agi_bridge = lambda: MockBridge()

def test_holographic_nexus():
    print("\n--- TEST: HOLOGRAPHIC NEXUS (TIME-TRAVEL REASONING) ---")
    
    # 1. Setup Memory and Cortex
    memory = HolographicMemory(dimensions=1024) # Smaller for test
    cortex = OmniCortex(memory=memory)
    
    # 2. Teach Memory a Lesson: "Chaos leads to Crash"
    # Create a "Crash State" vector (High fisher, stable prices that drop?)
    ctx_crash = {'fisher': 2.5, 'regime': 'CHAOTIC', 'last_price': 100.0}
    # Store outcome: -1.0 (Bad/Crash)
    # Actually, if we want "Bias", intuition should return the outcome.
    # If outcome is -1.0 (Loss on Long), it implies price went down.
    # So if we see this again, we should fear (-1).
    
    # Store 5 times to reinforce
    for _ in range(5):
        memory.store_experience(ctx_crash, outcome=-1.0, category='pattern')
    
    print("Memory Taught: High Chaos -> Crash (-1.0)")
    
    # 3. Perceive New Data (Identical Condition)
    market_data = {'prices': [100.0] * 30} # Dummy prices, but fisher mock returns 2.5
    cortex.perceive(market_data)
    
    # Check if Cortex recalled intuition
    last_intuition = getattr(cortex, 'last_intuition', 0.0)
    print(f"Cortex Recall Intuition: {last_intuition:.2f}")
    
    if last_intuition < -0.1:
        print("PASS: Cortex successfully recalled negative outcome (-1.0).")
    else:
        print(f"FAIL: Cortex amnesia. Intuition {last_intuition}")
        
    # 4. Deep Thought (Time-Travel Bias)
    print("Running Deep Thought...")
    thought = cortex.run_deep_thought({'current_price': 100.0, 'volatility': 0.01})
    
    print(f"Thought Hypothesis: {thought.get('hypothesis')}")
    
    if "Bias=-1" in thought.get('hypothesis', ''):
        print("PASS: MCTS was correctly biased by Memory (Fear of Crash).")
    else:
        print("FAIL: MCTS ignored memory bias.")

if __name__ == "__main__":
    test_holographic_nexus()
