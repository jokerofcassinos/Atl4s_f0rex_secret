import sys
import os
import time

sys.path.append(os.getcwd())

from core.agi.dialectic import DialecticEngine
from core.agi.omni_cortex import OmniCortex

def test_dialectic_reasoning():
    print("=== Testing Dialectic Engine (Debate AI) ===")
    
    # 1. Init
    cortex = OmniCortex()
    engine = DialecticEngine(cortex)
    
    # 2. Mock Market Conditions
    # We need a context that C++ bridge can use.
    # Bridge requires 'current_price' and 'volatility' to run MCTS.
    # We will simulate a situation.
    
    context = {
        'current_price': 2000.0,
        'volatility': 0.005
    }
    
    print(f"Scenario: Gold Price 2000.0, Volatility 0.5%")
    print("Initiating Debate: Thesis (Buy) vs Antithesis (Sell)...")
    
    start = time.time()
    result = engine.resolve_market_debate(context)
    duration = time.time() - start
    
    print(f"\n--- Debate Result ({duration:.3f}s) ---")
    print(f"Winner: {result.winner}")
    print(f"Decision: {result.decision}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Thesis Score (Bull): {result.thesis_score:.4f}")
    print(f"Antithesis Score (Bear): {result.antithesis_score:.4f}")
    print(f"Reasoning: {result.reasoning}")
    
    if result.decision != "UNKNOWN":
        print("\n[SUCCESS] Dialectic Engine successfully resolved the conflict.")
    else:
        print("\n[FAIL] Dialectic Engine returned UNKNOWN.")

if __name__ == "__main__":
    test_dialectic_reasoning()
