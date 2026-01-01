import pandas as pd
import numpy as np
import logging
from analysis.recursive_reasoner import RecursiveReasoner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify-Debate")

def test_debate():
    print("\n" + "="*50)
    print("TESTING RECURSIVE REASONER (THE DEBATER)")
    print("="*50 + "\n")
    
    debater = RecursiveReasoner()
    
    # Mock Data: Recent price drop
    dates = pd.date_range(start="2023-01-01", periods=20, freq="5min")
    # Strong downward velocity
    close = np.linspace(2010, 2000, 20) 
    df = pd.DataFrame({'close': close}, index=dates)
    data_map = {'M5': df}
    
    print("Scenario 1: Consensus says BUY, but Path is Bearish (Veto expected)")
    decision, score, log = debater.debate("BUY", 80, data_map)
    print(f"Final Decision: {decision}")
    print(f"Final Score: {score:.2f}")
    print(f"Debate Log: {log}")
    
    print("\nScenario 2: Consensus says SELL, and Path is Bearish (Reinforcement expected)")
    decision, score, log = debater.debate("SELL", 80, data_map)
    print(f"Final Decision: {decision}")
    print(f"Final Score: {score:.2f}")
    print(f"Debate Log: {log}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_debate()
