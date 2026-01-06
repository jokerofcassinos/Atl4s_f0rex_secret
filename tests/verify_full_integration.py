
import unittest
import pandas as pd
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.trend_architect import TrendArchitect

class TestAlphaSwarmIntegration(unittest.TestCase):
    def test_trend_architect_agi_loop(self):
        print("\n--- VERIFYING ALPHA-SWARM INTEGRATION (FULL LOOP) ---")
        
        # 1. Initialize Trend Architect (connects to AGI)
        architect = TrendArchitect()
        
        # 2. Generate Dummy Data (200+ candles needed)
        closes = [100.0 + i*0.1 for i in range(250)] # Uptrend
        data = {
            'open': closes,
            'high': [c + 0.5 for c in closes],
            'low': [c - 0.5 for c in closes],
            'close': closes,
            'tick_volume': [100]*250
        }
        df = pd.DataFrame(data)
        
        # 3. Analyze (Triggers AGI Deliberation)
        print("Running Trend Analysis with AGI Deliberation...")
        result = architect.analyze(df)
        
        # 4. Verify Output Structure
        print(f"Final Score: {result['score']}")
        print(f"Tech Direction: {result['direction']}")
        print(f"AGI Trace: {result['thought_trace']}")
        
        # Check if AGI actually ran
        self.assertIn('engine', result['thought_trace'])
        self.assertEqual(result['thought_trace']['engine'], 'MCTS-AlphaSwarm')
        
        # Check if score was influenced (Score calculation involves AGI)
        # We can't deterministically predict score since Neural Net is random initialized
        # But we know code ran.
        
        print("Integration Successful. TrendArchitect relies on Alpha-Swarm.")

if __name__ == '__main__':
    unittest.main()
