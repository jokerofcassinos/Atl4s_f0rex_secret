import unittest
import logging
import sys
import os
import numpy as np

# Need to path to find core
sys.path.append(os.getcwd())

from core.swarm_orchestrator import SwarmOrchestrator
from core.interfaces import SwarmSignal

# Mock dependencies
class MockBus:
    def get_recent_thoughts(self): return []
    def register_thought(self, t): pass

class MockEvolution: pass
class MockNeuro: 
    def get_dynamic_weights(self): return {}
class MockGM: pass
class MockAttention:
    def forward(self, vectors):
        return np.array([vectors[0]]), np.array([0.1])
        
class TestTrendExhaustion(unittest.TestCase):
    def setUp(self):
        self.orch = SwarmOrchestrator(
            bus=MockBus(),
            evolution=MockEvolution(),
            neuroplasticity=MockNeuro(),
            attention=MockAttention(),
            grandmaster=MockGM()
        )
        self.orch.attention.forward = lambda x: (np.array([[0.0] * 64]), np.array([0.1]))
        
        # Mock dependencies
        self.orch.holographic_memory.retrieve_intuition = lambda x: 0.0
        self.orch._resolve_active_inference = lambda t, c: None 

    def test_exhaustion_blocks_aikido(self):
        print("\n--- Testing Exhaustion Blocks Aikido ---")
        ts = 12345.0
        
        # Scenario: 
        # 1. Strong Trend UP (Normally would trigger Aikido)
        # 2. Reversal Signal (Sell)
        # 3. BUT Exhaustion Detected (Parabolic/Churn)
        
        thoughts = [
            # Strong Trend Agent, but flags EXHAUSTION
            SwarmSignal("Trending_Swarm", "BUY", 90.0, ts, {
                'reason': 'Strong Up',
                'exhaustion': True,
                'exhaustion_reason': 'PARABOLIC_EXTENSION'
            }),
            # Reversal Signal (e.g. RSI Divergence) - trying to Sell Top
            SwarmSignal("Sniper_Swarm", "SELL", 80.0, ts, {'reason': 'Divergence'}),
        ]
        
        # Mock Consensus to return SELL (the Reversal signal wins the vote initially)
        self.orch._transformer_consensus = lambda *args, **kwargs: ("SELL", 70.0, {})
        
        final_decision, final_score, meta = self.orch.synthesize_thoughts(thoughts, {}, mode="SNIPER")
        
        print(f"Decision: {final_decision}, Meta: {meta}")
        
        # Expectation: 
        # Aikido (Flip to BUY) should be BLOCKED because of Exhaustion.
        # It should fall through to VETO (WAIT) or Allow SELL?
        # Current logic: If outcome != Trend, and Trend != 80+Exhaustion, it goes to Veto.
        # So we expect WAIT.
        
        self.assertNotEqual(final_decision, "BUY") # MUST NOT BUY THE TOP
        self.assertEqual(final_decision, "WAIT") # Safety Veto is acceptable

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
