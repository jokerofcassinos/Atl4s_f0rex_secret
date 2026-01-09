
import unittest
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# Mocking the dependency classes locally to avoid importing heavy modules
class MockSwarmSignal:
    def __init__(self, source, signal_type, confidence, meta=None):
        self.source = source
        self.signal_type = signal_type
        self.confidence = confidence
        self.meta_data = meta if meta else {}

class MockAttention:
    def forward(self, vectors):
        # Return a mock consensus vector (dummy logic)
        # We only care about the logic *around* the consensus in the orchestrator
        return np.array([vectors[0]]), np.array([0.1])

class MockOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger("TestSwarm")
        self.attention = MockAttention()
        self.holographic_memory = self # Mock
    
    def retrieve_intuition(self, vec):
        return 0.0

    # We copy the RELEVANT methods from SwarmOrchestrator here
    # or we import the actual class if possible.
    # Given the complexity, let's try to import the actual class but mock its dependencies.

# Importing real class
import sys
import os
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

class TestSmartLogic(unittest.TestCase):
    def setUp(self):
        self.orch = SwarmOrchestrator(
            bus=MockBus(),
            evolution=MockEvolution(),
            neuroplasticity=MockNeuro(),
            attention=MockAttention(),
            grandmaster=MockGM()
        )
        # Mocking attention to return a neutral decision_score so we test the overrides
        self.orch.attention.forward = lambda x: (np.array([[0.0] * 64]), np.array([0.1]))

    def test_trend_aikido(self):
        print("\n--- Testing Trend Aikido ---")
        ts = 12345.0
        # Setup: Strong Trend UP (BUY), Signal DOWN (SELL)
        thoughts = [
            SwarmSignal("Trending_Swarm", "BUY", 95.0, ts, {'reason': 'Strong Up'}), # Strong Trend
            SwarmSignal("Sniper_Swarm", "SELL", 80.0, ts, {'reason': 'Overbought'}), # Counter Trend
        ]
        
        # We need to manually simulate what synthesize_thoughts does
        # It calls _transformer_consensus which returns a decision
        
        # For this test, we want to call synthesize_thoughts directly
        # But synthesize_thoughts logic for Veto is AFTER transformer consensus.
        # We need MockAttention to return the "SELL" decision first (as if Swarm decided SELL)
        
        # Let's mock _transformer_consensus to return SELL first
        original_consensus = self.orch._transformer_consensus
        self.orch._transformer_consensus = lambda *args, **kwargs: ("SELL", 70.0, {})
        
        final_decision, final_score, meta = self.orch.synthesize_thoughts(thoughts, {}, mode="SNIPER")
        
        print(f"Decision: {final_decision}, Score: {final_score}, Meta: {meta}")
        
        # We expect Aikido Inversion: SELL -> BUY
        self.assertEqual(final_decision, "BUY") 
        self.assertTrue(meta.get('aikido_flip'))
        self.orch._transformer_consensus = original_consensus # Restore

    def test_civil_war_smart_resolution(self):
        print("\n--- Testing Smart Civil War ---")
        ts = 12345.0
        # Setup: Gridlock (WAIT) but with Elite Support for BUY
        
        thoughts = []
        # 4 BUY agents @ 80 = 320 strength
        for i in range(4): thoughts.append(SwarmSignal(f"BuyAgent_{i}", "BUY", 80.0, ts, {}))
        # 4 SELL agents @ 80 = 320 strength
        for i in range(4): thoughts.append(SwarmSignal(f"SellAgent_{i}", "SELL", 80.0, ts, {}))
        
        # Add Elite Agent (Whale) for BUY
        thoughts.append(SwarmSignal("Whale_Swarm", "BUY", 90.0, ts, {}))
        
        # Total BUY > 400, Total SELL 320. 
        # Consensus Ratio ~ 410 / 730 = 0.56 roughly. 
        # Base confidence will be low (~50%). 
        # Attention vector is mocked to return 0.0 -> WAIT.
        
        # This should trigger Civil War check.
        # Elite Balance: Whale(+90) = +90. Wait, logic says > 100.
        # Need another elite or stronger whale.
        thoughts.append(SwarmSignal("SmartMoney_Swarm", "BUY", 90.0, ts, {}))
        # Elite Balance = 180.
        
        # Verify manually
        buy_s = sum([t.confidence for t in thoughts if t.signal_type == "BUY"])
        sell_s = sum([t.confidence for t in thoughts if t.signal_type == "SELL"])
        print(f"DEBUG: Buy Strength: {buy_s}, Sell Strength: {sell_s}")
        
        elite_agents = ["Whale_Swarm", "SmartMoney_Swarm", "Apex_Swarm", "Quantum_Grid_Swarm"]
        elite_bal = 0
        for t in thoughts:
             if t.source in elite_agents:
                 print(f"DEBUG: Elite found: {t.source} ({t.confidence})")
                 elite_bal += t.confidence
        print(f"DEBUG: Elite Balance: {elite_bal}")

        final_decision, final_score, meta = self.orch._transformer_consensus(
            thoughts, {}, {}, ["BUY", "SELL", "WAIT"], mode="SNIPER"
        )
        
        print(f"Decision: {final_decision}, Score: {final_score}, Meta: {meta}")
        
        self.assertEqual(final_decision, "BUY")
        self.assertEqual(meta.get('resolution'), "ELITE_VOTE")

    def test_deep_reasoning_rescue(self):
        print("\n--- Testing Deep Reasoning Rescue ---")
        # Setup: Decision made but Metacognition hates it (Score < 47)
        # We need to simulate the flow where reflect returns low score
        
        # Mock Metacognition.reflect to return bad score
        class MockMeta:
            def reflect(self, d, s, m, c):
                return {'adjusted_confidence': 30.0, 'notes': ['Bad reasoning']} # Fail!

        self.orch.metacognition = MockMeta()
        
        # Case 1: Standard failure (should WAIT)
        thoughts = [SwarmSignal("Newbie_Swarm", "BUY", 60.0, 12345.0, {})]
        
        # We need MockAttention to return BUY
        self.orch._transformer_consensus = lambda *args, **kwargs: ("BUY", 60.0, {})
        
        final_decision, final_score, meta = self.orch.synthesize_thoughts(thoughts, {}, mode="SNIPER")
        self.assertEqual(final_decision, "WAIT") # Should fail
        
        # Case 2: Rescue via Elite Vote (Simulated via metadata injection in consensus)
        # We modify the mock consensus to return the 'ELITE_VOTE' tag which triggers rescue
        self.orch._transformer_consensus = lambda *args, **kwargs: ("BUY", 60.0, {'resolution': 'ELITE_VOTE'})
        
        # FIX: Mock Active Inference so it doesn't veto our BUY signal before it gets to Metacognition
        self.orch._resolve_active_inference = lambda t, c: None
        
        final_decision, final_score, meta = self.orch.synthesize_thoughts(thoughts, {}, mode="SNIPER")
        print(f"Rescue Decision: {final_decision}, Reason: {meta.get('rescue_reason')}")
        
        self.assertEqual(final_decision, "BUY")
        self.assertTrue(meta.get('deep_reasoning'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
