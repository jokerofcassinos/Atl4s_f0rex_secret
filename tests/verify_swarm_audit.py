
import unittest
import asyncio
from typing import Dict, Any, Tuple
import sys
import os

import numpy as np

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import SwarmSignal
from core.swarm_orchestrator import SwarmOrchestrator
# Mock classes to avoid full initialization
class MockBus: 
    def register_thought(self, t): pass
    def get_recent_thoughts(self): return []
class MockEvo: pass
class MockNeuro: 
    def get_dynamic_weights(self): return {}
class MockAttn: 
    def forward(self, x): 
        # Return a mock matrix: [Context Vector, Weights] based on input size
        # Should return (context, weights)
        # context is expected to be [Batch, Dim] or similar. 
        # logic calls: swarm_vec = context_matrix[0] or mean...
        # Let's return a list capable of being indexed twice
        return ([[0.1, 0, 0, 0]], np.array([]))
class MockMCTS: pass

class TestSwarmAudit(unittest.TestCase):
    def test_audit_veto(self):
        print("\n--- VERIFYING SWARM AUDIT (ALPHA-SWARM VETO) ---")
        
        # 1. Initialize Orchestrator with Mocks
        orch = SwarmOrchestrator(MockBus(), MockEvo(), MockNeuro(), MockAttn(), MockMCTS())
        # Manual Patch for Microstructure which is usually init in constructor
        orch.microstructure = type('obj', (object,), {'process': asyncio.coroutine(lambda x: None)})

        
        # 2. Mock the AGIBrain to force a VETO scenario
        # We replace the deliberate method on the instance
        def mock_deliberate(context, focus_domain="GENERAL"):
            # AGI Says "SELL", forcing a conflict if Agent says BUY
            return {
                'consensus': 'SELL',
                'confidence': 0.95,
                'reasoning_trace': {'q': 'Mock Veto'}
            }
        
        orch.brain.deliberate = mock_deliberate
        
        # 3. Create a Dummy Signal (BUY)
        signal = SwarmSignal(
            source="TestAgent", 
            signal_type="BUY", 
            confidence=90.0, 
            meta_data={},
            timestamp=0
        )
        
        # 4. Run Synthesize
        # Since it's async, we run it synchronously for test
        loop = asyncio.get_event_loop()
        decision, score, meta = loop.run_until_complete(orch.synthesize_thoughts([signal], context={}))
        
        # 5. Verify VETO
        # Agent said BUY, AGI said SELL -> Result should be WAIT
        print(f"Agent: BUY | AGI: SELL -> Result: {decision}")
        
        self.assertEqual(decision, "WAIT")
        # Ensure it was vetoed (thoughts filtering worked)
        
    def test_audit_approval(self):
        print("\n--- VERIFYING SWARM AUDIT (ALPHA-SWARM APPROVAL) ---")
        orch = SwarmOrchestrator(MockBus(), MockEvo(), MockNeuro(), MockAttn(), MockMCTS())
        orch.microstructure = type('obj', (object,), {'process': asyncio.coroutine(lambda x: None)})
        
        def mock_deliberate_approve(context, focus_domain="GENERAL"):
            return {
                'consensus': 'BUY',
                'confidence': 0.95,
                'reasoning_trace': {}
            }
        orch.brain.deliberate = mock_deliberate_approve
        
        signal = SwarmSignal(source="TestAgent", signal_type="BUY", confidence=90.0, meta_data={}, timestamp=0)
        
        loop = asyncio.get_event_loop()
        decision, score, meta = loop.run_until_complete(orch.synthesize_thoughts([signal], context={}))
        
        print(f"Agent: BUY | AGI: BUY -> Result: {decision}")
        # Single agent BUY allowed through consensus
        self.assertEqual(decision, "BUY")

if __name__ == '__main__':
    unittest.main()
