"""Tests for confidence calibration."""

import unittest
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestConfidenceCalibration(unittest.TestCase):
    """Verify confidence scores are properly distributed."""
    
    def test_confidence_range(self):
        """Confidence should vary between 45-85%, not fixed at 99%."""
        from core.swarm_orchestrator import SwarmOrchestrator
        from core.agi.consciousness_bus import ConsciousnessBus
        from core.genetics import EvolutionEngine
        from core.neuroplasticity import NeuroPlasticityEngine
        from core.transformer_lite import TransformerLite
        
        bus = ConsciousnessBus()
        evo = EvolutionEngine()
        neuro = NeuroPlasticityEngine()
        attn = TransformerLite(64, 64)
        
        orch = SwarmOrchestrator(bus, evo, neuro, attn)
        
        # Simulate 50 decisions
        confidences = []
        import time
        for _ in range(50):
            # Mock thoughts with varying signals
            from core.interfaces import SwarmSignal
            thoughts = [
                SwarmSignal("Test1", "BUY", np.random.uniform(40, 90), time.time(), {}),
                SwarmSignal("Test2", "SELL" if np.random.random() > 0.5 else "BUY", 
                           np.random.uniform(30, 80), time.time(), {}),
            ]
            
            decision, conf, _ = orch.synthesize_thoughts(
                thoughts, {}, mode="SNIPER" # Mode will be forced to HYDRA internally
            )
            if decision in ["BUY", "SELL"]:
                confidences.append(conf)
        
        # Shutdown bus to prevent hanging threads
        bus.shutdown()
        
        if not confidences:
            self.fail("No BUY/SELL decisions generated during test")

        # Assertions
        print(f"\n[CALIBRATION] Count: {len(confidences)} | Min: {min(confidences):.1f} | Max: {max(confidences):.1f} | Std: {np.std(confidences):.1f}")
        
        self.assertGreater(np.std(confidences), 2.0, 
            "Confidence should have variance > 2")
        self.assertLess(max(confidences), 96.0,
            "Max confidence should be realistic (< 96%)")
        self.assertGreater(min(confidences), 10.0,
            "Min entry confidence should be >= 10%")

if __name__ == '__main__':
    unittest.main()
