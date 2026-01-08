
import logging
import unittest
from core.agi.synergy import AlphaSynergySwarm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestNeogenesis")

class TestNeogenesis(unittest.TestCase):
    def test_synergy_collapse(self):
        """
        Test the collapse of multiple AGI inputs into a Singularity Vector.
        """
        synergy = AlphaSynergySwarm()
        
        # Scenario: Swarm says BUY, but Temporal and History say SELL (Bearish divergence + Bad history)
        inputs = {
            'swarm_consensus': {'direction': 1, 'confidence': 0.7}, # Bullish Swarm
            'temporal_coherence': {'direction': -1, 'confidence': 0.8}, # Bearish Fractals (Time Dilation?)
            'history_bias': {'direction': -1, 'confidence': 0.9}, # History says "This setup fails"
            'causal_inference': {'direction': 0, 'confidence': 0.5}, # Neutral Logic
            'abstract_pattern': {'direction': -1, 'confidence': 0.6} # Bearish Structure (e.g. Head & Shoulders)
        }
        
        # Expected: The weight of Time (0.2), History (0.15), Abstract (0.15) should crush the Swarm (0.3).
        # Swarm Score: 1 * 0.7 * 0.3 = 0.21
        # Bearish Score: (-1*0.8*0.2) + (-1*0.9*0.15) + (-1*0.6*0.15) = -0.16 + -0.135 + -0.09 = -0.385
        # Net Score: 0.21 - 0.385 = -0.175
        # Verdict: SELL
        
        decision = synergy.synthesize_singularity_vector(inputs)
        logger.info(f"Singularity Decision: {decision}")
        
        self.assertEqual(decision['verdict'], "SELL")
        self.assertGreater(decision['confidence'], 0.15)
        logger.info("SUCCESS: Singularity correctly overrode the Swarm based on deep coherence.")

if __name__ == "__main__":
    unittest.main()
