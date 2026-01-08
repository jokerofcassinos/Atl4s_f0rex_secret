
import logging
import unittest
from core.agi.plasticity import SelfModificationHeuristic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPlasticity")

class TestPlasticity(unittest.TestCase):
    def test_risk_reduction(self):
        """
        Test that High Drawdown triggers Risk Reduction.
        """
        plasticity = SelfModificationHeuristic()
        
        # Scenario: 6% Drawdown
        metrics = {'drawdown': 0.06, 'win_rate': 0.4}
        config = {'risk_per_trade': 0.02} # 2% Risk
        
        mods = plasticity.evaluate_and_adapt(metrics, config)
        
        logger.info(f"Modifications Triggered: {mods}")
        
        self.assertIn('risk_per_trade', mods)
        self.assertEqual(mods['risk_per_trade'], 0.005) # Should slash to 0.5%
        logger.info("SUCCESS: Plasticity slashed risk due to drawdown.")

    def test_intelligence_adaptability(self):
        """
        Test that Low Win Rate triggers Higher Alpha Threshold (Bot becomes pickier).
        """
        plasticity = SelfModificationHeuristic()
        
        # Scenario: 30% Win Rate (Bot is stupid right now)
        metrics = {'drawdown': 0.01, 'win_rate': 0.3, 'trades_count': 10}
        config = {'alpha_threshold': 0.80}
        
        mods = plasticity.evaluate_and_adapt(metrics, config)
        
        logger.info(f"Modifications Triggered: {mods}")
        
        self.assertIn('alpha_threshold', mods)
        self.assertGreater(mods['alpha_threshold'], 0.80)
        logger.info("SUCCESS: Plasticity increased Alpha Threshold to trade less often.")

if __name__ == "__main__":
    unittest.main()
