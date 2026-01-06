import logging
import numpy as np
from typing import Dict, Any, List, Optional
from core.interfaces import SwarmSignal
from cpp_core.agi_bridge import get_agi_bridge

logger = logging.getLogger("OmniCortex")

class OmniCortex:
    """
    The Omni-Cortex: Higher-Order Reasoning Module.
    
    Responsibilities:
    1. Regime Detection (via C++ Fisher Information).
    2. Deep Thought Triggering (via C++ MCTS).
    3. Global 'Intuition' state management.
    """
    
    def __init__(self):
        self.bridge = get_agi_bridge()
        self.current_regime = "STABLE"
        self.fisher_metric = 0.0
        self.latest_thought = None
        self.consecutive_high_entropy_counts = 0
        
    def perceive(self, market_data: Dict[str, Any]) -> None:
        """
        Process incoming market data and update internal state using C++ Physics.
        """
        # We need a price array for physics calculations
        # Assuming market_data has 'close' history or we extract it
        # Ideally, main loop passes a window of closes. 
        # For now, let's assume we get a list of prices or extract from df if present
        
        prices = []
        if 'prices' in market_data:
            prices = market_data['prices']
        elif 'df' in market_data:
             # Extract last 100 closes
             df = market_data['df']
             if not df.empty and 'close' in df.columns:
                 prices = df['close'].tail(100).values
        
        if len(prices) < 20:
            return

        # 1. Calculate Fisher Information (Market Velocity/Regime Shift Speed)
        try:
            self.fisher_metric = self.bridge.physics.calculate_fisher(np.array(prices), window=10)
        except Exception as e:
            logger.error(f"Error calculating Fisher Info: {e}")
            self.fisher_metric = 0.0
            
        # 2. Update Regime State
        # Fisher > 1.5 usually implies rapid statistical change (Breakout or Crash)
        if self.fisher_metric > 2.0:
            self.current_regime = "CHAOTIC"
            self.consecutive_high_entropy_counts += 1
        elif self.fisher_metric > 1.0:
            self.current_regime = "VOLATILE"
            self.consecutive_high_entropy_counts = max(0, self.consecutive_high_entropy_counts - 1)
        else:
            self.current_regime = "STABLE"
            self.consecutive_high_entropy_counts = 0
            
        logger.debug(f"[OMNI-CORTEX] Fisher: {self.fisher_metric:.2f} | Regime: {self.current_regime}")

    def run_deep_thought(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Trigger a Deep Thought (Guided MCTS simulation) if warranted.
        Returns a 'Cognitive Packet' with simulation results.
        """
        if self.current_regime == "STABLE":
            return None # Don't waste compute on stable markets
            
        # If volatile, run a Guided Simulation to see if we should survive
        # We assume we have current price info
        current_price = context.get('current_price', 0.0)
        if current_price == 0: return None
        
        volatility = context.get('volatility', 0.001)
        
        # Scenario: "What if the trend reverses?"
        # We guide the MCTS to check for reversals (Bias towards Counter-Trend)
        # Assuming 'direction' is roughly known, e.g. 1 (Bull) or -1 (Bear)
        # We'll blindly test both or use a heuristic.
        
        # Test: BEARISH CRASH SCENARIO
        # Bias the simulation to SELL (-1) with strength 0.3
        result_crash = self.bridge.mcts.run_guided_mcts(
            current_price=current_price,
            entry_price=current_price, # Hypothetical entry now
            direction=1, # Assume we are Long, test if we die
            volatility=volatility,
            drift=0.0,
            iterations=5000, # Heavy compute
            depth=50,
            bias_strength=0.3,
            bias_direction=-1 # Force "Sell/Down" pressure
        )
        
        # Result analysis
        expected_value = result_crash.get('expected_value', 0.0)
        
        thought = {
             "hypothesis": "Crash Simulation",
             "regime": self.current_regime,
             "fisher_info": self.fisher_metric,
             "crash_expectation": expected_value,
             "recommendation": "HOLD"
        }
        
        if expected_value < -1.0: # If we expect to lose heavily even with just a 0.3 bias
             thought['recommendation'] = "VETO_LONG"
             
        self.latest_thought = thought
        return thought
