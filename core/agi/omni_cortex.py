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
    
    def __init__(self, memory=None):
        self.bridge = get_agi_bridge()
        self.memory = memory # Holographic Memory
        self.current_regime = "STABLE"
        self.fisher_metric = 0.0
        self.latest_thought = None
        self.consecutive_high_entropy_counts = 0
        
    def perceive(self, market_data: Dict[str, Any]) -> None:
        """
        Process incoming market data: Physics + History.
        """
        prices = []
        if 'prices' in market_data:
            prices = market_data['prices']
        elif 'df' in market_data:
             df = market_data['df']
             if not df.empty and 'close' in df.columns:
                 prices = df['close'].tail(100).values
        
        if len(prices) < 20: return

        # 1. Calculate Fisher Information (Market Velocity/Regime Shift Speed)
        try:
            self.fisher_metric = self.bridge.physics.calculate_fisher(np.array(prices), window=10)
        except Exception as e:
            logger.error(f"Error calculating Fisher Info: {e}")
            self.fisher_metric = 0.0
            
        # 2. Update Regime
        if self.fisher_metric > 2.0: self.current_regime = "CHAOTIC"
        elif self.fisher_metric > 1.0: self.current_regime = "VOLATILE"
        else: self.current_regime = "STABLE"
            
        logger.debug(f"[OMNI-CORTEX] Fisher: {self.fisher_metric:.2f} | Regime: {self.current_regime}")
        
        # 3. Holographic Memory Query (Time-Travel)
        # We query the memory to see if this pattern has happened before.
        if self.memory:
            # Construct a state vector context
            # (In production, we would use sophisticated feature extraction)
            ctx = {
                'fisher': self.fisher_metric,
                'regime': self.current_regime,
                'last_price': float(prices[-1])
            }
            intuition = self.memory.retrieve_intuition(ctx)
            # Store this "Feeling" for the deep thought process
            self.last_intuition = intuition
            if intuition != 0.0:
                 logger.info(f"HOLOGRAPHIC RECALL: Familiar Situation Detected. Intuition Score: {intuition:.2f}")

    def run_deep_thought(
        self, 
        context: Dict[str, Any],
        force_bias: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Trigger a Deep Thought (Guided MCTS simulation) if warranted.
        Biased by both Physics (Regime) and History (Memory).
        """
        if not force_bias and self.current_regime == "STABLE":
            # If memory says "Something big is coming" (high negative intuition), we might run anyway?
            # For now, keep it simple.
            return None 
            
        current_price = context.get('current_price', 0.0)
        if current_price == 0: return None
        volatility = context.get('volatility', 0.001)
        
        # Determine Bias Direction
        # Default: Fear of Crash (-1)
        bias_dir = -1 
        
        # 1. Force override
        if force_bias is not None:
             bias_dir = force_bias
             
        # 2. Memory Override (Time-Travel Bias)
        elif self.memory and hasattr(self, 'last_intuition'):
             # If memory strongly suggests UP (+0.5) or DOWN (-0.5), we bias the simulation
             if self.last_intuition > 0.3:
                 bias_dir = 1 # We remember this goes up
                 logger.info("MCTS BIAS: Using Historical Optimism (Intuition > 0.3)")
             elif self.last_intuition < -0.3:
                 bias_dir = -1 # We remember this crashes
                 logger.info("MCTS BIAS: Using Historical Fear (Intuition < -0.3)")
             
        # Run Simulation
        result = self.bridge.mcts.run_guided_mcts(
            current_price=current_price,
            entry_price=current_price, 
            direction=1, 
            volatility=volatility,
            drift=0.0,
            iterations=5000, 
            depth=50,
            bias_strength=0.3, 
            bias_direction=bias_dir
        )
        
        expected_value = result.get('expected_value', 0.0)
        confidence = result.get('visit_count', 0) / 5000.0 * 100.0 
        
        thought = {
             "hypothesis": f"Simulation Bias={bias_dir}",
             "regime": self.current_regime,
             "fisher_info": self.fisher_metric,
             "expected_value": expected_value,
             "confidence": confidence,
             "recommendation": "HOLD"
        }
        
        if not force_bias and expected_value < -1.0:
             thought['recommendation'] = "VETO_LONG"
             
        self.latest_thought = thought
        return thought

    def introspect(self, context: Dict[str, Any], prediction: float, reality: float):
        """
        Recursive Self-Correction (The 'Why' Loop).
        If Prediction != Reality, we inject a Correction Impulse into Memory.
        """
        error = abs(prediction - reality)
        
        # Threshold for "Surprise" (Metacognitive Trigger)
        if error > 0.5:
            logger.warning(f"METACOGNITION: Surprise Detected! Predicted {prediction:.2f} vs Real {reality:.2f}")
            
            # recursive analysis -> simplistic version:
            # We realize our intuition was wrong. We must store the TRUTH with high weight.
            # "I thought it was up, it was down. I must remember this context led to DOWN."
            
            if self.memory:
                # Store the CORRECTION
                # We use a special category "correction" to maybe give it higher weight later
                self.memory.store_experience(
                     context=context,
                     outcome=reality, 
                     category="correction",
                     temporal_level="long_term"
                )
                logger.info("METACOGNITION: Correction Impulse Stored. Neural path updated.")
