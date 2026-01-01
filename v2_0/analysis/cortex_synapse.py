
import json
import os
import logging

logger = logging.getLogger("Atl4s-CortexSynapse")

class CortexSynapse:
    """
    The Reinforcement Learning Core (RL Layer).
    Implements 'Synaptic Plasticity' - weights evolve based on trade outcomes.
    
    Mechanism: 
    - Online Learning (Bandit-style updates).
    - If Eye X contributes to a WIN -> Weight Incremented.
    - If Eye X contributes to a LOSS -> Weight Decremented.
    - Weights are persisted to JSON (Long-Term Memory).
    """
    def __init__(self, weights_file="brain/synaptic_weights.json"):
        self.weights_file = weights_file
        self.learning_rate = 0.05 # Alpha: Speed of adaptation
        self.max_weight = 0.50
        self.min_weight = 0.05
        
        # Default Genome
        self.default_weights = {
            'w_trend': 0.20,
            'w_sniper': 0.25, 
            'w_quant': 0.15,
            'w_pattern': 0.10,
            'w_cycle': 0.10,
            'w_sd': 0.10,
            'w_div': 0.05,
            'w_kin': 0.15,
            'w_fractal': 0.10
        }
        
        self.weights = self.load_weights()

    def load_weights(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    logger.info("Synaptic Weights Loaded from Long-Term Memory.")
                    return data
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
                return self.default_weights.copy()
        else:
            return self.default_weights.copy()

    def save_weights(self):
        try:
            os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=4)
            logger.info("Synaptic Weights Consolidated (Saved).")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")

    def evolve(self, trade_result, eye_snapshot):
        """
        Adjusts weights based on the outcome of a trade.
        trade_result: float (Profit in $)
        eye_snapshot: dict { 'Trend': 'BUY', 'Sniper': 'SELL', ... } - What each eye voted at entry.
        """
        outcome = 1 if trade_result > 0 else -1
        
        # Mapping generic names to weight keys
        key_map = {
            'Trend': 'w_trend',
            'Sniper': 'w_sniper', 
            'Quant': 'w_quant',
            'Pattern': 'w_pattern',
            'Cycle': 'w_cycle',
            'SupplyDemand': 'w_sd',
            'Divergence': 'w_div',
            'Kinematics': 'w_kin',
            'Fractal': 'w_fractal'
        }
        
        updates_log = []
        
        for eye_name, vote in eye_snapshot.items():
            if eye_name not in key_map: continue
            weight_key = key_map[eye_name]
            
            # Determine if this eye was "Right" or "Wrong"
            # If Trade was BUY (result > 0) AND Eye said BUY -> Right
            # If Trade was BUY (result < 0) AND Eye said BUY -> Wrong
            pass # We need to know the Trade Direction too!
            
            # Actually, simpler: 
            # If Trade was PROFITABLE, reinforce everyone who agreed with the Trade Direction.
            # If Trade was LOSS, punish everyone who agreed with the Trade Direction.
            
            # But wait, we need the Trade Direction (Buy/Sell) passed here.
            # Handled in updated signature below.
            
    def evolve_with_direction(self, profit, trade_direction, eye_snapshot):
        """
        profit: float ($)
        trade_direction: "BUY" or "SELL"
        eye_snapshot: dict of votes { 'Trend': 'BUY', ... }
        """
        if profit == 0: return # No learning on break-even
        
        is_win = profit > 0
        reward_sign = 1 if is_win else -1
        
        params_changed = False
        
        key_map = {
            'Trend': 'w_trend',
            'Sniper': 'w_sniper',
            'Quant': 'w_quant',
            'Pattern': 'w_pattern',
            'Cycle': 'w_cycle',
            'SupplyDemand': 'w_sd',
            'Divergence': 'w_div',
            'Kinematics': 'w_kin',
            'Fractal': 'w_fractal'
        }

        changes = []

        for eye_name, vote in eye_snapshot.items():
            if eye_name not in key_map: continue
            w_key = key_map[eye_name]
            
            # Did this eye agree with the trade?
            # E.g. Trade=BUY, Eye=BUY -> Agreed.
            # E.g. Trade=BUY, Eye=SELL -> Disagreed (It tried to save us!).
            
            agreed = (vote == trade_direction)
            
            adjustment = 0
            
            if is_win:
                if agreed:
                    # Good Eye! Increase Weight.
                    adjustment = self.learning_rate
                else:
                    # Bad Eye! It bet against a winner. Decrease Weight.
                    adjustment = -self.learning_rate
            else: # Loss
                if agreed:
                    # Bad Eye! It led us into a loss. Decrease Weight.
                    adjustment = -self.learning_rate
                else:
                    # Good Eye! It tried to warn us. Increase Weight.
                    adjustment = self.learning_rate
            
            if adjustment != 0:
                old_w = self.weights.get(w_key, 0.1)
                new_w = old_w + adjustment
                
                # Clamp
                new_w = max(self.min_weight, min(self.max_weight, new_w))
                
                self.weights[w_key] = round(new_w, 3)
                changes.append(f"{eye_name}: {old_w}->{new_w}")
                params_changed = True
        
        if params_changed:
            self.save_weights()
            logger.info(f"🧬 SYNAPTIC EVOLUTION ({'WIN' if is_win else 'LOSS'}): " + " | ".join(changes))

