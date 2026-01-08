
import logging
import random
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("NeuroPlasticity")

class NeuroPlasticityEngine:
    """
    Phase 132: Neuro-Plasticity Engine.
    
    Responsibilities:
    1. Synaptic Pruning: Disabling underperforming swarms/reasoning paths.
    2. Heuristic Evolution: Mutating parameters (SL/TP, Thresholds) to find optima.
    3. Plasticity Control: Adjusting learning rates based on global volatility (Chaos).
    """
    def __init__(self):
        self.synapses = {} # Map of Module -> Synaptic Weight (0.0 to 1.0)
        self.performance_history = {} # Map of Module -> List of outcomes
        self.mutation_rate = 0.05
        self.plasticity_index = 0.5 # 0=Rigid, 1=Liquid
        
        # Core Parameters that can be mutated
        self.gene_pool = {
            "risk_per_trade": {"val": 0.01, "min": 0.005, "max": 0.05},
            "veto_threshold": {"val": 0.7, "min": 0.5, "max": 0.95},
            "consensus_threshold": {"val": 0.6, "min": 0.51, "max": 0.9},
            "take_profit_mult": {"val": 2.0, "min": 1.0, "max": 5.0}
        }
        
    def adjust_plasticity(self, volatility_index: float):
        """
        System #13: Plasticity Controller.
        High Chaos (Vol) -> High Plasticity (Adapt fast).
        Low Chaos -> Low Plasticity (Crystalize best practices).
        """
        # Sigmoid-like adaptation
        self.plasticity_index = 1.0 / (1.0 + np.exp(-10 * (volatility_index - 0.5)))
        logger.info(f"Plasticity Adjusted to {self.plasticity_index:.2f} (Vol: {volatility_index:.2f})")

    def register_outcome(self, module_name: str, success: bool, impact: float = 1.0):
        """
        Reinforcement Learning Step.
        """
        if module_name not in self.synapses:
            self.synapses[module_name] = 1.0
            self.performance_history[module_name] = []
            
        # Update history
        self.performance_history[module_name].append(1 if success else 0)
        if len(self.performance_history[module_name]) > 100:
            self.performance_history[module_name].pop(0)
            
        # Update Synaptic Weight (Moving Average with Decay)
        # Weight increases with success, decays with failure
        current_weight = self.synapses[module_name]
        
        learning_rate = 0.1 * self.plasticity_index * impact
        
        if success:
            new_weight = current_weight + (learning_rate * (1.0 - current_weight))
        else:
            new_weight = current_weight - (learning_rate * current_weight)
            
        self.synapses[module_name] = max(0.01, min(1.5, new_weight)) # Clamp 0.01 - 1.5
        
        # System #11: Synaptic Pruning Check
        if self.synapses[module_name] < 0.2:
             logger.warning(f"SYNAPTIC PRUNING: Module {module_name} is withering (Weight: {new_weight:.2f})")
             # In a real system, we might disable execution of this module here.

    def mutate_genes(self):
        """
        System #14: Heuristic Evolver.
        Randomly mutates parameters based on plasticity index.
        """
        if random.random() > self.plasticity_index:
            return # Stability prevails
            
        for gene, data in self.gene_pool.items():
            if random.random() < self.mutation_rate:
                # Mutation!
                current = data['val']
                drift = (random.random() - 0.5) * 0.2 * current # +/- 10%
                new_val = current + drift
                
                # Clamp
                new_val = max(data['min'], min(data['max'], new_val))
                
                logger.info(f"🧬 MUTATION: {gene} evolved from {current:.3f} to {new_val:.3f}")
                self.gene_pool[gene]['val'] = new_val

    def get_gene(self, gene_name: str) -> float:
        return self.gene_pool.get(gene_name, {}).get('val', 0.0)

    def get_synaptic_weight(self, module_name: str) -> float:
         return self.synapses.get(module_name, 1.0)
