
import logging
import random
import copy
from typing import Dict, Any, List

logger = logging.getLogger("EvolutionaryHeuristics")

class EvolutionaryHeuristics:
    """
    Sistema D-6: Evolutionary Heuristics Engine
    Combina Algoritmos Genéticos com Neuroplasticidade.
    Gera 'Filhos' (Estratégias), testa em simulação e faz Hot-Swap se o filho for superior.
    """
    def __init__(self):
        self.genome = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "trend_ema_fast": 9,
            "trend_ema_slow": 90,
            "volatility_threshold": 1.5,
            "risk_multiplier": 1.0,
            "stop_loss_atr": 1.5
        }
        self.population = []
        self.generation = 1
        self.best_fitness = 0.0
        
    def spawn_generation(self, pop_size: int = 10):
        """
        Gera uma nova população de filhos mutantes baseados no genoma atual (O Pai).
        """
        self.population = []
        for _ in range(pop_size):
            child_genome = self._mutate(self.genome)
            self.population.append({"genome": child_genome, "fitness": 0.0, "id": random.randint(1000, 9999)})
            
        logger.info(f"EVOLUTION: Spawning Generation {self.generation} with {pop_size} candidates.")
        
    def evaluate_offspring(self, market_simulator_func) -> Dict[str, Any]:
        """
        Avalia a população usando uma função de simulação (SimulationSystem).
        Retorna o melhor genoma.
        """
        best_child = None
        best_score = -999.0
        
        for child in self.population:
            # Run Simulation (Mock latency for now)
            fitness = market_simulator_func(child['genome'])
            child['fitness'] = fitness
            
            if fitness > best_score:
                best_score = fitness
                best_child = child
                
        self.generation += 1
        return best_child
        
    def attempt_hotswap(self, best_child: Dict[str, Any]) -> bool:
        """
        Decide se substitui o cérebro atual pelo do filho.
        """
        if best_child and best_child['fitness'] > self.best_fitness * 1.1: # Must be 10% better
            logger.warning(f"EVOLUTION HOTSWAP: Replacing Genome! (Fitness {self.best_fitness:.2f} -> {best_child['fitness']:.2f})")
            self.genome = best_child['genome']
            self.best_fitness = best_child['fitness']
            return True
            
        return False

    def _mutate(self, parent_genome: Dict[str, Any]) -> Dict[str, Any]:
        child = copy.deepcopy(parent_genome)
        
        # Mutate random genes
        if random.random() < 0.3:
            child['rsi_period'] = max(2, child['rsi_period'] + random.randint(-2, 2))
            
        if random.random() < 0.3:
            child['stop_loss_atr'] = max(0.5, child['stop_loss_atr'] + random.uniform(-0.2, 0.2))
            
        if random.random() < 0.3:
            child['risk_multiplier'] = max(0.1, min(5.0, child['risk_multiplier'] + random.uniform(-0.1, 0.1)))
            
        return child
