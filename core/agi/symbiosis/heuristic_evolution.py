"""
Heuristic Evolution Engine - Autonomous Heuristic Evolution.

Implements genetic programming for autonomous evolution of
trading heuristics through mutation, crossover, and selection.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger("HeuristicEvolution")


@dataclass
class Heuristic:
    """A trading heuristic (gene)."""
    name: str
    parameters: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate parameters."""
        for key in self.parameters:
            if np.random.random() < mutation_rate:
                self.parameters[key] *= (1 + np.random.normal(0, 0.2))


@dataclass
class EvolutionResult:
    """Result of evolution step."""
    best_heuristic: Heuristic
    population_size: int
    generation: int
    avg_fitness: float
    diversity: float
    convergence: float


class HeuristicEvolution:
    """
    The Genetic Optimizer.
    
    Evolves trading heuristics through:
    - Fitness-based selection
    - Genetic crossover of successful heuristics
    - Random mutation for exploration
    - Elitism to preserve best performers
    """
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[Heuristic] = []
        self.generation = 0
        self.fitness_history: deque = deque(maxlen=100)
        
        # Evolution parameters
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.elite_count = 2
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"HeuristicEvolution initialized with pop_size={population_size}")
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        for i in range(self.population_size):
            heuristic = Heuristic(
                name=f"H_G0_{i}",
                parameters={
                    'trend_weight': np.random.uniform(0.2, 0.8),
                    'momentum_threshold': np.random.uniform(0.3, 0.7),
                    'stop_loss_ratio': np.random.uniform(0.01, 0.05),
                    'take_profit_ratio': np.random.uniform(0.02, 0.10),
                    'confidence_threshold': np.random.uniform(0.5, 0.8),
                    'session_weight': np.random.uniform(0.3, 0.7),
                    'liquidity_min': np.random.uniform(0.2, 0.6),
                },
                generation=0
            )
            self.population.append(heuristic)
    
    def evolve(self, fitness_scores: Dict[str, float]) -> EvolutionResult:
        """
        Evolve population based on fitness scores.
        
        Args:
            fitness_scores: Dict mapping heuristic name to fitness score
            
        Returns:
            EvolutionResult with best heuristic and stats.
        """
        # Update fitness
        for heuristic in self.population:
            if heuristic.name in fitness_scores:
                heuristic.fitness = fitness_scores[heuristic.name]
        
        # Record fitness
        avg_fitness = np.mean([h.fitness for h in self.population])
        self.fitness_history.append(avg_fitness)
        
        # Sort by fitness
        self.population.sort(key=lambda h: h.fitness, reverse=True)
        
        # Calculate diversity
        diversity = self._calculate_diversity()
        
        # New generation
        new_population = []
        
        # Elitism - keep best
        for i in range(self.elite_count):
            elite = copy.deepcopy(self.population[i])
            elite.name = f"H_G{self.generation + 1}_E{i}"
            elite.generation = self.generation + 1
            new_population.append(elite)
        
        # Crossover and mutation for rest
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            child.mutate(self.mutation_rate)
            
            # Set metadata
            child.name = f"H_G{self.generation + 1}_{len(new_population)}"
            child.generation = self.generation + 1
            child.fitness = 0.0
            child.parent_ids = [parent1.name, parent2.name]
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Calculate convergence
        convergence = self._calculate_convergence()
        
        return EvolutionResult(
            best_heuristic=self.population[0],
            population_size=len(self.population),
            generation=self.generation,
            avg_fitness=avg_fitness,
            diversity=diversity,
            convergence=convergence
        )
    
    def _select_parent(self) -> Heuristic:
        """Tournament selection."""
        tournament_size = 3
        candidates = np.random.choice(self.population, tournament_size, replace=False)
        return max(candidates, key=lambda h: h.fitness)
    
    def _crossover(self, parent1: Heuristic, parent2: Heuristic) -> Heuristic:
        """Uniform crossover of parameters."""
        child_params = {}
        
        for key in parent1.parameters:
            if np.random.random() < 0.5:
                child_params[key] = parent1.parameters[key]
            else:
                child_params[key] = parent2.parameters[key]
        
        return Heuristic(
            name="child",
            parameters=child_params
        )
    
    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity of population."""
        if len(self.population) < 2:
            return 1.0
        
        all_params = []
        for h in self.population:
            all_params.append(list(h.parameters.values()))
        
        params_array = np.array(all_params)
        stds = np.std(params_array, axis=0)
        
        return float(np.mean(stds))
    
    def _calculate_convergence(self) -> float:
        """Calculate how converged the population is."""
        if len(self.fitness_history) < 10:
            return 0.0
        
        recent = list(self.fitness_history)[-10:]
        variance = np.var(recent)
        
        # Low variance = high convergence
        return float(1.0 - min(1.0, variance * 10))
    
    def get_best_heuristic(self) -> Heuristic:
        """Get current best heuristic."""
        return max(self.population, key=lambda h: h.fitness)
    
    def inject_heuristic(self, heuristic: Heuristic):
        """Inject an external heuristic into population."""
        # Replace worst performer
        self.population.sort(key=lambda h: h.fitness)
        self.population[0] = heuristic
        logger.info(f"Injected heuristic: {heuristic.name}")
