"""
AGI Ultra-Complete: Optimizer AGI Components

Sistema de Otimização Evolutiva Avançada:
- MetaOptimizationEngine: Otimiza o próprio processo
- OptimizationTransferLearner: Transfer learning
- MultiObjectiveEvolutionaryAlgorithm: Evolução multi-objetivo
- AdaptivePopulationManager: População adaptativa
- AdvancedCMAES: CMA-ES avançado
"""

import logging
import time
import random
import copy
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("OptimizerAGI")


@dataclass
class Individual:
    """Individual in population."""
    genes: Dict[str, float]
    fitness: float = 0.0
    age: int = 0
    ancestors: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of optimization."""
    best_params: Dict[str, float]
    best_fitness: float
    generations: int
    evaluations: int
    convergence_history: List[float]


class MetaOptimizationEngine:
    """Optimizes the optimization process itself."""
    
    def __init__(self):
        self.optimization_history: deque = deque(maxlen=100)
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        
        self.current_strategy = {
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'selection_pressure': 2.0,
            'elitism': 0.1
        }
        
        logger.info("MetaOptimizationEngine initialized")
    
    def evaluate_strategy(self, result: OptimizationResult, strategy_name: str):
        """Evaluate optimization strategy performance."""
        efficiency = result.best_fitness / max(1, result.evaluations)
        self.strategy_performance[strategy_name].append(efficiency)
        
        self.optimization_history.append({
            'strategy': strategy_name,
            'result': result,
            'efficiency': efficiency,
            'timestamp': time.time()
        })
    
    def adapt_strategy(self) -> Dict[str, float]:
        """Adapt optimization strategy based on history."""
        if len(self.optimization_history) < 5:
            return self.current_strategy
        
        recent = list(self.optimization_history)[-10:]
        avg_efficiency = sum(r['efficiency'] for r in recent) / len(recent)
        
        if avg_efficiency < 0.5:
            self.current_strategy['mutation_rate'] *= 1.2
            self.current_strategy['population_size'] = min(100, self.current_strategy['population_size'] + 10)
        elif avg_efficiency > 0.8:
            self.current_strategy['mutation_rate'] *= 0.9
        
        return self.current_strategy


class OptimizationTransferLearner:
    """Transfers knowledge between optimization runs."""
    
    def __init__(self):
        self.knowledge_base: Dict[str, Dict] = {}
        self.parameter_correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("OptimizationTransferLearner initialized")
    
    def store_result(self, problem_id: str, result: OptimizationResult):
        """Store optimization result for transfer."""
        self.knowledge_base[problem_id] = {
            'best_params': result.best_params,
            'fitness': result.best_fitness,
            'timestamp': time.time()
        }
    
    def get_warm_start(self, problem_id: str, similar_problems: List[str] = None) -> Optional[Dict[str, float]]:
        """Get warm start from similar problems."""
        if problem_id in self.knowledge_base:
            return self.knowledge_base[problem_id]['best_params']
        
        if similar_problems:
            for similar in similar_problems:
                if similar in self.knowledge_base:
                    return self.knowledge_base[similar]['best_params']
        
        return None
    
    def learn_correlation(self, param1: str, param2: str, correlation: float):
        """Learn parameter correlations."""
        self.parameter_correlations[param1][param2] = correlation
        self.parameter_correlations[param2][param1] = correlation


class MultiObjectiveEvolutionaryAlgorithm:
    """NSGA-II style multi-objective optimization."""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ['profit', 'risk']
        self.pareto_front: List[Individual] = []
        
        logger.info("MultiObjectiveEvolutionaryAlgorithm initialized")
    
    def dominates(self, ind1: Dict[str, float], ind2: Dict[str, float]) -> bool:
        """Check if ind1 dominates ind2."""
        dominated = True
        for obj in self.objectives:
            if ind1.get(obj, 0) < ind2.get(obj, 0):
                return False
            if ind1.get(obj, 0) > ind2.get(obj, 0):
                dominated = True
        return dominated
    
    def compute_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """Compute Pareto front from population."""
        fronts = [[]]
        domination_count = {i: 0 for i in range(len(population))}
        dominated_solutions = {i: [] for i in range(len(population))}
        
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if self.dominates(p.genes, q.genes):
                        dominated_solutions[i].append(j)
                    elif self.dominates(q.genes, p.genes):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        self.pareto_front = [population[i] for i in fronts[0]]
        return self.pareto_front


class AdaptivePopulationManager:
    """Manages adaptive population size and diversity."""
    
    def __init__(self, initial_size: int = 50):
        self.population_size = initial_size
        self.min_size = 20
        self.max_size = 200
        self.diversity_history: deque = deque(maxlen=50)
        
        logger.info("AdaptivePopulationManager initialized")
    
    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        all_genes = [list(ind.genes.values()) for ind in population]
        
        variances = []
        for i in range(len(all_genes[0])):
            values = [g[i] for g in all_genes]
            variances.append(np.var(values) if values else 0.0)
        
        return float(np.mean(variances))
    
    def adapt_size(self, population: List[Individual], fitness_improvement: float):
        """Adapt population size."""
        diversity = self.calculate_diversity(population)
        self.diversity_history.append(diversity)
        
        if diversity < 0.1:
            self.population_size = min(self.max_size, self.population_size + 10)
            logger.info(f"Population increased to {self.population_size} (low diversity)")
        
        elif fitness_improvement > 0.1:
            self.population_size = max(self.min_size, self.population_size - 5)
            logger.info(f"Population decreased to {self.population_size} (good convergence)")


class AdvancedCMAES:
    """Advanced CMA-ES optimizer."""
    
    def __init__(self, dim: int = 5):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.sigma = 0.3
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        
        self.lambda_ = 4 + int(3 * np.log(dim))
        self.mu = self.lambda_ // 2
        
        self.cc = 4.0 / (dim + 4)
        self.cs = (2.0 + self.mu) / (5 + dim + self.mu)
        
        logger.info("AdvancedCMAES initialized")
    
    def ask(self, n: int = None) -> List[np.ndarray]:
        """Generate candidate solutions."""
        n = n or self.lambda_
        
        try:
            B, D = np.linalg.eigh(self.C)
            D = np.sqrt(np.maximum(D, 1e-10))
            BD = B * D
            
            samples = []
            for _ in range(n):
                z = np.random.randn(self.dim)
                x = self.mean + self.sigma * BD.dot(z)
                samples.append(x)
            
            return samples
        except:
            return [self.mean + self.sigma * np.random.randn(self.dim) for _ in range(n)]
    
    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]):
        """Update distribution based on evaluated solutions."""
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        best_solutions = [solutions[i] for i in sorted_indices[:self.mu]]
        
        old_mean = self.mean.copy()
        self.mean = np.mean(best_solutions, axis=0)
        
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu) * (self.mean - old_mean) / self.sigma
        
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (len(fitnesses) + 1))) < 1.4 + 2 / (self.dim + 1)
        
        if hsig:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu) * (self.mean - old_mean) / self.sigma
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) - 1.4) * 0.3)


class OptimizerAGI:
    """Main Optimizer AGI System."""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]] = None):
        self.param_bounds = param_bounds or {
            'sl': (5.0, 50.0),
            'tp': (1.0, 20.0),
            'threshold': (0.5, 0.95)
        }
        
        self.meta = MetaOptimizationEngine()
        self.transfer = OptimizationTransferLearner()
        self.moea = MultiObjectiveEvolutionaryAlgorithm()
        self.population_manager = AdaptivePopulationManager()
        self.cmaes = AdvancedCMAES(dim=len(self.param_bounds))
        
        self.best_result: Optional[OptimizationResult] = None
        
        logger.info("OptimizerAGI initialized")
    
    def optimize(self, fitness_func: Callable, generations: int = 50) -> OptimizationResult:
        """Run optimization."""
        population = self._initialize_population()
        
        convergence = []
        evaluations = 0
        
        for gen in range(generations):
            for ind in population:
                ind.fitness = fitness_func(ind.genes)
                evaluations += 1
            
            population.sort(key=lambda x: x.fitness, reverse=True)
            convergence.append(population[0].fitness)
            
            elite_count = max(2, int(len(population) * 0.1))
            new_pop = population[:elite_count]
            
            while len(new_pop) < self.population_manager.population_size:
                parent1, parent2 = random.sample(population[:len(population)//2], 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_pop.append(child)
            
            population = new_pop
            
            if gen > 0:
                improvement = convergence[-1] - convergence[-2] if len(convergence) > 1 else 0
                self.population_manager.adapt_size(population, improvement)
        
        best = max(population, key=lambda x: x.fitness)
        
        result = OptimizationResult(
            best_params=best.genes,
            best_fitness=best.fitness,
            generations=generations,
            evaluations=evaluations,
            convergence_history=convergence
        )
        
        self.best_result = result
        self.transfer.store_result('last', result)
        
        return result
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize population."""
        population = []
        
        warm_start = self.transfer.get_warm_start('last')
        if warm_start:
            population.append(Individual(genes=warm_start))
        
        while len(population) < self.population_manager.population_size:
            genes = {}
            for param, (low, high) in self.param_bounds.items():
                genes[param] = random.uniform(low, high)
            population.append(Individual(genes=genes))
        
        return population
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover two parents."""
        child_genes = {}
        for param in parent1.genes:
            if random.random() < 0.5:
                child_genes[param] = parent1.genes[param]
            else:
                child_genes[param] = parent2.genes[param]
        return Individual(genes=child_genes)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutate individual."""
        strategy = self.meta.current_strategy
        mutated_genes = individual.genes.copy()
        
        for param in mutated_genes:
            if random.random() < strategy['mutation_rate']:
                low, high = self.param_bounds.get(param, (0, 1))
                mutated_genes[param] += random.gauss(0, (high - low) * 0.1)
                mutated_genes[param] = max(low, min(high, mutated_genes[param]))
        
        return Individual(genes=mutated_genes)
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGI status."""
        return {
            'population_size': self.population_manager.population_size,
            'best_fitness': self.best_result.best_fitness if self.best_result else None,
            'knowledge_base_size': len(self.transfer.knowledge_base),
            'strategy': self.meta.current_strategy
        }
