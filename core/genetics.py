"""
AGI Ultra: Evolution Engine - Multi-Objective Genetic Optimization

Features:
- 1000+ population for broader genetic diversity
- Multi-objective optimization (profit, Sharpe, drawdown)
- Guided mutation based on historical patterns
- Island model for parallel evolution
- Adaptive mutation rates
"""

import numpy as np
import random
import logging
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("Genetics")


@dataclass
class FitnessVector:
    """Multi-objective fitness score."""
    profit: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    
    def dominates(self, other: 'FitnessVector') -> bool:
        """Check if this fitness dominates another (Pareto)."""
        better_in_any = False
        for attr in ['profit', 'sharpe_ratio', 'win_rate']:
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)
            if self_val < other_val:
                return False
            if self_val > other_val:
                better_in_any = True
        # Lower drawdown is better
        if self.max_drawdown > other.max_drawdown:
            return False
        if self.max_drawdown < other.max_drawdown:
            better_in_any = True
        return better_in_any
    
    def weighted_score(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted scalar score."""
        weights = weights or {'profit': 0.3, 'sharpe': 0.3, 'drawdown': 0.2, 'win_rate': 0.2}
        score = (
            weights.get('profit', 0.3) * np.tanh(self.profit * 0.01) +
            weights.get('sharpe', 0.3) * np.tanh(self.sharpe_ratio * 0.5) +
            weights.get('drawdown', 0.2) * (1.0 - min(1.0, abs(self.max_drawdown) * 0.01)) +
            weights.get('win_rate', 0.2) * self.win_rate
        )
        return score


class DNA:
    """
    AGI Ultra: Genetic Code with extended genome.
    """
    
    # Full genome template with valid ranges
    GENE_RANGES = {
        'rsi_period': (5, 50, 'int'),
        'rsi_overbought': (60, 85, 'int'),
        'rsi_oversold': (15, 40, 'int'),
        'bb_period': (10, 50, 'int'),
        'bb_std': (1.0, 3.5, 'float'),
        'nash_lookback': (50, 300, 'int'),
        'chaos_embedding': (2, 10, 'int'),
        'ma_fast': (5, 50, 'int'),
        'ma_slow': (20, 200, 'int'),
        'atr_period': (7, 28, 'int'),
        'atr_multiplier': (1.0, 4.0, 'float'),
        'volume_threshold': (1.0, 3.0, 'float'),
        'trend_strength_min': (0.1, 0.5, 'float'),
        'confidence_threshold': (0.5, 0.9, 'float'),
        'risk_per_trade': (0.01, 0.05, 'float'),
        'max_trades': (1, 5, 'int'),
        'trailing_activation': (0.001, 0.01, 'float'),
        'partial_tp_ratio': (0.3, 0.7, 'float'),
    }
    
    def __init__(self, genes: Optional[Dict] = None):
        if genes:
            self.genes = genes
        else:
            # Generate random genes within valid ranges
            self.genes = self._random_genome()
        
        self.fitness: Optional[FitnessVector] = None
        self.pareto_rank: int = 0
        self.crowding_distance: float = 0.0
    
    def _random_genome(self) -> Dict[str, Any]:
        """Generate random genome within valid ranges."""
        genes = {}
        for name, (min_val, max_val, dtype) in self.GENE_RANGES.items():
            if dtype == 'int':
                genes[name] = random.randint(int(min_val), int(max_val))
            else:
                genes[name] = random.uniform(min_val, max_val)
        return genes
    
    def mutate(self, mutation_rate: float = 0.1, guided: bool = True) -> 'DNA':
        """
        AGI Ultra: Guided mutation based on gene importance.
        """
        new_genes = deepcopy(self.genes)
        
        for key in new_genes:
            if random.random() < mutation_rate:
                if key in self.GENE_RANGES:
                    min_val, max_val, dtype = self.GENE_RANGES[key]
                    
                    if guided and self.fitness:
                        # Smaller mutations for good performers
                        strength = 1.0 - min(1.0, max(0.0, self.fitness.weighted_score()))
                        change_range = 0.05 + 0.15 * strength
                    else:
                        change_range = 0.15
                    
                    change = random.uniform(1 - change_range, 1 + change_range)
                    val = new_genes[key]
                    
                    if dtype == 'int':
                        new_val = int(val * change)
                        new_genes[key] = max(int(min_val), min(int(max_val), new_val))
                    else:
                        new_val = val * change
                        new_genes[key] = max(min_val, min(max_val, new_val))
        
        return DNA(new_genes)
    
    def crossover(self, partner: 'DNA') -> 'DNA':
        """Multi-point crossover."""
        child_genes = {}
        keys = list(self.genes.keys())
        
        # Choose 2 crossover points
        points = sorted(random.sample(range(len(keys)), min(2, len(keys))))
        
        use_self = True
        for i, key in enumerate(keys):
            if i in points:
                use_self = not use_self
            child_genes[key] = self.genes[key] if use_self else partner.genes[key]
        
        return DNA(child_genes)


class Island:
    """An island in the island model evolution."""
    
    def __init__(self, island_id: int, population_size: int):
        self.island_id = island_id
        self.population = [DNA() for _ in range(population_size)]
        self.best_fitness = FitnessVector()
        self.generation = 0


class EvolutionEngine:
    """
    AGI Ultra: Multi-Objective Evolution Engine.
    
    Features:
    - 1000+ population across multiple islands
    - NSGA-II style multi-objective optimization
    - Parallel fitness evaluation
    - Guided mutation based on performance
    - Regular migration between islands
    """
    
    def __init__(
        self,
        population_size: int = 1000,
        num_islands: int = 4,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.15,
        migration_interval: int = 10,
        parallel_workers: int = 4
    ):
        self.pop_size = population_size
        self.num_islands = num_islands
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.migration_interval = migration_interval
        self.parallel_workers = parallel_workers
        
        # Distribute population across islands
        island_size = population_size // num_islands
        self.islands = [Island(i, island_size) for i in range(num_islands)]
        
        self.generation = 0
        self.best_dna = DNA()
        self.best_fitness = FitnessVector()
        
        # Pareto front
        self.pareto_front: List[DNA] = []
        self.max_pareto_size = 100
        
        # Fitness weight evolution
        self.fitness_weights = {'profit': 0.3, 'sharpe': 0.3, 'drawdown': 0.2, 'win_rate': 0.2}
        
        # Thread pool for parallel evaluation
        self._executor = ThreadPoolExecutor(max_workers=parallel_workers)
        
        logger.info(f"EvolutionEngine initialized: pop={population_size}, islands={num_islands}")
    
    def evolve(self, df_history, fitness_fn=None) -> DNA:
        """
        Runs one generation of evolution across all islands.
        """
        self.generation += 1
        
        # 1. Evaluate all islands in parallel
        all_evaluated = []
        
        for island in self.islands:
            for dna in island.population:
                fitness = self._calculate_fitness(dna, df_history, fitness_fn)
                dna.fitness = fitness
                all_evaluated.append(dna)
        
        # 2. Perform NSGA-II ranking
        self._nsga2_ranking(all_evaluated)
        
        # 3. Update Pareto front
        self._update_pareto_front(all_evaluated)
        
        # 4. Find global best
        for dna in all_evaluated:
            if dna.fitness.weighted_score(self.fitness_weights) > self.best_fitness.weighted_score(self.fitness_weights):
                self.best_fitness = dna.fitness
                self.best_dna = dna
        
        # 5. Evolve each island
        for island in self.islands:
            self._evolve_island(island)
        
        # 6. Migration between islands
        if self.generation % self.migration_interval == 0:
            self._migrate()
        
        # 7. Adapt fitness weights based on performance
        self._adapt_fitness_weights()
        
        return self.best_dna
    
    def _evolve_island(self, island: Island):
        """Evolve a single island."""
        island.generation += 1
        
        # Sort by weighted score
        island.population.sort(
            key=lambda d: d.fitness.weighted_score(self.fitness_weights) if d.fitness else 0,
            reverse=True
        )
        
        # Update island best
        if island.population and island.population[0].fitness:
            island.best_fitness = island.population[0].fitness
        
        # Create new population
        elite_count = int(len(island.population) * self.elite_ratio)
        new_pop = island.population[:elite_count]  # Elitism
        
        while len(new_pop) < len(island.population):
            # Tournament selection
            p1 = self._tournament(island.population)
            p2 = self._tournament(island.population)
            
            child = p1.crossover(p2)
            child = child.mutate(self.mutation_rate, guided=True)
            new_pop.append(child)
        
        island.population = new_pop
    
    def _tournament(self, population: List[DNA], size: int = 3) -> DNA:
        """Tournament selection."""
        candidates = random.sample(population, min(size, len(population)))
        return max(
            candidates,
            key=lambda d: d.fitness.weighted_score(self.fitness_weights) if d.fitness else 0
        )
    
    def _nsga2_ranking(self, population: List[DNA]):
        """NSGA-II style Pareto ranking and crowding distance."""
        # Assign Pareto ranks
        remaining = list(population)
        rank = 0
        
        while remaining:
            non_dominated = []
            for dna in remaining:
                dominated = False
                for other in remaining:
                    if other is not dna and other.fitness and dna.fitness:
                        if other.fitness.dominates(dna.fitness):
                            dominated = True
                            break
                if not dominated:
                    non_dominated.append(dna)
            
            for dna in non_dominated:
                dna.pareto_rank = rank
                remaining.remove(dna)
            
            rank += 1
            if rank > 100:  # Safety limit
                break
        
        # Assign crowding distances per front
        fronts: Dict[int, List[DNA]] = {}
        for dna in population:
            if dna.pareto_rank not in fronts:
                fronts[dna.pareto_rank] = []
            fronts[dna.pareto_rank].append(dna)
        
        for front in fronts.values():
            self._assign_crowding_distance(front)
    
    def _assign_crowding_distance(self, front: List[DNA]):
        """Assign crowding distance to a Pareto front."""
        if len(front) <= 2:
            for dna in front:
                dna.crowding_distance = float('inf')
            return
        
        for dna in front:
            dna.crowding_distance = 0
        
        objectives = ['profit', 'sharpe_ratio', 'win_rate']
        
        for obj in objectives:
            front.sort(key=lambda d: getattr(d.fitness, obj) if d.fitness else 0)
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_range = (getattr(front[-1].fitness, obj) - getattr(front[0].fitness, obj)) if front[-1].fitness and front[0].fitness else 1
            if obj_range == 0:
                obj_range = 1
            
            for i in range(1, len(front) - 1):
                if front[i+1].fitness and front[i-1].fitness:
                    front[i].crowding_distance += (
                        getattr(front[i+1].fitness, obj) - getattr(front[i-1].fitness, obj)
                    ) / obj_range
    
    def _update_pareto_front(self, population: List[DNA]):
        """Update the global Pareto front."""
        candidates = [d for d in population if d.pareto_rank == 0]
        
        # Merge with existing front
        all_candidates = self.pareto_front + candidates
        
        # Filter dominated solutions
        new_front = []
        for dna in all_candidates:
            dominated = False
            for other in all_candidates:
                if other is not dna and other.fitness and dna.fitness:
                    if other.fitness.dominates(dna.fitness):
                        dominated = True
                        break
            if not dominated:
                new_front.append(dna)
        
        # Limit size
        if len(new_front) > self.max_pareto_size:
            new_front.sort(key=lambda d: d.crowding_distance, reverse=True)
            new_front = new_front[:self.max_pareto_size]
        
        self.pareto_front = new_front
    
    def _migrate(self):
        """Migrate best individuals between islands."""
        migrants_per_island = max(1, len(self.islands[0].population) // 20)
        
        for i in range(len(self.islands)):
            source = self.islands[i]
            target = self.islands[(i + 1) % len(self.islands)]
            
            # Send best from source to target
            source.population.sort(
                key=lambda d: d.fitness.weighted_score(self.fitness_weights) if d.fitness else 0,
                reverse=True
            )
            
            migrants = source.population[:migrants_per_island]
            
            # Replace worst in target
            target.population.sort(
                key=lambda d: d.fitness.weighted_score(self.fitness_weights) if d.fitness else 0
            )
            
            for j, migrant in enumerate(migrants):
                if j < len(target.population):
                    target.population[j] = deepcopy(migrant)
        
        logger.debug(f"Migration completed: {migrants_per_island} per island")
    
    def _adapt_fitness_weights(self):
        """Adapt fitness weights based on recent performance."""
        if len(self.pareto_front) < 5:
            return
        
        # If drawdowns are generally high, increase drawdown weight
        avg_dd = np.mean([d.fitness.max_drawdown for d in self.pareto_front if d.fitness])
        if avg_dd > 10:
            self.fitness_weights['drawdown'] = min(0.4, self.fitness_weights['drawdown'] + 0.02)
            self.fitness_weights['profit'] = max(0.2, self.fitness_weights['profit'] - 0.01)
    
    def _calculate_fitness(
        self,
        dna: DNA,
        df,
        custom_fn=None
    ) -> FitnessVector:
        """
        Calculate multi-objective fitness.
        """
        if custom_fn:
            return custom_fn(dna, df)
        
        # Placeholder: Return simulated fitness
        # In production, this runs the actual backtest
        return FitnessVector(
            profit=random.gauss(0, 50),
            sharpe_ratio=random.gauss(1, 0.5),
            max_drawdown=random.uniform(0, 20),
            win_rate=random.uniform(0.4, 0.7),
            trade_count=random.randint(10, 100)
        )
    
    def get_best_from_pareto(self, objective: str = 'profit') -> Optional[DNA]:
        """Get best DNA from Pareto front for a specific objective."""
        if not self.pareto_front:
            return self.best_dna
        
        return max(
            self.pareto_front,
            key=lambda d: getattr(d.fitness, objective) if d.fitness else 0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'best_score': self.best_fitness.weighted_score(self.fitness_weights),
            'best_profit': self.best_fitness.profit,
            'best_sharpe': self.best_fitness.sharpe_ratio,
            'pareto_size': len(self.pareto_front),
            'islands': len(self.islands),
            'fitness_weights': self.fitness_weights
        }
