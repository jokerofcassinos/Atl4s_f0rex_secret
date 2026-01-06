"""
AGI Fase 2: Creative Strategy Generator

Geração Criativa de Estratégias de Trading:
- Geração Original de estratégias
- Composição de Conceitos de múltiplos domínios
- Exploração de Espaço de Solução
- Evolução de Estratégias com mutação criativa
"""

import logging
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy

logger = logging.getLogger("StrategyGenerator")


class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    HYBRID = "hybrid"


class ConceptDomain(Enum):
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"
    PSYCHOLOGY = "psychology"
    ECONOMICS = "economics"
    GAME_THEORY = "game_theory"


@dataclass
class StrategyGene:
    """A gene representing a strategy component."""
    name: str
    value: Any
    domain: str
    mutability: float  # 0-1
    importance: float  # 0-1


@dataclass 
class Strategy:
    """A complete trading strategy."""
    strategy_id: str
    name: str
    strategy_type: StrategyType
    genes: List[StrategyGene]
    
    # Performance
    fitness: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    
    # Meta
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def get_gene(self, name: str) -> Optional[Any]:
        """Get gene value by name."""
        for gene in self.genes:
            if gene.name == name:
                return gene.value
        return None


@dataclass
class Concept:
    """A concept from any domain that can be applied to trading."""
    name: str
    domain: ConceptDomain
    description: str
    principles: List[str]
    trading_application: str


class ConceptLibrary:
    """Library of concepts from different domains."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self._initialize_concepts()
    
    def _initialize_concepts(self):
        """Initialize cross-domain concepts."""
        concepts = [
            # Physics
            Concept(
                "momentum", ConceptDomain.PHYSICS,
                "Object in motion tends to stay in motion",
                ["inertia", "mass", "velocity"],
                "Trend continuation - price momentum predicts future movement"
            ),
            Concept(
                "equilibrium", ConceptDomain.PHYSICS,
                "Systems tend toward balance",
                ["forces", "balance", "stability"],
                "Mean reversion - prices return to average"
            ),
            Concept(
                "resonance", ConceptDomain.PHYSICS,
                "Amplification at natural frequency",
                ["frequency", "amplitude", "coupling"],
                "Market cycles at specific timeframes"
            ),
            
            # Biology
            Concept(
                "evolution", ConceptDomain.BIOLOGY,
                "Survival of the fittest through adaptation",
                ["selection", "mutation", "fitness"],
                "Strategy evolution and parameter optimization"
            ),
            Concept(
                "predator_prey", ConceptDomain.BIOLOGY,
                "Cyclical population dynamics",
                ["cycles", "feedback", "population"],
                "Bull/bear market cycles"
            ),
            Concept(
                "swarm_intelligence", ConceptDomain.BIOLOGY,
                "Collective behavior from simple rules",
                ["emergence", "decentralization", "adaptation"],
                "Market consensus from individual traders"
            ),
            
            # Psychology
            Concept(
                "herd_behavior", ConceptDomain.PSYCHOLOGY,
                "Individuals follow the crowd",
                ["conformity", "fear", "greed"],
                "Trend following with crowd psychology"
            ),
            Concept(
                "anchoring", ConceptDomain.PSYCHOLOGY,
                "First information dominates judgment",
                ["reference", "bias", "adjustment"],
                "Support/resistance level psychology"
            ),
            
            # Mathematics
            Concept(
                "fractals", ConceptDomain.MATHEMATICS,
                "Self-similarity across scales",
                ["recursion", "scaling", "patterns"],
                "Multi-timeframe pattern analysis"
            ),
            Concept(
                "chaos", ConceptDomain.MATHEMATICS,
                "Sensitivity to initial conditions",
                ["butterfly_effect", "unpredictability", "attractors"],
                "Risk management in volatile markets"
            ),
            
            # Game Theory
            Concept(
                "nash_equilibrium", ConceptDomain.GAME_THEORY,
                "Stable state when no player can improve",
                ["strategy", "payoff", "stability"],
                "Optimal strategy against market participants"
            ),
        ]
        
        for concept in concepts:
            self.concepts[concept.name] = concept
    
    def get_concepts_by_domain(self, domain: ConceptDomain) -> List[Concept]:
        """Get all concepts from a domain."""
        return [c for c in self.concepts.values() if c.domain == domain]
    
    def get_random_concepts(self, n: int) -> List[Concept]:
        """Get random concepts for blending."""
        return random.sample(list(self.concepts.values()), min(n, len(self.concepts)))


class StrategyGenerator:
    """
    Creative Strategy Generator.
    
    Creates original trading strategies through:
    - Genetic programming
    - Concept blending
    - Divergent thinking
    """
    
    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.concept_library = ConceptLibrary()
        
        # Strategy population
        self.population: List[Strategy] = []
        self.generation = 0
        
        # Gene templates
        self.gene_templates = self._create_gene_templates()
        
        # Statistics
        self.strategies_created = 0
        self.best_fitness_history: List[float] = []
        
        logger.info(f"StrategyGenerator initialized (pop_size={population_size})")
    
    def _create_gene_templates(self) -> Dict[str, Dict]:
        """Create templates for strategy genes."""
        return {
            'entry_threshold': {'min': 0.1, 'max': 1.0, 'domain': 'signal'},
            'exit_threshold': {'min': 0.1, 'max': 1.0, 'domain': 'signal'},
            'stop_loss_atr': {'min': 0.5, 'max': 5.0, 'domain': 'risk'},
            'take_profit_atr': {'min': 1.0, 'max': 10.0, 'domain': 'risk'},
            'risk_per_trade': {'min': 0.001, 'max': 0.05, 'domain': 'risk'},
            'lookback_period': {'min': 5, 'max': 200, 'domain': 'timing'},
            'trend_filter': {'values': ['SMA', 'EMA', 'WMA', 'NONE'], 'domain': 'filter'},
            'volatility_filter': {'min': 0.5, 'max': 2.0, 'domain': 'filter'},
            'time_filter': {'values': ['LONDON', 'NY', 'ASIA', 'ALL'], 'domain': 'filter'},
            'momentum_weight': {'min': 0.0, 'max': 1.0, 'domain': 'weighting'},
            'reversion_weight': {'min': 0.0, 'max': 1.0, 'domain': 'weighting'},
        }
    
    def generate_original(self) -> Strategy:
        """Generate a completely new strategy."""
        strategy_id = f"strat_{self.strategies_created}"
        self.strategies_created += 1
        
        # Random type
        strategy_type = random.choice(list(StrategyType))
        
        # Generate genes
        genes = []
        for name, template in self.gene_templates.items():
            if 'min' in template:
                value = random.uniform(template['min'], template['max'])
            else:
                value = random.choice(template['values'])
            
            gene = StrategyGene(
                name=name,
                value=value,
                domain=template['domain'],
                mutability=random.uniform(0.3, 0.8),
                importance=random.uniform(0.3, 0.9)
            )
            genes.append(gene)
        
        # Name from concepts
        concepts = self.concept_library.get_random_concepts(2)
        name = f"{concepts[0].name}_{concepts[1].name}_{strategy_type.value[:3]}"
        
        return Strategy(
            strategy_id=strategy_id,
            name=name,
            strategy_type=strategy_type,
            genes=genes,
            generation=self.generation
        )
    
    def blend_concepts(self, concepts: List[Concept]) -> Strategy:
        """Create strategy by blending concepts from different domains."""
        strategy_id = f"blend_{self.strategies_created}"
        self.strategies_created += 1
        
        genes = []
        
        # Map concepts to strategy parameters
        for concept in concepts:
            if concept.domain == ConceptDomain.PHYSICS:
                if 'momentum' in concept.name:
                    genes.append(StrategyGene("momentum_weight", 0.7, "weighting", 0.5, 0.8))
                elif 'equilibrium' in concept.name:
                    genes.append(StrategyGene("reversion_weight", 0.7, "weighting", 0.5, 0.8))
            
            elif concept.domain == ConceptDomain.BIOLOGY:
                if 'swarm' in concept.name:
                    genes.append(StrategyGene("consensus_weight", 0.6, "weighting", 0.6, 0.7))
            
            elif concept.domain == ConceptDomain.PSYCHOLOGY:
                if 'herd' in concept.name:
                    genes.append(StrategyGene("trend_follow_strength", 0.8, "signal", 0.4, 0.8))
        
        # Fill in remaining genes
        for name, template in self.gene_templates.items():
            if not any(g.name == name for g in genes):
                if 'min' in template:
                    value = random.uniform(template['min'], template['max'])
                else:
                    value = random.choice(template['values'])
                genes.append(StrategyGene(name, value, template['domain'], 0.5, 0.5))
        
        # Determine type from concepts
        if any('momentum' in c.name for c in concepts):
            strategy_type = StrategyType.MOMENTUM
        elif any('equilibrium' in c.name for c in concepts):
            strategy_type = StrategyType.MEAN_REVERSION
        else:
            strategy_type = StrategyType.HYBRID
        
        name = "_".join([c.name[:5] for c in concepts])
        
        return Strategy(
            strategy_id=strategy_id,
            name=name,
            strategy_type=strategy_type,
            genes=genes,
            generation=self.generation
        )
    
    def crossover(self, parent1: Strategy, parent2: Strategy) -> Strategy:
        """Create child strategy from two parents."""
        strategy_id = f"cross_{self.strategies_created}"
        self.strategies_created += 1
        
        genes = []
        
        # Mix genes from parents
        all_gene_names = set(g.name for g in parent1.genes + parent2.genes)
        
        for name in all_gene_names:
            g1 = next((g for g in parent1.genes if g.name == name), None)
            g2 = next((g for g in parent2.genes if g.name == name), None)
            
            if g1 and g2:
                # Blend based on fitness
                if parent1.fitness > parent2.fitness:
                    weight = 0.7
                else:
                    weight = 0.3
                
                if isinstance(g1.value, (int, float)):
                    value = g1.value * weight + g2.value * (1 - weight)
                else:
                    value = g1.value if random.random() < weight else g2.value
                
                genes.append(StrategyGene(
                    name=name,
                    value=value,
                    domain=g1.domain,
                    mutability=(g1.mutability + g2.mutability) / 2,
                    importance=(g1.importance + g2.importance) / 2
                ))
            elif g1:
                genes.append(copy.deepcopy(g1))
            else:
                genes.append(copy.deepcopy(g2))
        
        return Strategy(
            strategy_id=strategy_id,
            name=f"child_{parent1.name[:5]}_{parent2.name[:5]}",
            strategy_type=parent1.strategy_type if parent1.fitness > parent2.fitness else parent2.strategy_type,
            genes=genes,
            generation=self.generation,
            parent_ids=[parent1.strategy_id, parent2.strategy_id]
        )
    
    def mutate(self, strategy: Strategy, rate: float = 0.1) -> Strategy:
        """Mutate a strategy."""
        mutated = copy.deepcopy(strategy)
        mutated.strategy_id = f"mut_{self.strategies_created}"
        self.strategies_created += 1
        
        for gene in mutated.genes:
            if random.random() < rate * gene.mutability:
                template = self.gene_templates.get(gene.name)
                if template:
                    if 'min' in template:
                        # Gaussian mutation
                        delta = (template['max'] - template['min']) * 0.2
                        gene.value = np.clip(
                            gene.value + random.gauss(0, delta),
                            template['min'],
                            template['max']
                        )
                    else:
                        # Random choice
                        gene.value = random.choice(template['values'])
        
        mutated.generation = self.generation
        return mutated
    
    def divergent_think(self, n: int = 5) -> List[Strategy]:
        """Generate multiple diverse strategies (divergent thinking)."""
        strategies = []
        
        # Ensure diversity
        types_used = set()
        
        for _ in range(n):
            # Try different creation methods
            method = random.choice(['original', 'blend', 'mutate'])
            
            if method == 'original':
                strat = self.generate_original()
            elif method == 'blend':
                concepts = self.concept_library.get_random_concepts(2)
                strat = self.blend_concepts(concepts)
            else:
                if self.population:
                    base = random.choice(self.population)
                    strat = self.mutate(base, rate=0.3)  # High mutation
                else:
                    strat = self.generate_original()
            
            # Ensure type diversity
            if strat.strategy_type not in types_used or len(types_used) >= len(StrategyType):
                strategies.append(strat)
                types_used.add(strat.strategy_type)
        
        return strategies
    
    def evaluate_strategy(
        self,
        strategy: Strategy,
        evaluator: Callable[[Strategy], Dict[str, float]]
    ):
        """Evaluate strategy fitness."""
        results = evaluator(strategy)
        
        strategy.fitness = results.get('fitness', 0.0)
        strategy.trades = results.get('trades', 0)
        strategy.win_rate = results.get('win_rate', 0.0)
        strategy.sharpe = results.get('sharpe', 0.0)
    
    def evolve(self, evaluator: Callable[[Strategy], Dict[str, float]]) -> Strategy:
        """Run one generation of evolution."""
        self.generation += 1
        
        # Initialize if empty
        if not self.population:
            self.population = [self.generate_original() for _ in range(self.population_size)]
        
        # Evaluate all
        for strat in self.population:
            if strat.fitness == 0:
                self.evaluate_strategy(strat, evaluator)
        
        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Record best
        if self.population:
            self.best_fitness_history.append(self.population[0].fitness)
        
        # Selection - keep top 50%
        survivors = self.population[:self.population_size // 2]
        
        # Create new generation
        new_population = list(survivors)
        
        # Crossover
        while len(new_population) < self.population_size * 0.8:
            p1, p2 = random.sample(survivors, 2)
            child = self.crossover(p1, p2)
            new_population.append(child)
        
        # Mutation
        for strat in new_population[len(survivors):]:
            self.mutate(strat, rate=0.1)
        
        # Add some fresh blood
        while len(new_population) < self.population_size:
            new_population.append(self.generate_original())
        
        self.population = new_population
        
        return self.population[0]  # Return best
    
    def get_best_strategies(self, n: int = 5) -> List[Strategy]:
        """Get top N strategies."""
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        return sorted_pop[:n]
