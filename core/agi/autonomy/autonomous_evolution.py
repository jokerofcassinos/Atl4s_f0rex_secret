"""
AGI Fase 2: Autonomous Evolution Engine

Motor de Evolução Autônoma:
- Evolução de Arquitetura
- Evolução de Objetivos
- Meta-Evolução
- Open-Ended Evolution
"""

import logging
import time
import random
import copy
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("AutonomousEvolution")


@dataclass
class Gene:
    """A single gene in the genome."""
    name: str
    value: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mutation_rate: float = 0.1
    immutable: bool = False


@dataclass
class Genome:
    """Complete genome of an individual."""
    genome_id: str
    genes: Dict[str, Gene]
    fitness: float = 0.0
    age: int = 0
    lineage: List[str] = field(default_factory=list)


@dataclass
class Objective:
    """An evolving objective."""
    name: str
    weight: float
    evaluator: Optional[Callable] = None
    evolved: bool = False


class ArchitectureEvolver:
    """
    Evolves the architecture of the system.
    
    Not just parameters, but structure.
    """
    
    def __init__(self):
        self.architectures: List[Dict] = []
        self.current_architecture: Optional[Dict] = None
        
        logger.info("ArchitectureEvolver initialized")
    
    def encode_architecture(self, modules: List[str], connections: List[Tuple[str, str]]) -> Dict:
        """Encode architecture as genome."""
        return {
            'id': f"arch_{len(self.architectures)}",
            'modules': modules,
            'connections': connections,
            'depth': self._calculate_depth(connections),
            'fitness': 0.0
        }
    
    def _calculate_depth(self, connections: List[Tuple[str, str]]) -> int:
        """Calculate architecture depth."""
        if not connections:
            return 1
        
        # Build adjacency
        adj = defaultdict(list)
        for src, dst in connections:
            adj[src].append(dst)
        
        # Find max depth
        def dfs(node, visited):
            if node in visited:
                return 0
            visited.add(node)
            if node not in adj:
                return 1
            return 1 + max((dfs(n, visited) for n in adj[node]), default=0)
        
        all_nodes = set([c[0] for c in connections] + [c[1] for c in connections])
        return max((dfs(n, set()) for n in all_nodes), default=1)
    
    def mutate_architecture(self, arch: Dict) -> Dict:
        """Mutate architecture."""
        new_arch = copy.deepcopy(arch)
        new_arch['id'] = f"arch_{len(self.architectures)}"
        
        mutation_type = random.choice(['add_module', 'remove_module', 'add_connection', 'remove_connection'])
        
        if mutation_type == 'add_module':
            new_module = f"module_{len(new_arch['modules'])}"
            new_arch['modules'].append(new_module)
            # Connect to random existing module
            if new_arch['modules']:
                existing = random.choice(new_arch['modules'][:-1]) if len(new_arch['modules']) > 1 else None
                if existing:
                    new_arch['connections'].append((existing, new_module))
        
        elif mutation_type == 'remove_module' and len(new_arch['modules']) > 1:
            to_remove = random.choice(new_arch['modules'])
            new_arch['modules'].remove(to_remove)
            new_arch['connections'] = [c for c in new_arch['connections'] if to_remove not in c]
        
        elif mutation_type == 'add_connection' and len(new_arch['modules']) >= 2:
            src = random.choice(new_arch['modules'])
            dst = random.choice([m for m in new_arch['modules'] if m != src])
            if (src, dst) not in new_arch['connections']:
                new_arch['connections'].append((src, dst))
        
        elif mutation_type == 'remove_connection' and new_arch['connections']:
            to_remove = random.choice(new_arch['connections'])
            new_arch['connections'].remove(to_remove)
        
        self.architectures.append(new_arch)
        return new_arch
    
    def crossover_architecture(self, arch1: Dict, arch2: Dict) -> Dict:
        """Crossover two architectures."""
        new_modules = list(set(arch1['modules'][:len(arch1['modules'])//2] + 
                               arch2['modules'][len(arch2['modules'])//2:]))
        
        # Take connections that are still valid
        new_connections = []
        for conn in arch1['connections'] + arch2['connections']:
            if conn[0] in new_modules and conn[1] in new_modules:
                if conn not in new_connections:
                    new_connections.append(conn)
        
        return self.encode_architecture(new_modules, new_connections)


class ObjectiveEvolver:
    """
    Evolves the objectives themselves.
    
    Objectives can change based on experience.
    """
    
    def __init__(self):
        self.objectives: Dict[str, Objective] = {}
        self.objective_history: List[Dict] = []
        
        # Initialize default objectives
        self._init_default_objectives()
        
        logger.info("ObjectiveEvolver initialized")
    
    def _init_default_objectives(self):
        """Initialize default objectives."""
        defaults = [
            ('maximize_profit', 0.4),
            ('minimize_risk', 0.3),
            ('maximize_consistency', 0.2),
            ('maximize_adaptability', 0.1),
        ]
        
        for name, weight in defaults:
            self.objectives[name] = Objective(name=name, weight=weight, evolved=False)
    
    def evolve_weights(self, performance: Dict[str, float]):
        """Evolve objective weights based on performance."""
        old_weights = {name: obj.weight for name, obj in self.objectives.items()}
        
        for name, obj in self.objectives.items():
            if name in performance:
                # Increase weight if performing well
                if performance[name] > 0.5:
                    obj.weight = min(0.8, obj.weight * 1.1)
                else:
                    obj.weight = max(0.05, obj.weight * 0.9)
                obj.evolved = True
        
        # Normalize
        total = sum(obj.weight for obj in self.objectives.values())
        for obj in self.objectives.values():
            obj.weight /= total
        
        self.objective_history.append({
            'timestamp': time.time(),
            'old': old_weights,
            'new': {name: obj.weight for name, obj in self.objectives.items()}
        })
    
    def discover_objective(self, name: str, initial_weight: float = 0.1):
        """Discover and add a new objective."""
        if name not in self.objectives:
            self.objectives[name] = Objective(name=name, weight=initial_weight, evolved=True)
            
            # Normalize
            total = sum(obj.weight for obj in self.objectives.values())
            for obj in self.objectives.values():
                obj.weight /= total
            
            logger.info(f"New objective discovered: {name}")


class MetaEvolver:
    """
    Meta-evolution: Evolution of evolution mechanisms.
    
    The evolutionary process itself evolves.
    """
    
    def __init__(self):
        self.evolution_params = {
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'selection_pressure': 1.5,
            'population_size': 50,
            'elitism_rate': 0.1,
        }
        
        self.param_history: List[Dict] = []
        
        logger.info("MetaEvolver initialized")
    
    def adapt_params(self, evolution_stats: Dict[str, float]):
        """Adapt evolution parameters based on progress."""
        old_params = copy.copy(self.evolution_params)
        
        diversity = evolution_stats.get('diversity', 0.5)
        improvement = evolution_stats.get('improvement_rate', 0.0)
        stagnation = evolution_stats.get('stagnation', 0)
        
        # If stagnating, increase mutation
        if stagnation > 5:
            self.evolution_params['mutation_rate'] = min(0.3, self.evolution_params['mutation_rate'] * 1.2)
            self.evolution_params['selection_pressure'] = max(1.0, self.evolution_params['selection_pressure'] * 0.9)
        
        # If improving, fine-tune
        if improvement > 0.1:
            self.evolution_params['mutation_rate'] = max(0.01, self.evolution_params['mutation_rate'] * 0.9)
        
        # If low diversity, increase exploration
        if diversity < 0.3:
            self.evolution_params['mutation_rate'] = min(0.3, self.evolution_params['mutation_rate'] * 1.3)
            self.evolution_params['crossover_rate'] = max(0.5, self.evolution_params['crossover_rate'] * 0.9)
        
        self.param_history.append({
            'timestamp': time.time(),
            'old': old_params,
            'new': copy.copy(self.evolution_params),
            'trigger': evolution_stats
        })
        
        logger.info(f"Evolution params adapted: mutation={self.evolution_params['mutation_rate']:.3f}")


class OpenEndedEvolution:
    """
    Open-ended evolution without predefined endpoint.
    
    Continuously generates novelty.
    """
    
    def __init__(self, max_population: int = 100):
        self.population: List[Genome] = []
        self.max_population = max_population
        self.generation = 0
        
        # Novelty archive
        self.novelty_archive: List[Genome] = []
        
        # Behavioral characteristics tracking
        self.behavior_space: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("OpenEndedEvolution initialized")
    
    def create_genome(self, genes: Dict[str, Any]) -> Genome:
        """Create a new genome."""
        gene_objects = {}
        for name, value in genes.items():
            gene_objects[name] = Gene(
                name=name,
                value=value,
                mutation_rate=0.1
            )
        
        genome = Genome(
            genome_id=f"gen_{self.generation}_{len(self.population)}",
            genes=gene_objects
        )
        
        return genome
    
    def mutate(self, genome: Genome) -> Genome:
        """Mutate a genome."""
        new_genome = copy.deepcopy(genome)
        new_genome.genome_id = f"gen_{self.generation}_{len(self.population)}"
        new_genome.lineage = genome.lineage + [genome.genome_id]
        new_genome.age = 0
        
        for gene in new_genome.genes.values():
            if not gene.immutable and random.random() < gene.mutation_rate:
                if isinstance(gene.value, (int, float)):
                    delta = (gene.max_val - gene.min_val) * 0.2 if gene.max_val and gene.min_val else gene.value * 0.2
                    gene.value += random.gauss(0, delta)
                    if gene.min_val is not None:
                        gene.value = max(gene.min_val, gene.value)
                    if gene.max_val is not None:
                        gene.value = min(gene.max_val, gene.value)
        
        return new_genome
    
    def calculate_novelty(self, genome: Genome, k: int = 15) -> float:
        """Calculate novelty of a genome compared to population and archive."""
        behavior = self._get_behavior_vector(genome)
        
        # Get behaviors of others
        all_behaviors = []
        for g in self.population + self.novelty_archive:
            all_behaviors.append(self._get_behavior_vector(g))
        
        if not all_behaviors:
            return 1.0
        
        # Calculate distances to k nearest neighbors
        distances = []
        for other_behavior in all_behaviors:
            dist = sum((a - b) ** 2 for a, b in zip(behavior, other_behavior)) ** 0.5
            distances.append(dist)
        
        distances.sort()
        k_nearest = distances[:min(k, len(distances))]
        
        return sum(k_nearest) / len(k_nearest) if k_nearest else 0.0
    
    def _get_behavior_vector(self, genome: Genome) -> List[float]:
        """Extract behavior vector from genome."""
        vector = []
        for gene in sorted(genome.genes.values(), key=lambda g: g.name):
            if isinstance(gene.value, (int, float)):
                vector.append(float(gene.value))
        return vector if vector else [0.0]
    
    def evolve_step(self, evaluator: Callable[[Genome], float]) -> Genome:
        """Run one step of open-ended evolution."""
        self.generation += 1
        
        # Evaluate population
        for genome in self.population:
            genome.fitness = evaluator(genome)
            genome.age += 1
        
        # Calculate novelty
        novelties = [(g, self.calculate_novelty(g)) for g in self.population]
        
        # Add highly novel individuals to archive
        for genome, novelty in novelties:
            if novelty > 0.5:  # Threshold
                self.novelty_archive.append(copy.deepcopy(genome))
        
        # Trim archive
        if len(self.novelty_archive) > self.max_population:
            self.novelty_archive = sorted(
                self.novelty_archive, 
                key=lambda g: g.fitness,
                reverse=True
            )[:self.max_population]
        
        # Selection: combine fitness and novelty
        scored = [(g, g.fitness * 0.5 + n * 0.5) for g, n in novelties]
        scored.sort(key=lambda x: -x[1])
        
        # Keep top half
        survivors = [g for g, _ in scored[:len(scored)//2]]
        
        # Generate new population
        new_population = list(survivors)
        while len(new_population) < self.max_population:
            if survivors:
                parent = random.choice(survivors)
                child = self.mutate(parent)
                new_population.append(child)
        
        self.population = new_population
        
        # Return best
        return max(self.population, key=lambda g: g.fitness) if self.population else None


class AutonomousEvolutionEngine:
    """
    Main Autonomous Evolution Engine.
    
    Combines:
    - ArchitectureEvolver
    - ObjectiveEvolver
    - MetaEvolver
    - OpenEndedEvolution
    """
    
    def __init__(self):
        self.architecture = ArchitectureEvolver()
        self.objectives = ObjectiveEvolver()
        self.meta = MetaEvolver()
        self.evolution = OpenEndedEvolution()
        
        self.generation = 0
        self.best_ever: Optional[Genome] = None
        
        logger.info("AutonomousEvolutionEngine initialized")
    
    def evolve(
        self,
        evaluator: Callable[[Genome], float],
        generations: int = 10
    ) -> Genome:
        """Run evolution for given generations."""
        for _ in range(generations):
            self.generation += 1
            
            # Evolve population
            best = self.evolution.evolve_step(evaluator)
            
            if best and (self.best_ever is None or best.fitness > self.best_ever.fitness):
                self.best_ever = copy.deepcopy(best)
            
            # Meta-evolution
            stats = {
                'diversity': len(set(g.genome_id for g in self.evolution.population)) / max(1, len(self.evolution.population)),
                'improvement_rate': (best.fitness - (self.best_ever.fitness if self.best_ever else 0)) if best else 0,
                'stagnation': 0  # Would track actual stagnation
            }
            self.meta.adapt_params(stats)
        
        return self.best_ever
    
    def get_status(self) -> Dict[str, Any]:
        """Get evolution status."""
        return {
            'generation': self.generation,
            'population_size': len(self.evolution.population),
            'archive_size': len(self.evolution.novelty_archive),
            'best_fitness': self.best_ever.fitness if self.best_ever else 0,
            'objectives': list(self.objectives.objectives.keys()),
            'mutation_rate': self.meta.evolution_params['mutation_rate']
        }
