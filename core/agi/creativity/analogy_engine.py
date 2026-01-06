"""
AGI Fase 2: Analogy Engine

Sistema de Analogias e Metáforas para pensamento criativo:
- Mapeamento Analógico entre domínios
- Transferência de Conhecimento
- Metáforas Estruturais
- Pensamento Lateral
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("AnalogyEngine")


@dataclass
class DomainConcept:
    """A concept within a domain."""
    name: str
    domain: str
    attributes: Dict[str, Any]
    relations: List[Tuple[str, str]]  # (relation_type, target_concept)


@dataclass
class AnalogicalMapping:
    """A mapping between source and target domain."""
    mapping_id: str
    source_domain: str
    target_domain: str
    concept_mappings: Dict[str, str]  # source_concept -> target_concept
    relation_mappings: Dict[str, str]  # source_relation -> target_relation
    confidence: float
    created_at: float = field(default_factory=time.time)


@dataclass
class StructuralMetaphor:
    """A structural metaphor for understanding."""
    metaphor_id: str
    source: str  # e.g., "WAR"
    target: str  # e.g., "TRADING"
    mappings: Dict[str, str]  # e.g., {"attack": "buy", "defend": "hedge"}
    entailments: List[str]  # Logical entailments from the metaphor


@dataclass
class LateralIdea:
    """An idea generated through lateral thinking."""
    idea_id: str
    trigger: str
    lateral_jump: str
    resulting_idea: str
    applicability: float  # 0-1


class DomainKnowledge:
    """
    Knowledge representation for a domain.
    
    Stores concepts and their relations.
    """
    
    def __init__(self, domain_name: str):
        self.name = domain_name
        self.concepts: Dict[str, DomainConcept] = {}
        self.hierarchies: Dict[str, List[str]] = {}  # parent -> children
    
    def add_concept(
        self,
        name: str,
        attributes: Dict[str, Any],
        relations: Optional[List[Tuple[str, str]]] = None
    ):
        """Add a concept to the domain."""
        self.concepts[name] = DomainConcept(
            name=name,
            domain=self.name,
            attributes=attributes,
            relations=relations or []
        )
    
    def add_hierarchy(self, parent: str, children: List[str]):
        """Add hierarchical relation."""
        self.hierarchies[parent] = children
    
    def get_similar_concepts(self, attributes: Dict[str, Any]) -> List[str]:
        """Find concepts with similar attributes."""
        matches = []
        for name, concept in self.concepts.items():
            overlap = sum(
                1 for k, v in attributes.items()
                if concept.attributes.get(k) == v
            )
            if overlap > 0:
                matches.append((name, overlap))
        
        return [m[0] for m in sorted(matches, key=lambda x: -x[1])]


class AnalogicalMapper:
    """
    Maps analogies between different domains.
    
    Implements Structure Mapping Theory.
    """
    
    def __init__(self):
        self.domains: Dict[str, DomainKnowledge] = {}
        self.mappings: List[AnalogicalMapping] = []
        self._mapping_counter = 0
        
        # Initialize with trading and common domains
        self._initialize_domains()
        
        logger.info("AnalogicalMapper initialized")
    
    def _initialize_domains(self):
        """Initialize common domains."""
        # Trading domain
        trading = DomainKnowledge("trading")
        trading.add_concept("trend", {"direction": True, "momentum": True})
        trading.add_concept("support", {"barrier": True, "price_level": True})
        trading.add_concept("resistance", {"barrier": True, "price_level": True})
        trading.add_concept("breakout", {"movement": True, "sudden": True})
        trading.add_concept("consolidation", {"stability": True, "range": True})
        trading.add_concept("volatility", {"variability": True, "risk": True})
        self.domains["trading"] = trading
        
        # Physics domain
        physics = DomainKnowledge("physics")
        physics.add_concept("velocity", {"direction": True, "momentum": True})
        physics.add_concept("floor", {"barrier": True, "level": True})
        physics.add_concept("ceiling", {"barrier": True, "level": True})
        physics.add_concept("explosion", {"movement": True, "sudden": True})
        physics.add_concept("equilibrium", {"stability": True, "balance": True})
        physics.add_concept("turbulence", {"variability": True, "chaos": True})
        self.domains["physics"] = physics
        
        # War domain
        war = DomainKnowledge("war")
        war.add_concept("advance", {"direction": True, "momentum": True})
        war.add_concept("line_of_defense", {"barrier": True, "position": True})
        war.add_concept("attack", {"movement": True, "aggressive": True})
        war.add_concept("siege", {"stability": True, "duration": True})
        war.add_concept("chaos", {"variability": True, "unpredictable": True})
        self.domains["war"] = war
        
        # Biology domain
        biology = DomainKnowledge("biology")
        biology.add_concept("growth", {"direction": True, "momentum": True})
        biology.add_concept("homeostasis", {"stability": True, "balance": True})
        biology.add_concept("mutation", {"change": True, "sudden": True})
        biology.add_concept("adaptation", {"change": True, "gradual": True})
        self.domains["biology"] = biology
    
    def map_domains(
        self,
        source_domain: str,
        target_domain: str
    ) -> Optional[AnalogicalMapping]:
        """
        Create mapping between two domains.
        
        Uses Structure Mapping Theory principles:
        - One-to-one mapping
        - Parallel connectivity
        - Systematicity
        """
        if source_domain not in self.domains or target_domain not in self.domains:
            return None
        
        source = self.domains[source_domain]
        target = self.domains[target_domain]
        
        concept_mappings = {}
        
        # Map concepts with similar attributes
        for s_name, s_concept in source.concepts.items():
            best_match = None
            best_score = 0
            
            for t_name, t_concept in target.concepts.items():
                if t_name in concept_mappings.values():
                    continue
                
                # Calculate attribute overlap
                overlap = sum(
                    1 for k in s_concept.attributes
                    if k in t_concept.attributes and s_concept.attributes[k] == t_concept.attributes[k]
                )
                
                if overlap > best_score:
                    best_score = overlap
                    best_match = t_name
            
            if best_match and best_score > 0:
                concept_mappings[s_name] = best_match
        
        if not concept_mappings:
            return None
        
        self._mapping_counter += 1
        mapping = AnalogicalMapping(
            mapping_id=f"map_{self._mapping_counter}",
            source_domain=source_domain,
            target_domain=target_domain,
            concept_mappings=concept_mappings,
            relation_mappings={},
            confidence=len(concept_mappings) / max(len(source.concepts), 1)
        )
        
        self.mappings.append(mapping)
        logger.info(f"Created mapping {source_domain} -> {target_domain} with {len(concept_mappings)} concepts")
        return mapping
    
    def transfer_insight(
        self,
        source_insight: str,
        source_domain: str,
        target_domain: str
    ) -> Optional[str]:
        """Transfer an insight from source to target domain."""
        # Find or create mapping
        mapping = next(
            (m for m in self.mappings 
             if m.source_domain == source_domain and m.target_domain == target_domain),
            None
        )
        
        if not mapping:
            mapping = self.map_domains(source_domain, target_domain)
        
        if not mapping:
            return None
        
        # Apply concept mappings to insight
        result = source_insight
        for source_concept, target_concept in mapping.concept_mappings.items():
            result = result.replace(source_concept, target_concept)
        
        return result


class MetaphorSystem:
    """
    System for structural metaphors.
    
    Uses conceptual metaphors to understand abstract concepts.
    """
    
    def __init__(self):
        self.metaphors: Dict[str, StructuralMetaphor] = {}
        self._metaphor_counter = 0
        
        # Initialize common trading metaphors
        self._init_trading_metaphors()
        
        logger.info("MetaphorSystem initialized")
    
    def _init_trading_metaphors(self):
        """Initialize common metaphors for trading."""
        metaphors = [
            {
                'source': 'WAR',
                'target': 'TRADING',
                'mappings': {
                    'attack': 'aggressive_buy',
                    'defend': 'hedge',
                    'retreat': 'stop_loss',
                    'advance': 'trend_follow',
                    'siege': 'range_trade',
                    'victory': 'profit',
                    'defeat': 'loss',
                    'territory': 'market_share',
                    'enemy': 'counterparty',
                    'weapon': 'strategy'
                },
                'entailments': [
                    'Trading requires strategy like war',
                    'Traders compete for resources (profit)',
                    'Defense (risk management) is as important as offense'
                ]
            },
            {
                'source': 'JOURNEY',
                'target': 'TRADING',
                'mappings': {
                    'path': 'trend',
                    'obstacle': 'resistance',
                    'destination': 'target_price',
                    'speed': 'momentum',
                    'detour': 'retracement',
                    'rest_stop': 'consolidation',
                    'compass': 'indicator'
                },
                'entailments': [
                    'Price has a direction',
                    'There are obstacles to overcome',
                    'Progress can be measured'
                ]
            },
            {
                'source': 'ORGANISM',
                'target': 'MARKET',
                'mappings': {
                    'health': 'market_strength',
                    'illness': 'crash',
                    'growth': 'bull_market',
                    'decay': 'bear_market',
                    'heartbeat': 'volatility',
                    'lifespan': 'cycle'
                },
                'entailments': [
                    'Markets have cycles like organisms',
                    'Markets can be healthy or sick',
                    'Markets grow and decay'
                ]
            }
        ]
        
        for m in metaphors:
            self._metaphor_counter += 1
            metaphor = StructuralMetaphor(
                metaphor_id=f"meta_{self._metaphor_counter}",
                source=m['source'],
                target=m['target'],
                mappings=m['mappings'],
                entailments=m['entailments']
            )
            self.metaphors[metaphor.metaphor_id] = metaphor
    
    def apply_metaphor(
        self,
        metaphor_id: str,
        source_concept: str
    ) -> Optional[str]:
        """Apply a metaphor to translate a concept."""
        if metaphor_id not in self.metaphors:
            return None
        
        metaphor = self.metaphors[metaphor_id]
        return metaphor.mappings.get(source_concept)
    
    def understand_via_metaphor(
        self,
        target_concept: str
    ) -> List[Tuple[str, str, str]]:
        """
        Understand a target concept through metaphors.
        
        Returns list of (metaphor_name, source_concept, entailment)
        """
        understandings = []
        
        for metaphor in self.metaphors.values():
            if metaphor.target in target_concept.upper():
                # Find relevant mappings
                for source, target in metaphor.mappings.items():
                    if target.lower() in target_concept.lower():
                        for entailment in metaphor.entailments:
                            understandings.append((
                                metaphor.source,
                                source,
                                entailment
                            ))
        
        return understandings


class LateralThinking:
    """
    Lateral thinking for creative problem solving.
    
    Uses:
    - Random stimulation
    - Reversal
    - Analogy jumping
    - Provocation
    """
    
    def __init__(self):
        self.ideas: List[LateralIdea] = []
        self._idea_counter = 0
        
        # Random word bank for stimulation
        self.random_words = [
            'water', 'tree', 'music', 'dance', 'river', 'mountain',
            'cloud', 'fire', 'ice', 'bridge', 'tunnel', 'mirror',
            'echo', 'shadow', 'light', 'wave', 'particle', 'network'
        ]
        
        logger.info("LateralThinking initialized")
    
    def random_stimulation(self, problem: str) -> LateralIdea:
        """Use random word to stimulate new connections."""
        import random
        
        random_word = random.choice(self.random_words)
        
        # Create connection
        if random_word == 'water':
            jump = "Like water finding path of least resistance"
            idea = "Trade in direction of easiest movement"
        elif random_word == 'tree':
            jump = "Like tree with roots and branches"
            idea = "Build trading on solid foundation with multiple strategies"
        elif random_word == 'music':
            jump = "Like music with rhythm and patterns"
            idea = "Find the rhythm/cycle of the market"
        elif random_word == 'bridge':
            jump = "Like bridge connecting two sides"
            idea = "Connect different timeframes for confirmation"
        else:
            jump = f"Random connection to '{random_word}'"
            idea = f"Apply properties of {random_word} to {problem}"
        
        self._idea_counter += 1
        lateral_idea = LateralIdea(
            idea_id=f"lat_{self._idea_counter}",
            trigger=random_word,
            lateral_jump=jump,
            resulting_idea=idea,
            applicability=0.5
        )
        
        self.ideas.append(lateral_idea)
        return lateral_idea
    
    def reversal(self, problem: str) -> LateralIdea:
        """Reverse the problem to find new perspectives."""
        # Create reversal
        if 'maximize' in problem.lower():
            reversed_problem = problem.replace('maximize', 'minimize')
            idea = "Instead of maximizing profit, minimize losses first"
        elif 'increase' in problem.lower():
            reversed_problem = problem.replace('increase', 'decrease')
            idea = "Focus on what to reduce rather than increase"
        elif 'buy' in problem.lower():
            reversed_problem = problem.replace('buy', 'sell')
            idea = "Consider selling perspective"
        else:
            reversed_problem = f"Opposite of: {problem}"
            idea = "Consider the opposite approach"
        
        self._idea_counter += 1
        lateral_idea = LateralIdea(
            idea_id=f"rev_{self._idea_counter}",
            trigger=problem,
            lateral_jump=f"Reverse: {reversed_problem}",
            resulting_idea=idea,
            applicability=0.6
        )
        
        self.ideas.append(lateral_idea)
        return lateral_idea
    
    def analogy_jump(
        self,
        problem: str,
        analogical_mapper: AnalogicalMapper
    ) -> LateralIdea:
        """Jump to different domain for insights."""
        import random
        
        # Pick random domain
        domains = list(analogical_mapper.domains.keys())
        domains.remove('trading')  # Don't jump to same domain
        
        if not domains:
            return self.random_stimulation(problem)
        
        target_domain = random.choice(domains)
        
        # Get insight from that domain
        target = analogical_mapper.domains[target_domain]
        concepts = list(target.concepts.keys())
        
        if concepts:
            concept = random.choice(concepts)
            jump = f"Jump to {target_domain} domain: {concept}"
            idea = f"Apply {concept} principle from {target_domain} to trading"
        else:
            jump = f"Jump to {target_domain}"
            idea = "Think about problem in different domain"
        
        self._idea_counter += 1
        lateral_idea = LateralIdea(
            idea_id=f"ajump_{self._idea_counter}",
            trigger=target_domain,
            lateral_jump=jump,
            resulting_idea=idea,
            applicability=0.5
        )
        
        self.ideas.append(lateral_idea)
        return lateral_idea


class AnalogyEngine:
    """
    Main Analogy Engine.
    
    Combines:
    - AnalogicalMapper: Cross-domain mappings
    - MetaphorSystem: Structural metaphors
    - LateralThinking: Creative problem solving
    """
    
    def __init__(self):
        self.mapper = AnalogicalMapper()
        self.metaphors = MetaphorSystem()
        self.lateral = LateralThinking()
        
        logger.info("AnalogyEngine initialized")
    
    def think_creatively(self, problem: str) -> Dict[str, Any]:
        """
        Apply multiple creative thinking techniques.
        
        Returns insights from different approaches.
        """
        insights = {
            'analogies': [],
            'metaphors': [],
            'lateral_ideas': []
        }
        
        # Try analogical mappings
        for source in ['physics', 'war', 'biology']:
            mapping = self.mapper.map_domains(source, 'trading')
            if mapping:
                transferred = self.mapper.transfer_insight(problem, 'trading', source)
                if transferred:
                    insights['analogies'].append({
                        'source': source,
                        'transferred_insight': transferred
                    })
        
        # Try metaphors
        for concept in problem.lower().split():
            understandings = self.metaphors.understand_via_metaphor(concept)
            for u in understandings:
                insights['metaphors'].append({
                    'metaphor': u[0],
                    'source_concept': u[1],
                    'entailment': u[2]
                })
        
        # Generate lateral ideas
        insights['lateral_ideas'].append(self.lateral.random_stimulation(problem).__dict__)
        insights['lateral_ideas'].append(self.lateral.reversal(problem).__dict__)
        insights['lateral_ideas'].append(self.lateral.analogy_jump(problem, self.mapper).__dict__)
        
        return insights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'domains': len(self.mapper.domains),
            'mappings': len(self.mapper.mappings),
            'metaphors': len(self.metaphors.metaphors),
            'lateral_ideas': len(self.lateral.ideas)
        }
