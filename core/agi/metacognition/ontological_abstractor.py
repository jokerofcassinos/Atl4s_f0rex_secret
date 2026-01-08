"""
Ontological Abstractor - Abstract Market Concept Extraction.

Extracts ontological relationships and abstract concepts from
market data for higher-level reasoning.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("OntologicalAbstractor")


@dataclass
class MarketConcept:
    """An abstract market concept."""
    name: str
    category: str  # 'TREND', 'REVERSAL', 'CONSOLIDATION', 'REGIME'
    strength: float
    relationships: List[str]  # Related concepts
    temporal_scope: str  # 'MICRO', 'MESO', 'MACRO'


@dataclass
class OntologyReading:
    """Ontological analysis result."""
    active_concepts: List[MarketConcept]
    dominant_narrative: str
    concept_hierarchy: Dict[str, List[str]]
    abstraction_level: int  # 1-5
    coherence_score: float


class OntologicalAbstractor:
    """
    The Concept Weaver.
    
    Extracts abstract concepts through:
    - Pattern categorization into ontological types
    - Relationship mapping between concepts
    - Hierarchical abstraction building
    - Narrative synthesis
    """
    
    def __init__(self):
        self.concept_library = {
            'MOMENTUM': MarketConcept('MOMENTUM', 'TREND', 0.0, ['TREND', 'DIRECTION'], 'MESO'),
            'EXHAUSTION': MarketConcept('EXHAUSTION', 'REVERSAL', 0.0, ['REVERSAL', 'VOLUME'], 'MICRO'),
            'ACCUMULATION': MarketConcept('ACCUMULATION', 'CONSOLIDATION', 0.0, ['VOLUME', 'RANGE'], 'MACRO'),
            'DISTRIBUTION': MarketConcept('DISTRIBUTION', 'CONSOLIDATION', 0.0, ['VOLUME', 'TOP'], 'MACRO'),
            'BREAKOUT': MarketConcept('BREAKOUT', 'TREND', 0.0, ['MOMENTUM', 'VOLUME'], 'MESO'),
            'REGIME_SHIFT': MarketConcept('REGIME_SHIFT', 'REGIME', 0.0, ['VOLATILITY', 'TREND'], 'MACRO'),
        }
        
        self.hierarchy = {
            'MACRO': ['REGIME', 'ACCUMULATION', 'DISTRIBUTION'],
            'MESO': ['TREND', 'BREAKOUT', 'MOMENTUM'],
            'MICRO': ['EXHAUSTION', 'REVERSAL', 'SPIKE']
        }
        
        logger.info("OntologicalAbstractor initialized")
    
    def abstract(self, market_data: Dict) -> OntologyReading:
        """Extract abstract concepts from market data."""
        # Identify active concepts
        active = self._identify_active_concepts(market_data)
        
        # Determine dominant narrative
        narrative = self._synthesize_narrative(active)
        
        # Calculate coherence
        coherence = self._calculate_coherence(active)
        
        # Determine abstraction level
        level = self._determine_abstraction_level(active)
        
        return OntologyReading(
            active_concepts=active,
            dominant_narrative=narrative,
            concept_hierarchy=self.hierarchy,
            abstraction_level=level,
            coherence_score=coherence
        )
    
    def _identify_active_concepts(self, data: Dict) -> List[MarketConcept]:
        """Identify which concepts are currently active."""
        active = []
        
        # Momentum detection
        trend_strength = data.get('trend_strength', 0)
        if abs(trend_strength) > 0.3:
            concept = MarketConcept('MOMENTUM', 'TREND', abs(trend_strength), ['TREND'], 'MESO')
            active.append(concept)
        
        # Exhaustion detection
        if data.get('volume_declining', False) and data.get('price_stalling', False):
            concept = MarketConcept('EXHAUSTION', 'REVERSAL', 0.7, ['VOLUME'], 'MICRO')
            active.append(concept)
        
        # Breakout detection
        if data.get('breaking_range', False):
            concept = MarketConcept('BREAKOUT', 'TREND', 0.8, ['RANGE', 'VOLUME'], 'MESO')
            active.append(concept)
        
        # Regime shift detection
        if data.get('volatility_change', 0) > 0.5:
            concept = MarketConcept('REGIME_SHIFT', 'REGIME', 0.6, ['VOLATILITY'], 'MACRO')
            active.append(concept)
        
        return active
    
    def _synthesize_narrative(self, concepts: List[MarketConcept]) -> str:
        """Synthesize a narrative from active concepts."""
        if not concepts:
            return "Market in undefined state"
        
        dominant = max(concepts, key=lambda c: c.strength)
        
        narratives = {
            'MOMENTUM': f"Strong {dominant.category.lower()} momentum playing out",
            'EXHAUSTION': "Trend exhaustion suggesting reversal potential",
            'BREAKOUT': "Breakout in progress with momentum confirmation",
            'REGIME_SHIFT': "Regime change detected - volatility transition",
            'ACCUMULATION': "Smart money accumulating at these levels",
            'DISTRIBUTION': "Distribution phase - potential top forming"
        }
        
        return narratives.get(dominant.name, f"Active: {dominant.name}")
    
    def _calculate_coherence(self, concepts: List[MarketConcept]) -> float:
        """Calculate how coherent the active concepts are."""
        if len(concepts) <= 1:
            return 1.0
        
        # Check for conflicting concepts
        categories = [c.category for c in concepts]
        if 'TREND' in categories and 'REVERSAL' in categories:
            return 0.4  # Conflicting narratives
        
        return 0.8
    
    def _determine_abstraction_level(self, concepts: List[MarketConcept]) -> int:
        """Determine appropriate abstraction level."""
        if not concepts:
            return 3
        
        scopes = [c.temporal_scope for c in concepts]
        if 'MACRO' in scopes:
            return 5
        elif 'MESO' in scopes:
            return 3
        return 1
