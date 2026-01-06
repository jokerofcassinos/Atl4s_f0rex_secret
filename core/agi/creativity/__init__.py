"""
AGI Fase 2: Creativity Package

Módulos de Criatividade e Inovação:
- StrategyGenerator: Geração criativa de estratégias
- InnovationEngine: Detecção de limitações e inovação
- AnalogyEngine: Analogias e pensamento lateral
"""

from .strategy_generator import StrategyGenerator, ConceptLibrary, Strategy
from .innovation_engine import InnovationEngine, LimitationDetector, ViabilityTester
from .analogy_engine import AnalogyEngine, AnalogicalMapper, MetaphorSystem

__all__ = [
    'StrategyGenerator',
    'ConceptLibrary',
    'Strategy',
    'InnovationEngine',
    'LimitationDetector',
    'ViabilityTester',
    'AnalogyEngine',
    'AnalogicalMapper',
    'MetaphorSystem',
]
