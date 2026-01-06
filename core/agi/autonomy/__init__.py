"""
AGI Fase 2: Autonomy Package

Módulos de Autonomia e Auto-Evolução:
- SelfModificationSystem: Auto-modificação segura
- AutonomousEvolutionEngine: Evolução autônoma
- ContinuousLearningSystem: Aprendizado contínuo
"""

from .self_modification import SelfModificationSystem, SandboxEnvironment, CodeModifier
from .autonomous_evolution import AutonomousEvolutionEngine, ArchitectureEvolver, OpenEndedEvolution
from .continuous_learning import ContinuousLearningSystem, UnsupervisedLearner, ActiveLearner

__all__ = [
    'SelfModificationSystem',
    'SandboxEnvironment',
    'CodeModifier',
    'AutonomousEvolutionEngine',
    'ArchitectureEvolver',
    'OpenEndedEvolution',
    'ContinuousLearningSystem',
    'UnsupervisedLearner',
    'ActiveLearner',
]
