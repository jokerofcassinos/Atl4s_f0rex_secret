"""
AGI Ultra: Module Initialization

Exports all AGI Ultra components for easy importing.
"""

from .thought_tree import (
    ThoughtNode,
    ThoughtTree,
    GlobalThoughtOrchestrator,
    ThoughtGraph,
    ConfidencePropagator,
    TreeCompressor
)

from .infinite_why_engine import (
    InfiniteWhyEngine,
    MemoryEvent,
    WhyNode,
    ScenarioBranch
)

from .pattern_library import (
    PatternLibrary,
    Pattern,
    PatternCluster
)

from .meta_learning import (
    MetaLearningEngine,
    LearningEpisode,
    TransferTask,
    CurriculumStage
)

from .unified_reasoning import (
    UnifiedReasoningLayer,
    ModuleInsight,
    UnifiedDecision,
    ConflictType
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    HealthAlert,
    AlertType,
    ComponentHealth
)

from .swarm_adapter import (
    AGISwarmAdapter,
    AGIAnalysis,
    SimpleAGISwarm
)

from .memory_integration import (
    MemoryIntegrationLayer,
    MemoryQuery,
    MemoryResult
)


__all__ = [
    # ThoughtTree
    'ThoughtNode', 'ThoughtTree', 'GlobalThoughtOrchestrator',
    'ThoughtGraph', 'ConfidencePropagator', 'TreeCompressor',
    
    # InfiniteWhyEngine
    'InfiniteWhyEngine', 'MemoryEvent', 'WhyNode', 'ScenarioBranch',
    
    # PatternLibrary  
    'PatternLibrary', 'Pattern', 'PatternCluster',
    
    # MetaLearning
    'MetaLearningEngine', 'LearningEpisode', 'TransferTask', 'CurriculumStage',
    
    # UnifiedReasoning
    'UnifiedReasoningLayer', 'ModuleInsight', 'UnifiedDecision', 'ConflictType',
    
    # HealthMonitor
    'HealthMonitor', 'HealthStatus', 'HealthAlert', 'AlertType', 'ComponentHealth',
    
    # SwarmAdapter
    'AGISwarmAdapter', 'AGIAnalysis', 'SimpleAGISwarm',
    
    # MemoryIntegration
    'MemoryIntegrationLayer', 'MemoryQuery', 'MemoryResult'
]
