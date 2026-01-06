"""
AGI Ultra: Memory Integration Layer

Coordinates memory access across all AGI modules:
- Cross-module memory queries
- Memory consolidation
- Correlation analysis between memories
- Global learning integration
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("MemoryIntegration")


@dataclass
class MemoryQuery:
    """A query to the memory system."""
    query_id: str
    module_name: str
    query_type: str  # similar, temporal, categorical, cross
    context: Dict[str, Any]
    vector: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryResult:
    """Result from memory query."""
    memories: List[Dict[str, Any]]
    confidence: float
    source_modules: List[str]
    correlation_score: float = 0.0
    processing_time_ms: float = 0.0


class MemoryIntegrationLayer:
    """
    AGI Ultra: Memory Integration Layer.
    
    Provides unified access to memories across all modules:
    - HolographicMemory
    - PatternLibrary
    - DecisionMemory
    - ThoughtTree archives
    
    Features:
    - Cross-module queries
    - Memory correlation analysis
    - Consolidated learning
    - Memory health monitoring
    """
    
    def __init__(self):
        # Memory sources (lazy loaded)
        self._holographic = None
        self._patterns = None
        self._decision = None
        self._thoughts = None
        
        # Cross-module correlations
        self.correlations: Dict[Tuple[str, str], float] = {}
        
        # Query history for learning
        self.query_history: List[MemoryQuery] = []
        self.max_history = 1000
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        
        # Simple cache
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 60.0  # 60 seconds
        
        logger.info("MemoryIntegrationLayer initialized")
    
    # -------------------------------------------------------------------------
    # LAZY LOADING
    # -------------------------------------------------------------------------
    @property
    def holographic(self):
        if self._holographic is None:
            try:
                from core.memory.holographic import HolographicMemory
                self._holographic = HolographicMemory()
            except ImportError:
                logger.warning("HolographicMemory not available")
        return self._holographic
    
    @property
    def patterns(self):
        if self._patterns is None:
            try:
                from core.agi.pattern_library import PatternLibrary
                self._patterns = PatternLibrary()
            except ImportError:
                logger.warning("PatternLibrary not available")
        return self._patterns
    
    @property
    def decision(self):
        if self._decision is None:
            try:
                from core.memory.decision import GlobalDecisionMemory
                self._decision = GlobalDecisionMemory()
            except ImportError:
                logger.warning("DecisionMemory not available")
        return self._decision
    
    @property
    def thoughts(self):
        if self._thoughts is None:
            try:
                from core.agi.thought_tree import GlobalThoughtOrchestrator
                self._thoughts = GlobalThoughtOrchestrator()
            except ImportError:
                logger.warning("ThoughtOrchestrator not available")
        return self._thoughts
    
    # -------------------------------------------------------------------------
    # UNIFIED QUERIES
    # -------------------------------------------------------------------------
    def query(
        self,
        module_name: str,
        context: Dict[str, Any],
        query_type: str = "cross",
        top_k: int = 10
    ) -> MemoryResult:
        """
        Query memories across all sources.
        
        Args:
            module_name: Requesting module
            context: Query context
            query_type: Type of query (similar, temporal, categorical, cross)
            top_k: Number of results
            
        Returns:
            MemoryResult with consolidated memories
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Check cache
        cache_key = f"{module_name}:{query_type}:{hash(str(context))}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self.cache_hits += 1
                return cached_result
        
        results = []
        source_modules = []
        
        # Query holographic memory
        if self.holographic and query_type in ['similar', 'cross']:
            try:
                intuition = self.holographic.intuit(context)
                if isinstance(intuition, dict):
                    results.append({
                        'source': 'holographic',
                        'type': 'intuition',
                        'score': intuition.get('combined_score', 0.5),
                        'data': intuition
                    })
                    source_modules.append('holographic')
            except Exception as e:
                logger.debug(f"Holographic query error: {e}")
        
        # Query pattern library
        if self.patterns and query_type in ['similar', 'categorical', 'cross']:
            try:
                # Build vector from context
                vector = self._context_to_vector(context)
                category = context.get('category', 'general')
                
                similar = self.patterns.search_similar(vector, category=category, top_k=top_k)
                for pattern in similar:
                    results.append({
                        'source': 'patterns',
                        'type': 'pattern',
                        'score': getattr(pattern, 'success_rate', 0.5),
                        'data': pattern
                    })
                if similar:
                    source_modules.append('patterns')
            except Exception as e:
                logger.debug(f"Pattern query error: {e}")
        
        # Query decision memory
        if self.decision and query_type in ['temporal', 'cross']:
            try:
                module_mem = self.decision.get_or_create_memory(module_name)
                recent = module_mem.get_recent_decisions(limit=top_k)
                for dec in recent:
                    results.append({
                        'source': 'decision',
                        'type': 'decision',
                        'score': dec.get('confidence', 0.5),
                        'data': dec
                    })
                if recent:
                    source_modules.append('decision')
            except Exception as e:
                logger.debug(f"Decision query error: {e}")
        
        # Calculate overall confidence
        if results:
            avg_score = np.mean([r['score'] for r in results])
        else:
            avg_score = 0.0
        
        # Calculate correlation score
        correlation = self._compute_correlation(results)
        
        result = MemoryResult(
            memories=results[:top_k],
            confidence=float(avg_score),
            source_modules=source_modules,
            correlation_score=correlation,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Cache result
        self._cache[cache_key] = (time.time(), result)
        
        return result
    
    def _context_to_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context to vector for similarity search."""
        vector = np.zeros(256)
        idx = 0
        
        for key, value in context.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if idx < 256:
                    vector[idx] = np.tanh(value * 0.1)
                    idx += 1
        
        return vector
    
    def _compute_correlation(self, results: List[Dict]) -> float:
        """Compute correlation between memory results."""
        if len(results) < 2:
            return 0.0
        
        scores = [r['score'] for r in results]
        
        # Check agreement between sources
        sources = set(r['source'] for r in results)
        if len(sources) >= 2:
            # Multiple sources agreeing = higher correlation
            score_std = np.std(scores)
            agreement = 1.0 - min(1.0, score_std)
            return agreement
        
        return 0.5
    
    # -------------------------------------------------------------------------
    # CROSS-MODULE LEARNING
    # -------------------------------------------------------------------------
    def propagate_learning(
        self,
        module_name: str,
        context: Dict[str, Any],
        outcome: float,
        decision: str
    ):
        """
        Propagate learning outcome to all memory systems.
        
        Args:
            module_name: Module that made the decision
            context: Decision context
            outcome: Outcome score (-1 to 1)
            decision: The decision made
        """
        # Learn in holographic memory
        if self.holographic:
            try:
                self.holographic.learn(context, outcome)
            except Exception as e:
                logger.debug(f"Holographic learning error: {e}")
        
        # Update pattern statistics
        if self.patterns:
            try:
                # Find and update matching patterns
                vector = self._context_to_vector(context)
                similar = self.patterns.search_similar(vector, top_k=3)
                for pattern in similar:
                    if hasattr(pattern, 'pattern_id'):
                        self.patterns.update_outcome(pattern.pattern_id, outcome)
            except Exception as e:
                logger.debug(f"Pattern update error: {e}")
        
        logger.debug(f"Learning propagated from {module_name}: outcome={outcome}")
    
    # -------------------------------------------------------------------------
    # MEMORY CONSOLIDATION
    # -------------------------------------------------------------------------
    def consolidate(self):
        """Consolidate and compress memories across all systems."""
        if self.holographic:
            try:
                self.holographic.consolidate_temporal_memory()
            except Exception as e:
                logger.debug(f"Holographic consolidation error: {e}")
        
        if self.patterns:
            try:
                for category in ['trend', 'reversal', 'breakout', 'range']:
                    self.patterns._compact_category(category)
            except Exception as e:
                logger.debug(f"Pattern compaction error: {e}")
        
        # Clear old cache entries
        now = time.time()
        self._cache = {
            k: v for k, v in self._cache.items()
            if now - v[0] < self._cache_ttl * 2
        }
        
        logger.info("Memory consolidation completed")
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory integration statistics."""
        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.total_queries if self.total_queries > 0 else 0,
            'cache_size': len(self._cache),
            'available_sources': {
                'holographic': self._holographic is not None,
                'patterns': self._patterns is not None,
                'decision': self._decision is not None,
                'thoughts': self._thoughts is not None
            }
        }
