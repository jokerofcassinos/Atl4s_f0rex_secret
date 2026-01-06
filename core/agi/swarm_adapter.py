"""
AGI Ultra: Swarm AGI Adapter Base

Base class that adds AGI capabilities to any Swarm module:
- ThoughtTree integration for recursive reasoning
- Meta-reasoning through InfiniteWhyEngine
- Pattern matching via PatternLibrary
- Holographic memory intuition
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger("AGISwarmAdapter")


@dataclass
class AGIAnalysis:
    """Result of AGI-enhanced analysis."""
    decision: str  # BUY, SELL, WAIT
    confidence: float  # 0.0 to 1.0
    
    # Reasoning chain
    reasoning: str
    thought_path: List[str] = field(default_factory=list)
    why_chain: List[str] = field(default_factory=list)
    
    # Evidence
    supporting_patterns: List[str] = field(default_factory=list)
    memory_intuition: float = 0.0
    
    # Meta-cognition
    meta_confidence: float = 0.0
    uncertainty_sources: List[str] = field(default_factory=list)
    
    # Timing
    processing_time_ms: float = 0.0


class AGISwarmAdapter(ABC):
    """
    AGI Ultra: Base class for AGI-enhanced Swarms.
    
    Any swarm can inherit from this to gain:
    - Recursive reasoning through InfiniteWhyEngine
    - Memory integration through HolographicMemory
    - Pattern matching through PatternLibrary
    - ThoughtTree tracking
    - Meta-reasoning capabilities
    """
    
    def __init__(
        self,
        swarm_name: str,
        enable_meta_reasoning: bool = True,
        enable_memory: bool = True,
        enable_patterns: bool = True
    ):
        self.swarm_name = swarm_name
        self.enable_meta_reasoning = enable_meta_reasoning
        self.enable_memory = enable_memory
        self.enable_patterns = enable_patterns
        
        # Lazy loading of AGI components
        self._why_engine = None
        self._holographic_memory = None
        self._pattern_library = None
        self._thought_tree = None
        
        # Statistics
        self.analysis_count = 0
        self.total_processing_time = 0.0
        
        logger.info(f"AGISwarmAdapter initialized for: {swarm_name}")
    
    # -------------------------------------------------------------------------
    # LAZY LOADING
    # -------------------------------------------------------------------------
    @property
    def why_engine(self):
        """Lazy load InfiniteWhyEngine."""
        if self._why_engine is None:
            try:
                from core.agi.infinite_why_engine import InfiniteWhyEngine
                self._why_engine = InfiniteWhyEngine()
            except ImportError:
                logger.warning(f"InfiniteWhyEngine not available for {self.swarm_name}")
        return self._why_engine
    
    @property
    def holographic_memory(self):
        """Lazy load HolographicMemory."""
        if self._holographic_memory is None:
            try:
                from core.memory.holographic import HolographicMemory
                self._holographic_memory = HolographicMemory()
            except ImportError:
                logger.warning(f"HolographicMemory not available for {self.swarm_name}")
        return self._holographic_memory
    
    @property
    def pattern_library(self):
        """Lazy load PatternLibrary."""
        if self._pattern_library is None:
            try:
                from core.agi.pattern_library import PatternLibrary
                self._pattern_library = PatternLibrary()
            except ImportError:
                logger.warning(f"PatternLibrary not available for {self.swarm_name}")
        return self._pattern_library
    
    @property
    def thought_tree(self):
        """Lazy load or get ThoughtTree for this swarm."""
        if self._thought_tree is None:
            try:
                from core.agi.thought_tree import GlobalThoughtOrchestrator
                orchestrator = GlobalThoughtOrchestrator()
                self._thought_tree = orchestrator.get_or_create_tree(self.swarm_name)
            except ImportError:
                logger.warning(f"ThoughtTree not available for {self.swarm_name}")
        return self._thought_tree
    
    # -------------------------------------------------------------------------
    # ABSTRACT METHODS (To be implemented by specific swarms)
    # -------------------------------------------------------------------------
    @abstractmethod
    def _core_analysis(
        self,
        symbol: str,
        timeframe: str,
        df,
        context: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """
        Core swarm analysis logic.
        
        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        pass
    
    # -------------------------------------------------------------------------
    # AGI-ENHANCED ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(
        self,
        symbol: str,
        timeframe: str,
        df,
        context: Optional[Dict[str, Any]] = None
    ) -> AGIAnalysis:
        """
        Perform AGI-enhanced analysis.
        
        This method:
        1. Runs the core swarm analysis
        2. Enhances with memory intuition
        3. Matches against known patterns
        4. Runs meta-reasoning
        5. Records in thought tree
        
        Returns:
            AGIAnalysis with full reasoning chain
        """
        start_time = time.time()
        context = context or {}
        
        # 1. Core analysis
        decision, confidence, reasoning = self._core_analysis(symbol, timeframe, df, context)
        
        # Initialize result
        result = AGIAnalysis(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # 2. Memory intuition
        if self.enable_memory and self.holographic_memory:
            memory_intuition = self._get_memory_intuition(symbol, timeframe, df, context)
            result.memory_intuition = memory_intuition
            
            # Adjust confidence based on memory
            if memory_intuition > 0.5:
                result.confidence = min(1.0, result.confidence * (1 + 0.2 * memory_intuition))
            else:
                result.confidence = result.confidence * (0.8 + 0.4 * memory_intuition)
        
        # 3. Pattern matching
        if self.enable_patterns and self.pattern_library:
            patterns = self._match_patterns(symbol, timeframe, df, context)
            result.supporting_patterns = patterns
            
            # Boost confidence if strong pattern matches
            if len(patterns) >= 2:
                result.confidence = min(1.0, result.confidence * 1.1)
        
        # 4. Meta-reasoning
        if self.enable_meta_reasoning and self.why_engine:
            meta_result = self._run_meta_reasoning(symbol, timeframe, decision, context)
            result.meta_confidence = meta_result.get('meta_confidence', 0.5)
            result.why_chain = meta_result.get('why_chain', [])
            result.uncertainty_sources = meta_result.get('uncertainty', [])
            
            # Adjust confidence based on meta-reasoning
            result.confidence = (result.confidence + result.meta_confidence) / 2
        
        # 5. Record in thought tree
        if self.thought_tree:
            self._record_thought(symbol, timeframe, result)
        
        # Finalize
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.analysis_count += 1
        self.total_processing_time += result.processing_time_ms
        
        return result
    
    def _get_memory_intuition(
        self,
        symbol: str,
        timeframe: str,
        df,
        context: Dict[str, Any]
    ) -> float:
        """Get intuition from holographic memory."""
        try:
            # Build state vector
            state = {
                'symbol': symbol,
                'timeframe': timeframe,
                **context
            }
            
            # Add price features if available
            if df is not None and len(df) > 0:
                close = df['close'].values
                state['price_change'] = (close[-1] / close[-min(10, len(close))] - 1) * 100
                state['volatility'] = np.std(close[-20:]) / np.mean(close[-20:]) if len(close) >= 20 else 0
            
            intuition = self.holographic_memory.intuit(state)
            
            # Extract score from intuition result
            if isinstance(intuition, dict):
                return intuition.get('combined_score', intuition.get('main_score', 0.5))
            elif isinstance(intuition, (int, float)):
                return float(intuition)
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Memory intuition error: {e}")
            return 0.5
    
    def _match_patterns(
        self,
        symbol: str,
        timeframe: str,
        df,
        context: Dict[str, Any]
    ) -> List[str]:
        """Match current situation against known patterns."""
        try:
            # Build pattern vector from current data
            if df is None or len(df) < 10:
                return []
            
            close = df['close'].values[-50:]
            
            # Simple pattern encoding
            pattern_vector = np.zeros(256)
            
            # Price momentum
            if len(close) >= 20:
                pattern_vector[0] = (close[-1] / close[-5] - 1) * 100
                pattern_vector[1] = (close[-1] / close[-10] - 1) * 100
                pattern_vector[2] = (close[-1] / close[-20] - 1) * 100
            
            # Volatility
            if len(close) >= 20:
                pattern_vector[10] = np.std(close[-10:]) / np.mean(close[-10:]) * 100
                pattern_vector[11] = np.std(close[-20:]) / np.mean(close[-20:]) * 100
            
            # Search for similar patterns
            similar = self.pattern_library.search_similar(
                pattern_vector,
                category='trend',
                top_k=5
            )
            
            return [p.pattern_id for p in similar if hasattr(p, 'pattern_id')]
            
        except Exception as e:
            logger.debug(f"Pattern matching error: {e}")
            return []
    
    def _run_meta_reasoning(
        self,
        symbol: str,
        timeframe: str,
        decision: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run meta-reasoning on the decision."""
        try:
            # Create event for why engine
            event = self.why_engine.capture_event(
                symbol=symbol,
                timeframe=timeframe,
                market_state=context.get('market_state', {}),
                analysis_state=context.get('analysis_state', {}),
                decision=decision,
                decision_score=context.get('score', 50.0),
                decision_meta={'swarm': self.swarm_name},
                module_name=self.swarm_name
            )
            
            # Run deep scan
            result = self.why_engine.deep_scan_recursive(
                module_name=self.swarm_name,
                query_event=event,
                max_depth=8  # Limited depth for swarm analysis
            )
            
            # Extract meta-confidence
            meta_conf = result.get('combined_intuition', 0.5)
            if isinstance(meta_conf, dict):
                meta_conf = meta_conf.get('combined_score', 0.5)
            
            return {
                'meta_confidence': float(meta_conf) if not np.isnan(meta_conf) else 0.5,
                'why_chain': result.get('reasoning_chain', [])[:5],
                'uncertainty': result.get('uncertainty_sources', [])
            }
            
        except Exception as e:
            logger.debug(f"Meta-reasoning error: {e}")
            return {'meta_confidence': 0.5, 'why_chain': [], 'uncertainty': []}
    
    def _record_thought(self, symbol: str, timeframe: str, result: AGIAnalysis):
        """Record analysis in thought tree."""
        try:
            question = f"What decision for {symbol} {timeframe}?"
            self.thought_tree.create_node(
                question=question,
                context={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'decision': result.decision,
                    'reasoning': result.reasoning
                },
                confidence=result.confidence
            )
        except Exception as e:
            logger.debug(f"Thought recording error: {e}")
    
    # -------------------------------------------------------------------------
    # FEEDBACK
    # -------------------------------------------------------------------------
    def register_outcome(self, symbol: str, actual_outcome: float, prediction: str):
        """
        Register the actual outcome for learning.
        
        Args:
            symbol: Trading symbol
            actual_outcome: Actual price movement
            prediction: The prediction made (BUY/SELL)
        """
        was_correct = (
            (prediction == 'BUY' and actual_outcome > 0) or
            (prediction == 'SELL' and actual_outcome < 0)
        )
        
        if self.holographic_memory:
            try:
                state = {'symbol': symbol, 'prediction': prediction}
                outcome_score = 1.0 if was_correct else -1.0
                self.holographic_memory.learn(state, outcome_score)
            except Exception as e:
                logger.debug(f"Memory learning error: {e}")
        
        logger.debug(f"{self.swarm_name} outcome: {'correct' if was_correct else 'wrong'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics."""
        avg_time = self.total_processing_time / self.analysis_count if self.analysis_count > 0 else 0
        
        return {
            'swarm_name': self.swarm_name,
            'analysis_count': self.analysis_count,
            'avg_processing_time_ms': avg_time,
            'agi_enabled': {
                'meta_reasoning': self.enable_meta_reasoning,
                'memory': self.enable_memory,
                'patterns': self.enable_patterns
            }
        }


class SimpleAGISwarm(AGISwarmAdapter):
    """
    A simple AGI-enabled swarm for demonstration.
    Can be used as template for converting existing swarms.
    """
    
    def __init__(self, swarm_name: str = "SimpleAGISwarm"):
        super().__init__(swarm_name)
    
    def _core_analysis(
        self,
        symbol: str,
        timeframe: str,
        df,
        context: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """Simple trend-following analysis."""
        if df is None or len(df) < 20:
            return 'WAIT', 0.3, "Insufficient data"
        
        close = df['close'].values
        
        # Simple momentum check
        short_ma = np.mean(close[-5:])
        long_ma = np.mean(close[-20:])
        
        if short_ma > long_ma * 1.001:
            return 'BUY', 0.6, f"Short MA ({short_ma:.2f}) > Long MA ({long_ma:.2f})"
        elif short_ma < long_ma * 0.999:
            return 'SELL', 0.6, f"Short MA ({short_ma:.2f}) < Long MA ({long_ma:.2f})"
        else:
            return 'WAIT', 0.4, "No clear trend"
