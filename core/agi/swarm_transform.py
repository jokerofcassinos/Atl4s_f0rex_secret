"""
AGI Ultra: Swarm Transformation Template

Provides templates and utilities for quickly transforming
existing swarm modules to use AGI capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

# AGI Imports
from core.agi.thought_tree import ThoughtTree
from core.agi.decision_memory import ModuleDecisionMemory
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("SwarmTransform")


class AGIEnabledSwarm(SubconsciousUnit, ABC):
    """
    Base class for AGI-enabled swarms.
    
    Provides standard AGI integration:
    - ThoughtTree for recursive thinking
    - DecisionMemory for learning
    - AGISwarmAdapter for deep scanning
    
    Subclasses only need to implement:
    - _analyze(): Core analysis logic
    - get_description(): Swarm description
    """
    
    def __init__(self, swarm_name: str, max_thought_depth: int = 5, max_memory: int = 500):
        super().__init__(swarm_name)
        
        # AGI Components
        self.thought_tree = ThoughtTree(swarm_name, max_depth=max_thought_depth)
        self.decision_memory = ModuleDecisionMemory(swarm_name, max_memory=max_memory)
        self.agi_adapter = AGISwarmAdapter(swarm_name)
        
        # Statistics
        self.analysis_count = 0
        self.positive_signals = 0
        self.agi_thoughts = 0
        
        logger.info(f"AGIEnabledSwarm initialized: {swarm_name}")
    
    @abstractmethod
    def _analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core analysis logic. Override this in subclasses.
        
        Args:
            context: Analysis context with market data
            
        Returns:
            Dict with: decision, score, reasoning
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return description of what this swarm analyzes."""
        pass
    
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        """
        Main processing entry point.
        Handles AGI integration automatically.
        """
        self.analysis_count += 1
        
        # 1. Run core analysis
        result = self._analyze(context)
        
        if result is None:
            return None
        
        decision = result.get('decision', 'WAIT')
        score = result.get('score', 0.0)
        reasoning = result.get('reasoning', '')
        
        # 2. Record in thought tree
        root_id = self.thought_tree.create_node(
            question=f"Why did I decide {decision}?",
            context={'result': result, 'market_context': self._extract_context(context)},
            confidence=abs(score) / 100.0 if score != 0 else 0.0
        )
        
        self.thought_tree.answer_node(root_id, reasoning, confidence=abs(score) / 100.0)
        
        # 3. Check similar past decisions
        similar = self.decision_memory.find_similar_decisions(
            {'decision': decision, 'score': score},
            limit=5
        )
        
        if similar:
            success_rate = len([d for d in similar if d.result == "WIN"]) / len(similar)
            
            # Create child thought about past performance
            child_id = self.thought_tree.create_node(
                question="How did similar decisions perform?",
                parent_id=root_id,
                context={'similar_count': len(similar)}
            )
            self.thought_tree.answer_node(
                child_id, 
                f"Similar decisions had {success_rate:.1%} success rate",
                confidence=success_rate
            )
        
        # 4. Record decision
        decision_id = self.decision_memory.record_decision(
            decision=decision,
            score=score,
            context=self._extract_context(context),
            reasoning=reasoning,
            confidence=abs(score) / 100.0
        )
        
        # 5. Run AGI deep scan for non-WAIT decisions
        if decision != "WAIT":
            try:
                swarm_thought = self.agi_adapter.think_on_swarm_output(
                    symbol=context.get('symbol', 'UNKNOWN'),
                    timeframe=context.get('timeframe', 'M5'),
                    market_state=self._extract_context(context),
                    swarm_output={
                        'decision': decision,
                        'score': score,
                        'reasoning': reasoning
                    }
                )
                self.agi_thoughts += 1
                
                return SwarmSignal(
                    source=self.name,
                    signal_type=decision,
                    confidence=abs(score),
                    timestamp=0,
                    meta_data={
                        'reasoning': reasoning,
                        'thought_nodes': len(self.thought_tree.nodes),
                        'agi_thought_id': swarm_thought.thought_root_id,
                        'agi_scenarios': swarm_thought.meta.get('scenario_count', 0)
                    }
                )
            except Exception as e:
                logger.error(f"AGI adapter error: {e}")
        
        # Return signal if not WAIT
        if decision != "WAIT" and abs(score) > 30:
            self.positive_signals += 1
            return SwarmSignal(
                source=self.name,
                signal_type=decision,
                confidence=abs(score),
                timestamp=0,
                meta_data={'reasoning': reasoning}
            )
        
        return None
    
    def _extract_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context for memory."""
        df = context.get('df_m5')
        
        extracted = {
            'symbol': context.get('symbol', 'UNKNOWN'),
            'timeframe': context.get('timeframe', 'M5'),
            'regime': context.get('market_state', {}).get('regime', 'unknown')
        }
        
        if df is not None and len(df) > 0:
            extracted['price'] = float(df['close'].iloc[-1])
            extracted['volume'] = float(df['volume'].iloc[-1]) if 'volume' in df else 0
        
        return extracted
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics."""
        return {
            'name': self.name,
            'analysis_count': self.analysis_count,
            'positive_signals': self.positive_signals,
            'agi_thoughts': self.agi_thoughts,
            'thought_tree_nodes': len(self.thought_tree.nodes),
            'decision_memory_size': self.decision_memory.total_decisions
        }


# =============================================================================
# EXAMPLE IMPLEMENTATIONS
# =============================================================================

class ExampleTrendSwarm(AGIEnabledSwarm):
    """Example: Trend-following swarm with AGI."""
    
    def __init__(self):
        super().__init__("ExampleTrendSwarm")
    
    def get_description(self) -> str:
        return "Analyzes trend direction and strength"
    
    def _analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get('df_m5')
        
        if df is None or len(df) < 20:
            return {'decision': 'WAIT', 'score': 0, 'reasoning': 'Insufficient data'}
        
        # Simple trend analysis
        close = df['close'].values
        sma_fast = close[-5:].mean()
        sma_slow = close[-20:].mean()
        
        diff = (sma_fast - sma_slow) / sma_slow * 100
        
        if diff > 0.1:
            return {
                'decision': 'BUY',
                'score': min(100, diff * 100),
                'reasoning': f'Uptrend: Fast MA above Slow by {diff:.2%}'
            }
        elif diff < -0.1:
            return {
                'decision': 'SELL',
                'score': -min(100, abs(diff) * 100),
                'reasoning': f'Downtrend: Fast MA below Slow by {abs(diff):.2%}'
            }
        
        return {'decision': 'WAIT', 'score': 0, 'reasoning': 'No clear trend'}


class ExampleMomentumSwarm(AGIEnabledSwarm):
    """Example: Momentum-based swarm with AGI."""
    
    def __init__(self):
        super().__init__("ExampleMomentumSwarm")
    
    def get_description(self) -> str:
        return "Analyzes price momentum and acceleration"
    
    def _analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get('df_m5')
        
        if df is None or len(df) < 10:
            return {'decision': 'WAIT', 'score': 0, 'reasoning': 'Insufficient data'}
        
        # Simple momentum
        close = df['close'].values
        momentum = (close[-1] - close[-5]) / close[-5] * 100
        
        if momentum > 0.2:
            return {
                'decision': 'BUY',
                'score': min(100, momentum * 50),
                'reasoning': f'Strong bullish momentum: {momentum:.2%}'
            }
        elif momentum < -0.2:
            return {
                'decision': 'SELL',
                'score': -min(100, abs(momentum) * 50),
                'reasoning': f'Strong bearish momentum: {momentum:.2%}'
            }
        
        return {'decision': 'WAIT', 'score': 0, 'reasoning': 'Weak momentum'}


# =============================================================================
# TRANSFORMATION UTILITY
# =============================================================================

def transform_swarm_to_agi(
    original_class: type,
    swarm_name: str,
    analysis_method: str = 'analyze'
) -> type:
    """
    Factory function to quickly transform an existing swarm to AGI-enabled.
    
    Usage:
        AGIMySwarm = transform_swarm_to_agi(MySwarm, "MySwarm", "analyze")
        swarm = AGIMySwarm()
    """
    
    class AGITransformedSwarm(AGIEnabledSwarm):
        def __init__(self):
            super().__init__(swarm_name)
            self._original = original_class()
        
        def get_description(self) -> str:
            return f"AGI-transformed {swarm_name}"
        
        def _analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
            # Call original analysis method
            original_result = getattr(self._original, analysis_method)(context)
            
            # Normalize result
            if isinstance(original_result, tuple):
                if len(original_result) == 2:
                    return {'decision': 'UNKNOWN', 'score': original_result[0], 'reasoning': str(original_result[1])}
                elif len(original_result) == 3:
                    return {'decision': str(original_result[2]), 'score': original_result[0], 'reasoning': str(original_result[1])}
            elif isinstance(original_result, dict):
                return {
                    'decision': original_result.get('decision', 'WAIT'),
                    'score': original_result.get('score', 0),
                    'reasoning': original_result.get('reason', str(original_result))
                }
            
            return {'decision': 'WAIT', 'score': 0, 'reasoning': 'Unknown result format'}
    
    AGITransformedSwarm.__name__ = f"AGI{original_class.__name__}"
    return AGITransformedSwarm
