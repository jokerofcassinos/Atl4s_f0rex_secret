"""
AGI Ultra: Consciousness Bus

Inter-module communication system for AGI components:
- Parallel thought streams
- Intelligent prioritization
- Thought fusion
- Temporal coherence
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import heapq

logger = logging.getLogger("ConsciousnessBus")


class ThoughtPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass(order=True)
class Thought:
    """A thought message on the consciousness bus."""
    priority: int
    timestamp: float = field(compare=False)
    source_module: str = field(compare=False)
    content: Dict[str, Any] = field(compare=False)
    thought_type: str = field(compare=False, default="insight")
    context: Dict[str, Any] = field(compare=False, default_factory=dict)
    related_thoughts: List[str] = field(compare=False, default_factory=list)
    
    def __post_init__(self):
        self.thought_id = f"{self.source_module}:{int(self.timestamp * 1000)}"


@dataclass
class ThoughtStream:
    """A parallel stream of related thoughts."""
    stream_id: str
    topic: str
    thoughts: List[Thought] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    def add_thought(self, thought: Thought):
        self.thoughts.append(thought)
        self.last_active = time.time()


class ConsciousnessBus:
    """
    AGI Ultra: Consciousness Bus
    
    Central hub for inter-module communication.
    
    Features:
    - Priority queue for thought processing
    - Parallel thought streams
    - Thought fusion for combining insights
    - Temporal coherence tracking
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        stream_timeout: float = 300.0,
        fusion_threshold: float = 0.7
    ):
        self.max_queue_size = max_queue_size
        self.stream_timeout = stream_timeout
        self.fusion_threshold = fusion_threshold
        
        # Priority queue for thoughts
        self._thought_queue: List[Thought] = []
        self._queue_lock = threading.Lock()
        
        # Parallel streams
        self._streams: Dict[str, ThoughtStream] = {}
        self._stream_lock = threading.Lock()
        
        # Subscribers
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Temporal coherence tracking
        self._recent_thoughts: List[Thought] = []
        self._coherence_window = 100
        
        # Statistics
        self.thoughts_published = 0
        self.thoughts_processed = 0
        self.fusions_performed = 0
        
        # Background processing
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processor_thread.start()
        
        logger.info("ConsciousnessBus initialized")
    
    # -------------------------------------------------------------------------
    # PUBLISHING
    # -------------------------------------------------------------------------
    def publish(
        self,
        source_module: str,
        content: Dict[str, Any],
        thought_type: str = "insight",
        priority: ThoughtPriority = ThoughtPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
        stream_topic: Optional[str] = None
    ) -> str:
        """
        Publish a thought to the consciousness bus.
        
        Args:
            source_module: Module publishing the thought
            content: Thought content
            thought_type: Type of thought (insight, decision, question, etc.)
            priority: Priority level
            context: Additional context
            stream_topic: Optional topic to add to a stream
            
        Returns:
            Thought ID
        """
        thought = Thought(
            priority=priority.value,
            timestamp=time.time(),
            source_module=source_module,
            content=content,
            thought_type=thought_type,
            context=context or {}
        )
        
        # Add to queue
        with self._queue_lock:
            if len(self._thought_queue) < self.max_queue_size:
                heapq.heappush(self._thought_queue, thought)
                self.thoughts_published += 1
            else:
                logger.warning("Thought queue full, dropping thought")
        
        # Add to stream if specified
        if stream_topic:
            self._add_to_stream(stream_topic, thought)
        
        # Track for coherence
        self._track_coherence(thought)
        
        return thought.thought_id
    
    def _add_to_stream(self, topic: str, thought: Thought):
        """Add thought to a parallel stream."""
        with self._stream_lock:
            if topic not in self._streams:
                self._streams[topic] = ThoughtStream(
                    stream_id=f"stream:{topic}:{int(time.time())}",
                    topic=topic
                )
            self._streams[topic].add_thought(thought)
    
    def _track_coherence(self, thought: Thought):
        """Track thought for temporal coherence."""
        self._recent_thoughts.append(thought)
        if len(self._recent_thoughts) > self._coherence_window:
            self._recent_thoughts.pop(0)
    
    # -------------------------------------------------------------------------
    # SUBSCRIPTION
    # -------------------------------------------------------------------------
    def subscribe(
        self,
        thought_type: str,
        callback: Callable[[Thought], None]
    ):
        """
        Subscribe to thoughts of a specific type.
        
        Args:
            thought_type: Type to subscribe to ("*" for all)
            callback: Function to call when thought is published
        """
        self._subscribers[thought_type].append(callback)
        logger.debug(f"Subscription added for type: {thought_type}")
    
    def unsubscribe(self, thought_type: str, callback: Callable):
        """Unsubscribe from thought type."""
        if callback in self._subscribers[thought_type]:
            self._subscribers[thought_type].remove(callback)
    
    # -------------------------------------------------------------------------
    # PROCESSING
    # -------------------------------------------------------------------------
    def _process_loop(self):
        """Background thread for processing thoughts."""
        while self._running:
            thought = self._get_next_thought()
            if thought:
                self._dispatch_thought(thought)
                self.thoughts_processed += 1
            else:
                time.sleep(0.01)  # Avoid busy waiting
    
    def _get_next_thought(self) -> Optional[Thought]:
        """Get next thought from priority queue."""
        with self._queue_lock:
            if self._thought_queue:
                return heapq.heappop(self._thought_queue)
        return None
    
    def _dispatch_thought(self, thought: Thought):
        """Dispatch thought to subscribers."""
        # Dispatch to type-specific subscribers
        for callback in self._subscribers.get(thought.thought_type, []):
            try:
                callback(thought)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
        
        # Dispatch to wildcard subscribers
        for callback in self._subscribers.get("*", []):
            try:
                callback(thought)
            except Exception as e:
                logger.error(f"Wildcard subscriber error: {e}")
    
    # -------------------------------------------------------------------------
    # THOUGHT FUSION
    # -------------------------------------------------------------------------
    def fuse_thoughts(
        self,
        thoughts: List[Thought],
        fusion_strategy: str = "weighted_merge"
    ) -> Dict[str, Any]:
        """
        Fuse multiple thoughts into a combined insight.
        
        Args:
            thoughts: Thoughts to fuse
            fusion_strategy: How to combine ("weighted_merge", "consensus", "conflict_resolution")
            
        Returns:
            Fused insight
        """
        if not thoughts:
            return {}
        
        self.fusions_performed += 1
        
        if fusion_strategy == "weighted_merge":
            return self._weighted_merge(thoughts)
        elif fusion_strategy == "consensus":
            return self._consensus_fusion(thoughts)
        elif fusion_strategy == "conflict_resolution":
            return self._conflict_resolution_fusion(thoughts)
        else:
            return self._weighted_merge(thoughts)
    
    def _weighted_merge(self, thoughts: List[Thought]) -> Dict[str, Any]:
        """Merge thoughts with priority weighting."""
        weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
        
        merged = {
            'fused': True,
            'source_count': len(thoughts),
            'sources': [t.source_module for t in thoughts],
            'contents': []
        }
        
        for thought in thoughts:
            weight = weights.get(thought.priority, 0.5)
            merged['contents'].append({
                'source': thought.source_module,
                'content': thought.content,
                'weight': weight
            })
        
        return merged
    
    def _consensus_fusion(self, thoughts: List[Thought]) -> Dict[str, Any]:
        """Fuse thoughts by finding consensus."""
        # Count decisions
        decisions = defaultdict(int)
        for thought in thoughts:
            decision = thought.content.get('decision', 'UNKNOWN')
            decisions[decision] += 1
        
        # Find majority
        if decisions:
            majority = max(decisions.items(), key=lambda x: x[1])
            agreement = majority[1] / len(thoughts)
        else:
            majority = ("UNKNOWN", 0)
            agreement = 0
        
        return {
            'fused': True,
            'consensus': majority[0],
            'agreement': agreement,
            'votes': dict(decisions)
        }
    
    def _conflict_resolution_fusion(self, thoughts: List[Thought]) -> Dict[str, Any]:
        """Resolve conflicts between thoughts."""
        # Group by decision
        by_decision = defaultdict(list)
        for thought in thoughts:
            decision = thought.content.get('decision', 'UNKNOWN')
            by_decision[decision].append(thought)
        
        # If no conflict
        if len(by_decision) <= 1:
            return self._weighted_merge(thoughts)
        
        # Resolve by priority and confidence
        best_decision = None
        best_score = -1
        
        for decision, thought_list in by_decision.items():
            # Score = sum of (1/priority) * confidence
            score = sum(
                (1 / t.priority) * t.content.get('confidence', 0.5)
                for t in thought_list
            )
            if score > best_score:
                best_score = score
                best_decision = decision
        
        return {
            'fused': True,
            'resolved_decision': best_decision,
            'conflict_count': len(by_decision),
            'resolution_score': best_score,
            'alternatives': list(by_decision.keys())
        }
    
    # -------------------------------------------------------------------------
    # STREAMS
    # -------------------------------------------------------------------------
    def get_stream(self, topic: str) -> Optional[ThoughtStream]:
        """Get a thought stream by topic."""
        with self._stream_lock:
            return self._streams.get(topic)
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream topics."""
        with self._stream_lock:
            now = time.time()
            return [
                topic for topic, stream in self._streams.items()
                if now - stream.last_active < self.stream_timeout
            ]
    
    def cleanup_stale_streams(self):
        """Remove stale streams."""
        with self._stream_lock:
            now = time.time()
            stale = [
                topic for topic, stream in self._streams.items()
                if now - stream.last_active > self.stream_timeout
            ]
            for topic in stale:
                del self._streams[topic]
            return len(stale)
    
    # -------------------------------------------------------------------------
    # COHERENCE
    # -------------------------------------------------------------------------
    def get_temporal_coherence(self) -> float:
        """
        Calculate temporal coherence of recent thoughts.
        
        High coherence = thoughts are consistent over time
        Low coherence = thoughts are contradictory/erratic
        """
        if len(self._recent_thoughts) < 2:
            return 1.0
        
        # Check for contradictions
        decisions = []
        for thought in self._recent_thoughts:
            if 'decision' in thought.content:
                decisions.append(thought.content['decision'])
        
        if not decisions:
            return 1.0
        
        # Count changes
        changes = sum(1 for i in range(1, len(decisions)) if decisions[i] != decisions[i-1])
        change_rate = changes / len(decisions)
        
        # Coherence = 1 - change_rate
        return max(0.0, 1.0 - change_rate)
    
    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Get bus statistics."""
        return {
            'thoughts_published': self.thoughts_published,
            'thoughts_processed': self.thoughts_processed,
            'fusions_performed': self.fusions_performed,
            'queue_size': len(self._thought_queue),
            'stream_count': len(self._streams),
            'subscriber_count': sum(len(s) for s in self._subscribers.values()),
            'temporal_coherence': self.get_temporal_coherence()
        }
    
    def shutdown(self):
        """Shutdown the consciousness bus."""
        self._running = False
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=1.0)
        logger.info("ConsciousnessBus shutdown")


# Global instance
_global_bus: Optional[ConsciousnessBus] = None

def get_consciousness_bus() -> ConsciousnessBus:
    """Get global consciousness bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = ConsciousnessBus()
    return _global_bus
