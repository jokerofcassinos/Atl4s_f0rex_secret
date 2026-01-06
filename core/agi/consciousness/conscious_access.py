"""
AGI Fase 2: Conscious Access Layer

Global Workspace for conscious information processing:
- Global Workspace of Consciousness
- Attentional Selectivity
- Temporal Bindings
- Stream of Consciousness
"""

import logging
import time
import threading
import heapq
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("ConsciousAccess")


class AttentionPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class ConsciousItem:
    """An item in conscious awareness."""
    priority: int
    timestamp: float = field(compare=False)
    source: str = field(compare=False)
    content: Dict[str, Any] = field(compare=False)
    item_type: str = field(compare=False)
    salience: float = field(compare=False, default=0.5)  # 0-1
    duration_ms: float = field(compare=False, default=0.0)
    
    def __post_init__(self):
        self.item_id = f"{self.source}:{int(self.timestamp * 1000)}"


@dataclass
class TemporalBinding:
    """Binding of information from different modules at a moment."""
    binding_id: str
    timestamp: float
    items: List[ConsciousItem]
    context: Dict[str, Any]
    coherence: float  # 0-1


@dataclass
class ConsciousMoment:
    """A single moment in the stream of consciousness."""
    timestamp: float
    focus: Optional[ConsciousItem]
    background: List[ConsciousItem]
    bindings: List[TemporalBinding]
    meta_state: Dict[str, Any]


class GlobalWorkspace:
    """
    Global Workspace of Consciousness.
    
    Based on Global Workspace Theory:
    - Limited capacity focus of attention
    - Broadcasting to all modules
    - Competition for access
    """
    
    def __init__(self, workspace_capacity: int = 7):
        self.capacity = workspace_capacity
        
        # Current conscious contents
        self._workspace: List[ConsciousItem] = []
        self._lock = threading.Lock()
        
        # Competition queue
        self._competition_queue: List[ConsciousItem] = []
        
        # Broadcast subscribers
        self._subscribers: List[Callable[[ConsciousItem], None]] = []
        
        # History
        self._access_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.items_processed = 0
        self.broadcasts = 0
        
        logger.info(f"GlobalWorkspace initialized (capacity={workspace_capacity})")
    
    def compete_for_access(
        self,
        item: ConsciousItem,
        force: bool = False
    ) -> bool:
        """
        Submit item to compete for conscious access.
        
        Returns True if item gains access.
        """
        with self._lock:
            # Check if already at capacity
            if len(self._workspace) < self.capacity or force:
                self._workspace.append(item)
                self._broadcast(item)
                self.items_processed += 1
                return True
            
            # Competition - compare with lowest salience item
            if not self._workspace:
                return False
            
            lowest = min(self._workspace, key=lambda i: i.salience)
            
            if item.salience > lowest.salience:
                self._workspace.remove(lowest)
                self._workspace.append(item)
                self._broadcast(item)
                self.items_processed += 1
                return True
            
            # Add to queue for later
            heapq.heappush(self._competition_queue, item)
            return False
    
    def _broadcast(self, item: ConsciousItem):
        """Broadcast item to all subscribers."""
        for callback in self._subscribers:
            try:
                callback(item)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
        
        self.broadcasts += 1
        self._access_history.append({
            'item_id': item.item_id,
            'timestamp': time.time(),
            'source': item.source
        })
    
    def subscribe(self, callback: Callable[[ConsciousItem], None]):
        """Subscribe to conscious broadcasts."""
        self._subscribers.append(callback)
    
    def get_current_focus(self) -> List[ConsciousItem]:
        """Get current conscious contents."""
        with self._lock:
            return list(self._workspace)
    
    def decay_old_items(self, max_age_seconds: float = 5.0):
        """Remove items that have been in workspace too long."""
        now = time.time()
        with self._lock:
            self._workspace = [
                item for item in self._workspace
                if now - item.timestamp < max_age_seconds
            ]
            
            # Promote from queue if space
            while len(self._workspace) < self.capacity and self._competition_queue:
                item = heapq.heappop(self._competition_queue)
                self._workspace.append(item)
                self._broadcast(item)
    
    def clear(self):
        """Clear workspace."""
        with self._lock:
            self._workspace.clear()
            self._competition_queue.clear()


class AttentionalSelectivity:
    """
    Decides what deserves conscious attention.
    
    Implements:
    - Bottom-up (salience-driven) attention
    - Top-down (goal-driven) attention
    - Attention filters
    """
    
    def __init__(self):
        # Top-down goals
        self.current_goals: List[str] = []
        self.goal_keywords: Dict[str, List[str]] = {}
        
        # Attention filters
        self.filters: Dict[str, Callable[[Dict], bool]] = {}
        
        # Salience modifiers
        self.source_weights: Dict[str, float] = {}
        self.type_weights: Dict[str, float] = {}
        
        # Statistics
        self.items_filtered = 0
        self.items_passed = 0
        
        logger.info("AttentionalSelectivity initialized")
    
    def set_goal(self, goal: str, keywords: Optional[List[str]] = None):
        """Set current attention goal."""
        self.current_goals.append(goal)
        if keywords:
            self.goal_keywords[goal] = keywords
    
    def clear_goals(self):
        """Clear all goals."""
        self.current_goals.clear()
        self.goal_keywords.clear()
    
    def add_filter(self, name: str, filter_fn: Callable[[Dict], bool]):
        """Add attention filter."""
        self.filters[name] = filter_fn
    
    def set_source_weight(self, source: str, weight: float):
        """Set weight for a source (affects salience)."""
        self.source_weights[source] = weight
    
    def calculate_salience(
        self,
        source: str,
        content: Dict[str, Any],
        base_priority: AttentionPriority
    ) -> float:
        """
        Calculate salience score for an item.
        
        Combines bottom-up and top-down factors.
        """
        # Base salience from priority
        priority_salience = 1.0 - (base_priority.value / 4.0)
        
        # Source weight
        source_weight = self.source_weights.get(source, 1.0)
        
        # Goal relevance (top-down)
        goal_boost = 0.0
        content_str = str(content).lower()
        
        for goal, keywords in self.goal_keywords.items():
            matches = sum(1 for kw in keywords if kw.lower() in content_str)
            if matches > 0:
                goal_boost += 0.1 * matches
        
        # Combine
        salience = (priority_salience * 0.5 + source_weight * 0.3 + goal_boost * 0.2)
        return min(1.0, max(0.0, salience))
    
    def should_attend(self, content: Dict[str, Any]) -> bool:
        """Check if content passes all filters."""
        for name, filter_fn in self.filters.items():
            try:
                if not filter_fn(content):
                    self.items_filtered += 1
                    return False
            except Exception as e:
                logger.error(f"Filter {name} error: {e}")
        
        self.items_passed += 1
        return True


class TemporalBinder:
    """
    Binds information from different modules at specific moments.
    
    Creates unified conscious experiences from distributed processing.
    """
    
    def __init__(self, binding_window_ms: float = 100.0):
        self.binding_window = binding_window_ms / 1000.0
        
        # Pending items for binding
        self._pending: List[ConsciousItem] = []
        self._lock = threading.Lock()
        
        # Active bindings
        self._bindings: deque = deque(maxlen=100)
        
        # Binding counter
        self._binding_count = 0
        
        logger.info(f"TemporalBinder initialized (window={binding_window_ms}ms)")
    
    def add_item(self, item: ConsciousItem):
        """Add item for potential binding."""
        now = time.time()
        
        with self._lock:
            self._pending.append(item)
            
            # Remove old items
            self._pending = [
                i for i in self._pending
                if now - i.timestamp < self.binding_window
            ]
    
    def create_binding(self, context: Optional[Dict] = None) -> Optional[TemporalBinding]:
        """Create a binding from pending items."""
        with self._lock:
            if len(self._pending) < 2:
                return None
            
            # Group items within window
            now = time.time()
            to_bind = [i for i in self._pending if now - i.timestamp < self.binding_window]
            
            if len(to_bind) < 2:
                return None
            
            self._binding_count += 1
            binding_id = f"bind:{self._binding_count}"
            
            # Calculate coherence
            sources = set(i.source for i in to_bind)
            coherence = 1.0 / len(sources) if sources else 0  # Lower with more sources
            
            binding = TemporalBinding(
                binding_id=binding_id,
                timestamp=now,
                items=to_bind,
                context=context or {},
                coherence=coherence
            )
            
            self._bindings.append(binding)
            
            # Clear pending
            self._pending = [i for i in self._pending if i not in to_bind]
            
            return binding
    
    def get_recent_bindings(self, n: int = 10) -> List[TemporalBinding]:
        """Get recent bindings."""
        return list(self._bindings)[-n:]


class ConsciousnessStream:
    """
    Stream of conscious moments over time.
    
    Tracks the flow of consciousness:
    - What is in focus now?
    - What was in focus before?
    - How does consciousness evolve?
    """
    
    def __init__(self, max_moments: int = 1000):
        self.max_moments = max_moments
        
        # Stream of moments
        self._stream: deque = deque(maxlen=max_moments)
        
        # Current moment
        self._current: Optional[ConsciousMoment] = None
        
        # Moment sampling rate
        self._sample_interval = 0.1  # 100ms
        self._last_sample = 0.0
        
        logger.info("ConsciousnessStream initialized")
    
    def sample_moment(
        self,
        focus: Optional[ConsciousItem],
        background: List[ConsciousItem],
        bindings: List[TemporalBinding],
        meta_state: Optional[Dict] = None
    ) -> ConsciousMoment:
        """Sample and record a moment of consciousness."""
        now = time.time()
        
        moment = ConsciousMoment(
            timestamp=now,
            focus=focus,
            background=background,
            bindings=bindings,
            meta_state=meta_state or {}
        )
        
        self._stream.append(moment)
        self._current = moment
        self._last_sample = now
        
        return moment
    
    def get_current(self) -> Optional[ConsciousMoment]:
        """Get current moment."""
        return self._current
    
    def get_recent_stream(self, n: int = 20) -> List[ConsciousMoment]:
        """Get recent moments."""
        return list(self._stream)[-n:]
    
    def get_focus_history(self) -> List[str]:
        """Get history of what was in focus."""
        return [
            m.focus.item_id if m.focus else "empty"
            for m in self._stream
        ]
    
    def analyze_stream(self) -> Dict[str, Any]:
        """Analyze the stream of consciousness."""
        if not self._stream:
            return {'error': 'No stream data'}
        
        # Focus duration analysis
        focus_counts: Dict[str, int] = {}
        for moment in self._stream:
            if moment.focus:
                source = moment.focus.source
                focus_counts[source] = focus_counts.get(source, 0) + 1
        
        total = len(self._stream)
        focus_distribution = {k: v/total for k, v in focus_counts.items()}
        
        # Binding frequency
        total_bindings = sum(len(m.bindings) for m in self._stream)
        
        return {
            'total_moments': total,
            'focus_distribution': focus_distribution,
            'total_bindings': total_bindings,
            'avg_bindings_per_moment': total_bindings / total if total > 0 else 0
        }


class ConsciousAccessLayer:
    """
    Main Conscious Access Layer.
    
    Integrates:
    - GlobalWorkspace
    - AttentionalSelectivity
    - TemporalBinder
    - ConsciousnessStream
    """
    
    def __init__(self):
        self.workspace = GlobalWorkspace(workspace_capacity=7)
        self.attention = AttentionalSelectivity()
        self.binder = TemporalBinder(binding_window_ms=100)
        self.stream = ConsciousnessStream()
        
        # Background processing thread
        self._running = True
        self._processor = threading.Thread(target=self._process_loop, daemon=True)
        self._processor.start()
        
        logger.info("ConsciousAccessLayer initialized")
    
    def present(
        self,
        source: str,
        content: Dict[str, Any],
        item_type: str = "info",
        priority: AttentionPriority = AttentionPriority.NORMAL,
        force: bool = False
    ) -> bool:
        """
        Present information for conscious access.
        
        Returns True if information gained conscious access.
        """
        # Check attention filter
        if not self.attention.should_attend(content):
            return False
        
        # Calculate salience
        salience = self.attention.calculate_salience(source, content, priority)
        
        # Create conscious item
        item = ConsciousItem(
            priority=priority.value,
            timestamp=time.time(),
            source=source,
            content=content,
            item_type=item_type,
            salience=salience
        )
        
        # Compete for workspace access
        success = self.workspace.compete_for_access(item, force)
        
        # Add to binder regardless
        self.binder.add_item(item)
        
        return success
    
    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            # Decay old items
            self.workspace.decay_old_items()
            
            # Create bindings
            binding = self.binder.create_binding()
            
            # Sample moment
            focus_items = self.workspace.get_current_focus()
            focus = focus_items[0] if focus_items else None
            background = focus_items[1:] if len(focus_items) > 1 else []
            bindings = self.binder.get_recent_bindings(3)
            
            self.stream.sample_moment(focus, background, bindings)
            
            time.sleep(0.1)
    
    def get_current_consciousness(self) -> Dict[str, Any]:
        """Get current state of consciousness."""
        focus = self.workspace.get_current_focus()
        stream_analysis = self.stream.analyze_stream()
        
        return {
            'focus_count': len(focus),
            'focus_items': [f.item_id for f in focus],
            'stream_analysis': stream_analysis
        }
    
    def shutdown(self):
        """Shutdown the conscious access layer."""
        self._running = False
        if self._processor.is_alive():
            self._processor.join(timeout=1.0)
        logger.info("ConsciousAccessLayer shutdown")
