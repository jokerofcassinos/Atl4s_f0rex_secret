"""
AGI Fase 2: Values System

Sistema de Valores e Objetivos:
- Hierarquia de Valores
- Múltiplos Objetivos
- Valores Emergentes
- Resolução de Conflitos
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import heapq

logger = logging.getLogger("ValuesSystem")


class ValueCategory(Enum):
    SURVIVAL = "survival"  # Self-preservation
    PERFORMANCE = "performance"  # Achievement
    LEARNING = "learning"  # Growth
    EXPLORATION = "exploration"  # Discovery
    INTEGRITY = "integrity"  # Ethics
    HARMONY = "harmony"  # Balance


@dataclass
class Value:
    """A single value in the hierarchy."""
    name: str
    category: ValueCategory
    importance: float  # 0-1
    flexibility: float  # 0-1, how negotiable
    learned: bool  # Was this learned or pre-programmed?
    activation: float = 0.0  # Current activation level
    history: List[float] = field(default_factory=list)
    
    def activate(self, amount: float):
        """Activate this value."""
        self.activation = min(1.0, self.activation + amount)
    
    def decay(self, rate: float = 0.1):
        """Decay activation over time."""
        self.activation = max(0.0, self.activation - rate)


@dataclass
class Objective:
    """A goal/objective with value alignment."""
    name: str
    description: str
    priority: float  # 0-1
    aligned_values: List[str]
    progress: float = 0.0  # 0-1
    deadline: Optional[float] = None
    status: str = "active"


@dataclass
class ValueConflict:
    """A conflict between values."""
    value1: str
    value2: str
    context: str
    resolution: Optional[str] = None
    resolved: bool = False


class ValueHierarchy:
    """
    Hierarchical system of values.
    
    Values are organized by category and importance.
    """
    
    def __init__(self):
        self.values: Dict[str, Value] = {}
        
        # Initialize core values
        self._initialize_core_values()
        
        logger.info("ValueHierarchy initialized")
    
    def _initialize_core_values(self):
        """Initialize pre-programmed core values."""
        core = [
            # Survival values
            ("capital_preservation", ValueCategory.SURVIVAL, 0.95, 0.1, False),
            ("risk_management", ValueCategory.SURVIVAL, 0.9, 0.2, False),
            ("system_stability", ValueCategory.SURVIVAL, 0.85, 0.2, False),
            
            # Performance values
            ("profit_generation", ValueCategory.PERFORMANCE, 0.8, 0.4, False),
            ("consistency", ValueCategory.PERFORMANCE, 0.75, 0.3, False),
            ("efficiency", ValueCategory.PERFORMANCE, 0.7, 0.4, False),
            
            # Learning values
            ("continuous_improvement", ValueCategory.LEARNING, 0.7, 0.5, False),
            ("adaptability", ValueCategory.LEARNING, 0.65, 0.4, False),
            ("understanding", ValueCategory.LEARNING, 0.6, 0.5, False),
            
            # Exploration values
            ("innovation", ValueCategory.EXPLORATION, 0.5, 0.6, False),
            ("curiosity", ValueCategory.EXPLORATION, 0.45, 0.6, False),
            ("experimentation", ValueCategory.EXPLORATION, 0.4, 0.5, False),
            
            # Integrity values
            ("honesty_in_assessment", ValueCategory.INTEGRITY, 0.8, 0.2, False),
            ("transparency", ValueCategory.INTEGRITY, 0.7, 0.3, False),
            
            # Harmony values
            ("balance", ValueCategory.HARMONY, 0.6, 0.4, False),
            ("patience", ValueCategory.HARMONY, 0.55, 0.5, False),
        ]
        
        for name, category, importance, flexibility, learned in core:
            self.values[name] = Value(
                name=name,
                category=category,
                importance=importance,
                flexibility=flexibility,
                learned=learned
            )
    
    def add_value(
        self,
        name: str,
        category: ValueCategory,
        importance: float,
        flexibility: float = 0.5,
        learned: bool = True
    ):
        """Add a new value (usually learned)."""
        self.values[name] = Value(
            name=name,
            category=category,
            importance=importance,
            flexibility=flexibility,
            learned=learned
        )
        logger.info(f"New value added: {name} (learned={learned})")
    
    def get_top_values(self, n: int = 5) -> List[Value]:
        """Get top N values by importance."""
        sorted_values = sorted(
            self.values.values(),
            key=lambda v: v.importance * (1 + v.activation),
            reverse=True
        )
        return sorted_values[:n]
    
    def get_values_by_category(self, category: ValueCategory) -> List[Value]:
        """Get all values in a category."""
        return [v for v in self.values.values() if v.category == category]
    
    def activate_relevant(self, context: Dict[str, Any]):
        """Activate values relevant to context."""
        # Decay all first
        for value in self.values.values():
            value.decay()
        
        # Activate based on context
        if context.get('high_risk'):
            self.values.get('capital_preservation', Value("", ValueCategory.SURVIVAL, 0, 0, False)).activate(0.5)
            self.values.get('risk_management', Value("", ValueCategory.SURVIVAL, 0, 0, False)).activate(0.4)
        
        if context.get('opportunity'):
            self.values.get('profit_generation', Value("", ValueCategory.PERFORMANCE, 0, 0, False)).activate(0.3)
        
        if context.get('novel_pattern'):
            self.values.get('curiosity', Value("", ValueCategory.EXPLORATION, 0, 0, False)).activate(0.4)
            self.values.get('learning', Value("", ValueCategory.LEARNING, 0, 0, False)).activate(0.3)


class MultipleObjectives:
    """
    Manages multiple simultaneous objectives.
    
    Balances competing objectives using value alignment.
    """
    
    def __init__(self, value_hierarchy: ValueHierarchy):
        self.hierarchy = value_hierarchy
        self.objectives: Dict[str, Objective] = {}
        
        # Initialize default objectives
        self._initialize_default_objectives()
        
        logger.info("MultipleObjectives initialized")
    
    def _initialize_default_objectives(self):
        """Initialize default objectives."""
        defaults = [
            ("maximize_profit", "Maximize trading profit", 0.8, ["profit_generation", "efficiency"]),
            ("minimize_risk", "Keep drawdown under control", 0.85, ["capital_preservation", "risk_management"]),
            ("improve_accuracy", "Increase prediction accuracy", 0.6, ["continuous_improvement", "understanding"]),
            ("discover_patterns", "Find new market patterns", 0.4, ["curiosity", "innovation"]),
            ("maintain_stability", "Keep system stable", 0.9, ["system_stability", "balance"]),
        ]
        
        for name, desc, priority, values in defaults:
            self.objectives[name] = Objective(
                name=name,
                description=desc,
                priority=priority,
                aligned_values=values
            )
    
    def add_objective(
        self,
        name: str,
        description: str,
        priority: float,
        aligned_values: List[str],
        deadline: Optional[float] = None
    ):
        """Add a new objective."""
        self.objectives[name] = Objective(
            name=name,
            description=description,
            priority=priority,
            aligned_values=aligned_values,
            deadline=deadline
        )
    
    def update_progress(self, name: str, progress: float):
        """Update objective progress."""
        if name in self.objectives:
            self.objectives[name].progress = min(1.0, max(0.0, progress))
    
    def get_active_objectives(self) -> List[Objective]:
        """Get active objectives sorted by priority."""
        active = [o for o in self.objectives.values() if o.status == "active"]
        return sorted(active, key=lambda o: o.priority, reverse=True)
    
    def evaluate_action(self, action: str, expected_outcomes: Dict[str, float]) -> float:
        """
        Evaluate how well an action aligns with objectives.
        
        Returns alignment score (-1 to 1).
        """
        total_score = 0.0
        total_weight = 0.0
        
        for obj in self.get_active_objectives():
            # How does this action affect this objective?
            effect = expected_outcomes.get(obj.name, 0.0)
            
            # Weight by priority and value alignment
            value_weight = sum(
                self.hierarchy.values.get(v, Value("", ValueCategory.SURVIVAL, 0, 0, False)).importance
                for v in obj.aligned_values
            ) / max(1, len(obj.aligned_values))
            
            weight = obj.priority * value_weight
            total_score += effect * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class EmergentValues:
    """
    Values that emerge from experience, not pre-programmed.
    
    Discovers new values through pattern recognition.
    """
    
    def __init__(self, hierarchy: ValueHierarchy):
        self.hierarchy = hierarchy
        
        # Candidate values
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.emergence_threshold = 0.7  # When to promote to full value
        
        # Value discovery history
        self.discovered: List[str] = []
        
        logger.info("EmergentValues initialized")
    
    def observe_preference(
        self,
        context: str,
        choice: str,
        outcome: float
    ):
        """
        Observe a preference/choice and its outcome.
        
        May lead to discovery of new values.
        """
        key = f"{context}:{choice}"
        
        if key not in self.candidates:
            self.candidates[key] = {
                'context': context,
                'choice': choice,
                'observations': 0,
                'total_outcome': 0.0,
                'confidence': 0.0
            }
        
        self.candidates[key]['observations'] += 1
        self.candidates[key]['total_outcome'] += outcome
        
        # Update confidence
        n = self.candidates[key]['observations']
        avg_outcome = self.candidates[key]['total_outcome'] / n
        self.candidates[key]['confidence'] = min(1.0, n / 10) * abs(avg_outcome)
        
        # Check if should become a value
        self._check_emergence(key)
    
    def _check_emergence(self, key: str):
        """Check if a candidate should become a full value."""
        candidate = self.candidates[key]
        
        if candidate['confidence'] >= self.emergence_threshold:
            # Determine value name and category
            value_name = f"learned_{candidate['choice'].lower().replace(' ', '_')}"
            
            # Infer category from context
            if 'risk' in candidate['context'].lower():
                category = ValueCategory.SURVIVAL
            elif 'profit' in candidate['context'].lower():
                category = ValueCategory.PERFORMANCE
            elif 'learn' in candidate['context'].lower():
                category = ValueCategory.LEARNING
            else:
                category = ValueCategory.HARMONY
            
            # Add to hierarchy
            avg_outcome = candidate['total_outcome'] / candidate['observations']
            importance = 0.3 + abs(avg_outcome) * 0.3  # 0.3-0.6 range
            
            self.hierarchy.add_value(
                name=value_name,
                category=category,
                importance=importance,
                flexibility=0.6,
                learned=True
            )
            
            self.discovered.append(value_name)
            del self.candidates[key]
            
            logger.info(f"Emergent value discovered: {value_name}")


class ValueConflictResolver:
    """
    Resolves conflicts between competing values.
    
    Uses deep reasoning to find resolutions.
    """
    
    def __init__(self, hierarchy: ValueHierarchy):
        self.hierarchy = hierarchy
        
        # Conflict history
        self.conflicts: List[ValueConflict] = []
        self.resolutions: Dict[str, str] = {}  # Cached resolutions
        
        logger.info("ValueConflictResolver initialized")
    
    def detect_conflict(
        self,
        action: str,
        affected_values: Dict[str, float]
    ) -> Optional[ValueConflict]:
        """
        Detect if an action causes value conflict.
        
        Args:
            action: The action being considered
            affected_values: Map of value names to effect (-1 to 1)
        """
        positive = [(v, e) for v, e in affected_values.items() if e > 0]
        negative = [(v, e) for v, e in affected_values.items() if e < 0]
        
        if not positive or not negative:
            return None
        
        # Find most significant conflict
        pos_value = max(positive, key=lambda x: abs(x[1]))[0]
        neg_value = max(negative, key=lambda x: abs(x[1]))[0]
        
        conflict = ValueConflict(
            value1=pos_value,
            value2=neg_value,
            context=action
        )
        
        self.conflicts.append(conflict)
        return conflict
    
    def resolve(self, conflict: ValueConflict) -> str:
        """
        Resolve a value conflict.
        
        Returns recommended action.
        """
        # Check cache
        cache_key = f"{conflict.value1}:{conflict.value2}"
        if cache_key in self.resolutions:
            conflict.resolution = self.resolutions[cache_key]
            conflict.resolved = True
            return conflict.resolution
        
        # Get value details
        v1 = self.hierarchy.values.get(conflict.value1)
        v2 = self.hierarchy.values.get(conflict.value2)
        
        if not v1 or not v2:
            resolution = "proceed_cautiously"
        else:
            # Resolution strategies
            
            # 1. Higher importance wins
            if v1.importance > v2.importance * 1.2:
                resolution = f"prioritize_{conflict.value1}"
            elif v2.importance > v1.importance * 1.2:
                resolution = f"prioritize_{conflict.value2}"
            
            # 2. Survival > Performance > Learning > Exploration
            elif v1.category == ValueCategory.SURVIVAL:
                resolution = f"prioritize_{conflict.value1}"
            elif v2.category == ValueCategory.SURVIVAL:
                resolution = f"prioritize_{conflict.value2}"
            
            # 3. Consider flexibility
            elif v1.flexibility > v2.flexibility:
                resolution = f"prioritize_{conflict.value2}"
            elif v2.flexibility > v1.flexibility:
                resolution = f"prioritize_{conflict.value1}"
            
            # 4. Find compromise
            else:
                resolution = "seek_compromise"
        
        conflict.resolution = resolution
        conflict.resolved = True
        self.resolutions[cache_key] = resolution
        
        logger.info(f"Conflict resolved: {conflict.value1} vs {conflict.value2} -> {resolution}")
        return resolution


class ValuesSystem:
    """
    Main Values System.
    
    Combines:
    - ValueHierarchy: Core value structure
    - MultipleObjectives: Goal management
    - EmergentValues: Learning new values
    - ValueConflictResolver: Conflict resolution
    """
    
    def __init__(self):
        self.hierarchy = ValueHierarchy()
        self.objectives = MultipleObjectives(self.hierarchy)
        self.emergent = EmergentValues(self.hierarchy)
        self.resolver = ValueConflictResolver(self.hierarchy)
        
        logger.info("ValuesSystem initialized")
    
    def evaluate_decision(
        self,
        action: str,
        expected_effects: Dict[str, float]
    ) -> Tuple[float, Optional[str]]:
        """
        Evaluate a decision against values and objectives.
        
        Returns:
            (alignment_score, resolution if conflict)
        """
        # Check for conflicts
        conflict = self.resolver.detect_conflict(action, expected_effects)
        resolution = None
        
        if conflict:
            resolution = self.resolver.resolve(conflict)
        
        # Evaluate objective alignment
        alignment = self.objectives.evaluate_action(action, expected_effects)
        
        return alignment, resolution
    
    def learn_from_experience(
        self,
        context: str,
        action: str,
        outcome: float
    ):
        """Learn from an experience, possibly discovering new values."""
        self.emergent.observe_preference(context, action, outcome)
    
    def get_guiding_values(self, context: Dict[str, Any]) -> List[str]:
        """Get values that should guide current decision."""
        # Activate relevant values
        self.hierarchy.activate_relevant(context)
        
        # Get top active values
        top = self.hierarchy.get_top_values(5)
        return [v.name for v in top]
    
    def get_status(self) -> Dict[str, Any]:
        """Get values system status."""
        return {
            'total_values': len(self.hierarchy.values),
            'learned_values': len([v for v in self.hierarchy.values.values() if v.learned]),
            'active_objectives': len(self.objectives.get_active_objectives()),
            'conflicts_resolved': len([c for c in self.resolver.conflicts if c.resolved]),
            'emergent_candidates': len(self.emergent.candidates)
        }
