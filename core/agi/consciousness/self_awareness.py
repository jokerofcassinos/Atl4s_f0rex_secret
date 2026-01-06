"""
AGI Fase 2: Self-Awareness Engine

Sistema de Auto-Consciência que mantém modelo interno do sistema:
- Modelo de Si Mesmo (capacidades, limitações, estado)
- Auto-Reflexão Contínua
- Narrativa Interna
- Teoria da Mente
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("SelfAwareness")


class CapabilityLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SystemState(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    LEARNING = "learning"
    REFLECTING = "reflecting"
    ERROR = "error"


@dataclass
class Capability:
    """A single capability of the system."""
    name: str
    level: CapabilityLevel
    confidence: float  # 0-1
    last_used: float
    success_rate: float
    limitations: List[str] = field(default_factory=list)


@dataclass
class SelfNarrative:
    """A narrative about self actions and decisions."""
    timestamp: float
    action: str
    reasoning: str
    outcome: Optional[str] = None
    emotional_valence: float = 0.0  # -1 to 1


@dataclass
class AgentModel:
    """Model of another agent's mental state."""
    agent_id: str
    inferred_goal: str
    inferred_strategy: str
    confidence: float
    last_updated: float
    beliefs: Dict[str, Any] = field(default_factory=dict)


class SelfModel:
    """
    Internal model of the system itself.
    
    Tracks:
    - Current state
    - Capabilities and limitations
    - Recent history
    - Performance metrics
    """
    
    def __init__(self):
        self.current_state = SystemState.IDLE
        self.state_history: deque = deque(maxlen=100)
        
        # Identity
        self.name = "AGI Ultra Trading System"
        self.version = "2.0.0"
        self.creation_time = time.time()
        
        # Capabilities
        self.capabilities: Dict[str, Capability] = {}
        
        # Performance
        self.total_decisions = 0
        self.successful_decisions = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Beliefs about self
        self.self_beliefs: Dict[str, Any] = {
            'is_capable': True,
            'is_learning': True,
            'is_improving': True,
            'risk_tolerance': 'moderate'
        }
        
        logger.info("SelfModel initialized")
    
    def update_state(self, new_state: SystemState):
        """Update current system state."""
        self.state_history.append({
            'from': self.current_state.value,
            'to': new_state.value,
            'time': time.time()
        })
        self.current_state = new_state
    
    def register_capability(
        self,
        name: str,
        level: CapabilityLevel,
        limitations: Optional[List[str]] = None
    ):
        """Register a system capability."""
        self.capabilities[name] = Capability(
            name=name,
            level=level,
            confidence=0.5,
            last_used=0.0,
            success_rate=0.5,
            limitations=limitations or []
        )
    
    def update_capability(self, name: str, success: bool):
        """Update capability based on usage."""
        if name not in self.capabilities:
            return
        
        cap = self.capabilities[name]
        cap.last_used = time.time()
        
        # Update success rate
        alpha = 0.1
        cap.success_rate = cap.success_rate * (1 - alpha) + float(success) * alpha
        cap.confidence = min(1.0, cap.confidence + 0.01 if success else cap.confidence - 0.02)
    
    def get_self_assessment(self) -> Dict[str, Any]:
        """Generate self-assessment report."""
        accuracy = self.successful_decisions / self.total_decisions if self.total_decisions > 0 else 0
        win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'state': self.current_state.value,
            'uptime': time.time() - self.creation_time,
            'total_decisions': self.total_decisions,
            'decision_accuracy': accuracy,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'capability_count': len(self.capabilities),
            'self_beliefs': self.self_beliefs
        }


class CapabilityMap:
    """
    Map of all system capabilities and limitations.
    
    Allows the system to understand what it can and cannot do.
    """
    
    def __init__(self, self_model: SelfModel):
        self.model = self_model
        
        # Initialize core capabilities
        self._register_core_capabilities()
        
        logger.info("CapabilityMap initialized")
    
    def _register_core_capabilities(self):
        """Register core system capabilities."""
        capabilities = [
            ("trend_analysis", CapabilityLevel.EXPERT, ["Weak in ranging markets"]),
            ("pattern_recognition", CapabilityLevel.ADVANCED, ["Limited by training data"]),
            ("risk_management", CapabilityLevel.ADVANCED, ["May underestimate black swans"]),
            ("timing", CapabilityLevel.INTERMEDIATE, ["Latency dependent"]),
            ("adaptation", CapabilityLevel.ADVANCED, ["Slow in regime changes"]),
            ("meta_reasoning", CapabilityLevel.INTERMEDIATE, ["Resource intensive"]),
            ("memory_recall", CapabilityLevel.EXPERT, ["Older memories less reliable"]),
            ("creativity", CapabilityLevel.BASIC, ["Limited to known patterns"]),
            ("self_reflection", CapabilityLevel.INTERMEDIATE, ["May have blind spots"]),
        ]
        
        for name, level, limits in capabilities:
            self.model.register_capability(name, level, limits)
    
    def can_do(self, capability: str) -> bool:
        """Check if system has a capability."""
        if capability not in self.model.capabilities:
            return False
        return self.model.capabilities[capability].level != CapabilityLevel.NONE
    
    def get_limitations(self, capability: str) -> List[str]:
        """Get limitations for a capability."""
        if capability not in self.model.capabilities:
            return ["Capability not found"]
        return self.model.capabilities[capability].limitations
    
    def get_best_capabilities(self, n: int = 5) -> List[str]:
        """Get top N capabilities by success rate."""
        sorted_caps = sorted(
            self.model.capabilities.values(),
            key=lambda c: c.success_rate,
            reverse=True
        )
        return [c.name for c in sorted_caps[:n]]
    
    def get_weak_capabilities(self, threshold: float = 0.5) -> List[str]:
        """Get capabilities below threshold."""
        return [
            name for name, cap in self.model.capabilities.items()
            if cap.success_rate < threshold
        ]


class NarrativeBuilder:
    """
    Constructs coherent narrative about system's actions and decisions.
    
    Creates a story of "what I did and why" for self-understanding.
    """
    
    def __init__(self, max_history: int = 500):
        self.narratives: deque = deque(maxlen=max_history)
        self.current_thread = ""  # Current narrative thread
        
        logger.info("NarrativeBuilder initialized")
    
    def add_chapter(
        self,
        action: str,
        reasoning: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new chapter to the narrative."""
        narrative = SelfNarrative(
            timestamp=time.time(),
            action=action,
            reasoning=reasoning
        )
        
        self.narratives.append(narrative)
        
        # Update current thread
        self.current_thread = f"I {action.lower()} because {reasoning}"
        
        return self.current_thread
    
    def close_chapter(self, outcome: str, emotional_valence: float = 0.0):
        """Close the most recent chapter with outcome."""
        if self.narratives:
            self.narratives[-1].outcome = outcome
            self.narratives[-1].emotional_valence = emotional_valence
    
    def get_recent_narrative(self, n: int = 10) -> str:
        """Generate readable narrative of recent actions."""
        recent = list(self.narratives)[-n:]
        
        if not recent:
            return "No recent actions to narrate."
        
        lines = []
        for nar in recent:
            line = f"At {time.strftime('%H:%M', time.localtime(nar.timestamp))}, I {nar.action}"
            if nar.reasoning:
                line += f" because {nar.reasoning}"
            if nar.outcome:
                line += f". The outcome was: {nar.outcome}"
            lines.append(line)
        
        return " ".join(lines)
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get emotional summary of recent narrative."""
        if not self.narratives:
            return {'average_valence': 0, 'trend': 'neutral'}
        
        recent = list(self.narratives)[-20:]
        valences = [n.emotional_valence for n in recent if n.emotional_valence != 0]
        
        if not valences:
            return {'average_valence': 0, 'trend': 'neutral'}
        
        avg = sum(valences) / len(valences)
        
        # Check trend
        if len(valences) >= 2:
            first_half = sum(valences[:len(valences)//2]) / (len(valences)//2)
            second_half = sum(valences[len(valences)//2:]) / len(valences[len(valences)//2:])
            trend = 'improving' if second_half > first_half else 'declining'
        else:
            trend = 'neutral'
        
        return {'average_valence': avg, 'trend': trend}


class TheoryOfMind:
    """
    Models mental states of external agents (other traders, market participants).
    
    Attempts to understand:
    - What do they believe?
    - What are their goals?
    - What strategy are they using?
    """
    
    def __init__(self, max_agents: int = 100):
        self.agent_models: Dict[str, AgentModel] = {}
        self.max_agents = max_agents
        
        # Market-level mental model
        self.market_sentiment = 0.0  # -1 (fear) to 1 (greed)
        self.market_belief = "neutral"
        
        logger.info("TheoryOfMind initialized")
    
    def model_agent(
        self,
        agent_id: str,
        observed_actions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> AgentModel:
        """Create or update model of an agent."""
        # Infer goal from actions
        inferred_goal = self._infer_goal(observed_actions)
        
        # Infer strategy
        inferred_strategy = self._infer_strategy(observed_actions, context)
        
        model = AgentModel(
            agent_id=agent_id,
            inferred_goal=inferred_goal,
            inferred_strategy=inferred_strategy,
            confidence=0.5,
            last_updated=time.time()
        )
        
        self.agent_models[agent_id] = model
        
        # Cleanup old agents
        if len(self.agent_models) > self.max_agents:
            oldest = min(self.agent_models.values(), key=lambda a: a.last_updated)
            del self.agent_models[oldest.agent_id]
        
        return model
    
    def _infer_goal(self, actions: List[Dict[str, Any]]) -> str:
        """Infer agent's goal from actions."""
        if not actions:
            return "unknown"
        
        # Simple heuristics
        buy_count = sum(1 for a in actions if a.get('type') == 'BUY')
        sell_count = sum(1 for a in actions if a.get('type') == 'SELL')
        
        if buy_count > sell_count * 2:
            return "accumulation"
        elif sell_count > buy_count * 2:
            return "distribution"
        else:
            return "trading"
    
    def _infer_strategy(self, actions: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Infer agent's strategy from actions and context."""
        if not actions:
            return "unknown"
        
        # Check timing patterns
        times = [a.get('time', 0) for a in actions]
        
        if len(times) >= 2:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            if avg_interval < 60:  # Less than 1 minute
                return "scalping"
            elif avg_interval < 3600:  # Less than 1 hour
                return "day_trading"
            else:
                return "swing_trading"
        
        return "unknown"
    
    def update_market_sentiment(self, fear_greed: float, belief: str):
        """Update overall market mental model."""
        self.market_sentiment = max(-1, min(1, fear_greed))
        self.market_belief = belief
    
    def predict_agent_action(self, agent_id: str) -> Optional[str]:
        """Predict what an agent might do next."""
        if agent_id not in self.agent_models:
            return None
        
        model = self.agent_models[agent_id]
        
        if model.inferred_goal == "accumulation":
            return "likely_buy"
        elif model.inferred_goal == "distribution":
            return "likely_sell"
        else:
            return "uncertain"
    
    def predict_market_reaction(self, event: str) -> Dict[str, Any]:
        """Predict how market participants might react to an event."""
        # Simple model based on sentiment
        if self.market_sentiment > 0.5:
            # Greedy market
            return {
                'likely_reaction': 'buy_the_dip',
                'confidence': abs(self.market_sentiment)
            }
        elif self.market_sentiment < -0.5:
            # Fearful market
            return {
                'likely_reaction': 'sell_the_rally',
                'confidence': abs(self.market_sentiment)
            }
        else:
            return {
                'likely_reaction': 'mixed',
                'confidence': 0.3
            }


class SelfAwarenessEngine:
    """
    Main Self-Awareness Engine.
    
    Combines:
    - SelfModel: Internal representation
    - CapabilityMap: What can I do?
    - NarrativeBuilder: What have I done?
    - TheoryOfMind: What do others think?
    """
    
    def __init__(self):
        self.self_model = SelfModel()
        self.capability_map = CapabilityMap(self.self_model)
        self.narrative = NarrativeBuilder()
        self.theory_of_mind = TheoryOfMind()
        
        # Reflection state
        self._reflection_interval = 60.0  # seconds
        self._last_reflection = time.time()
        
        # Self-questions
        self._pending_questions = [
            "What do I know?",
            "What don't I know?",
            "What are my limitations?",
            "Am I improving?"
        ]
        
        logger.info("SelfAwarenessEngine initialized")
    
    def reflect(self) -> Dict[str, Any]:
        """
        Perform self-reflection.
        
        Answers fundamental questions about self.
        """
        now = time.time()
        
        if now - self._last_reflection < self._reflection_interval:
            return {}
        
        self._last_reflection = now
        self.self_model.update_state(SystemState.REFLECTING)
        
        reflection = {
            'timestamp': now,
            'questions': {}
        }
        
        # What do I know?
        best_caps = self.capability_map.get_best_capabilities(5)
        reflection['questions']['what_i_know'] = {
            'answer': f"I am best at: {', '.join(best_caps)}",
            'confidence': 0.8
        }
        
        # What don't I know?
        weak_caps = self.capability_map.get_weak_capabilities(0.5)
        reflection['questions']['what_i_dont_know'] = {
            'answer': f"I struggle with: {', '.join(weak_caps) or 'nothing major'}",
            'confidence': 0.7
        }
        
        # What are my limitations?
        all_limits = []
        for cap in self.self_model.capabilities.values():
            all_limits.extend(cap.limitations)
        reflection['questions']['my_limitations'] = {
            'answer': f"Key limitations: {'; '.join(all_limits[:5])}",
            'confidence': 0.6
        }
        
        # Am I improving?
        assessment = self.self_model.get_self_assessment()
        emotional = self.narrative.get_emotional_summary()
        reflection['questions']['am_i_improving'] = {
            'answer': f"Trend: {emotional['trend']}, Accuracy: {assessment['decision_accuracy']:.1%}",
            'confidence': 0.7
        }
        
        logger.info(f"Self-reflection complete: {reflection}")
        return reflection
    
    def record_action(self, action: str, reasoning: str, context: Optional[Dict] = None):
        """Record an action for narrative."""
        self.narrative.add_chapter(action, reasoning, context)
    
    def record_outcome(self, outcome: str, success: bool):
        """Record outcome of last action."""
        valence = 0.5 if success else -0.5
        self.narrative.close_chapter(outcome, valence)
        
        # Update self model
        self.self_model.total_decisions += 1
        if success:
            self.self_model.successful_decisions += 1
    
    def get_self_summary(self) -> str:
        """Generate a self-summary statement."""
        assessment = self.self_model.get_self_assessment()
        narrative = self.narrative.get_recent_narrative(3)
        
        summary = f"""
I am {self.self_model.name} v{self.self_model.version}.
I have been running for {assessment['uptime']/3600:.1f} hours.
I have made {assessment['total_decisions']} decisions with {assessment['decision_accuracy']:.1%} accuracy.
My current state is {assessment['state']}.

Recent activity: {narrative}
        """
        
        return summary.strip()
    
    def understand_other(self, agent_id: str, actions: List[Dict]) -> str:
        """Try to understand what another agent is thinking."""
        model = self.theory_of_mind.model_agent(agent_id, actions, {})
        prediction = self.theory_of_mind.predict_agent_action(agent_id)
        
        return f"Agent {agent_id} seems to be {model.inferred_strategy} with goal of {model.inferred_goal}. Prediction: {prediction}"
