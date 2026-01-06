"""
AGI Fase 2: Artificial Emotions System

Emoções Artificiais Funcionais:
- Curiosidade, Satisfação, Frustração
- Influência nas Decisões
- Regulação Emocional
- Memória Emocional
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("Emotions")


class EmotionType(Enum):
    CURIOSITY = "curiosity"
    SATISFACTION = "satisfaction"
    FRUSTRATION = "frustration"
    EXCITEMENT = "excitement"
    CAUTION = "caution"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"


@dataclass
class Emotion:
    """An emotional state."""
    emotion_type: EmotionType
    intensity: float  # 0-1
    valence: float  # -1 to 1 (negative to positive)
    triggered_by: str
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0  # seconds active


@dataclass
class EmotionalMemory:
    """Memory tagged with emotion."""
    memory_id: str
    content: Dict[str, Any]
    emotions: List[Emotion]
    recall_count: int = 0
    last_recalled: float = 0.0


class EmotionalState:
    """Current emotional state of the system."""
    
    def __init__(self):
        self.active_emotions: Dict[EmotionType, Emotion] = {}
        self.baseline: Dict[EmotionType, float] = {
            EmotionType.CURIOSITY: 0.5,
            EmotionType.CAUTION: 0.5,
            EmotionType.CONFIDENCE: 0.5,
        }
        self.mood: float = 0.0  # Overall mood: -1 to 1
        
        logger.info("EmotionalState initialized")
    
    def feel(self, emotion_type: EmotionType, intensity: float, trigger: str):
        """Experience an emotion."""
        valence = self._get_valence(emotion_type)
        
        emotion = Emotion(
            emotion_type=emotion_type,
            intensity=min(1.0, intensity),
            valence=valence,
            triggered_by=trigger
        )
        
        self.active_emotions[emotion_type] = emotion
        self._update_mood()
    
    def _get_valence(self, emotion_type: EmotionType) -> float:
        """Get valence for emotion type."""
        valences = {
            EmotionType.CURIOSITY: 0.3,
            EmotionType.SATISFACTION: 0.8,
            EmotionType.FRUSTRATION: -0.6,
            EmotionType.EXCITEMENT: 0.7,
            EmotionType.CAUTION: -0.2,
            EmotionType.CONFIDENCE: 0.5,
            EmotionType.UNCERTAINTY: -0.3,
        }
        return valences.get(emotion_type, 0.0)
    
    def _update_mood(self):
        """Update overall mood from active emotions."""
        if not self.active_emotions:
            self.mood = 0.0
            return
        
        weighted_sum = sum(
            e.valence * e.intensity 
            for e in self.active_emotions.values()
        )
        total_intensity = sum(e.intensity for e in self.active_emotions.values())
        
        self.mood = weighted_sum / total_intensity if total_intensity > 0 else 0.0
    
    def decay(self, rate: float = 0.1):
        """Decay emotion intensities over time."""
        to_remove = []
        
        for etype, emotion in self.active_emotions.items():
            emotion.intensity = max(0, emotion.intensity - rate)
            emotion.duration += 1.0
            
            if emotion.intensity < 0.1:
                to_remove.append(etype)
        
        for etype in to_remove:
            del self.active_emotions[etype]
        
        self._update_mood()
    
    def get_dominant(self) -> Optional[EmotionType]:
        """Get dominant emotion."""
        if not self.active_emotions:
            return None
        return max(self.active_emotions.items(), key=lambda x: x[1].intensity)[0]


class EmotionGenerator:
    """Generates emotions from events."""
    
    def __init__(self, state: EmotionalState):
        self.state = state
        self.event_history: deque = deque(maxlen=100)
        
        logger.info("EmotionGenerator initialized")
    
    def process_event(self, event: Dict[str, Any]):
        """Process event and generate emotions."""
        event_type = event.get('type', 'unknown')
        outcome = event.get('outcome', 0.0)
        
        self.event_history.append(event)
        
        if event_type == 'trade_result':
            if outcome > 0:
                self.state.feel(EmotionType.SATISFACTION, outcome, 'profitable_trade')
                self.state.feel(EmotionType.CONFIDENCE, 0.3, 'success')
            else:
                self.state.feel(EmotionType.FRUSTRATION, abs(outcome), 'losing_trade')
                self.state.feel(EmotionType.UNCERTAINTY, 0.3, 'failure')
        
        elif event_type == 'new_pattern':
            self.state.feel(EmotionType.CURIOSITY, 0.6, 'discovery')
            self.state.feel(EmotionType.EXCITEMENT, 0.4, 'novelty')
        
        elif event_type == 'high_risk':
            self.state.feel(EmotionType.CAUTION, 0.7, 'risk_detection')
        
        elif event_type == 'opportunity':
            self.state.feel(EmotionType.EXCITEMENT, 0.5, 'opportunity')
    
    def process_repeated_failure(self, failures: int):
        """Process repeated failures."""
        frustration = min(1.0, failures * 0.2)
        self.state.feel(EmotionType.FRUSTRATION, frustration, 'repeated_failure')
        self.state.feel(EmotionType.UNCERTAINTY, frustration * 0.5, 'loss_of_confidence')
    
    def process_success_streak(self, successes: int):
        """Process success streak."""
        confidence = min(1.0, successes * 0.15)
        self.state.feel(EmotionType.CONFIDENCE, confidence, 'success_streak')
        self.state.feel(EmotionType.SATISFACTION, confidence * 0.8, 'winning')


class EmotionalDecisionInfluence:
    """Emotions influence decisions."""
    
    def __init__(self, state: EmotionalState):
        self.state = state
        
        logger.info("EmotionalDecisionInfluence initialized")
    
    def modify_confidence(self, base_confidence: float) -> float:
        """Modify decision confidence based on emotions."""
        modifier = 1.0
        
        if EmotionType.CONFIDENCE in self.state.active_emotions:
            conf_emotion = self.state.active_emotions[EmotionType.CONFIDENCE]
            modifier += conf_emotion.intensity * 0.2
        
        if EmotionType.UNCERTAINTY in self.state.active_emotions:
            unc_emotion = self.state.active_emotions[EmotionType.UNCERTAINTY]
            modifier -= unc_emotion.intensity * 0.3
        
        if EmotionType.FRUSTRATION in self.state.active_emotions:
            frus_emotion = self.state.active_emotions[EmotionType.FRUSTRATION]
            modifier -= frus_emotion.intensity * 0.1
        
        return max(0.1, min(1.0, base_confidence * modifier))
    
    def modify_risk_tolerance(self, base_tolerance: float) -> float:
        """Modify risk tolerance based on emotions."""
        modifier = 1.0
        
        if EmotionType.CAUTION in self.state.active_emotions:
            caut_emotion = self.state.active_emotions[EmotionType.CAUTION]
            modifier -= caut_emotion.intensity * 0.3
        
        if EmotionType.EXCITEMENT in self.state.active_emotions:
            exc_emotion = self.state.active_emotions[EmotionType.EXCITEMENT]
            modifier += exc_emotion.intensity * 0.2
        
        if self.state.mood < -0.3:
            modifier -= 0.2
        
        return max(0.1, min(1.0, base_tolerance * modifier))
    
    def should_explore(self) -> bool:
        """Determine if should explore based on emotions."""
        if EmotionType.CURIOSITY in self.state.active_emotions:
            return self.state.active_emotions[EmotionType.CURIOSITY].intensity > 0.5
        return False
    
    def get_emotional_bias(self) -> Dict[str, float]:
        """Get current emotional biases."""
        return {
            'confidence_bias': self.modify_confidence(0.5) - 0.5,
            'risk_bias': self.modify_risk_tolerance(0.5) - 0.5,
            'exploration_bias': 0.3 if self.should_explore() else 0.0,
            'overall_mood': self.state.mood
        }


class EmotionRegulator:
    """Regulates emotions to prevent extremes."""
    
    def __init__(self, state: EmotionalState):
        self.state = state
        self.regulation_history: List[Dict] = []
        
        logger.info("EmotionRegulator initialized")
    
    def regulate(self) -> List[str]:
        """Regulate extreme emotions."""
        interventions = []
        
        for etype, emotion in list(self.state.active_emotions.items()):
            if emotion.intensity > 0.9:
                emotion.intensity = 0.7
                interventions.append(f"Dampened extreme {etype.value}")
            
            if emotion.duration > 300 and emotion.intensity > 0.5:
                emotion.intensity *= 0.8
                interventions.append(f"Reduced prolonged {etype.value}")
        
        if self.state.mood < -0.7:
            for etype in [EmotionType.FRUSTRATION, EmotionType.UNCERTAINTY]:
                if etype in self.state.active_emotions:
                    self.state.active_emotions[etype].intensity *= 0.7
            interventions.append("Uplifted negative mood")
        
        if interventions:
            self.regulation_history.append({
                'timestamp': time.time(),
                'interventions': interventions
            })
        
        return interventions
    
    def cognitive_reappraisal(self, situation: str) -> str:
        """Reappraise situation to change emotional response."""
        if 'loss' in situation.lower():
            return "This loss is a learning opportunity"
        if 'failure' in situation.lower():
            return "Failure is feedback for improvement"
        if 'risk' in situation.lower():
            return "Risk is manageable with proper controls"
        return situation


class EmotionalMemorySystem:
    """Memory tagged with emotions."""
    
    def __init__(self):
        self.memories: Dict[str, EmotionalMemory] = {}
        self._memory_counter = 0
        
        logger.info("EmotionalMemorySystem initialized")
    
    def store(self, content: Dict, emotions: List[Emotion]) -> str:
        """Store emotionally tagged memory."""
        self._memory_counter += 1
        memory_id = f"emem_{self._memory_counter}"
        
        self.memories[memory_id] = EmotionalMemory(
            memory_id=memory_id,
            content=content,
            emotions=emotions
        )
        
        return memory_id
    
    def recall_by_emotion(self, emotion_type: EmotionType) -> List[EmotionalMemory]:
        """Recall memories by emotion."""
        matches = []
        
        for memory in self.memories.values():
            if any(e.emotion_type == emotion_type for e in memory.emotions):
                memory.recall_count += 1
                memory.last_recalled = time.time()
                matches.append(memory)
        
        return sorted(matches, key=lambda m: max(e.intensity for e in m.emotions), reverse=True)
    
    def get_most_emotional(self, n: int = 5) -> List[EmotionalMemory]:
        """Get most emotionally intense memories."""
        scored = []
        for memory in self.memories.values():
            intensity = sum(e.intensity for e in memory.emotions)
            scored.append((memory, intensity))
        
        return [m for m, _ in sorted(scored, key=lambda x: -x[1])[:n]]


class ArtificialEmotionsSystem:
    """Main Artificial Emotions System."""
    
    def __init__(self):
        self.state = EmotionalState()
        self.generator = EmotionGenerator(self.state)
        self.influence = EmotionalDecisionInfluence(self.state)
        self.regulator = EmotionRegulator(self.state)
        self.memory = EmotionalMemorySystem()
        
        logger.info("ArtificialEmotionsSystem initialized")
    
    def process_and_feel(self, event: Dict[str, Any]):
        """Process event and generate emotions."""
        self.generator.process_event(event)
        
        self.regulator.regulate()
        
        self.memory.store(event, list(self.state.active_emotions.values()))
    
    def get_emotional_context(self) -> Dict[str, Any]:
        """Get current emotional context for decisions."""
        return {
            'mood': self.state.mood,
            'dominant_emotion': self.state.get_dominant().value if self.state.get_dominant() else None,
            'active_emotions': {
                e.value: self.state.active_emotions[e].intensity
                for e in self.state.active_emotions
            },
            'biases': self.influence.get_emotional_bias()
        }
    
    def tick(self):
        """Time tick for emotion decay."""
        self.state.decay()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'mood': self.state.mood,
            'active_emotions': len(self.state.active_emotions),
            'emotional_memories': len(self.memory.memories),
            'regulations': len(self.regulator.regulation_history)
        }
