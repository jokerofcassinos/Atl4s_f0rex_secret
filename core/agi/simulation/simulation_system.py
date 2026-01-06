"""
AGI Fase 2: Mental Simulation and Dreaming

Simulação Mental e "Sonhos":
- Simulação de Cenários
- Mental Time Travel
- Sistema de Sonhos (Offline Processing)
- Imaginação e Visualização
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("Simulation")


class ScenarioType(Enum):
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    REALISTIC = "realistic"
    EXTREME = "extreme"


@dataclass
class Scenario:
    """A simulated scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    initial_state: Dict[str, Any]
    simulated_steps: List[Dict]
    final_state: Dict[str, Any]
    outcome: float
    probability: float


@dataclass
class DreamContent:
    """Content of a dream/offline processing session."""
    dream_id: str
    themes: List[str]
    recombinations: List[Dict]
    insights: List[str]
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MentalImage:
    """A mental visualization."""
    image_id: str
    subject: str
    features: Dict[str, Any]
    manipulations: List[str]
    derived_insights: List[str]


class ScenarioSimulator:
    """Simulates future scenarios."""
    
    def __init__(self, steps_per_scenario: int = 10):
        self.steps = steps_per_scenario
        self.scenarios: List[Scenario] = []
        self._scenario_counter = 0
        
        logger.info("ScenarioSimulator initialized")
    
    def simulate(
        self,
        initial_state: Dict[str, Any],
        scenario_type: ScenarioType = ScenarioType.REALISTIC
    ) -> Scenario:
        """Simulate a scenario."""
        self._scenario_counter += 1
        
        current_state = dict(initial_state)
        steps = []
        
        for i in range(self.steps):
            new_state, event = self._simulate_step(current_state, scenario_type)
            steps.append({
                'step': i,
                'event': event,
                'state': dict(new_state)
            })
            current_state = new_state
        
        outcome = self._evaluate_outcome(initial_state, current_state)
        probability = self._estimate_probability(scenario_type, steps)
        
        scenario = Scenario(
            scenario_id=f"scen_{self._scenario_counter}",
            scenario_type=scenario_type,
            initial_state=initial_state,
            simulated_steps=steps,
            final_state=current_state,
            outcome=outcome,
            probability=probability
        )
        
        self.scenarios.append(scenario)
        return scenario
    
    def _simulate_step(
        self,
        state: Dict[str, Any],
        scenario_type: ScenarioType
    ) -> Tuple[Dict[str, Any], str]:
        """Simulate one step."""
        new_state = dict(state)
        
        if scenario_type == ScenarioType.OPTIMISTIC:
            delta = random.uniform(0, 0.05)
            event = "positive_movement"
        elif scenario_type == ScenarioType.PESSIMISTIC:
            delta = random.uniform(-0.05, 0)
            event = "negative_movement"
        elif scenario_type == ScenarioType.EXTREME:
            delta = random.uniform(-0.1, 0.1)
            event = "volatile_movement"
        else:
            delta = random.gauss(0, 0.02)
            event = "normal_movement"
        
        if 'price' in new_state:
            new_state['price'] = new_state['price'] * (1 + delta)
        if 'value' in new_state:
            new_state['value'] = new_state['value'] * (1 + delta)
        
        return new_state, event
    
    def _evaluate_outcome(self, initial: Dict, final: Dict) -> float:
        """Evaluate scenario outcome."""
        if 'price' in initial and 'price' in final:
            return (final['price'] - initial['price']) / initial['price']
        if 'value' in initial and 'value' in final:
            return (final['value'] - initial['value']) / max(0.01, initial['value'])
        return 0.0
    
    def _estimate_probability(self, scenario_type: ScenarioType, steps: List) -> float:
        """Estimate scenario probability."""
        base = {
            ScenarioType.REALISTIC: 0.6,
            ScenarioType.OPTIMISTIC: 0.2,
            ScenarioType.PESSIMISTIC: 0.15,
            ScenarioType.EXTREME: 0.05
        }
        return base.get(scenario_type, 0.5)
    
    def monte_carlo(self, initial_state: Dict, n_simulations: int = 100) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        outcomes = []
        
        for _ in range(n_simulations):
            scenario = self.simulate(initial_state, ScenarioType.REALISTIC)
            outcomes.append(scenario.outcome)
        
        return {
            'mean': sum(outcomes) / len(outcomes),
            'min': min(outcomes),
            'max': max(outcomes),
            'std': (sum((o - sum(outcomes)/len(outcomes))**2 for o in outcomes) / len(outcomes)) ** 0.5,
            'positive_probability': sum(1 for o in outcomes if o > 0) / len(outcomes)
        }


class MentalTimeTravel:
    """Travel mentally through time."""
    
    def __init__(self):
        self.past_experiences: deque = deque(maxlen=1000)
        self.future_projections: List[Dict] = []
        
        logger.info("MentalTimeTravel initialized")
    
    def remember_past(self, time_ago_hours: float) -> List[Dict]:
        """Remember past experiences."""
        threshold = time.time() - (time_ago_hours * 3600)
        
        memories = [
            exp for exp in self.past_experiences
            if exp.get('timestamp', 0) >= threshold
        ]
        
        return memories
    
    def record_experience(self, experience: Dict):
        """Record an experience for later recall."""
        experience['timestamp'] = time.time()
        self.past_experiences.append(experience)
    
    def project_future(
        self,
        current_state: Dict,
        hours_ahead: float,
        conditions: Dict
    ) -> Dict:
        """Project into the future."""
        projection = {
            'from_state': current_state,
            'hours_ahead': hours_ahead,
            'conditions': conditions,
            'projected_state': dict(current_state),
            'confidence': 0.5 - (hours_ahead * 0.02)
        }
        
        rate = conditions.get('expected_change_rate', 0)
        for key in projection['projected_state']:
            if isinstance(projection['projected_state'][key], (int, float)):
                projection['projected_state'][key] *= (1 + rate * hours_ahead)
        
        self.future_projections.append(projection)
        return projection
    
    def counterfactual(self, past_decision: Dict, alternative: str) -> Dict:
        """Think counterfactually."""
        return {
            'original_decision': past_decision,
            'alternative': alternative,
            'counterfactual_outcome': 'Would depend on market conditions',
            'lesson': 'Consider alternatives before deciding'
        }


class DreamingSystem:
    """Offline processing during idle time."""
    
    def __init__(self):
        self.dreams: List[DreamContent] = []
        self.consolidated_memories: List[Dict] = []
        self.is_dreaming = False
        self._dream_counter = 0
        
        logger.info("DreamingSystem initialized")
    
    def enter_dream_state(self, memories: List[Dict], duration_seconds: float = 60):
        """Enter dream/offline processing state."""
        self.is_dreaming = True
        self._dream_counter += 1
        
        themes = self._extract_themes(memories)
        recombinations = self._recombine_creatively(memories)
        insights = self._discover_insights(recombinations)
        
        dream = DreamContent(
            dream_id=f"dream_{self._dream_counter}",
            themes=themes,
            recombinations=recombinations,
            insights=insights,
            duration_seconds=duration_seconds
        )
        
        self.dreams.append(dream)
        self._consolidate_memories(memories)
        
        self.is_dreaming = False
        return dream
    
    def _extract_themes(self, memories: List[Dict]) -> List[str]:
        """Extract themes from memories."""
        themes = []
        
        word_counts = {}
        for mem in memories:
            for key in mem.keys():
                word_counts[key] = word_counts.get(key, 0) + 1
        
        themes = sorted(word_counts.keys(), key=lambda k: -word_counts[k])[:5]
        return themes
    
    def _recombine_creatively(self, memories: List[Dict]) -> List[Dict]:
        """Recombine memories creatively."""
        recombinations = []
        
        if len(memories) >= 2:
            for i in range(min(5, len(memories) - 1)):
                m1 = random.choice(memories)
                m2 = random.choice(memories)
                
                combined = {}
                for key in set(m1.keys()) | set(m2.keys()):
                    if key in m1 and key in m2:
                        if isinstance(m1[key], (int, float)):
                            combined[key] = (m1[key] + m2[key]) / 2
                        else:
                            combined[key] = random.choice([m1[key], m2[key]])
                    elif key in m1:
                        combined[key] = m1[key]
                    else:
                        combined[key] = m2[key]
                
                recombinations.append(combined)
        
        return recombinations
    
    def _discover_insights(self, recombinations: List[Dict]) -> List[str]:
        """Discover insights from recombinations."""
        insights = []
        
        for rec in recombinations:
            if 'outcome' in rec and rec.get('outcome', 0) > 0:
                insights.append("Positive outcomes associated with this pattern")
        
        return insights
    
    def _consolidate_memories(self, memories: List[Dict]):
        """Consolidate important memories."""
        for mem in memories:
            importance = mem.get('importance', 0.5)
            if importance > 0.7 or random.random() < importance:
                self.consolidated_memories.append(mem)


class ImaginationEngine:
    """Mental visualization and manipulation."""
    
    def __init__(self):
        self.images: List[MentalImage] = []
        self._image_counter = 0
        
        logger.info("ImaginationEngine initialized")
    
    def visualize(self, subject: str, features: Dict[str, Any]) -> MentalImage:
        """Create mental visualization."""
        self._image_counter += 1
        
        image = MentalImage(
            image_id=f"img_{self._image_counter}",
            subject=subject,
            features=features,
            manipulations=[],
            derived_insights=[]
        )
        
        self.images.append(image)
        return image
    
    def manipulate(self, image: MentalImage, transformation: str) -> MentalImage:
        """Mentally manipulate an image."""
        new_features = dict(image.features)
        
        if transformation == 'rotate':
            if 'direction' in new_features:
                new_features['direction'] = -new_features['direction']
        elif transformation == 'scale_up':
            for key in new_features:
                if isinstance(new_features[key], (int, float)):
                    new_features[key] *= 1.5
        elif transformation == 'scale_down':
            for key in new_features:
                if isinstance(new_features[key], (int, float)):
                    new_features[key] *= 0.5
        
        image.manipulations.append(transformation)
        image.features = new_features
        
        return image
    
    def explore_possibility(self, base_image: MentalImage, variation: str) -> MentalImage:
        """Explore a possibility through imagination."""
        new_image = self.visualize(
            subject=f"{base_image.subject}_{variation}",
            features=dict(base_image.features)
        )
        
        new_image.derived_insights.append(f"Variation: {variation}")
        
        return new_image


class SimulationSystem:
    """Main Mental Simulation System."""
    
    def __init__(self):
        self.simulator = ScenarioSimulator()
        self.time_travel = MentalTimeTravel()
        self.dreaming = DreamingSystem()
        self.imagination = ImaginationEngine()
        
        logger.info("SimulationSystem initialized")
    
    def full_mental_simulation(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive mental simulation."""
        optimistic = self.simulator.simulate(current_state, ScenarioType.OPTIMISTIC)
        pessimistic = self.simulator.simulate(current_state, ScenarioType.PESSIMISTIC)
        realistic = self.simulator.simulate(current_state, ScenarioType.REALISTIC)
        
        monte_carlo = self.simulator.monte_carlo(current_state, 50)
        
        future = self.time_travel.project_future(current_state, 24, {'expected_change_rate': 0.001})
        
        return {
            'scenarios': {
                'optimistic': optimistic.outcome,
                'pessimistic': pessimistic.outcome,
                'realistic': realistic.outcome
            },
            'monte_carlo': monte_carlo,
            'future_projection': future,
            'recommendation': 'BUY' if monte_carlo['positive_probability'] > 0.6 else 'WAIT'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'scenarios_simulated': len(self.simulator.scenarios),
            'experiences_recorded': len(self.time_travel.past_experiences),
            'dreams': len(self.dreaming.dreams),
            'consolidated_memories': len(self.dreaming.consolidated_memories),
            'mental_images': len(self.imagination.images)
        }
