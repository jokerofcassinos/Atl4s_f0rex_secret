"""
AGI Fase 2: Emergence and Self-Organization

Sistemas Emergentes e Auto-Organização:
- Formação de Hierarquias
- Formação de Coalizões
- Divisão de Trabalho
- Comportamento Emergente Complexo
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("Emergence")


class AgentRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"
    SCOUT = "scout"


@dataclass
class Agent:
    """A self-organizing agent."""
    agent_id: str
    role: AgentRole = AgentRole.GENERALIST
    capabilities: Set[str] = field(default_factory=set)
    performance: float = 0.5
    connections: Set[str] = field(default_factory=set)
    coalition: Optional[str] = None


@dataclass
class Coalition:
    """A coalition of agents."""
    coalition_id: str
    members: Set[str]
    goal: str
    formed_at: float
    dissolved_at: Optional[float] = None
    success_rate: float = 0.0


@dataclass
class EmergentPattern:
    """An emergent pattern from agent interactions."""
    pattern_id: str
    pattern_type: str
    participating_agents: List[str]
    first_observed: float
    occurrences: int = 1
    strength: float = 0.5


class SwarmSelfOrganization:
    """Self-organization of swarm modules."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("SwarmSelfOrganization initialized")
    
    def add_agent(self, agent_id: str, capabilities: Set[str]) -> Agent:
        """Add a new agent to the swarm."""
        agent = Agent(agent_id=agent_id, capabilities=capabilities)
        self.agents[agent_id] = agent
        return agent
    
    def form_hierarchy(self):
        """Form hierarchy based on performance."""
        sorted_agents = sorted(
            self.agents.values(),
            key=lambda a: a.performance,
            reverse=True
        )
        
        self.hierarchy.clear()
        
        if len(sorted_agents) >= 1:
            leader = sorted_agents[0]
            leader.role = AgentRole.LEADER
            
            for follower in sorted_agents[1:]:
                follower.role = AgentRole.FOLLOWER
                self.hierarchy[leader.agent_id].append(follower.agent_id)
                follower.connections.add(leader.agent_id)
    
    def form_coalition(
        self,
        goal: str,
        required_capabilities: Set[str]
    ) -> Optional[Coalition]:
        """Form coalition based on complementary capabilities."""
        candidates = []
        
        for agent in self.agents.values():
            if agent.capabilities & required_capabilities:
                candidates.append(agent)
        
        if len(candidates) < 2:
            return None
        
        coalition_id = f"coal_{int(time.time() * 1000)}"
        member_ids = {a.agent_id for a in candidates}
        
        coalition = Coalition(
            coalition_id=coalition_id,
            members=member_ids,
            goal=goal,
            formed_at=time.time()
        )
        
        for agent_id in member_ids:
            self.agents[agent_id].coalition = coalition_id
        
        self.coalitions[coalition_id] = coalition
        return coalition
    
    def divide_labor(self, task: str, subtasks: List[str]):
        """Dynamically divide labor based on capabilities."""
        assignments = {}
        
        for subtask in subtasks:
            best_agent = None
            best_score = 0
            
            for agent in self.agents.values():
                if subtask in agent.capabilities:
                    score = agent.performance
                    if score > best_score:
                        best_score = score
                        best_agent = agent.agent_id
            
            if best_agent:
                assignments[subtask] = best_agent
        
        return assignments
    
    def update_performance(self, agent_id: str, delta: float):
        """Update agent performance and reorganize."""
        if agent_id in self.agents:
            self.agents[agent_id].performance += delta
            self.agents[agent_id].performance = max(0, min(1, self.agents[agent_id].performance))
            
            if abs(delta) > 0.1:
                self.form_hierarchy()


class EmergentBehavior:
    """Tracks and analyzes emergent behavior."""
    
    def __init__(self):
        self.patterns: Dict[str, EmergentPattern] = {}
        self.interactions: List[Dict] = []
        self.phase: str = "normal"
        
        logger.info("EmergentBehavior initialized")
    
    def record_interaction(
        self,
        agents: List[str],
        interaction_type: str,
        outcome: float
    ):
        """Record an interaction between agents."""
        self.interactions.append({
            'agents': agents,
            'type': interaction_type,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
        self._detect_patterns()
    
    def _detect_patterns(self):
        """Detect emergent patterns from interactions."""
        if len(self.interactions) < 5:
            return
        
        recent = self.interactions[-20:]
        
        type_counts = defaultdict(list)
        for i in recent:
            key = (tuple(sorted(i['agents'])), i['type'])
            type_counts[key].append(i['outcome'])
        
        for key, outcomes in type_counts.items():
            if len(outcomes) >= 3:
                agents, itype = key
                pattern_id = f"pat_{hash(key) % 10000}"
                
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = EmergentPattern(
                        pattern_id=pattern_id,
                        pattern_type=itype,
                        participating_agents=list(agents),
                        first_observed=time.time()
                    )
                else:
                    self.patterns[pattern_id].occurrences += 1
                    self.patterns[pattern_id].strength = sum(outcomes) / len(outcomes)
    
    def detect_phase_transition(self) -> Optional[str]:
        """Detect if system is undergoing phase transition."""
        if len(self.interactions) < 10:
            return None
        
        recent = self.interactions[-10:]
        older = self.interactions[-20:-10] if len(self.interactions) >= 20 else []
        
        if not older:
            return None
        
        recent_avg = sum(i['outcome'] for i in recent) / len(recent)
        older_avg = sum(i['outcome'] for i in older) / len(older)
        
        diff = abs(recent_avg - older_avg)
        
        if diff > 0.3:
            new_phase = "high_activity" if recent_avg > older_avg else "low_activity"
            if new_phase != self.phase:
                self.phase = new_phase
                return new_phase
        
        return None


class CollectiveAdaptation:
    """Collective learning and adaptation."""
    
    def __init__(self):
        self.collective_memory: Dict[str, List[float]] = defaultdict(list)
        self.collective_strategy: Dict[str, float] = {}
        
        logger.info("CollectiveAdaptation initialized")
    
    def share_experience(self, agent_id: str, experience: Dict[str, float]):
        """Share individual experience with collective."""
        for key, value in experience.items():
            self.collective_memory[key].append(value)
            
            if len(self.collective_memory[key]) > 100:
                self.collective_memory[key] = self.collective_memory[key][-100:]
    
    def get_collective_knowledge(self, key: str) -> float:
        """Get collective knowledge about something."""
        if key not in self.collective_memory:
            return 0.5
        
        values = self.collective_memory[key]
        return sum(values) / len(values) if values else 0.5
    
    def update_collective_strategy(self):
        """Update collective strategy from individual experiences."""
        for key, values in self.collective_memory.items():
            if values:
                self.collective_strategy[key] = sum(values) / len(values)
    
    def get_collective_intelligence(self) -> Dict[str, Any]:
        """Measure collective intelligence."""
        if not self.collective_strategy:
            return {'iq': 0.5, 'diversity': 0, 'consensus': 0}
        
        values = list(self.collective_strategy.values())
        
        avg = sum(values) / len(values) if values else 0.5
        variance = sum((v - avg) ** 2 for v in values) / len(values) if values else 0
        
        return {
            'iq': avg,
            'diversity': variance ** 0.5,
            'consensus': 1 - variance,
            'knowledge_areas': len(self.collective_strategy)
        }


class EmergenceSystem:
    """Main Emergence and Self-Organization System."""
    
    def __init__(self):
        self.swarm = SwarmSelfOrganization()
        self.behavior = EmergentBehavior()
        self.collective = CollectiveAdaptation()
        
        logger.info("EmergenceSystem initialized")
    
    def initialize_swarm(self, agent_configs: List[Dict[str, Any]]):
        """Initialize swarm with agents."""
        for config in agent_configs:
            self.swarm.add_agent(
                agent_id=config['id'],
                capabilities=set(config.get('capabilities', []))
            )
        self.swarm.form_hierarchy()
    
    def tick(self):
        """Run one tick of emergence simulation."""
        agents = list(self.swarm.agents.keys())
        
        if len(agents) >= 2:
            a1, a2 = random.sample(agents, 2)
            outcome = random.random()
            
            self.behavior.record_interaction([a1, a2], "collaboration", outcome)
            
            self.swarm.update_performance(a1, (outcome - 0.5) * 0.1)
            self.swarm.update_performance(a2, (outcome - 0.5) * 0.1)
            
            self.collective.share_experience(a1, {'collaboration': outcome})
        
        phase = self.behavior.detect_phase_transition()
        if phase:
            logger.info(f"Phase transition detected: {phase}")
        
        self.collective.update_collective_strategy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get emergence system status."""
        return {
            'agents': len(self.swarm.agents),
            'coalitions': len(self.swarm.coalitions),
            'patterns': len(self.behavior.patterns),
            'phase': self.behavior.phase,
            'collective_intelligence': self.collective.get_collective_intelligence()
        }
