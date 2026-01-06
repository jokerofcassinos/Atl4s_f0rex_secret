"""
AGI Fase 2: Collaboration System

Colaboração entre Instâncias AGI:
- Comunicação Multi-Instância
- Co-Evolução
- Compartilhamento de Conhecimento
- Inteligência Distribuída
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import hashlib

logger = logging.getLogger("Collaboration")


class InstanceRole(Enum):
    LEADER = "leader"
    SPECIALIST = "specialist"
    EXPLORER = "explorer"
    VALIDATOR = "validator"
    LEARNER = "learner"


@dataclass
class AGIInstance:
    """An AGI instance in the collective."""
    instance_id: str
    role: InstanceRole
    specialization: str
    capabilities: Set[str]
    performance: float = 0.5
    trust_score: float = 0.5
    last_active: float = field(default_factory=time.time)


@dataclass
class SharedKnowledge:
    """Knowledge shared between instances."""
    knowledge_id: str
    content: Dict[str, Any]
    source_instance: str
    validators: List[str]
    confidence: float
    created_at: float = field(default_factory=time.time)


@dataclass
class CollaborativeTask:
    """A task for collaborative execution."""
    task_id: str
    description: str
    assigned_instances: List[str]
    subtasks: List[Dict]
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)


class MultiInstanceCommunication:
    """Communication between AGI instances."""
    
    def __init__(self):
        self.instances: Dict[str, AGIInstance] = {}
        self.message_queue: deque = deque(maxlen=1000)
        self.broadcasts: List[Dict] = []
        
        logger.info("MultiInstanceCommunication initialized")
    
    def register_instance(
        self,
        instance_id: str,
        role: InstanceRole,
        specialization: str,
        capabilities: Set[str]
    ) -> AGIInstance:
        """Register a new AGI instance."""
        instance = AGIInstance(
            instance_id=instance_id,
            role=role,
            specialization=specialization,
            capabilities=capabilities
        )
        
        self.instances[instance_id] = instance
        return instance
    
    def send_message(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        content: Dict[str, Any]
    ):
        """Send message between instances."""
        self.message_queue.append({
            'sender': sender,
            'receiver': receiver,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        })
    
    def broadcast(self, sender: str, message_type: str, content: Dict[str, Any]):
        """Broadcast to all instances."""
        broadcast = {
            'sender': sender,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        }
        
        self.broadcasts.append(broadcast)
        
        for instance_id in self.instances:
            if instance_id != sender:
                self.send_message(sender, instance_id, message_type, content)
    
    def get_messages(self, instance_id: str) -> List[Dict]:
        """Get messages for an instance."""
        return [
            m for m in self.message_queue
            if m['receiver'] == instance_id
        ]
    
    def elect_leader(self) -> Optional[str]:
        """Elect a leader instance."""
        if not self.instances:
            return None
        
        candidates = [
            i for i in self.instances.values()
            if i.role == InstanceRole.LEADER or i.performance > 0.7
        ]
        
        if candidates:
            leader = max(candidates, key=lambda i: i.performance * i.trust_score)
            return leader.instance_id
        
        return max(self.instances.values(), key=lambda i: i.performance).instance_id


class CoEvolutionEngine:
    """Co-evolution of multiple AGI instances."""
    
    def __init__(self, communication: MultiInstanceCommunication):
        self.communication = communication
        self.evolution_history: List[Dict] = []
        self.shared_genome: Dict[str, Any] = {}
        
        logger.info("CoEvolutionEngine initialized")
    
    def share_improvement(self, instance_id: str, improvement: Dict[str, Any]):
        """Share an improvement with the collective."""
        improvement_msg = {
            'improvement': improvement,
            'improvement_type': improvement.get('type', 'unknown'),
            'benefit': improvement.get('benefit', 0.0)
        }
        
        self.communication.broadcast(instance_id, 'improvement', improvement_msg)
        
        if improvement.get('benefit', 0) > 0.3:
            key = improvement.get('type', 'unknown')
            self.shared_genome[key] = improvement
    
    def adopt_improvement(self, instance_id: str, improvement: Dict) -> bool:
        """Adopt an improvement from another instance."""
        if improvement.get('benefit', 0) < 0.2:
            return False
        
        if instance_id in self.communication.instances:
            instance = self.communication.instances[instance_id]
            instance.performance = min(1.0, instance.performance + improvement.get('benefit', 0) * 0.5)
            
            self.evolution_history.append({
                'instance': instance_id,
                'adopted': improvement.get('type'),
                'timestamp': time.time()
            })
            
            return True
        
        return False
    
    def get_collective_genome(self) -> Dict[str, Any]:
        """Get the collective genome."""
        return self.shared_genome


class KnowledgeSharing:
    """System for sharing knowledge between instances."""
    
    def __init__(self, communication: MultiInstanceCommunication):
        self.communication = communication
        self.shared_knowledge: Dict[str, SharedKnowledge] = {}
        self._knowledge_counter = 0
        
        logger.info("KnowledgeSharing initialized")
    
    def share(
        self,
        source_instance: str,
        content: Dict[str, Any]
    ) -> str:
        """Share knowledge with the collective."""
        self._knowledge_counter += 1
        knowledge_id = f"shared_{self._knowledge_counter}"
        
        knowledge = SharedKnowledge(
            knowledge_id=knowledge_id,
            content=content,
            source_instance=source_instance,
            validators=[],
            confidence=0.5
        )
        
        self.shared_knowledge[knowledge_id] = knowledge
        
        self.communication.broadcast(source_instance, 'knowledge_share', {
            'knowledge_id': knowledge_id,
            'summary': str(list(content.keys())[:5])
        })
        
        return knowledge_id
    
    def validate(self, knowledge_id: str, validator_instance: str, is_valid: bool):
        """Validate shared knowledge."""
        if knowledge_id not in self.shared_knowledge:
            return
        
        knowledge = self.shared_knowledge[knowledge_id]
        
        if validator_instance not in knowledge.validators:
            knowledge.validators.append(validator_instance)
            
            if is_valid:
                knowledge.confidence = min(1.0, knowledge.confidence + 0.1)
            else:
                knowledge.confidence = max(0.0, knowledge.confidence - 0.2)
    
    def get_high_confidence(self, threshold: float = 0.7) -> List[SharedKnowledge]:
        """Get high confidence shared knowledge."""
        return [k for k in self.shared_knowledge.values() if k.confidence >= threshold]
    
    def consensus_knowledge(self) -> Dict[str, Any]:
        """Get knowledge with consensus."""
        consensus = {}
        
        for knowledge in self.shared_knowledge.values():
            if knowledge.confidence > 0.7 and len(knowledge.validators) >= 2:
                for key, value in knowledge.content.items():
                    consensus[key] = value
        
        return consensus


class DistributedIntelligence:
    """Distributed intelligence across instances."""
    
    def __init__(self, communication: MultiInstanceCommunication):
        self.communication = communication
        self.tasks: Dict[str, CollaborativeTask] = {}
        self._task_counter = 0
        
        logger.info("DistributedIntelligence initialized")
    
    def distribute_task(
        self,
        description: str,
        subtasks: List[Dict]
    ) -> CollaborativeTask:
        """Distribute a task across instances."""
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"
        
        assignments = self._assign_subtasks(subtasks)
        
        task = CollaborativeTask(
            task_id=task_id,
            description=description,
            assigned_instances=list(assignments.keys()),
            subtasks=subtasks,
            status="assigned"
        )
        
        self.tasks[task_id] = task
        
        for instance_id, assigned_subtasks in assignments.items():
            self.communication.send_message(
                sender="coordinator",
                receiver=instance_id,
                message_type="task_assignment",
                content={'task_id': task_id, 'subtasks': assigned_subtasks}
            )
        
        return task
    
    def _assign_subtasks(self, subtasks: List[Dict]) -> Dict[str, List[Dict]]:
        """Assign subtasks to instances based on capabilities."""
        assignments = defaultdict(list)
        
        instances = list(self.communication.instances.values())
        if not instances:
            return assignments
        
        for i, subtask in enumerate(subtasks):
            required_cap = subtask.get('required_capability')
            
            matching = [
                inst for inst in instances
                if required_cap in inst.capabilities
            ] if required_cap else instances
            
            if matching:
                best = max(matching, key=lambda x: x.performance)
                assignments[best.instance_id].append(subtask)
            else:
                assignments[instances[i % len(instances)].instance_id].append(subtask)
        
        return dict(assignments)
    
    def report_result(self, task_id: str, instance_id: str, result: Dict):
        """Report subtask result."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.results[instance_id] = result
            
            if len(task.results) == len(task.assigned_instances):
                task.status = "completed"
    
    def aggregate_results(self, task_id: str) -> Dict[str, Any]:
        """Aggregate results from all instances."""
        if task_id not in self.tasks:
            return {}
        
        task = self.tasks[task_id]
        
        aggregated = {
            'task_id': task_id,
            'status': task.status,
            'instance_results': task.results,
            'consensus': {}
        }
        
        all_keys = set()
        for result in task.results.values():
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = [r.get(key) for r in task.results.values() if key in r]
            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    aggregated['consensus'][key] = sum(values) / len(values)
        
        return aggregated


class CollaborationSystem:
    """Main Collaboration System."""
    
    def __init__(self):
        self.communication = MultiInstanceCommunication()
        self.coevolution = CoEvolutionEngine(self.communication)
        self.knowledge = KnowledgeSharing(self.communication)
        self.distributed = DistributedIntelligence(self.communication)
        
        logger.info("CollaborationSystem initialized")
    
    def create_instance(
        self,
        specialization: str,
        capabilities: Set[str],
        role: InstanceRole = InstanceRole.SPECIALIST
    ) -> str:
        """Create a new AGI instance."""
        instance_id = hashlib.md5(f"{specialization}{time.time()}".encode()).hexdigest()[:12]
        
        self.communication.register_instance(
            instance_id=instance_id,
            role=role,
            specialization=specialization,
            capabilities=capabilities
        )
        
        return instance_id
    
    def collective_decision(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Make a collective decision."""
        votes = defaultdict(list)
        
        for instance in self.communication.instances.values():
            vote = options[hash(instance.instance_id + question) % len(options)]
            votes[vote].append({
                'instance': instance.instance_id,
                'weight': instance.trust_score * instance.performance
            })
        
        weighted_votes = {}
        for option, vote_list in votes.items():
            weighted_votes[option] = sum(v['weight'] for v in vote_list)
        
        winner = max(weighted_votes.items(), key=lambda x: x[1])[0] if weighted_votes else options[0]
        
        return {
            'question': question,
            'decision': winner,
            'votes': dict(votes),
            'weighted_votes': weighted_votes
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'instances': len(self.communication.instances),
            'messages': len(self.communication.message_queue),
            'shared_knowledge': len(self.knowledge.shared_knowledge),
            'active_tasks': len([t for t in self.distributed.tasks.values() if t.status != 'completed']),
            'evolution_events': len(self.coevolution.evolution_history)
        }
