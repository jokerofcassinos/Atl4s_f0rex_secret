"""
AGI Fase 2: Internal Language System

Linguagem Interna entre Módulos:
- Linguagem Estruturada
- Semântica Compartilhada
- Negociação entre Módulos
- Persuasão por Argumentação
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("InternalLanguage")


class MessageType(Enum):
    ASSERTION = "assertion"
    QUERY = "query"
    PROPOSAL = "proposal"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    ARGUMENT = "argument"
    EVIDENCE = "evidence"


class ArgumentStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    COMPELLING = 4


@dataclass
class Proposition:
    """A logical proposition."""
    subject: str
    predicate: str
    object: Optional[str] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        if self.object:
            return f"{self.subject} {self.predicate} {self.object}"
        return f"{self.subject} {self.predicate}"


@dataclass
class Message:
    """A message between modules."""
    sender: str
    receiver: str
    msg_type: MessageType
    content: Proposition
    context: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None
    msg_id: str = ""
    
    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"msg_{int(time.time() * 1000)}"


@dataclass
class Argument:
    """An argument for persuasion."""
    claim: Proposition
    premises: List[Proposition]
    evidence: List[Dict[str, Any]]
    strength: ArgumentStrength = ArgumentStrength.MODERATE


class SharedSemantics:
    """Shared meaning of concepts between modules."""
    
    def __init__(self):
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.synonyms: Dict[str, List[str]] = defaultdict(list)
        
        self._init_trading_semantics()
        logger.info("SharedSemantics initialized")
    
    def _init_trading_semantics(self):
        """Initialize trading domain semantics."""
        self.concepts = {
            'BUY': {'type': 'action', 'direction': 'long', 'confidence_min': 0.6},
            'SELL': {'type': 'action', 'direction': 'short', 'confidence_min': 0.6},
            'WAIT': {'type': 'action', 'direction': 'neutral', 'confidence_min': 0.0},
            'BULLISH': {'type': 'state', 'direction': 'up', 'sentiment': 'positive'},
            'BEARISH': {'type': 'state', 'direction': 'down', 'sentiment': 'negative'},
            'HIGH_RISK': {'type': 'risk', 'level': 'high', 'action_modifier': 'reduce_size'},
            'LOW_RISK': {'type': 'risk', 'level': 'low', 'action_modifier': 'increase_size'},
        }
        
        self.synonyms = {
            'BUY': ['LONG', 'BULLISH_ENTRY'],
            'SELL': ['SHORT', 'BEARISH_ENTRY'],
            'HIGH_RISK': ['DANGER', 'CAUTION'],
        }
    
    def define_concept(self, name: str, attributes: Dict[str, Any]):
        """Define a new concept."""
        self.concepts[name] = attributes
    
    def get_meaning(self, term: str) -> Optional[Dict[str, Any]]:
        """Get meaning of a term."""
        if term in self.concepts:
            return self.concepts[term]
        
        for canonical, syns in self.synonyms.items():
            if term in syns:
                return self.concepts.get(canonical)
        
        return None
    
    def are_compatible(self, term1: str, term2: str) -> bool:
        """Check if two terms are compatible."""
        m1 = self.get_meaning(term1)
        m2 = self.get_meaning(term2)
        
        if not m1 or not m2:
            return True
        
        if m1.get('direction') and m2.get('direction'):
            return m1['direction'] == m2['direction']
        
        return True


class Negotiator:
    """Handles negotiation between modules."""
    
    def __init__(self, semantics: SharedSemantics):
        self.semantics = semantics
        self.active_negotiations: Dict[str, Dict] = {}
        self.resolved: List[Dict] = []
        
        logger.info("Negotiator initialized")
    
    def start_negotiation(
        self,
        topic: str,
        parties: List[str],
        initial_positions: Dict[str, Proposition]
    ) -> str:
        """Start a negotiation session."""
        neg_id = f"neg_{int(time.time() * 1000)}"
        
        self.active_negotiations[neg_id] = {
            'topic': topic,
            'parties': parties,
            'positions': initial_positions,
            'rounds': 0,
            'status': 'active',
            'started_at': time.time()
        }
        
        return neg_id
    
    def propose(
        self,
        negotiation_id: str,
        party: str,
        proposal: Proposition
    ) -> Dict[str, Any]:
        """Make a proposal in negotiation."""
        if negotiation_id not in self.active_negotiations:
            return {'success': False, 'reason': 'Negotiation not found'}
        
        neg = self.active_negotiations[negotiation_id]
        neg['positions'][party] = proposal
        neg['rounds'] += 1
        
        # Check for consensus
        if self._check_consensus(neg):
            neg['status'] = 'resolved'
            self.resolved.append(neg)
            return {'success': True, 'consensus': True}
        
        return {'success': True, 'consensus': False}
    
    def _check_consensus(self, negotiation: Dict) -> bool:
        """Check if parties have reached consensus."""
        positions = list(negotiation['positions'].values())
        if len(positions) < 2:
            return False
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if not self.semantics.are_compatible(
                    positions[i].predicate,
                    positions[j].predicate
                ):
                    return False
        
        return True
    
    def find_compromise(self, negotiation_id: str) -> Optional[Proposition]:
        """Find a compromise position."""
        if negotiation_id not in self.active_negotiations:
            return None
        
        neg = self.active_negotiations[negotiation_id]
        positions = neg['positions']
        
        if not positions:
            return None
        
        avg_confidence = sum(p.confidence for p in positions.values()) / len(positions)
        
        if avg_confidence < 0.5:
            return Proposition(
                subject="consensus",
                predicate="WAIT",
                confidence=avg_confidence
            )
        
        return list(positions.values())[0]


class Persuader:
    """Handles persuasion through arguments."""
    
    def __init__(self):
        self.arguments: List[Argument] = []
        
        logger.info("Persuader initialized")
    
    def build_argument(
        self,
        claim: Proposition,
        premises: List[Proposition],
        evidence: List[Dict[str, Any]]
    ) -> Argument:
        """Build an argument to support a claim."""
        strength = self._evaluate_strength(premises, evidence)
        
        arg = Argument(
            claim=claim,
            premises=premises,
            evidence=evidence,
            strength=strength
        )
        
        self.arguments.append(arg)
        return arg
    
    def _evaluate_strength(
        self,
        premises: List[Proposition],
        evidence: List[Dict[str, Any]]
    ) -> ArgumentStrength:
        """Evaluate argument strength."""
        score = 0
        
        score += len(premises) * 0.3
        score += len(evidence) * 0.4
        
        avg_premise_conf = sum(p.confidence for p in premises) / len(premises) if premises else 0
        score += avg_premise_conf * 0.3
        
        if score >= 0.8:
            return ArgumentStrength.COMPELLING
        elif score >= 0.6:
            return ArgumentStrength.STRONG
        elif score >= 0.4:
            return ArgumentStrength.MODERATE
        return ArgumentStrength.WEAK
    
    def counter_argument(self, argument: Argument) -> Optional[Argument]:
        """Attempt to counter an argument."""
        weak_premise = min(argument.premises, key=lambda p: p.confidence) if argument.premises else None
        
        if weak_premise and weak_premise.confidence < 0.5:
            counter_claim = Proposition(
                subject=weak_premise.subject,
                predicate="NOT " + weak_premise.predicate,
                confidence=1 - weak_premise.confidence
            )
            
            return Argument(
                claim=counter_claim,
                premises=[],
                evidence=[{'type': 'rebuttal'}],
                strength=ArgumentStrength.MODERATE
            )
        
        return None


class InternalLanguageSystem:
    """Main Internal Language System."""
    
    def __init__(self):
        self.semantics = SharedSemantics()
        self.negotiator = Negotiator(self.semantics)
        self.persuader = Persuader()
        
        self.message_log: List[Message] = []
        
        logger.info("InternalLanguageSystem initialized")
    
    def send_message(
        self,
        sender: str,
        receiver: str,
        msg_type: MessageType,
        subject: str,
        predicate: str,
        **kwargs
    ) -> Message:
        """Send a message between modules."""
        prop = Proposition(
            subject=subject,
            predicate=predicate,
            object=kwargs.get('object'),
            confidence=kwargs.get('confidence', 1.0)
        )
        
        msg = Message(
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            content=prop,
            context=kwargs.get('context', {}),
            reply_to=kwargs.get('reply_to')
        )
        
        self.message_log.append(msg)
        return msg
    
    def get_status(self) -> Dict[str, Any]:
        """Get language system status."""
        return {
            'concepts_defined': len(self.semantics.concepts),
            'messages_sent': len(self.message_log),
            'active_negotiations': len(self.negotiator.active_negotiations),
            'resolved_negotiations': len(self.negotiator.resolved),
            'arguments_made': len(self.persuader.arguments)
        }
