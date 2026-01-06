"""
AGI Fase 2: Knowledge Transfer System

Transferência de Conhecimento:
- Transferência Inter-Domínio
- Generalização Extrema
- Biblioteca de Conhecimento
- Abstração de Conceitos
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("Transfer")


class AbstractionLevel(Enum):
    CONCRETE = 1
    SPECIFIC = 2
    GENERAL = 3
    ABSTRACT = 4
    UNIVERSAL = 5


@dataclass
class Concept:
    """An abstracted concept."""
    concept_id: str
    name: str
    abstraction_level: AbstractionLevel
    features: Dict[str, Any]
    relations: List[str]
    domains: Set[str] = field(default_factory=set)


@dataclass
class KnowledgeItem:
    """An item in the knowledge library."""
    item_id: str
    domain: str
    content: Dict[str, Any]
    abstractions: List[str]
    applications: List[str]
    confidence: float = 0.5


@dataclass
class TransferResult:
    """Result of knowledge transfer."""
    source_domain: str
    target_domain: str
    knowledge_transferred: List[str]
    adaptation_needed: List[str]
    success_likelihood: float


class ConceptAbstractor:
    """Abstracts concepts to higher levels."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)
        self._concept_counter = 0
        
        logger.info("ConceptAbstractor initialized")
    
    def abstract(self, concrete_items: List[Dict], domain: str) -> Concept:
        """Abstract from concrete items."""
        common_features = self._find_common_features(concrete_items)
        
        self._concept_counter += 1
        concept = Concept(
            concept_id=f"concept_{self._concept_counter}",
            name=f"abstraction_of_{len(concrete_items)}_items",
            abstraction_level=AbstractionLevel.GENERAL,
            features=common_features,
            relations=[],
            domains={domain}
        )
        
        self.concepts[concept.concept_id] = concept
        return concept
    
    def _find_common_features(self, items: List[Dict]) -> Dict[str, Any]:
        """Find common features across items."""
        if not items:
            return {}
        
        common_keys = set(items[0].keys())
        for item in items[1:]:
            common_keys &= set(item.keys())
        
        common_features = {}
        for key in common_keys:
            values = [item[key] for item in items]
            
            if all(isinstance(v, (int, float)) for v in values):
                common_features[key] = sum(values) / len(values)
            elif len(set(str(v) for v in values)) == 1:
                common_features[key] = values[0]
        
        return common_features
    
    def elevate(self, concept: Concept) -> Concept:
        """Elevate concept to higher abstraction."""
        if concept.abstraction_level == AbstractionLevel.UNIVERSAL:
            return concept
        
        self._concept_counter += 1
        elevated = Concept(
            concept_id=f"concept_{self._concept_counter}",
            name=f"elevated_{concept.name}",
            abstraction_level=AbstractionLevel(concept.abstraction_level.value + 1),
            features={k: v for k, v in concept.features.items() if k in ['type', 'pattern']},
            relations=[concept.concept_id],
            domains=concept.domains.copy()
        )
        
        self.hierarchy[elevated.concept_id].append(concept.concept_id)
        self.concepts[elevated.concept_id] = elevated
        
        return elevated
    
    def find_similar(self, concept: Concept) -> List[Concept]:
        """Find similar concepts across domains."""
        similar = []
        
        for other in self.concepts.values():
            if other.concept_id == concept.concept_id:
                continue
            
            overlap = set(concept.features.keys()) & set(other.features.keys())
            if len(overlap) >= len(concept.features) * 0.5:
                similar.append(other)
        
        return similar


class InterDomainTransfer:
    """Transfer knowledge between domains."""
    
    def __init__(self, abstractor: ConceptAbstractor):
        self.abstractor = abstractor
        self.transfers: List[TransferResult] = []
        self.domain_mappings: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        logger.info("InterDomainTransfer initialized")
    
    def transfer(
        self,
        knowledge: KnowledgeItem,
        target_domain: str
    ) -> TransferResult:
        """Transfer knowledge to new domain."""
        abstract_concepts = [
            self.abstractor.concepts[c_id] 
            for c_id in knowledge.abstractions 
            if c_id in self.abstractor.concepts
        ]
        
        transferred = []
        adaptations = []
        
        for concept in abstract_concepts:
            if concept.abstraction_level.value >= AbstractionLevel.GENERAL.value:
                transferred.append(concept.concept_id)
                concept.domains.add(target_domain)
            else:
                adaptations.append(f"Adapt {concept.name} for {target_domain}")
        
        success = len(transferred) / max(1, len(abstract_concepts))
        
        result = TransferResult(
            source_domain=knowledge.domain,
            target_domain=target_domain,
            knowledge_transferred=transferred,
            adaptation_needed=adaptations,
            success_likelihood=success
        )
        
        self.transfers.append(result)
        return result
    
    def map_domains(self, source: str, target: str, mappings: Dict[str, str]):
        """Create explicit domain mappings."""
        self.domain_mappings[f"{source}->{target}"] = mappings
    
    def apply_mapping(self, knowledge: Dict, source: str, target: str) -> Dict:
        """Apply domain mapping to knowledge."""
        mapping_key = f"{source}->{target}"
        
        if mapping_key not in self.domain_mappings:
            return knowledge
        
        mappings = self.domain_mappings[mapping_key]
        transferred = {}
        
        for key, value in knowledge.items():
            new_key = mappings.get(key, key)
            transferred[new_key] = value
        
        return transferred


class KnowledgeLibrary:
    """Library of transferable knowledge."""
    
    def __init__(self):
        self.items: Dict[str, KnowledgeItem] = {}
        self.by_domain: Dict[str, List[str]] = defaultdict(list)
        self.by_abstraction: Dict[str, List[str]] = defaultdict(list)
        self._item_counter = 0
        
        logger.info("KnowledgeLibrary initialized")
    
    def store(
        self,
        domain: str,
        content: Dict[str, Any],
        abstractions: List[str]
    ) -> str:
        """Store knowledge item."""
        self._item_counter += 1
        item_id = f"know_{self._item_counter}"
        
        item = KnowledgeItem(
            item_id=item_id,
            domain=domain,
            content=content,
            abstractions=abstractions,
            applications=[domain]
        )
        
        self.items[item_id] = item
        self.by_domain[domain].append(item_id)
        
        for abstract in abstractions:
            self.by_abstraction[abstract].append(item_id)
        
        return item_id
    
    def retrieve(self, domain: str) -> List[KnowledgeItem]:
        """Retrieve knowledge for a domain."""
        item_ids = self.by_domain.get(domain, [])
        return [self.items[i] for i in item_ids if i in self.items]
    
    def retrieve_by_abstraction(self, abstraction: str) -> List[KnowledgeItem]:
        """Retrieve by abstraction."""
        item_ids = self.by_abstraction.get(abstraction, [])
        return [self.items[i] for i in item_ids if i in self.items]
    
    def find_applicable(self, target_domain: str) -> List[KnowledgeItem]:
        """Find knowledge applicable to domain."""
        applicable = []
        
        for item in self.items.values():
            if item.domain != target_domain and item.confidence > 0.6:
                applicable.append(item)
        
        return applicable
    
    def mark_application(self, item_id: str, domain: str, success: bool):
        """Mark knowledge application."""
        if item_id in self.items:
            item = self.items[item_id]
            item.applications.append(domain)
            
            alpha = 0.2
            item.confidence = item.confidence * (1 - alpha) + float(success) * alpha


class ExtremeGeneralizer:
    """Generalizes to extreme levels."""
    
    def __init__(self, abstractor: ConceptAbstractor):
        self.abstractor = abstractor
        self.universal_patterns: List[Dict] = []
        
        logger.info("ExtremeGeneralizer initialized")
    
    def generalize_to_universal(self, concepts: List[Concept]) -> Dict[str, Any]:
        """Generalize to universal pattern."""
        all_features = defaultdict(list)
        
        for concept in concepts:
            for key, value in concept.features.items():
                all_features[key].append(value)
        
        universal = {
            'pattern_type': 'universal',
            'domains': set().union(*(c.domains for c in concepts)),
            'core_features': {}
        }
        
        for key, values in all_features.items():
            if len(values) >= len(concepts) * 0.8:
                if all(isinstance(v, (int, float)) for v in values):
                    universal['core_features'][key] = 'numeric_value'
                else:
                    universal['core_features'][key] = 'categorical_value'
        
        self.universal_patterns.append(universal)
        return universal
    
    def find_universal_truths(self) -> List[str]:
        """Find universal truths from patterns."""
        truths = []
        
        for pattern in self.universal_patterns:
            if len(pattern.get('domains', set())) >= 3:
                truths.append(f"Universal pattern across {len(pattern['domains'])} domains")
        
        return truths


class KnowledgeTransferSystem:
    """Main Knowledge Transfer System."""
    
    def __init__(self):
        self.abstractor = ConceptAbstractor()
        self.transfer = InterDomainTransfer(self.abstractor)
        self.library = KnowledgeLibrary()
        self.generalizer = ExtremeGeneralizer(self.abstractor)
        
        logger.info("KnowledgeTransferSystem initialized")
    
    def learn_and_abstract(
        self,
        domain: str,
        items: List[Dict]
    ) -> str:
        """Learn from items and abstract."""
        concept = self.abstractor.abstract(items, domain)
        
        combined = {}
        for item in items:
            combined.update(item)
        
        item_id = self.library.store(domain, combined, [concept.concept_id])
        
        return item_id
    
    def transfer_to_domain(
        self,
        source_domain: str,
        target_domain: str
    ) -> List[TransferResult]:
        """Transfer all applicable knowledge to target."""
        source_items = self.library.retrieve(source_domain)
        results = []
        
        for item in source_items:
            result = self.transfer.transfer(item, target_domain)
            results.append(result)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'concepts': len(self.abstractor.concepts),
            'knowledge_items': len(self.library.items),
            'domains': len(self.library.by_domain),
            'transfers': len(self.transfer.transfers),
            'universal_patterns': len(self.generalizer.universal_patterns)
        }
