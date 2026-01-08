"""
Cognitive Symbiosis Bridge - Human-Machine Cognitive Fusion.

Bridges human trader intuition with machine analysis
for enhanced decision-making through symbiotic fusion.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("CognitiveSymbiosis")


@dataclass
class SymbioticDecision:
    """A decision made through human-machine symbiosis."""
    direction: str
    human_weight: float
    machine_weight: float
    fusion_confidence: float
    disagreement_resolved: bool
    resolution_method: str


@dataclass
class SymbiosisState:
    """Current symbiosis state."""
    alignment: float  # How aligned are human and machine
    trust_level: float  # Machine's trust in human input
    deference_ratio: float  # When to defer to human
    learning_rate: float
    recent_outcomes: List[bool]


class CognitiveSymbiosisBridge:
    """
    The Mind Meld.
    
    Bridges human and machine cognition through:
    - Trust calibration between human and machine
    - Disagreement resolution protocols
    - Adaptive weighting based on performance
    - Intuition quantification
    """
    
    def __init__(self):
        self.state = SymbiosisState(
            alignment=0.5,
            trust_level=0.5,
            deference_ratio=0.5,
            learning_rate=0.1,
            recent_outcomes=[]
        )
        
        self.decision_history: deque = deque(maxlen=100)
        
        # Track performance by source
        self.performance = {
            'human_only': {'wins': 0, 'total': 0},
            'machine_only': {'wins': 0, 'total': 0},
            'fused': {'wins': 0, 'total': 0},
        }
        
        logger.info("CognitiveSymbiosisBridge initialized")
    
    def fuse(self, human_signal: Optional[Dict], 
            machine_signal: Dict) -> SymbioticDecision:
        """
        Fuse human and machine signals.
        
        Args:
            human_signal: Optional human input {'direction': str, 'confidence': float}
            machine_signal: Machine analysis {'direction': str, 'confidence': float}
            
        Returns:
            SymbioticDecision with fused result.
        """
        machine_dir = machine_signal.get('direction', 'WAIT')
        machine_conf = machine_signal.get('confidence', 0.5)
        
        if human_signal is None:
            # Pure machine decision
            return SymbioticDecision(
                direction=machine_dir,
                human_weight=0.0,
                machine_weight=1.0,
                fusion_confidence=machine_conf,
                disagreement_resolved=True,
                resolution_method='MACHINE_ONLY'
            )
        
        human_dir = human_signal.get('direction', 'WAIT')
        human_conf = human_signal.get('confidence', 0.5)
        
        # Check for agreement
        if human_dir == machine_dir:
            # Agreement - boost confidence
            fusion_conf = min(0.95, (human_conf + machine_conf) / 2 * 1.2)
            return SymbioticDecision(
                direction=machine_dir,
                human_weight=0.5,
                machine_weight=0.5,
                fusion_confidence=fusion_conf,
                disagreement_resolved=True,
                resolution_method='CONSENSUS'
            )
        
        # Disagreement - need resolution
        return self._resolve_disagreement(
            human_dir, human_conf, machine_dir, machine_conf
        )
    
    def _resolve_disagreement(self, human_dir: str, human_conf: float,
                              machine_dir: str, machine_conf: float) -> SymbioticDecision:
        """Resolve disagreement between human and machine."""
        # Calculate weights based on trust and performance
        human_weight = self.state.trust_level * human_conf
        machine_weight = (1 - self.state.deference_ratio) * machine_conf
        
        total = human_weight + machine_weight
        human_weight /= total
        machine_weight /= total
        
        # Decide based on weighted confidence
        if human_weight > 0.6:
            final_dir = human_dir
            resolution = 'HUMAN_OVERRIDE'
        elif machine_weight > 0.6:
            final_dir = machine_dir
            resolution = 'MACHINE_OVERRIDE'
        else:
            # Neither strong enough - wait
            final_dir = 'WAIT'
            resolution = 'DEADLOCK_WAIT'
        
        fusion_conf = max(human_conf, machine_conf) * 0.8  # Reduce confidence due to disagreement
        
        decision = SymbioticDecision(
            direction=final_dir,
            human_weight=human_weight,
            machine_weight=machine_weight,
            fusion_confidence=fusion_conf,
            disagreement_resolved=final_dir != 'WAIT',
            resolution_method=resolution
        )
        
        self.decision_history.append(decision)
        return decision
    
    def update_from_outcome(self, source: str, success: bool):
        """Update symbiosis state from trade outcome."""
        # Update performance tracking
        if source in self.performance:
            self.performance[source]['total'] += 1
            if success:
                self.performance[source]['wins'] += 1
        
        # Update trust levels
        self._update_trust(source, success)
        
        # Update alignment
        self._update_alignment()
        
        # Track recent outcomes
        self.state.recent_outcomes.append(success)
        if len(self.state.recent_outcomes) > 20:
            self.state.recent_outcomes = self.state.recent_outcomes[-20:]
    
    def _update_trust(self, source: str, success: bool):
        """Update trust levels based on outcome."""
        lr = self.state.learning_rate
        
        if source == 'human_only' or 'HUMAN' in source:
            if success:
                self.state.trust_level = min(0.9, self.state.trust_level + lr)
            else:
                self.state.trust_level = max(0.1, self.state.trust_level - lr)
        
        if source == 'machine_only' or 'MACHINE' in source:
            if success:
                self.state.deference_ratio = max(0.1, self.state.deference_ratio - lr)
            else:
                self.state.deference_ratio = min(0.9, self.state.deference_ratio + lr)
    
    def _update_alignment(self):
        """Update alignment based on recent decisions."""
        if len(self.decision_history) < 5:
            return
        
        recent = list(self.decision_history)[-10:]
        agreements = sum(1 for d in recent if d.resolution_method == 'CONSENSUS')
        self.state.alignment = agreements / len(recent)
    
    def get_recommended_action(self) -> str:
        """Get recommended action based on symbiosis state."""
        if self.state.alignment > 0.7:
            return "FOLLOW_CONSENSUS"
        elif self.state.trust_level > 0.7:
            return "DEFER_TO_HUMAN"
        else:
            return "TRUST_MACHINE"
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary by source."""
        summary = {}
        for source, data in self.performance.items():
            if data['total'] > 0:
                summary[source] = {
                    'win_rate': data['wins'] / data['total'],
                    'total_trades': data['total']
                }
        return summary
