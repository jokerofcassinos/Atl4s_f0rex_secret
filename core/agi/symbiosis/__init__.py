# Symbiotic Fusion Package
"""
Advanced systems for human-machine cognitive symbiosis and pattern synthesis.
"""

import logging
import random
from typing import Dict, Any

from .pattern_synthesis_matrix import PatternSynthesisMatrix
from .heuristic_evolution import HeuristicEvolution
from .cognitive_symbiosis_bridge import CognitiveSymbiosisBridge
from .temporal_abstraction import TemporalAbstraction
from .cross_domain_reasoner import CrossDomainReasoner

logger = logging.getLogger("Symbiosis")

# === Backward Compatibility Classes (from original symbiosis.py) ===

class UserIntentModeler:
    """
    System 11: User Intent Modeler.
    Infers the hidden state of the User (Goals, Risk Tolerance, Mood).
    """
    def __init__(self):
        self.profile = {
            'risk_tolerance': 'MODERATE',
            'focus': 'PROFIT',
            'interaction_style': 'DIRECT'
        }
        
    def analyze_command(self, command: str):
        cmd = command.lower()
        if "aggressive" in cmd or "max" in cmd or "full" in cmd:
            self.profile['risk_tolerance'] = 'AGGRESSIVE'
        elif "safe" in cmd or "cautious" in cmd or "risk" in cmd: 
            self.profile['risk_tolerance'] = 'CAUTIOUS'
        logger.info(f"SYMBIOSIS: Updated User Profile -> {self.profile}")

    def get_context_modifiers(self) -> Dict[str, float]:
        mods = {'risk_multiplier': 1.0}
        if self.profile['risk_tolerance'] == 'AGGRESSIVE':
            mods['risk_multiplier'] = 1.5
        elif self.profile['risk_tolerance'] == 'CAUTIOUS':
            mods['risk_multiplier'] = 0.5
        return mods


class ExplanabilityGenerator:
    """
    System 13: Explanability Generator (The Translator).
    Translates AGI internal states into Natural Language.
    """
    def __init__(self):
        self.phrases = {
            'BUY': ["Detecting bullish momentum.", "Accumulation phase identified."],
            'SELL': ["Bearish divergence noted.", "Distribution pattern emerging."],
            'WAIT': ["Market noise is high.", "Conflicting signals detected."],
            'VETO': ["Safety override engaged.", "Risk exceeds protocol limits."]
        }
        
    def generate_narrative(self, decision: str, confidence: float, meta_data: Dict[str, Any]) -> str:
        base = random.choice(self.phrases.get(decision, ["Processing..."]))
        reflection_notes = meta_data.get('reflection_notes', [])
        reflection_text = ""
        if reflection_notes:
            reflection_text = " However, " + "; ".join(reflection_notes).lower() + "."
        reason = ""
        if 'active_inference_G' in meta_data:
            reason = f" (Active Inference G-Score: {meta_data['active_inference_G']:.2f})"
        return f"{base} Confidence: {confidence:.1f}%.{reason}{reflection_text}"


__all__ = [
    'PatternSynthesisMatrix',
    'HeuristicEvolution',
    'CognitiveSymbiosisBridge',
    'TemporalAbstraction',
    'CrossDomainReasoner',
    'UserIntentModeler',
    'ExplanabilityGenerator'
]

