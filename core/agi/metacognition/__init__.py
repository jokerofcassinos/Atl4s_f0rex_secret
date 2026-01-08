# AGI Metacognition Package
"""
Advanced metacognitive reasoning systems for AGI-level analysis.
"""

from .recursive_reflection import RecursiveReflection
from .empathic_resonance import EmpathicResonance
from .ontological_abstractor import OntologicalAbstractor
from .causal_web_navigator import CausalWebNavigator
from .neural_plasticity_core import NeuralPlasticityCore

# Backward compatibility aliases for omega_agi_core imports
RecursiveReflectionLoop = RecursiveReflection

# ConfidenceCalibrator class for metacognitive confidence assessment
class ConfidenceCalibrator:
    """Calibrates confidence scores based on historical accuracy."""
    
    def __init__(self):
        self.history = []
        self.calibration_factor = 1.0
    
    def calibrate(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score."""
        calibrated = raw_confidence * self.calibration_factor
        return min(0.95, max(0.05, calibrated))
    
    def update(self, predicted_conf: float, actual_correct: bool):
        """Update calibration based on outcome."""
        self.history.append({'pred': predicted_conf, 'correct': actual_correct})
        if len(self.history) >= 20:
            self._recalibrate()
    
    def _recalibrate(self):
        """Recalibrate based on recent history."""
        if not self.history:
            return
        recent = self.history[-50:]
        avg_pred = sum(h['pred'] for h in recent) / len(recent)
        accuracy = sum(1 for h in recent if h['correct']) / len(recent)
        if avg_pred > 0:
            self.calibration_factor = accuracy / avg_pred
            self.calibration_factor = min(1.5, max(0.5, self.calibration_factor))

__all__ = [
    'RecursiveReflection',
    'RecursiveReflectionLoop',  # Alias
    'EmpathicResonance',
    'OntologicalAbstractor',
    'CausalWebNavigator',
    'NeuralPlasticityCore',
    'ConfidenceCalibrator'
]
