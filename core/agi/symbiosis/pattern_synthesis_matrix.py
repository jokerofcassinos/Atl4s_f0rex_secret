"""
Pattern Synthesis Matrix - Hyper-Complex Pattern Synthesis.

Synthesizes patterns from multiple data streams into unified
higher-dimensional representations for advanced reasoning.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("PatternSynthesis")


@dataclass
class SynthesizedPattern:
    """A synthesized hyper-pattern."""
    name: str
    components: List[str]  # Source patterns
    strength: float
    dimensionality: int
    novelty_score: float
    actionable: bool
    suggested_action: Optional[str]


@dataclass
class SynthesisResult:
    """Pattern synthesis analysis result."""
    patterns: List[SynthesizedPattern]
    coherence: float
    complexity_level: int
    dominant_pattern: Optional[SynthesizedPattern]
    synthesis_quality: float


class PatternSynthesisMatrix:
    """
    The Pattern Alchemist.
    
    Synthesizes hyper-complex patterns through:
    - Multi-stream pattern fusion
    - Dimensionality expansion for richer representations
    - Cross-modal pattern matching
    - Emergent pattern detection
    """
    
    def __init__(self, encoding_dim: int = 128):
        self.encoding_dim = encoding_dim
        self.pattern_memory: Dict[str, np.ndarray] = {}
        self.synthesis_history: deque = deque(maxlen=50)
        
        # Base pattern templates
        self.templates = {
            'TREND_IMPULSE': np.random.randn(encoding_dim) * 0.5,
            'REVERSAL_PATTERN': np.random.randn(encoding_dim) * 0.5,
            'CONSOLIDATION': np.random.randn(encoding_dim) * 0.5,
            'BREAKOUT': np.random.randn(encoding_dim) * 0.5,
            'EXHAUSTION': np.random.randn(encoding_dim) * 0.5,
        }
        
        logger.info(f"PatternSynthesisMatrix initialized with dim={encoding_dim}")
    
    def synthesize(self, input_streams: Dict[str, Dict]) -> SynthesisResult:
        """
        Synthesize patterns from multiple input streams.
        
        Args:
            input_streams: Dict of stream_name -> stream_data
            
        Returns:
            SynthesisResult with synthesized patterns.
        """
        # Encode each stream
        encoded_streams = {}
        for name, data in input_streams.items():
            encoded_streams[name] = self._encode_stream(name, data)
        
        # Fuse encodings
        fused = self._fuse_encodings(encoded_streams)
        
        # Detect patterns in fused representation
        patterns = self._detect_patterns(fused, list(encoded_streams.keys()))
        
        # Calculate coherence
        coherence = self._calculate_coherence(encoded_streams)
        
        # Determine complexity
        complexity = min(5, len(patterns) + len(encoded_streams) // 2)
        
        # Find dominant pattern
        dominant = max(patterns, key=lambda p: p.strength) if patterns else None
        
        # Synthesis quality
        quality = np.mean([p.strength for p in patterns]) if patterns else 0.0
        
        result = SynthesisResult(
            patterns=patterns,
            coherence=coherence,
            complexity_level=complexity,
            dominant_pattern=dominant,
            synthesis_quality=quality
        )
        
        self.synthesis_history.append(result)
        return result
    
    def _encode_stream(self, name: str, data: Dict) -> np.ndarray:
        """Encode a data stream into the pattern space."""
        encoding = np.zeros(self.encoding_dim)
        
        # Extract features from data
        for i, (key, value) in enumerate(data.items()):
            if isinstance(value, (int, float)):
                idx = hash(key) % self.encoding_dim
                encoding[idx] += float(value)
            elif isinstance(value, str):
                idx = hash(f"{key}_{value}") % self.encoding_dim
                encoding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding /= norm
        
        return encoding
    
    def _fuse_encodings(self, encodings: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse multiple encodings into one."""
        if not encodings:
            return np.zeros(self.encoding_dim)
        
        # Weighted sum with attention-like mechanism
        fused = np.zeros(self.encoding_dim)
        weights = []
        
        for name, enc in encodings.items():
            # Weight by encoding magnitude
            weight = np.linalg.norm(enc) + 0.1
            weights.append((name, weight))
            fused += enc * weight
        
        # Normalize
        total_weight = sum(w for _, w in weights)
        if total_weight > 0:
            fused /= total_weight
        
        return fused
    
    def _detect_patterns(self, fused: np.ndarray, 
                        sources: List[str]) -> List[SynthesizedPattern]:
        """Detect patterns in fused representation."""
        patterns = []
        
        # Match against templates
        for name, template in self.templates.items():
            similarity = np.dot(fused, template) / (
                np.linalg.norm(fused) * np.linalg.norm(template) + 1e-8
            )
            
            if similarity > 0.3:
                action = self._get_action_for_pattern(name, similarity)
                patterns.append(SynthesizedPattern(
                    name=name,
                    components=sources,
                    strength=float(similarity),
                    dimensionality=len(sources),
                    novelty_score=1 - similarity,  # More different = more novel
                    actionable=similarity > 0.5,
                    suggested_action=action
                ))
        
        # Check for novel patterns
        novel = self._detect_novel_patterns(fused, sources)
        patterns.extend(novel)
        
        return sorted(patterns, key=lambda p: p.strength, reverse=True)
    
    def _detect_novel_patterns(self, fused: np.ndarray, 
                              sources: List[str]) -> List[SynthesizedPattern]:
        """Detect novel patterns not matching templates."""
        # Compare to pattern memory
        for name, stored in self.pattern_memory.items():
            similarity = np.dot(fused, stored) / (
                np.linalg.norm(fused) * np.linalg.norm(stored) + 1e-8
            )
            
            if similarity < 0.3:  # Very different from any known
                return [SynthesizedPattern(
                    name='NOVEL_PATTERN',
                    components=sources,
                    strength=0.5,
                    dimensionality=len(sources),
                    novelty_score=1 - similarity,
                    actionable=False,
                    suggested_action='OBSERVE'
                )]
        
        # Store for future comparison
        key = '_'.join(sorted(sources))
        self.pattern_memory[key] = fused.copy()
        
        return []
    
    def _calculate_coherence(self, encodings: Dict[str, np.ndarray]) -> float:
        """Calculate coherence between streams."""
        if len(encodings) < 2:
            return 1.0
        
        similarities = []
        enc_list = list(encodings.values())
        
        for i in range(len(enc_list)):
            for j in range(i + 1, len(enc_list)):
                sim = np.dot(enc_list[i], enc_list[j]) / (
                    np.linalg.norm(enc_list[i]) * np.linalg.norm(enc_list[j]) + 1e-8
                )
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    def _get_action_for_pattern(self, pattern_name: str, strength: float) -> Optional[str]:
        """Get suggested action for a detected pattern."""
        actions = {
            'TREND_IMPULSE': 'FOLLOW_TREND',
            'REVERSAL_PATTERN': 'PREPARE_REVERSAL',
            'CONSOLIDATION': 'WAIT',
            'BREAKOUT': 'ENTER_BREAKOUT',
            'EXHAUSTION': 'TAKE_PROFIT',
        }
        
        if strength > 0.5:
            return actions.get(pattern_name)
        return None
