
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("Abstraction")

class AbstractPatternSynthesizer:
    """
    System 5: Abstract Pattern Synthesizer.
    Identifies "Isomorphisms" in market structure.
    "It's not exactly a Head and Shoulders, but it rhymes with one."
    """
    def __init__(self):
        # Known recursive definitions
        self.definitions = {
            'ACCUMULATION': ['D', 'F', 'U', 'F', 'U', 'U'], # Down, Flat, Up, Flat, Up, Up (Simplified)
            'DISTRIBUTION': ['U', 'F', 'D', 'F', 'D', 'D'],
            'COMPRESSION': ['F', 'F', 'F', 'F'] # Tight consolidation
        }
        
    def _discretize_move(self, change: float, threshold: float) -> str:
        if change > threshold: return 'U' # Up
        if change < -threshold: return 'D' # Down
        return 'F' # Flat
        
    def synthesize_signature(self, prices: List[float], window: int = 5) -> str:
        """
        Converts array of prices into a string DNA sequences.
        e.g. [10, 11, 11, 12, 11] -> "U F U D"
        """
        if len(prices) < window: return ""
        
        # Calculate changes
        arr = np.array(prices)
        changes = np.diff(arr)
        threshold = np.mean(np.abs(changes)) * 0.5 # Dynamic threshold
        
        signature = []
        for c in changes[-window:]:
            signature.append(self._discretize_move(c, threshold))
            
        return "".join(signature)
        
    def find_structural_similarity(self, current_sig: str) -> Tuple[str, float]:
        """
        Matches current signature against definitions using Levenshtein distance (Fuzzy Match).
        """
        best_match = "NO_PATTERN"
        best_score = 0.0
        
        for name, pattern_list in self.definitions.items():
            pattern_sig = "".join(pattern_list)
            
            # Simple Substring match for MVP
            # In full version, use Sequence Alignment Algo
            if pattern_sig in current_sig or current_sig in pattern_sig:
                # Crude "contains" check
                score = min(len(current_sig), len(pattern_sig)) / max(len(current_sig), len(pattern_sig))
                if score > best_score:
                    best_score = score
                    best_match = name
                    
        # Check repetitive patterns (Compression)
        # If we have alternating U/D without major direction, it's also compression
        if current_sig.count('F') / len(current_sig) > 0.6:
             return "COMPRESSION", 0.9
             
        if current_sig.count('F') / len(current_sig) > 0.6:
             return "COMPRESSION", 0.9
             
        # Alternating Tight: UDUDU for 5+ OR Chop (UDUUD)
        # If U and D count are roughly equal (within 1) and length > 4 -> Compression/Chop
        u_count = current_sig.count('U')
        d_count = current_sig.count('D')
        
        if len(current_sig) >= 4 and abs(u_count - d_count) <= 1:
             return "COMPRESSION", 0.8 # Chop is a form of compression
             
        return best_match, best_score
