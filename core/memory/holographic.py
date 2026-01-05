
import numpy as np
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("HolographicMemory")

@dataclass
class MemoryImpulse:
    """A single 'thought' or state to be encoded."""
    timestamp: float
    vector: np.ndarray
    metadata: Dict[str, Any]

class HolographicPlate:
    """
    Phase 117: Holographic Associative Memory (HAM).
    
    Implements Hyperdimensional Computing (HDC) principles to store infinite
    market states in a fixed-size superposition vector.
    
    Principles:
    1. Orthogonality: Random high-dim vectors are nearly orthogonal.
    2. Superposition: A + B = C (C contains both A and B).
    3. Binding: A * B = D (D is unique to the pair A-B).
    
    "The Universe is a Hologram."
    """
    
    def __init__(self, dimensions: int = 10000):
        self.d = dimensions
        # The "Plate" is the sum of all experiences
        self.memory_plate = np.zeros(self.d)
        
        # Item Memory: Fixed random vectors for concepts (e.g., "Bullish", "Bearish", "HighVol")
        self.concept_space = {} 
        
        # Decay factor (forgetting old irrelevant memories)
        self.decay = 0.999 
        
    def _get_concept_vector(self, concept: str) -> np.ndarray:
        """Returns (or creates) a static random vector for a concept."""
        if concept not in self.concept_space:
            # Create a bipolar vector {-1, 1} for better mathematical properties
            rng = np.random.default_rng(seed=abs(hash(concept)))
            self.concept_space[concept] = rng.choice([-1, 1], size=self.d)
        return self.concept_space[concept]
    
    def encode_state(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Encodes a dictionary of market data into a single hypervector.
        Uses "Role-Filler" binding.
        Vector = (Role_Price * Val_Price) + (Role_Vol * Val_Vol) ...
        """
        state_vector = np.zeros(self.d)
        
        for key, value in market_state.items():
            if isinstance(value, (int, float)):
                # Continuous Value Encoding
                # We normalize value to reasonable range and scale a "Base" vector?
                # Or simpler: Role Vector * Scalar Value.
                # Scalar multiplication in HDC is... tricky. 
                # Better: Binning. "Price_High", "Price_Low". 
                # OR: Linear projection.
                
                # Method: Role * Value (Projection)
                role_vec = self._get_concept_vector(f"Role_{key}")
                
                # Normalize scalar roughly to -1 to 1 range (tanh)
                # We assume standard scaling was done elsewhere, doing simple tanh here
                norm_val = np.tanh(value) 
                
                # Bind Role to Value
                # In this continuous scheme, we just scale the vector.
                encoded_feature = role_vec * norm_val 
                
                # Superposition (Add to state)
                state_vector += encoded_feature
                
            elif isinstance(value, str):
                # Discrete Value Encoding
                # Bind Role * ValueVector
                role_vec = self._get_concept_vector(f"Role_{key}")
                val_vec = self._get_concept_vector(f"Val_{value}")
                
                # Binding Operation (Element-wise XOR for binary, Mul for real)
                # Since we use {-1, 1}, element-wise multiplication is XOR equivalent.
                bound_vec = role_vec * val_vec
                
                state_vector += bound_vec
                
        # Normalize the result to keep values constrained
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        return state_vector

    def learn(self, state_vector: np.ndarray, outcome_score: float):
        """
        Absorbs a new experience into the Hologram.
        Outcome Score: +1.0 (Profit), -1.0 (Loss), 0.0 (Neutral).
        
        We store: State * Outcome.
        So if we query State later, we retrieve Outcome.
        """
        # Encode Outcome as a scalar weight on the state
        # Experience = State * Outcome
        experience = state_vector * outcome_score
        
        # Add to Plate (Superposition)
        # Apply decay to existing memory first (Time-weighted)
        self.memory_plate = (self.memory_plate * self.decay) + experience
        
    def intuit(self, current_state_vector: np.ndarray) -> float:
        """
        Queries the Hologram. "How did this turn out in the past?"
        
        Operation: Dot Product (Similarity) between Current State and Plate.
        Since Plate = Sum(State_i * Outcome_i),
        Dot(Current, Plate) = Sum(Dot(Current, State_i) * Outcome_i)
        
        If Current is similar to State_i, Dot is high -> We retrieve Outcome_i.
        """
        # Resonance (Cosine Similarity-ish)
        # We don't strictly need to normalize plate, but it helps stability
        resonance = np.dot(current_state_vector, self.memory_plate)
        
        # Scale resonance to valid -1 to 1 range approx?
        # Actually proper cosine similarity is better:
        plate_norm = np.linalg.norm(self.memory_plate)
        state_norm = np.linalg.norm(current_state_vector)
        
        if plate_norm == 0 or state_norm == 0:
            return 0.0
            
        similarity = resonance / (plate_norm * state_norm)
        
        # Amplification: HDC similarities are usually small (~0.0 for orthogonal).
        # A similarity of 0.05 is HUGE in high dimensions if N is large.
        # We amplify relevant signals.
        
        # Heuristic amplification for decision making
        intuition_strength = similarity * np.sqrt(self.d) # Scale by sqrt(Dimensions)
        
        return float(intuition_strength)

class HolographicMemory:
    """Wrapper for the System."""
    def __init__(self, persistence_file="brain/holographic_plate.npy"):
        self.persistence_file = persistence_file
        self.plate = HolographicPlate(dimensions=4096) 
        self.load_memory()
        
    def store_experience(self, context: Dict, outcome: float):
        vec = self.plate.encode_state(context)
        self.plate.learn(vec, outcome)
        
    def retrieve_intuition(self, context: Dict) -> float:
        vec = self.plate.encode_state(context)
        return self.plate.intuit(vec)

    def save_memory(self):
        """Persists the Holographic Plate to disk."""
        try:
             os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
             np.save(self.persistence_file, self.plate.memory_plate)
             # We should also save the concept space if it's random but static?
             # For now, concept space is rebuilt deterministically from seed (hash(concept)).
             # So only the Plate needs saving.
             logger.info(f"HOLOGRAPHIC MEMORY SAVED to {self.persistence_file}")
        except Exception as e:
             logger.error(f"Failed to Save Hologram: {e}")

    def load_memory(self):
        """Loads the Holographic Plate."""
        if os.path.exists(self.persistence_file):
             try:
                 self.plate.memory_plate = np.load(self.persistence_file)
                 logger.info(f"HOLOGRAPHIC MEMORY LOADED from {self.persistence_file}")
             except Exception as e:
                 logger.error(f"Failed to Load Hologram: {e}")

