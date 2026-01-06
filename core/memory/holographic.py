"""
AGI Ultra: Holographic Associative Memory (HAM) - Massive Scale

Implements Hyperdimensional Computing (HDC) with:
- 65536 dimensions for ultra-high resolution pattern matching
- Temporal hierarchy (recent, medium, long-term memory)
- FAISS integration for fast vector search (billions of patterns)
- Multi-dimensional indexing (patterns, trends, sessions, news)
- Adaptive compression for memory efficiency
"""

import numpy as np
import logging
import os
import pickle
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger("HolographicMemory")

# Try importing FAISS - fall back gracefully if not available
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available - ultra-fast vector search enabled")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using numpy fallback for vector search")


@dataclass
class MemoryImpulse:
    """A single 'thought' or state to be encoded."""
    timestamp: float
    vector: np.ndarray
    metadata: Dict[str, Any]
    outcome: Optional[float] = None
    category: str = "general"  # pattern, trend, session, news


@dataclass
class TemporalMemoryLevel:
    """A single level in the temporal memory hierarchy."""
    name: str
    max_size: int
    decay_rate: float
    plate: np.ndarray
    impulses: deque = field(default_factory=deque)
    access_count: int = 0
    last_access: float = 0.0


class HolographicPlateUltra:
    """
    AGI Ultra: Holographic Associative Memory with massive scale.
    
    Implements Hyperdimensional Computing (HDC) principles:
    1. Orthogonality: Random high-dim vectors are nearly orthogonal
    2. Superposition: A + B = C (C contains both A and B)
    3. Binding: A * B = D (D is unique to the pair A-B)
    
    AGI Ultra Expansions:
    - 65536 dimensions for ultra-resolution
    - Temporal hierarchy for time-aware memory
    - FAISS index for O(log n) nearest neighbor search
    - Multi-index for categorical memory organization
    """
    
    def __init__(self, dimensions: int = 65536):
        """
        Initialize the ultra-scale holographic plate.
        
        Args:
            dimensions: Vector dimensionality (default: 65536 for AGI Ultra)
        """
        self.d = dimensions
        
        # Main holographic plate (superposition of all experiences)
        self.memory_plate = np.zeros(self.d, dtype=np.float32)
        
        # Concept space: semantic vectors for concepts
        self.concept_space: Dict[str, np.ndarray] = {}
        
        # AGI Ultra: Temporal hierarchy
        self.temporal_levels = {
            'recent': TemporalMemoryLevel(
                name='recent',
                max_size=10000,
                decay_rate=0.99,
                plate=np.zeros(self.d, dtype=np.float32)
            ),
            'medium': TemporalMemoryLevel(
                name='medium',
                max_size=100000,
                decay_rate=0.999,
                plate=np.zeros(self.d, dtype=np.float32)
            ),
            'long_term': TemporalMemoryLevel(
                name='long_term',
                max_size=1000000,
                decay_rate=0.9999,
                plate=np.zeros(self.d, dtype=np.float32)
            )
        }
        
        # AGI Ultra: Multi-dimensional indices
        self.category_plates: Dict[str, np.ndarray] = {
            'pattern': np.zeros(self.d, dtype=np.float32),
            'trend': np.zeros(self.d, dtype=np.float32),
            'session': np.zeros(self.d, dtype=np.float32),
            'news': np.zeros(self.d, dtype=np.float32),
        }
        
        # AGI Ultra: FAISS index for nearest neighbor search
        self.faiss_index = None
        self.faiss_vectors: List[np.ndarray] = []
        self.faiss_metadata: List[Dict[str, Any]] = []
        self._init_faiss_index()
        
        # Statistics
        self.total_experiences = 0
        self.total_queries = 0
        
        logger.info(f"HolographicPlateUltra initialized: {dimensions} dimensions, FAISS={FAISS_AVAILABLE}")
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search."""
        if FAISS_AVAILABLE:
            # Using IVF index for efficient search in large datasets
            # nlist = number of clusters
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.d)  # Inner product for cosine similarity
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.nprobe = 10  # Search 10 clusters
        else:
            self.faiss_index = None
    
    def _get_concept_vector(self, concept: str) -> np.ndarray:
        """Returns (or creates) a static random vector for a concept."""
        if concept not in self.concept_space:
            rng = np.random.default_rng(seed=abs(hash(concept)) % (2**31))
            self.concept_space[concept] = rng.choice([-1, 1], size=self.d).astype(np.float32)
        return self.concept_space[concept]
    
    def encode_state(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Encodes a dictionary of market data into a single hypervector.
        Uses "Role-Filler" binding with improved encoding.
        """
        state_vector = np.zeros(self.d, dtype=np.float32)
        
        for key, value in market_state.items():
            if value is None:
                continue
                
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    continue
                    
                role_vec = self._get_concept_vector(f"Role_{key}")
                
                # Improved normalization with sigmoid for better gradient
                norm_val = 2.0 / (1.0 + np.exp(-value * 0.1)) - 1.0
                encoded_feature = role_vec * norm_val
                state_vector += encoded_feature
                
            elif isinstance(value, str):
                role_vec = self._get_concept_vector(f"Role_{key}")
                val_vec = self._get_concept_vector(f"Val_{value}")
                bound_vec = role_vec * val_vec
                state_vector += bound_vec
                
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Handle arrays by encoding mean and std
                try:
                    arr = np.array(value, dtype=float)
                    role_vec = self._get_concept_vector(f"Role_{key}")
                    mean_val = 2.0 / (1.0 + np.exp(-np.mean(arr) * 0.1)) - 1.0
                    state_vector += role_vec * mean_val
                except (ValueError, TypeError):
                    pass
        
        # L2 normalize
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        return state_vector
    
    def learn(
        self,
        state_vector: np.ndarray,
        outcome_score: float,
        category: str = "general",
        temporal_level: str = "recent",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        AGI Ultra: Absorbs a new experience into multiple memory structures.
        
        Args:
            state_vector: Encoded market state
            outcome_score: +1.0 (profit) to -1.0 (loss)
            category: Memory category (pattern, trend, session, news)
            temporal_level: Which temporal level to prioritize
            metadata: Additional metadata to store
        """
        experience = state_vector * outcome_score
        
        # 1. Update main plate
        self.memory_plate = (self.memory_plate * 0.999) + experience
        
        # 2. Update temporal hierarchy
        if temporal_level in self.temporal_levels:
            level = self.temporal_levels[temporal_level]
            level.plate = (level.plate * level.decay_rate) + experience
            level.impulses.append(MemoryImpulse(
                timestamp=time.time(),
                vector=state_vector.copy(),
                metadata=metadata or {},
                outcome=outcome_score,
                category=category
            ))
            # Maintain size limit
            while len(level.impulses) > level.max_size:
                level.impulses.popleft()
        
        # 3. Update category-specific plate
        if category in self.category_plates:
            self.category_plates[category] = (
                self.category_plates[category] * 0.999
            ) + experience
        
        # 4. Add to FAISS index
        self._add_to_faiss(state_vector, outcome_score, metadata)
        
        self.total_experiences += 1
    
    def _add_to_faiss(self, vector: np.ndarray, outcome: float, metadata: Optional[Dict] = None):
        """Add vector to FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        # Normalize for cosine similarity
        normalized = vector / (np.linalg.norm(vector) + 1e-8)
        normalized = normalized.reshape(1, -1).astype(np.float32)
        
        self.faiss_vectors.append(vector)
        self.faiss_metadata.append({
            'outcome': outcome,
            'timestamp': time.time(),
            **(metadata or {})
        })
        
        # Train and add when we have enough vectors
        if len(self.faiss_vectors) >= 1000 and not self.faiss_index.is_trained:
            training_data = np.vstack([v / (np.linalg.norm(v) + 1e-8) for v in self.faiss_vectors[-10000:]]).astype(np.float32)
            self.faiss_index.train(training_data)
            logger.info("FAISS index trained")
        
        if self.faiss_index.is_trained:
            self.faiss_index.add(normalized)
    
    def intuit(
        self,
        current_state_vector: np.ndarray,
        temporal_weight: Optional[Dict[str, float]] = None,
        category_weight: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        AGI Ultra: Query the hologram with multi-level intuition.
        
        Returns comprehensive intuition from all memory structures.
        """
        self.total_queries += 1
        
        # Default weights
        if temporal_weight is None:
            temporal_weight = {'recent': 0.5, 'medium': 0.3, 'long_term': 0.2}
        if category_weight is None:
            category_weight = {'pattern': 0.3, 'trend': 0.3, 'session': 0.2, 'news': 0.2}
        
        # Normalize query vector
        state_norm = np.linalg.norm(current_state_vector)
        if state_norm == 0:
            return {'main': 0.0, 'temporal': {}, 'categorical': {}, 'combined': 0.0}
        
        normalized_query = current_state_vector / state_norm
        
        # 1. Main plate intuition
        plate_norm = np.linalg.norm(self.memory_plate)
        if plate_norm > 0:
            main_intuition = np.dot(normalized_query, self.memory_plate) / plate_norm
            main_intuition *= np.sqrt(self.d)
        else:
            main_intuition = 0.0
        
        # 2. Temporal level intuitions
        temporal_intuitions = {}
        for level_name, level in self.temporal_levels.items():
            level_norm = np.linalg.norm(level.plate)
            if level_norm > 0:
                intuition = np.dot(normalized_query, level.plate) / level_norm * np.sqrt(self.d)
                level.access_count += 1
                level.last_access = time.time()
            else:
                intuition = 0.0
            temporal_intuitions[level_name] = float(intuition)
        
        # 3. Categorical intuitions
        categorical_intuitions = {}
        for cat_name, cat_plate in self.category_plates.items():
            cat_norm = np.linalg.norm(cat_plate)
            if cat_norm > 0:
                intuition = np.dot(normalized_query, cat_plate) / cat_norm * np.sqrt(self.d)
            else:
                intuition = 0.0
            categorical_intuitions[cat_name] = float(intuition)
        
        # 4. Weighted combination
        temporal_combined = sum(
            temporal_intuitions.get(k, 0) * temporal_weight.get(k, 0)
            for k in temporal_weight
        )
        categorical_combined = sum(
            categorical_intuitions.get(k, 0) * category_weight.get(k, 0)
            for k in category_weight
        )
        
        combined = 0.4 * float(main_intuition) + 0.3 * temporal_combined + 0.3 * categorical_combined
        
        return {
            'main': float(main_intuition),
            'temporal': temporal_intuitions,
            'categorical': categorical_intuitions,
            'combined': combined
        }
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        AGI Ultra: Fast similarity search using FAISS.
        
        Returns:
            List of (similarity_score, metadata) tuples
        """
        if not FAISS_AVAILABLE or self.faiss_index is None or not self.faiss_index.is_trained:
            return self._numpy_search_similar(query_vector, top_k)
        
        # Normalize query
        normalized = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        normalized = normalized.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(top_k, self.faiss_index.ntotal)
        if k == 0:
            return []
        
        distances, indices = self.faiss_index.search(normalized, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.faiss_metadata):
                results.append((float(dist), self.faiss_metadata[idx]))
        
        return results
    
    def _numpy_search_similar(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Fallback similarity search using numpy."""
        if not self.faiss_vectors:
            return []
        
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        
        normalized_query = query_vector / query_norm
        
        similarities = []
        for i, vec in enumerate(self.faiss_vectors):
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                sim = np.dot(normalized_query, vec) / vec_norm
                similarities.append((float(sim), i))
        
        similarities.sort(reverse=True)
        
        return [
            (sim, self.faiss_metadata[idx])
            for sim, idx in similarities[:top_k]
            if idx < len(self.faiss_metadata)
        ]
    
    def consolidate_temporal_memory(self):
        """
        AGI Ultra: Move memories between temporal levels based on age.
        
        Recent -> Medium (after 1 hour)
        Medium -> Long-term (after 1 day)
        """
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        
        # Move from recent to medium
        recent = self.temporal_levels['recent']
        medium = self.temporal_levels['medium']
        
        while recent.impulses and recent.impulses[0].timestamp < hour_ago:
            impulse = recent.impulses.popleft()
            medium.impulses.append(impulse)
            if impulse.outcome:
                experience = impulse.vector * impulse.outcome
                medium.plate = (medium.plate * medium.decay_rate) + experience
        
        # Move from medium to long-term
        long_term = self.temporal_levels['long_term']
        
        while medium.impulses and medium.impulses[0].timestamp < day_ago:
            impulse = medium.impulses.popleft()
            long_term.impulses.append(impulse)
            if impulse.outcome:
                experience = impulse.vector * impulse.outcome
                long_term.plate = (long_term.plate * long_term.decay_rate) + experience
        
        logger.debug(f"Memory consolidation: recent={len(recent.impulses)}, "
                     f"medium={len(medium.impulses)}, long_term={len(long_term.impulses)}")


class HolographicMemory:
    """
    AGI Ultra: Wrapper for the Holographic Memory System.
    
    Provides simple interface for storing experiences and retrieving intuitions.
    """
    
    def __init__(
        self,
        persistence_file: str = "brain/holographic_plate.npy",
        dimensions: int = 65536  # AGI Ultra: 65536 dimensions
    ):
        self.persistence_file = persistence_file
        self.dimensions = dimensions
        self.plate = HolographicPlateUltra(dimensions=dimensions)
        self.load_memory()
        
        logger.info(f"HolographicMemory initialized: {dimensions} dimensions")
    
    def store_experience(
        self,
        context: Dict,
        outcome: float,
        category: str = "general",
        temporal_level: str = "recent"
    ):
        """Store a market experience with outcome."""
        vec = self.plate.encode_state(context)
        self.plate.learn(vec, outcome, category=category, temporal_level=temporal_level, metadata=context)
    
    def retrieve_intuition(self, context: Dict) -> float:
        """
        Retrieve intuition for a context.
        
        Returns combined intuition score (legacy interface).
        """
        vec = self.plate.encode_state(context)
        result = self.plate.intuit(vec)
        return result['combined']
    
    def retrieve_full_intuition(self, context: Dict) -> Dict[str, Any]:
        """
        AGI Ultra: Retrieve full multi-level intuition.
        """
        vec = self.plate.encode_state(context)
        return self.plate.intuit(vec)
    
    def search_similar_experiences(self, context: Dict, top_k: int = 10) -> List[Tuple[float, Dict]]:
        """
        AGI Ultra: Search for similar historical experiences.
        """
        vec = self.plate.encode_state(context)
        return self.plate.search_similar(vec, top_k)
    
    def consolidate(self):
        """Consolidate temporal memories."""
        self.plate.consolidate_temporal_memory()
    
    def save_memory(self):
        """Persists the Holographic Memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.persistence_file) or '.', exist_ok=True)
            
            # Save main data
            save_data = {
                'dimensions': self.dimensions,
                'memory_plate': self.plate.memory_plate,
                'temporal_plates': {
                    name: {
                        'plate': level.plate,
                        'decay_rate': level.decay_rate,
                        'access_count': level.access_count
                    }
                    for name, level in self.plate.temporal_levels.items()
                },
                'category_plates': self.plate.category_plates,
                'total_experiences': self.plate.total_experiences,
                'total_queries': self.plate.total_queries
            }
            
            with open(self.persistence_file.replace('.npy', '.pkl'), 'wb') as f:
                pickle.dump(save_data, f)
            
            # Also save main plate as numpy for compatibility
            np.save(self.persistence_file, self.plate.memory_plate)
            
            logger.info(f"HOLOGRAPHIC MEMORY SAVED ({self.plate.total_experiences} experiences)")
        except Exception as e:
            logger.error(f"Failed to Save Hologram: {e}")
    
    def load_memory(self):
        """Loads the Holographic Memory."""
        pkl_file = self.persistence_file.replace('.npy', '.pkl')
        
        # Try loading full state first
        if os.path.exists(pkl_file):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                if data.get('dimensions') == self.dimensions:
                    self.plate.memory_plate = data['memory_plate']
                    
                    for name, level_data in data.get('temporal_plates', {}).items():
                        if name in self.plate.temporal_levels:
                            self.plate.temporal_levels[name].plate = level_data['plate']
                            self.plate.temporal_levels[name].access_count = level_data.get('access_count', 0)
                    
                    self.plate.category_plates = data.get('category_plates', self.plate.category_plates)
                    self.plate.total_experiences = data.get('total_experiences', 0)
                    self.plate.total_queries = data.get('total_queries', 0)
                    
                    logger.info(f"HOLOGRAPHIC MEMORY LOADED ({self.plate.total_experiences} experiences)")
                    return
            except Exception as e:
                logger.warning(f"Failed to load full state: {e}")
        
        # Fallback to numpy file
        if os.path.exists(self.persistence_file):
            try:
                loaded_plate = np.load(self.persistence_file)
                if loaded_plate.shape[0] == self.dimensions:
                    self.plate.memory_plate = loaded_plate
                    logger.info(f"HOLOGRAPHIC MEMORY LOADED from {self.persistence_file}")
                else:
                    logger.warning(f"Dimension mismatch: expected {self.dimensions}, got {loaded_plate.shape[0]}")
            except Exception as e:
                logger.error(f"Failed to Load Hologram: {e}")
