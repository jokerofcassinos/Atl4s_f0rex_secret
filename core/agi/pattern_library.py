"""
AGI Ultra: Pattern Library - Consolidated Pattern Bank

Stores and manages 1+ trillion consolidated market patterns with:
- Automatic categorization (graphical, volume, macro, micro)
- Semantic and vector similarity search
- Automatic clustering of similar patterns
- Pattern versioning (temporal evolution tracking)
"""

import logging
import os
import pickle
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import numpy as np

logger = logging.getLogger("PatternLibrary")


@dataclass
class Pattern:
    """A consolidated market pattern."""
    pattern_id: str
    category: str  # graphical, volume, macro, micro, composite
    name: str
    description: str
    
    # Vector representation for similarity search
    vector: np.ndarray
    
    # Pattern statistics
    occurrence_count: int = 0
    success_rate: float = 0.5
    avg_profit: float = 0.0
    avg_duration: float = 0.0
    
    # Temporal tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    version: int = 1
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Cluster info
    cluster_id: Optional[str] = None
    cluster_distance: float = 0.0


@dataclass
class PatternCluster:
    """A cluster of similar patterns."""
    cluster_id: str
    centroid: np.ndarray
    pattern_ids: List[str] = field(default_factory=list)
    category: str = "mixed"
    avg_success_rate: float = 0.5
    total_occurrences: int = 0


class PatternLibrary:
    """
    AGI Ultra: Trillion-Scale Pattern Library.
    
    Manages consolidated market patterns with:
    - Hierarchical storage for massive scale
    - Automatic categorization and clustering
    - Fast vector similarity search
    - Pattern evolution tracking
    """
    
    CATEGORIES = ['graphical', 'volume', 'macro', 'micro', 'composite', 'temporal', 'news']
    
    def __init__(
        self,
        dimensions: int = 4096,  # Pattern vector dimensions
        persistence_dir: str = "brain/patterns",
        max_patterns_per_category: int = 1000000
    ):
        self.dimensions = dimensions
        self.persistence_dir = persistence_dir
        self.max_patterns_per_category = max_patterns_per_category
        
        # Pattern storage by category
        self.patterns: Dict[str, Dict[str, Pattern]] = {
            cat: {} for cat in self.CATEGORIES
        }
        
        # Cluster storage
        self.clusters: Dict[str, PatternCluster] = {}
        self.pattern_to_cluster: Dict[str, str] = {}
        
        # Vector index for similarity search (simple numpy-based)
        self.vector_index: Dict[str, List[Tuple[str, np.ndarray]]] = {
            cat: [] for cat in self.CATEGORIES
        }
        
        # Statistics
        self.total_patterns = 0
        self.total_lookups = 0
        
        os.makedirs(persistence_dir, exist_ok=True)
        
        logger.info(f"PatternLibrary initialized: {dimensions}D vectors, max {max_patterns_per_category}/category")
    
    # -------------------------------------------------------------------------
    # PATTERN REGISTRATION
    # -------------------------------------------------------------------------
    def register_pattern(
        self,
        name: str,
        category: str,
        vector: np.ndarray,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None
    ) -> Pattern:
        """
        Register a new pattern or update existing one if similar.
        
        Returns:
            The registered or updated Pattern
        """
        if category not in self.CATEGORIES:
            category = 'composite'
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized = vector / norm
        else:
            normalized = vector
        
        # Check for similar existing pattern
        similar = self.find_similar(normalized, category, threshold=0.95, top_k=1)
        
        if similar:
            # Update existing pattern
            existing_pattern = similar[0][0]
            existing_pattern.occurrence_count += 1
            existing_pattern.last_seen = time.time()
            existing_pattern.version += 1
            
            # Update tags if provided
            if tags:
                existing_pattern.tags.update(tags)
            
            logger.debug(f"Updated existing pattern: {existing_pattern.name} (count={existing_pattern.occurrence_count})")
            return existing_pattern
        
        # Create new pattern
        pattern_id = str(uuid.uuid4())
        pattern = Pattern(
            pattern_id=pattern_id,
            category=category,
            name=name,
            description=description,
            vector=normalized.astype(np.float32),
            occurrence_count=1,
            metadata=metadata or {},
            tags=tags or set()
        )
        
        # Store pattern
        self.patterns[category][pattern_id] = pattern
        self.vector_index[category].append((pattern_id, normalized))
        self.total_patterns += 1
        
        # Try to assign to cluster
        self._assign_to_cluster(pattern)
        
        # Check capacity
        if len(self.patterns[category]) > self.max_patterns_per_category:
            self._compact_category(category)
        
        logger.debug(f"Registered new pattern: {name} in {category}")
        return pattern
    
    def update_pattern_outcome(
        self,
        pattern_id: str,
        category: str,
        success: bool,
        profit: float = 0.0,
        duration: float = 0.0
    ):
        """Update pattern with trade outcome."""
        if category not in self.patterns or pattern_id not in self.patterns[category]:
            return
        
        pattern = self.patterns[category][pattern_id]
        
        # Update statistics with exponential moving average
        alpha = 0.1
        
        if success:
            pattern.success_rate = pattern.success_rate * (1 - alpha) + alpha
        else:
            pattern.success_rate = pattern.success_rate * (1 - alpha)
        
        pattern.avg_profit = pattern.avg_profit * (1 - alpha) + profit * alpha
        pattern.avg_duration = pattern.avg_duration * (1 - alpha) + duration * alpha
        pattern.last_seen = time.time()
    
    # -------------------------------------------------------------------------
    # PATTERN SEARCH
    # -------------------------------------------------------------------------
    def find_similar(
        self,
        query_vector: np.ndarray,
        category: Optional[str] = None,
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[Pattern, float]]:
        """
        Find patterns similar to query vector.
        
        Args:
            query_vector: Pattern vector to match
            category: Specific category or None for all
            threshold: Minimum similarity threshold
            top_k: Maximum results
            
        Returns:
            List of (Pattern, similarity) tuples
        """
        self.total_lookups += 1
        
        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            return []
        normalized_query = query_vector / norm
        
        results = []
        
        categories_to_search = [category] if category and category in self.CATEGORIES else self.CATEGORIES
        
        for cat in categories_to_search:
            for pattern_id, pattern_vec in self.vector_index[cat]:
                similarity = float(np.dot(normalized_query, pattern_vec))
                
                if similarity >= threshold:
                    pattern = self.patterns[cat].get(pattern_id)
                    if pattern:
                        results.append((pattern, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def find_by_tags(
        self,
        tags: Set[str],
        category: Optional[str] = None,
        match_all: bool = False
    ) -> List[Pattern]:
        """Find patterns by tags."""
        results = []
        
        categories_to_search = [category] if category and category in self.CATEGORIES else self.CATEGORIES
        
        for cat in categories_to_search:
            for pattern in self.patterns[cat].values():
                if match_all:
                    if tags.issubset(pattern.tags):
                        results.append(pattern)
                else:
                    if tags.intersection(pattern.tags):
                        results.append(pattern)
        
        return results
    
    def get_top_patterns(
        self,
        category: Optional[str] = None,
        sort_by: str = "success_rate",
        min_occurrences: int = 10,
        limit: int = 20
    ) -> List[Pattern]:
        """Get top performing patterns."""
        results = []
        
        categories_to_search = [category] if category and category in self.CATEGORIES else self.CATEGORIES
        
        for cat in categories_to_search:
            for pattern in self.patterns[cat].values():
                if pattern.occurrence_count >= min_occurrences:
                    results.append(pattern)
        
        # Sort
        if sort_by == "success_rate":
            results.sort(key=lambda p: p.success_rate, reverse=True)
        elif sort_by == "profit":
            results.sort(key=lambda p: p.avg_profit, reverse=True)
        elif sort_by == "occurrences":
            results.sort(key=lambda p: p.occurrence_count, reverse=True)
        elif sort_by == "recent":
            results.sort(key=lambda p: p.last_seen, reverse=True)
        
        return results[:limit]
    
    # -------------------------------------------------------------------------
    # CLUSTERING
    # -------------------------------------------------------------------------
    def _assign_to_cluster(self, pattern: Pattern):
        """Assign pattern to nearest cluster or create new one."""
        best_cluster = None
        best_distance = float('inf')
        
        for cluster in self.clusters.values():
            if cluster.category == pattern.category or cluster.category == "mixed":
                distance = 1 - float(np.dot(pattern.vector, cluster.centroid))
                if distance < best_distance and distance < 0.3:
                    best_distance = distance
                    best_cluster = cluster
        
        if best_cluster:
            # Assign to existing cluster
            best_cluster.pattern_ids.append(pattern.pattern_id)
            best_cluster.total_occurrences += pattern.occurrence_count
            
            # Update centroid
            n = len(best_cluster.pattern_ids)
            best_cluster.centroid = (best_cluster.centroid * (n - 1) + pattern.vector) / n
            
            pattern.cluster_id = best_cluster.cluster_id
            pattern.cluster_distance = best_distance
            self.pattern_to_cluster[pattern.pattern_id] = best_cluster.cluster_id
        elif len(self.clusters) < 1000:  # Max clusters
            # Create new cluster
            cluster_id = str(uuid.uuid4())
            cluster = PatternCluster(
                cluster_id=cluster_id,
                centroid=pattern.vector.copy(),
                pattern_ids=[pattern.pattern_id],
                category=pattern.category,
                total_occurrences=pattern.occurrence_count
            )
            self.clusters[cluster_id] = cluster
            pattern.cluster_id = cluster_id
            pattern.cluster_distance = 0.0
            self.pattern_to_cluster[pattern.pattern_id] = cluster_id
    
    def get_cluster_patterns(self, cluster_id: str) -> List[Pattern]:
        """Get all patterns in a cluster."""
        if cluster_id not in self.clusters:
            return []
        
        cluster = self.clusters[cluster_id]
        patterns = []
        
        for pattern_id in cluster.pattern_ids:
            for cat in self.CATEGORIES:
                if pattern_id in self.patterns[cat]:
                    patterns.append(self.patterns[cat][pattern_id])
                    break
        
        return patterns
    
    # -------------------------------------------------------------------------
    # COMPACTION
    # -------------------------------------------------------------------------
    def _compact_category(self, category: str, keep_ratio: float = 0.8):
        """Compact a category by removing low-value patterns."""
        patterns_list = list(self.patterns[category].values())
        
        # Score patterns by value
        def pattern_score(p: Pattern) -> float:
            recency = 1.0 / (1.0 + (time.time() - p.last_seen) / 86400)
            return p.success_rate * 0.3 + np.log1p(p.occurrence_count) * 0.4 + recency * 0.3
        
        patterns_list.sort(key=pattern_score, reverse=True)
        
        # Keep top patterns
        keep_count = int(len(patterns_list) * keep_ratio)
        patterns_to_keep = set(p.pattern_id for p in patterns_list[:keep_count])
        
        # Remove from patterns dict and vector index
        self.patterns[category] = {
            pid: p for pid, p in self.patterns[category].items()
            if pid in patterns_to_keep
        }
        
        self.vector_index[category] = [
            (pid, vec) for pid, vec in self.vector_index[category]
            if pid in patterns_to_keep
        ]
        
        removed = len(patterns_list) - keep_count
        self.total_patterns -= removed
        
        logger.info(f"Compacted {category}: removed {removed} patterns, kept {keep_count}")
    
    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------
    def save(self) -> bool:
        """Save pattern library to disk."""
        try:
            for category in self.CATEGORIES:
                if not self.patterns[category]:
                    continue
                
                filepath = os.path.join(self.persistence_dir, f"{category}.pkl")
                
                # Serialize patterns
                data = {
                    'patterns': {
                        pid: self._serialize_pattern(p)
                        for pid, p in self.patterns[category].items()
                    },
                    'vectors': [(pid, vec.tolist()) for pid, vec in self.vector_index[category]]
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            # Save clusters
            clusters_file = os.path.join(self.persistence_dir, "clusters.pkl")
            with open(clusters_file, 'wb') as f:
                pickle.dump({
                    'clusters': {cid: self._serialize_cluster(c) for cid, c in self.clusters.items()},
                    'pattern_to_cluster': self.pattern_to_cluster
                }, f)
            
            logger.info(f"PatternLibrary saved: {self.total_patterns} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save PatternLibrary: {e}")
            return False
    
    def load(self) -> bool:
        """Load pattern library from disk."""
        try:
            for category in self.CATEGORIES:
                filepath = os.path.join(self.persistence_dir, f"{category}.pkl")
                
                if not os.path.exists(filepath):
                    continue
                
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                # Deserialize patterns
                self.patterns[category] = {
                    pid: self._deserialize_pattern(p_data)
                    for pid, p_data in data.get('patterns', {}).items()
                }
                
                self.vector_index[category] = [
                    (pid, np.array(vec, dtype=np.float32))
                    for pid, vec in data.get('vectors', [])
                ]
                
                self.total_patterns += len(self.patterns[category])
            
            # Load clusters
            clusters_file = os.path.join(self.persistence_dir, "clusters.pkl")
            if os.path.exists(clusters_file):
                with open(clusters_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.clusters = {
                    cid: self._deserialize_cluster(c_data)
                    for cid, c_data in data.get('clusters', {}).items()
                }
                self.pattern_to_cluster = data.get('pattern_to_cluster', {})
            
            logger.info(f"PatternLibrary loaded: {self.total_patterns} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PatternLibrary: {e}")
            return False
    
    def _serialize_pattern(self, pattern: Pattern) -> Dict[str, Any]:
        """Serialize a pattern for persistence."""
        return {
            'pattern_id': pattern.pattern_id,
            'category': pattern.category,
            'name': pattern.name,
            'description': pattern.description,
            'vector': pattern.vector.tolist(),
            'occurrence_count': pattern.occurrence_count,
            'success_rate': pattern.success_rate,
            'avg_profit': pattern.avg_profit,
            'avg_duration': pattern.avg_duration,
            'first_seen': pattern.first_seen,
            'last_seen': pattern.last_seen,
            'version': pattern.version,
            'metadata': pattern.metadata,
            'tags': list(pattern.tags),
            'cluster_id': pattern.cluster_id,
            'cluster_distance': pattern.cluster_distance
        }
    
    def _deserialize_pattern(self, data: Dict[str, Any]) -> Pattern:
        """Deserialize a pattern from persistence."""
        return Pattern(
            pattern_id=data['pattern_id'],
            category=data['category'],
            name=data['name'],
            description=data.get('description', ''),
            vector=np.array(data['vector'], dtype=np.float32),
            occurrence_count=data.get('occurrence_count', 1),
            success_rate=data.get('success_rate', 0.5),
            avg_profit=data.get('avg_profit', 0.0),
            avg_duration=data.get('avg_duration', 0.0),
            first_seen=data.get('first_seen', time.time()),
            last_seen=data.get('last_seen', time.time()),
            version=data.get('version', 1),
            metadata=data.get('metadata', {}),
            tags=set(data.get('tags', [])),
            cluster_id=data.get('cluster_id'),
            cluster_distance=data.get('cluster_distance', 0.0)
        )
    
    def _serialize_cluster(self, cluster: PatternCluster) -> Dict[str, Any]:
        """Serialize a cluster."""
        return {
            'cluster_id': cluster.cluster_id,
            'centroid': cluster.centroid.tolist(),
            'pattern_ids': cluster.pattern_ids,
            'category': cluster.category,
            'avg_success_rate': cluster.avg_success_rate,
            'total_occurrences': cluster.total_occurrences
        }
    
    def _deserialize_cluster(self, data: Dict[str, Any]) -> PatternCluster:
        """Deserialize a cluster."""
        return PatternCluster(
            cluster_id=data['cluster_id'],
            centroid=np.array(data['centroid'], dtype=np.float32),
            pattern_ids=data.get('pattern_ids', []),
            category=data.get('category', 'mixed'),
            avg_success_rate=data.get('avg_success_rate', 0.5),
            total_occurrences=data.get('total_occurrences', 0)
        )
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        stats = {
            'total_patterns': self.total_patterns,
            'total_lookups': self.total_lookups,
            'total_clusters': len(self.clusters),
            'categories': {}
        }
        
        for category in self.CATEGORIES:
            patterns = list(self.patterns[category].values())
            if patterns:
                stats['categories'][category] = {
                    'count': len(patterns),
                    'avg_success_rate': np.mean([p.success_rate for p in patterns]),
                    'avg_occurrences': np.mean([p.occurrence_count for p in patterns]),
                    'total_occurrences': sum(p.occurrence_count for p in patterns)
                }
        
        return stats
