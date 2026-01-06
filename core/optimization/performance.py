"""
AGI Ultra: Performance Optimization Module

Features:
- Intelligent caching with LRU and TTL
- Lazy evaluation for expensive computations
- Batch processing for parallel execution
- Memory-efficient storage
- CPU optimization helpers
"""

import logging
import time
import hashlib
import functools
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

logger = logging.getLogger("Optimization")

T = TypeVar('T')


# =============================================================================
# INTELLIGENT CACHE
# =============================================================================
@dataclass
class CacheEntry:
    """A cached value with metadata."""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl_seconds: float


class IntelligentCache:
    """
    Intelligent cache with LRU eviction, TTL support, and hit tracking.
    
    Features:
    - LRU eviction when full
    - TTL-based expiration
    - Hit rate tracking
    - Memory-aware eviction
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 300.0,
        max_memory_mb: float = 100.0
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_memory = 0
        
        logger.info(f"IntelligentCache initialized: size={max_size}, ttl={default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry.created_at > entry.ttl_seconds:
                del self._cache[key]
                self.total_memory -= entry.size_bytes
                self.misses += 1
                return None
            
            # Update access
            entry.accessed_at = time.time()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self._lock:
            # Estimate size
            size = self._estimate_size(value)
            
            # Evict if necessary
            while (len(self._cache) >= self.max_size or 
                   self.total_memory + size > self.max_memory_bytes):
                if not self._cache:
                    break
                self._evict_one()
            
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                size_bytes=size,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            self.total_memory += size
    
    def _evict_one(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Remove from beginning (least recently used)
        key, entry = self._cache.popitem(last=False)
        self.total_memory -= entry.size_bytes
        self.evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        import sys
        try:
            return sys.getsizeof(value)
        except:
            return 100  # Default estimate
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.total_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'evictions': self.evictions,
            'memory_mb': self.total_memory / (1024 * 1024)
        }


def cached(cache: IntelligentCache, key_fn: Optional[Callable] = None, ttl: Optional[float] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator


# =============================================================================
# LAZY EVALUATION
# =============================================================================
class Lazy(Generic[T]):
    """
    Lazy evaluation wrapper.
    Value is computed only when first accessed.
    """
    
    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._value: Optional[T] = None
        self._computed = False
        self._lock = threading.Lock()
    
    @property
    def value(self) -> T:
        """Get value, computing if necessary."""
        if not self._computed:
            with self._lock:
                if not self._computed:
                    self._value = self._factory()
                    self._computed = True
        return self._value
    
    def is_computed(self) -> bool:
        """Check if value has been computed."""
        return self._computed
    
    def reset(self):
        """Reset to uncomputed state."""
        with self._lock:
            self._value = None
            self._computed = False


# =============================================================================
# BATCH PROCESSOR
# =============================================================================
class BatchProcessor:
    """
    Batch processor for parallel execution.
    
    Features:
    - Thread-based parallelism
    - Process-based parallelism for CPU-bound tasks
    - Progress tracking
    - Error handling
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Statistics
        self.batches_processed = 0
        self.items_processed = 0
        self.errors = 0
        
        logger.info(f"BatchProcessor initialized: workers={max_workers}, processes={use_processes}")
    
    def process(
        self,
        items: List[Any],
        func: Callable[[Any], Any],
        batch_size: int = 100,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            func: Function to apply to each item
            batch_size: Size of each batch
            timeout: Optional timeout per batch
            
        Returns:
            List of results
        """
        results = []
        
        # Create batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            for batch in batches:
                # Submit batch
                futures = {executor.submit(func, item): i for i, item in enumerate(batch)}
                
                batch_results = [None] * len(batch)
                
                for future in as_completed(futures, timeout=timeout):
                    idx = futures[future]
                    try:
                        batch_results[idx] = future.result()
                        self.items_processed += 1
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
                        self.errors += 1
                        batch_results[idx] = None
                
                results.extend(batch_results)
                self.batches_processed += 1
        
        return results
    
    def process_async(
        self,
        items: List[Any],
        func: Callable[[Any], Any]
    ) -> List[Any]:
        """Process items asynchronously using threads."""
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, items))
            self.items_processed += len(items)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'batches_processed': self.batches_processed,
            'items_processed': self.items_processed,
            'errors': self.errors,
            'error_rate': self.errors / self.items_processed if self.items_processed > 0 else 0
        }


# =============================================================================
# MEMORY-EFFICIENT STORAGE
# =============================================================================
class CompressedStorage:
    """
    Memory-efficient storage with compression.
    
    Uses zlib compression for large values.
    """
    
    def __init__(self, compression_threshold: int = 1000):
        self._data: Dict[str, bytes] = {}
        self._compressed: Dict[str, bool] = {}
        self.compression_threshold = compression_threshold
        
        # Statistics
        self.original_bytes = 0
        self.compressed_bytes = 0
    
    def store(self, key: str, value: Any):
        """Store value with optional compression."""
        import pickle
        import zlib
        
        serialized = pickle.dumps(value)
        original_size = len(serialized)
        self.original_bytes += original_size
        
        if original_size > self.compression_threshold:
            compressed = zlib.compress(serialized)
            self._data[key] = compressed
            self._compressed[key] = True
            self.compressed_bytes += len(compressed)
        else:
            self._data[key] = serialized
            self._compressed[key] = False
            self.compressed_bytes += original_size
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value, decompressing if necessary."""
        import pickle
        import zlib
        
        if key not in self._data:
            return None
        
        data = self._data[key]
        
        if self._compressed[key]:
            data = zlib.decompress(data)
        
        return pickle.loads(data)
    
    def delete(self, key: str):
        """Delete stored value."""
        if key in self._data:
            del self._data[key]
            del self._compressed[key]
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        if self.original_bytes == 0:
            return 1.0
        return self.compressed_bytes / self.original_bytes


# =============================================================================
# GLOBAL OPTIMIZER
# =============================================================================
class GlobalOptimizer:
    """
    Centralized optimization manager.
    
    Provides access to caching, batch processing, and storage.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.cache = IntelligentCache(max_size=50000, default_ttl=600.0)
        self.batch_processor = BatchProcessor(max_workers=4)
        self.storage = CompressedStorage()
        
        self._initialized = True
        logger.info("GlobalOptimizer initialized")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all optimization statistics."""
        return {
            'cache': self.cache.get_stats(),
            'batch': self.batch_processor.get_stats(),
            'storage': {
                'compression_ratio': self.storage.get_compression_ratio(),
                'items': len(self.storage._data)
            }
        }
