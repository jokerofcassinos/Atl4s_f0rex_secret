"""
AGI Ultra-Complete: DataLoader AGI Components

Sistema de Dados Inteligente:
- DataQualityAnalyzer: Avalia qualidade de dados
- PredictiveCacheManager: Cache preditivo
- AdaptiveDataFetcher: Fetch adaptativo
- DataAnomalyDetector: Detecção de anomalias
- IntelligentImputationEngine: Imputação inteligente
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("DataLoaderAGI")


class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class QualityReport:
    """Data quality report."""
    symbol: str
    timeframe: str
    quality: DataQuality
    completeness: float
    accuracy: float
    timeliness: float
    issues: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    predicted_next_access: float = 0.0


class DataQualityAnalyzer:
    """Analyzes data quality automatically."""
    
    def __init__(self):
        self.reports: Dict[str, QualityReport] = {}
        self.thresholds = {
            'completeness': 0.95,
            'price_deviation': 0.02,
            'max_gap_seconds': 60
        }
        
        logger.info("DataQualityAnalyzer initialized")
    
    def analyze(self, symbol: str, timeframe: str, data: Any) -> QualityReport:
        """Analyze data quality."""
        issues = []
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            return QualityReport(
                symbol=symbol, timeframe=timeframe,
                quality=DataQuality.UNUSABLE,
                completeness=0.0, accuracy=0.0, timeliness=0.0,
                issues=["No data available"]
            )
        
        completeness = self._check_completeness(data)
        accuracy = self._check_accuracy(data)
        timeliness = self._check_timeliness(data)
        
        if completeness < self.thresholds['completeness']:
            issues.append(f"Low completeness: {completeness:.0%}")
        
        if accuracy < 0.95:
            issues.append(f"Potential accuracy issues")
        
        quality = self._determine_quality(completeness, accuracy, timeliness)
        
        report = QualityReport(
            symbol=symbol, timeframe=timeframe,
            quality=quality,
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            issues=issues
        )
        
        self.reports[f"{symbol}_{timeframe}"] = report
        return report
    
    def _check_completeness(self, data) -> float:
        """Check data completeness."""
        if hasattr(data, 'isnull'):
            total_cells = data.size
            missing = data.isnull().sum().sum()
            return 1.0 - (missing / max(1, total_cells))
        return 1.0
    
    def _check_accuracy(self, data) -> float:
        """Check data accuracy."""
        if hasattr(data, 'columns'):
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    if (data['high'] < data['low']).any():
                        return 0.5
                    if (data['close'] > data['high']).any():
                        return 0.7
        return 1.0
    
    def _check_timeliness(self, data) -> float:
        """Check data timeliness."""
        if hasattr(data, 'index') and len(data) > 0:
            try:
                latest = data.index[-1]
                now = time.time()
                if hasattr(latest, 'timestamp'):
                    age = now - latest.timestamp()
                    if age > 60:
                        return max(0.5, 1.0 - (age / 3600))
            except:
                pass
        return 1.0
    
    def _determine_quality(self, completeness: float, accuracy: float, timeliness: float) -> DataQuality:
        """Determine overall quality."""
        avg = (completeness + accuracy + timeliness) / 3
        
        if avg > 0.95:
            return DataQuality.EXCELLENT
        elif avg > 0.85:
            return DataQuality.GOOD
        elif avg > 0.7:
            return DataQuality.ACCEPTABLE
        elif avg > 0.5:
            return DataQuality.POOR
        return DataQuality.UNUSABLE


class PredictiveCacheManager:
    """Cache with prediction of future access."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("PredictiveCacheManager initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.access_patterns[key].append(time.time())
            return entry.data
        return None
    
    def put(self, key: str, data: Any):
        """Put in cache with prediction."""
        if len(self.cache) >= self.max_size:
            self._evict()
        
        predicted = self._predict_next_access(key)
        
        self.cache[key] = CacheEntry(
            key=key,
            data=data,
            created_at=time.time(),
            last_accessed=time.time(),
            predicted_next_access=predicted
        )
    
    def _predict_next_access(self, key: str) -> float:
        """Predict when key will be accessed next."""
        if key not in self.access_patterns:
            return time.time() + 60
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return time.time() + 60
        
        intervals = [accesses[i+1] - accesses[i] for i in range(len(accesses)-1)]
        avg_interval = sum(intervals) / len(intervals)
        
        return time.time() + avg_interval
    
    def _evict(self):
        """Evict least valuable entry."""
        if not self.cache:
            return
        
        now = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            recency = 1.0 / (1 + now - entry.last_accessed)
            frequency = entry.access_count / max(1, now - entry.created_at)
            predicted_value = 1.0 if entry.predicted_next_access > now else 0.5
            
            scores[key] = recency * 0.3 + frequency * 0.4 + predicted_value * 0.3
        
        victim = min(scores, key=scores.get)
        del self.cache[victim]
    
    def preload_predicted(self, loader_func, keys: List[str]):
        """Preload data predicted to be needed."""
        now = time.time()
        
        for key, entry in self.cache.items():
            if entry.predicted_next_access < now + 30:
                if key not in self.cache:
                    try:
                        data = loader_func(key)
                        self.put(key, data)
                    except:
                        pass


class AdaptiveDataFetcher:
    """Adapts fetch frequency based on market conditions."""
    
    def __init__(self):
        self.fetch_intervals: Dict[str, float] = {}
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.default_interval = 1.0
        
        logger.info("AdaptiveDataFetcher initialized")
    
    def get_interval(self, symbol: str) -> float:
        """Get adaptive fetch interval."""
        if symbol in self.fetch_intervals:
            return self.fetch_intervals[symbol]
        return self.default_interval
    
    def update_volatility(self, symbol: str, volatility: float):
        """Update volatility and adjust interval."""
        self.volatility_history[symbol].append(volatility)
        
        avg_vol = sum(self.volatility_history[symbol]) / len(self.volatility_history[symbol])
        
        if avg_vol > 0.02:
            self.fetch_intervals[symbol] = 0.1
        elif avg_vol > 0.01:
            self.fetch_intervals[symbol] = 0.5
        elif avg_vol > 0.005:
            self.fetch_intervals[symbol] = 1.0
        else:
            self.fetch_intervals[symbol] = 2.0


class DataAnomalyDetector:
    """Detects anomalies in data."""
    
    def __init__(self):
        self.baselines: Dict[str, Dict] = {}
        self.anomalies: List[Dict] = []
        
        logger.info("DataAnomalyDetector initialized")
    
    def set_baseline(self, symbol: str, data: Any):
        """Set baseline for anomaly detection."""
        if hasattr(data, 'describe'):
            stats = data.describe()
            self.baselines[symbol] = {
                'mean': stats.get('mean', {}),
                'std': stats.get('std', {}),
                'min': stats.get('min', {}),
                'max': stats.get('max', {})
            }
    
    def detect(self, symbol: str, data: Any) -> List[Dict]:
        """Detect anomalies."""
        detected = []
        
        if symbol not in self.baselines:
            self.set_baseline(symbol, data)
            return []
        
        baseline = self.baselines[symbol]
        
        if hasattr(data, 'iloc') and len(data) > 0:
            latest = data.iloc[-1]
            
            for col in ['close', 'volume']:
                if col in latest and col in baseline.get('mean', {}):
                    mean = baseline['mean'].get(col, 0)
                    std = baseline['std'].get(col, 1)
                    
                    z_score = abs(latest[col] - mean) / max(0.001, std)
                    
                    if z_score > 3:
                        anomaly = {
                            'symbol': symbol,
                            'column': col,
                            'value': latest[col],
                            'z_score': z_score,
                            'timestamp': time.time()
                        }
                        detected.append(anomaly)
                        self.anomalies.append(anomaly)
        
        return detected


class IntelligentImputationEngine:
    """Intelligent missing data imputation."""
    
    def __init__(self):
        self.patterns: Dict[str, Dict] = {}
        
        logger.info("IntelligentImputationEngine initialized")
    
    def impute(self, data: Any, symbol: str = "") -> Any:
        """Impute missing values intelligently."""
        if not hasattr(data, 'isnull') or not data.isnull().any().any():
            return data
        
        result = data.copy()
        
        for col in result.columns:
            if result[col].isnull().any():
                if col in ['open', 'high', 'low', 'close']:
                    result[col] = result[col].fillna(method='ffill')
                    result[col] = result[col].fillna(method='bfill')
                elif col == 'volume':
                    result[col] = result[col].fillna(result[col].median())
                else:
                    result[col] = result[col].fillna(method='ffill')
        
        return result


class DataLoaderAGI:
    """
    Main DataLoader AGI System.
    
    Integrates all intelligent data components.
    """
    
    def __init__(self):
        self.quality = DataQualityAnalyzer()
        self.cache = PredictiveCacheManager()
        self.fetcher = AdaptiveDataFetcher()
        self.anomaly = DataAnomalyDetector()
        self.imputation = IntelligentImputationEngine()
        
        logger.info("DataLoaderAGI initialized")
    
    def process_data(self, symbol: str, timeframe: str, data: Any) -> Tuple[Any, QualityReport]:
        """Process data with AGI capabilities."""
        report = self.quality.analyze(symbol, timeframe, data)
        
        if report.quality == DataQuality.UNUSABLE:
            cached = self.cache.get(f"{symbol}_{timeframe}")
            if cached is not None:
                return cached, report
        
        data = self.imputation.impute(data, symbol)
        
        anomalies = self.anomaly.detect(symbol, data)
        if anomalies:
            logger.warning(f"Anomalies detected in {symbol}: {anomalies}")
        
        self.cache.put(f"{symbol}_{timeframe}", data)
        
        return data, report
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGI status."""
        return {
            'cache_size': len(self.cache.cache),
            'quality_reports': len(self.quality.reports),
            'anomalies_detected': len(self.anomaly.anomalies),
            'symbols_monitored': len(self.fetcher.fetch_intervals)
        }
