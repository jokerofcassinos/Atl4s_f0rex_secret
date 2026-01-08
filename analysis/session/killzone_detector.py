"""
Killzone Detector - High-Probability Trading Window Identification.

Implements heuristic evolution and pattern synthesis for detecting
institutional trading windows with maximum liquidity and momentum.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("KillzoneDetector")


class KillzoneType(Enum):
    ASIA_OPEN = "ASIA_OPEN"
    ASIA_CLOSE = "ASIA_CLOSE"
    LONDON_OPEN = "LONDON_OPEN"
    LONDON_CLOSE = "LONDON_CLOSE"
    NY_OPEN = "NY_OPEN"
    NY_CLOSE = "NY_CLOSE"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    ASIA_LONDON_OVERLAP = "ASIA_LONDON_OVERLAP"
    FIX_TIME = "FIX_TIME"  # 4PM London Fix
    OPTION_EXPIRY = "OPTION_EXPIRY"


@dataclass
class Killzone:
    """Represents a high-probability trading window."""
    zone_type: KillzoneType
    start_hour: int
    start_minute: int
    duration_minutes: int
    strength: float  # 0-1 probability of good trades
    volatility_expected: float
    liquidity_expected: float
    best_pairs: List[str]
    
    # Evolved heuristics
    historical_win_rate: float = 0.0
    pattern_score: float = 0.0
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if killzone is currently active."""
        hour = current_time.hour
        minute = current_time.minute
        current_minutes = hour * 60 + minute
        
        start_minutes = self.start_hour * 60 + self.start_minute
        end_minutes = start_minutes + self.duration_minutes
        
        # Handle overnight wrapping
        if end_minutes >= 1440:
            return current_minutes >= start_minutes or current_minutes < (end_minutes - 1440)
        
        return start_minutes <= current_minutes < end_minutes
    
    def time_until_start(self, current_time: datetime) -> int:
        """Get minutes until killzone starts."""
        hour = current_time.hour
        minute = current_time.minute
        current_minutes = hour * 60 + minute
        start_minutes = self.start_hour * 60 + self.start_minute
        
        if current_minutes < start_minutes:
            return start_minutes - current_minutes
        else:
            return (1440 - current_minutes) + start_minutes


@dataclass
class KillzoneAnalysis:
    """Analysis result for killzone detection."""
    active_zones: List[Killzone]
    upcoming_zones: List[Tuple[Killzone, int]]  # (zone, minutes_until)
    total_strength: float
    recommended_action: str
    confidence: float
    
    # Metacognitive metrics
    pattern_synthesis_score: float = 0.0
    heuristic_evolution_state: Dict = None


class KillzoneDetector:
    """
    The Temporal Predator.
    
    Identifies high-probability trading windows through:
    - Pattern synthesis from historical data
    - Heuristic evolution for threshold optimization
    - Cross-domain reasoning across sessions and instruments
    """
    
    def __init__(self):
        self.killzones = self._initialize_killzones()
        self.pattern_memory: List[Dict] = []
        self.evolution_generation = 0
        
        # Heuristic evolution parameters
        self.heuristic_genome = {
            'strength_threshold': 0.6,
            'liquidity_weight': 0.4,
            'volatility_weight': 0.3,
            'pattern_weight': 0.3,
            'overlap_bonus': 1.5,
        }
        
        # Pattern synthesis state
        self.synthesized_patterns: Dict[str, np.ndarray] = {}
        
        logger.info("KillzoneDetector initialized with heuristic evolution")
    
    def _initialize_killzones(self) -> List[Killzone]:
        """Initialize predefined killzones with institutional knowledge."""
        return [
            # Asian Session
            Killzone(
                zone_type=KillzoneType.ASIA_OPEN,
                start_hour=0, start_minute=0,
                duration_minutes=60,
                strength=0.5, volatility_expected=0.4, liquidity_expected=40,
                best_pairs=['USDJPY', 'AUDUSD', 'NZDUSD']
            ),
            Killzone(
                zone_type=KillzoneType.ASIA_CLOSE,
                start_hour=8, start_minute=0,
                duration_minutes=60,
                strength=0.4, volatility_expected=0.3, liquidity_expected=35,
                best_pairs=['USDJPY', 'AUDJPY']
            ),
            
            # London Session (MOST IMPORTANT)
            Killzone(
                zone_type=KillzoneType.LONDON_OPEN,
                start_hour=7, start_minute=0,
                duration_minutes=120,  # 7:00 - 9:00 UTC
                strength=0.85, volatility_expected=0.9, liquidity_expected=90,
                best_pairs=['GBPUSD', 'EURUSD', 'EURGBP', 'GBPJPY']
            ),
            Killzone(
                zone_type=KillzoneType.LONDON_CLOSE,
                start_hour=15, start_minute=30,
                duration_minutes=60,
                strength=0.7, volatility_expected=0.7, liquidity_expected=75,
                best_pairs=['GBPUSD', 'EURUSD']
            ),
            
            # New York Session
            Killzone(
                zone_type=KillzoneType.NY_OPEN,
                start_hour=12, start_minute=30,
                duration_minutes=120,  # 12:30 - 14:30 UTC
                strength=0.9, volatility_expected=0.95, liquidity_expected=95,
                best_pairs=['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
            ),
            Killzone(
                zone_type=KillzoneType.NY_CLOSE,
                start_hour=20, start_minute=0,
                duration_minutes=60,
                strength=0.5, volatility_expected=0.5, liquidity_expected=50,
                best_pairs=['EURUSD', 'USDJPY']
            ),
            
            # Overlaps (MAXIMUM PROBABILITY)
            Killzone(
                zone_type=KillzoneType.LONDON_NY_OVERLAP,
                start_hour=12, start_minute=0,
                duration_minutes=240,  # 12:00 - 16:00 UTC
                strength=1.0, volatility_expected=1.0, liquidity_expected=100,
                best_pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD']
            ),
            Killzone(
                zone_type=KillzoneType.ASIA_LONDON_OVERLAP,
                start_hour=7, start_minute=0,
                duration_minutes=120,
                strength=0.75, volatility_expected=0.75, liquidity_expected=75,
                best_pairs=['USDJPY', 'GBPJPY', 'EURJPY']
            ),
            
            # Special Times
            Killzone(
                zone_type=KillzoneType.FIX_TIME,
                start_hour=15, start_minute=50,
                duration_minutes=20,  # 4PM London Fix
                strength=0.65, volatility_expected=0.85, liquidity_expected=60,
                best_pairs=['EURUSD', 'GBPUSD']
            ),
        ]
    
    def analyze(self, current_time: Optional[datetime] = None) -> KillzoneAnalysis:
        """
        Analyze current killzone status with pattern synthesis.
        
        Returns:
            KillzoneAnalysis with active zones and recommendations.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Find active killzones
        active_zones = [kz for kz in self.killzones if kz.is_active(current_time)]
        
        # Find upcoming killzones (within 2 hours)
        upcoming_zones = []
        for kz in self.killzones:
            if not kz.is_active(current_time):
                minutes_until = kz.time_until_start(current_time)
                if minutes_until <= 120:
                    upcoming_zones.append((kz, minutes_until))
        
        upcoming_zones.sort(key=lambda x: x[1])
        
        # Calculate total strength
        total_strength = self._calculate_total_strength(active_zones)
        
        # Pattern synthesis
        pattern_score = self._synthesize_patterns(active_zones, current_time)
        
        # Generate recommendation
        if active_zones:
            if total_strength >= 0.8:
                action = "AGGRESSIVE_ENTRY"
            elif total_strength >= 0.6:
                action = "NORMAL_ENTRY"
            else:
                action = "CAUTIOUS_ENTRY"
        elif upcoming_zones and upcoming_zones[0][1] <= 15:
            action = "PREPARE_ENTRY"
        else:
            action = "WAIT_FOR_KILLZONE"
        
        # Confidence based on evolved heuristics
        confidence = self._calculate_confidence(active_zones, pattern_score)
        
        return KillzoneAnalysis(
            active_zones=active_zones,
            upcoming_zones=upcoming_zones[:3],  # Top 3 upcoming
            total_strength=total_strength,
            recommended_action=action,
            confidence=confidence,
            pattern_synthesis_score=pattern_score,
            heuristic_evolution_state=self.heuristic_genome.copy()
        )
    
    def _calculate_total_strength(self, active_zones: List[Killzone]) -> float:
        """Calculate combined strength of active killzones."""
        if not active_zones:
            return 0.0
        
        # Overlaps get bonus
        strengths = []
        for zone in active_zones:
            strength = zone.strength
            if 'OVERLAP' in zone.zone_type.value:
                strength *= self.heuristic_genome['overlap_bonus']
            strengths.append(strength)
        
        # Combined with diminishing returns
        total = strengths[0] if strengths else 0
        for s in strengths[1:]:
            total = total + s * (1 - total)
        
        return float(np.clip(total, 0, 1))
    
    def _synthesize_patterns(self, active_zones: List[Killzone], 
                            current_time: datetime) -> float:
        """
        Pattern synthesis from historical data.
        
        Synthesizes hyper-complex patterns across multiple dimensions.
        """
        if not active_zones:
            return 0.0
        
        # Create pattern vector
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        pattern_features = np.array([
            hour / 24,  # Time normalization
            day_of_week / 7,  # Day normalization
            len(active_zones) / 3,  # Zone count
            np.mean([z.strength for z in active_zones]),  # Avg strength
            np.mean([z.liquidity_expected for z in active_zones]) / 100,  # Liquidity
        ])
        
        # Compare to stored patterns (using cosine similarity)
        best_similarity = 0.0
        for pattern_name, stored_pattern in self.synthesized_patterns.items():
            if len(stored_pattern) == len(pattern_features):
                similarity = np.dot(pattern_features, stored_pattern) / (
                    np.linalg.norm(pattern_features) * np.linalg.norm(stored_pattern) + 1e-8
                )
                best_similarity = max(best_similarity, similarity)
        
        # Store current pattern for future synthesis
        pattern_key = f"{hour}_{day_of_week}"
        self.synthesized_patterns[pattern_key] = pattern_features
        
        return float(best_similarity)
    
    def _calculate_confidence(self, active_zones: List[Killzone], 
                             pattern_score: float) -> float:
        """Calculate confidence using evolved heuristics."""
        if not active_zones:
            return 0.2
        
        g = self.heuristic_genome
        
        liquidity = np.mean([z.liquidity_expected for z in active_zones]) / 100
        volatility = np.mean([z.volatility_expected for z in active_zones])
        strength = np.mean([z.strength for z in active_zones])
        
        confidence = (
            liquidity * g['liquidity_weight'] +
            volatility * g['volatility_weight'] +
            pattern_score * g['pattern_weight'] +
            strength * (1 - g['liquidity_weight'] - g['volatility_weight'] - g['pattern_weight'])
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def evolve_heuristics(self, performance_feedback: float):
        """
        Heuristic evolution through genetic programming.
        
        Args:
            performance_feedback: -1 to 1, negative for losses, positive for wins.
        """
        self.evolution_generation += 1
        
        # Mutation rate based on performance
        mutation_rate = 0.1 if performance_feedback > 0 else 0.2
        
        for key in self.heuristic_genome:
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, 0.1) * (1 if performance_feedback > 0 else -1)
                self.heuristic_genome[key] *= (1 + mutation)
                self.heuristic_genome[key] = np.clip(self.heuristic_genome[key], 0.1, 3.0)
        
        # Normalize weights
        weight_keys = ['liquidity_weight', 'volatility_weight', 'pattern_weight']
        total_weight = sum(self.heuristic_genome[k] for k in weight_keys)
        for k in weight_keys:
            self.heuristic_genome[k] /= total_weight
        
        logger.debug(f"EVOLUTION Gen {self.evolution_generation}: {self.heuristic_genome}")
    
    def get_active_killzones_for_pair(self, pair: str, 
                                      current_time: Optional[datetime] = None) -> List[Killzone]:
        """Get active killzones relevant for a specific currency pair."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        return [
            kz for kz in self.killzones 
            if kz.is_active(current_time) and pair in kz.best_pairs
        ]
    
    def should_trade_now(self, pair: str, min_strength: float = 0.5) -> Tuple[bool, str]:
        """Quick check if we should trade a pair right now."""
        analysis = self.analyze()
        
        # Check if pair is in any active zone's best pairs
        for zone in analysis.active_zones:
            if pair in zone.best_pairs and zone.strength >= min_strength:
                return True, f"Active in {zone.zone_type.value} (strength: {zone.strength:.2f})"
        
        if analysis.active_zones:
            return True, f"General killzone active (total: {analysis.total_strength:.2f})"
        
        if analysis.upcoming_zones and analysis.upcoming_zones[0][1] <= 5:
            zone, minutes = analysis.upcoming_zones[0]
            return False, f"Wait {minutes}m for {zone.zone_type.value}"
        
        return False, "No active killzone - consider waiting"
