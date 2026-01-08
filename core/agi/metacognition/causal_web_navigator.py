"""
Causal Web Navigator - Multi-dimensional Causal Inference.

Navigates causal relationships across temporal and asset dimensions
for predictive reasoning and intervention analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("CausalWebNavigator")


@dataclass
class CausalLink:
    """A causal relationship between variables."""
    cause: str
    effect: str
    strength: float  # 0-1
    delay_periods: int  # Time delay
    confidence: float
    is_bidirectional: bool = False


@dataclass
class CausalPath:
    """A path through the causal web."""
    links: List[CausalLink]
    total_strength: float
    total_delay: int
    intervention_effect: float


@dataclass
class CausalAnalysis:
    """Causal web analysis result."""
    relevant_paths: List[CausalPath]
    root_causes: List[str]
    downstream_effects: List[str]
    intervention_recommendations: List[Dict]
    causal_certainty: float


class CausalWebNavigator:
    """
    The Causal Detective.
    
    Navigates causal relationships through:
    - Multi-dimensional causal graph construction
    - Path finding for cause-effect chains
    - Intervention effect prediction
    - Counterfactual reasoning
    """
    
    def __init__(self):
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)
        self._build_default_graph()
        
        logger.info("CausalWebNavigator initialized")
    
    def _build_default_graph(self):
        """Build default causal relationships for forex."""
        links = [
            CausalLink('INTEREST_RATE_HIKE', 'CURRENCY_STRENGTH', 0.8, 0, 0.9),
            CausalLink('CURRENCY_STRENGTH', 'TREND_UP', 0.7, 1, 0.85),
            CausalLink('HIGH_VOLATILITY', 'WIDE_SPREADS', 0.9, 0, 0.95),
            CausalLink('WIDE_SPREADS', 'REDUCED_LIQUIDITY', 0.8, 0, 0.9),
            CausalLink('SESSION_OVERLAP', 'HIGH_LIQUIDITY', 0.85, 0, 0.9),
            CausalLink('HIGH_LIQUIDITY', 'TIGHT_SPREADS', 0.8, 0, 0.85),
            CausalLink('NEWS_RELEASE', 'VOLATILITY_SPIKE', 0.9, 0, 0.95),
            CausalLink('VOLATILITY_SPIKE', 'PRICE_MOVE', 0.75, 0, 0.8),
            CausalLink('TREND_UP', 'HIGHER_HIGHS', 0.7, 1, 0.8),
            CausalLink('TREND_DOWN', 'LOWER_LOWS', 0.7, 1, 0.8),
            CausalLink('EXHAUSTION', 'REVERSAL', 0.6, 2, 0.7),
            CausalLink('ACCUMULATION', 'BREAKOUT', 0.65, 5, 0.7),
        ]
        
        for link in links:
            self.causal_graph[link.cause].append(link)
    
    def navigate(self, current_state: Dict, target: Optional[str] = None) -> CausalAnalysis:
        """Navigate the causal web from current state."""
        # Identify active causes
        active_causes = self._identify_active_causes(current_state)
        
        # Find all relevant paths
        paths = self._find_causal_paths(active_causes, target)
        
        # Identify root causes
        roots = self._find_root_causes(active_causes)
        
        # Predict downstream effects
        effects = self._predict_downstream_effects(active_causes)
        
        # Generate intervention recommendations
        interventions = self._recommend_interventions(active_causes, effects)
        
        # Calculate overall causal certainty
        certainty = np.mean([p.total_strength for p in paths]) if paths else 0.5
        
        return CausalAnalysis(
            relevant_paths=paths[:5],  # Top 5 paths
            root_causes=roots,
            downstream_effects=effects,
            intervention_recommendations=interventions,
            causal_certainty=certainty
        )
    
    def _identify_active_causes(self, state: Dict) -> List[str]:
        """Identify which causes are currently active."""
        active = []
        
        if state.get('session_overlap'):
            active.append('SESSION_OVERLAP')
        if state.get('news_pending'):
            active.append('NEWS_RELEASE')
        if state.get('high_volatility'):
            active.append('HIGH_VOLATILITY')
        if state.get('trend_up'):
            active.append('TREND_UP')
        if state.get('trend_down'):
            active.append('TREND_DOWN')
        if state.get('exhaustion_detected'):
            active.append('EXHAUSTION')
        
        return active
    
    def _find_causal_paths(self, causes: List[str], target: Optional[str]) -> List[CausalPath]:
        """Find causal paths from causes to effects."""
        paths = []
        
        for cause in causes:
            path = self._trace_path(cause, target, [], 0, 1.0)
            if path:
                paths.extend(path)
        
        # Sort by strength
        paths.sort(key=lambda p: p.total_strength, reverse=True)
        return paths
    
    def _trace_path(self, node: str, target: Optional[str], 
                    visited: List[str], depth: int, cumulative_strength: float) -> List[CausalPath]:
        """Recursively trace causal paths."""
        if depth > 5 or node in visited:
            return []
        
        paths = []
        visited = visited + [node]
        
        for link in self.causal_graph.get(node, []):
            new_strength = cumulative_strength * link.strength
            
            if target is None or link.effect == target:
                paths.append(CausalPath(
                    links=[link],
                    total_strength=new_strength,
                    total_delay=link.delay_periods,
                    intervention_effect=new_strength
                ))
            
            # Continue tracing
            child_paths = self._trace_path(
                link.effect, target, visited, depth + 1, new_strength
            )
            for child in child_paths:
                child.links.insert(0, link)
                child.total_delay += link.delay_periods
            paths.extend(child_paths)
        
        return paths
    
    def _find_root_causes(self, active: List[str]) -> List[str]:
        """Find root causes (no incoming edges)."""
        all_effects = set()
        for links in self.causal_graph.values():
            for link in links:
                all_effects.add(link.effect)
        
        return [c for c in active if c not in all_effects]
    
    def _predict_downstream_effects(self, causes: List[str]) -> List[str]:
        """Predict downstream effects from active causes."""
        effects = set()
        
        for cause in causes:
            for link in self.causal_graph.get(cause, []):
                if link.strength > 0.5:
                    effects.add(link.effect)
        
        return list(effects)
    
    def _recommend_interventions(self, causes: List[str], 
                                 effects: List[str]) -> List[Dict]:
        """Recommend interventions based on causal analysis."""
        interventions = []
        
        if 'WIDE_SPREADS' in effects:
            interventions.append({
                'action': 'REDUCE_POSITION_SIZE',
                'reason': 'Wide spreads expected due to low liquidity',
                'priority': 'HIGH'
            })
        
        if 'VOLATILITY_SPIKE' in effects:
            interventions.append({
                'action': 'WIDEN_STOPS',
                'reason': 'Volatility spike expected',
                'priority': 'MEDIUM'
            })
        
        if 'REVERSAL' in effects:
            interventions.append({
                'action': 'TAKE_PROFIT',
                'reason': 'Reversal likely due to exhaustion',
                'priority': 'HIGH'
            })
        
        return interventions
