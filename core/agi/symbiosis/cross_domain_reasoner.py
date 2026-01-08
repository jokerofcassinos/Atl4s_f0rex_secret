"""
Cross-Domain Reasoner - Cross-Asset and Cross-Market Reasoning.

Implements causal reasoning across different assets and markets
for enhanced predictive power.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("CrossDomainReasoner")


@dataclass
class CrossDomainLink:
    """A link between two domains."""
    source_domain: str
    target_domain: str
    correlation: float
    lead_lag_minutes: int  # Positive = source leads
    causal_strength: float
    link_type: str  # 'POSITIVE', 'NEGATIVE', 'CONDITIONAL'


@dataclass
class CrossDomainInsight:
    """An insight from cross-domain analysis."""
    insight_type: str
    source: str
    implication: str
    confidence: float
    suggested_action: Optional[str]


@dataclass
class CrossDomainReading:
    """Cross-domain analysis result."""
    active_links: List[CrossDomainLink]
    insights: List[CrossDomainInsight]
    inter_market_bias: str
    risk_adjustment: float
    opportunity_score: float


class CrossDomainReasoner:
    """
    The Market Polyglot.
    
    Reasons across domains through:
    - Inter-market correlation tracking
    - Lead-lag relationship detection
    - Cross-asset causal inference
    - Multi-market opportunity scoring
    """
    
    def __init__(self):
        self.domain_data: Dict[str, deque] = {}
        self.known_links = self._initialize_links()
        
        logger.info("CrossDomainReasoner initialized")
    
    def _initialize_links(self) -> List[CrossDomainLink]:
        """Initialize known cross-domain relationships."""
        return [
            # USD Index correlations
            CrossDomainLink('DXY', 'EURUSD', -0.95, 0, 0.9, 'NEGATIVE'),
            CrossDomainLink('DXY', 'GBPUSD', -0.85, 0, 0.85, 'NEGATIVE'),
            CrossDomainLink('DXY', 'USDJPY', 0.75, 0, 0.8, 'POSITIVE'),
            
            # Commodity correlations
            CrossDomainLink('OIL', 'USDCAD', -0.60, 15, 0.7, 'NEGATIVE'),
            CrossDomainLink('GOLD', 'USDJPY', -0.45, 0, 0.6, 'NEGATIVE'),
            
            # Risk sentiment
            CrossDomainLink('SPX', 'USDJPY', 0.70, 0, 0.75, 'POSITIVE'),
            CrossDomainLink('VIX', 'USDJPY', -0.65, -5, 0.7, 'NEGATIVE'),
            
            # Yield correlations
            CrossDomainLink('US10Y', 'USDJPY', 0.55, 30, 0.6, 'POSITIVE'),
            CrossDomainLink('US10Y', 'EURUSD', -0.40, 30, 0.5, 'NEGATIVE'),
            
            # Currency crosses
            CrossDomainLink('EURUSD', 'EURGBP', 0.50, 0, 0.6, 'CONDITIONAL'),
            CrossDomainLink('USDJPY', 'EURJPY', 0.85, 0, 0.85, 'POSITIVE'),
        ]
    
    def reason(self, market_data: Dict[str, Dict]) -> CrossDomainReading:
        """
        Perform cross-domain reasoning.
        
        Args:
            market_data: Dict mapping asset/index to current data
            
        Returns:
            CrossDomainReading with insights.
        """
        # Update domain data
        for domain, data in market_data.items():
            if domain not in self.domain_data:
                self.domain_data[domain] = deque(maxlen=100)
            self.domain_data[domain].append(data)
        
        # Find active links
        active_links = self._find_active_links(market_data)
        
        # Generate insights
        insights = self._generate_insights(active_links, market_data)
        
        # Determine inter-market bias
        bias = self._calculate_inter_market_bias(active_links, market_data)
        
        # Risk adjustment
        risk_adj = self._calculate_risk_adjustment(active_links)
        
        # Opportunity score
        opp_score = self._calculate_opportunity_score(insights)
        
        return CrossDomainReading(
            active_links=active_links,
            insights=insights,
            inter_market_bias=bias,
            risk_adjustment=risk_adj,
            opportunity_score=opp_score
        )
    
    def _find_active_links(self, data: Dict[str, Dict]) -> List[CrossDomainLink]:
        """Find links that are currently relevant."""
        active = []
        
        for link in self.known_links:
            source = link.source_domain
            target = link.target_domain
            
            # Check if we have data for both
            source_present = source in data or any(source in d for d in data)
            target_present = target in data or any(target in d for d in data)
            
            if source_present or target_present:
                active.append(link)
        
        return active
    
    def _generate_insights(self, links: List[CrossDomainLink],
                          data: Dict[str, Dict]) -> List[CrossDomainInsight]:
        """Generate insights from active links."""
        insights = []
        
        for link in links:
            source_data = data.get(link.source_domain, {})
            
            if not source_data:
                continue
            
            # Check for significant moves in source
            change = source_data.get('change_pct', 0)
            
            if abs(change) > 0.5:  # Significant move
                if link.link_type == 'POSITIVE':
                    implication = f"{link.target_domain} likely to follow"
                    direction = 'UP' if change > 0 else 'DOWN'
                elif link.link_type == 'NEGATIVE':
                    implication = f"{link.target_domain} likely to move opposite"
                    direction = 'DOWN' if change > 0 else 'UP'
                else:
                    implication = f"Monitor {link.target_domain} for reaction"
                    direction = None
                
                action = f"BIAS_{direction}" if direction else None
                
                insights.append(CrossDomainInsight(
                    insight_type='CORRELATION_SIGNAL',
                    source=link.source_domain,
                    implication=implication,
                    confidence=abs(link.correlation) * link.causal_strength,
                    suggested_action=action
                ))
        
        return insights
    
    def _calculate_inter_market_bias(self, links: List[CrossDomainLink],
                                     data: Dict[str, Dict]) -> str:
        """Calculate overall inter-market bias."""
        bullish = 0
        bearish = 0
        
        for link in links:
            source_data = data.get(link.source_domain, {})
            change = source_data.get('change_pct', 0)
            
            if link.link_type == 'POSITIVE':
                if change > 0:
                    bullish += abs(link.correlation)
                else:
                    bearish += abs(link.correlation)
            elif link.link_type == 'NEGATIVE':
                if change > 0:
                    bearish += abs(link.correlation)
                else:
                    bullish += abs(link.correlation)
        
        if bullish > bearish * 1.3:
            return 'RISK_ON'
        elif bearish > bullish * 1.3:
            return 'RISK_OFF'
        return 'NEUTRAL'
    
    def _calculate_risk_adjustment(self, links: List[CrossDomainLink]) -> float:
        """Calculate risk adjustment factor."""
        if not links:
            return 1.0
        
        avg_correlation = np.mean([abs(l.correlation) for l in links])
        
        # High correlation = lower risk adjustment needed
        return float(1.0 + (1 - avg_correlation) * 0.5)
    
    def _calculate_opportunity_score(self, insights: List[CrossDomainInsight]) -> float:
        """Calculate opportunity score from insights."""
        if not insights:
            return 0.5
        
        return float(np.mean([i.confidence for i in insights]))
