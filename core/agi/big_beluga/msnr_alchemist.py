
import logging
from typing import List, Dict, Any
from core.agi.big_beluga.snr_matrix import SNRLevel

logger = logging.getLogger("MSNRAlchemist")

class GoldenZone:
    def __init__(self, price_min: float, price_max: float, score: float):
        self.min = price_min
        self.max = price_max
        self.score = score
        self.type = "HOSTILE" # or FRIENDLY (Support/Resistance)

class MSNRAlchemist:
    """
    Phase 22: MSNR Alchemist (Multi-Scale Nexus Resonance).
    Fuses multiple weak levels into Indestructible Golden Zones.
    """
    def __init__(self):
        self.fusion_threshold = 0.0005 # 5 pips (approx for Forex)
        
    def transmute(self, levels: List[SNRLevel], current_price: float) -> List[GoldenZone]:
        """
        Alchemizes raw levels into Golden Zones.
        """
        if not levels: return []
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        
        zones: List[GoldenZone] = []
        cluster: List[SNRLevel] = []
        
        for lvl in sorted_levels:
            if not cluster:
                cluster.append(lvl)
                continue
                
            # Check distance to cluster
            avg_price = sum(l.price for l in cluster) / len(cluster)
            
            # If close enough, add to cluster
            if abs(lvl.price - avg_price) < self.fusion_threshold:
                cluster.append(lvl)
            else:
                # Seal the previous cluster
                self._forge_zone(cluster, zones)
                cluster = [lvl] # Start new
                
        # Seal final cluster
        if cluster:
            self._forge_zone(cluster, zones)
            
        return zones
        
    def _forge_zone(self, cluster: List[SNRLevel], zones: List[GoldenZone]):
        """
        Creates a Golden Zone from a cluster of levels.
        """
        if not cluster: return
        
        prices = [l.price for l in cluster]
        min_p = min(prices)
        max_p = max(prices)
        
        # Alchemic Score Calculation
        # Base score = 1.0 per level
        # Bonus for Diversity (Support + Psych)
        # Bonus for Freshness
        
        score = sum(l.strength for l in cluster)
        
        # Diversity Bonus
        sources = set(l.source for l in cluster)
        if len(sources) > 1: score *= 1.5
        
        zones.append(GoldenZone(min_p, max_p, score))

    def detect_confluence(self, zones: List[GoldenZone], current_price: float) -> Dict:
        """
        Checks if we are INSIDE or NEAR a Golden Zone.
        """
        nearest_zone = None
        min_dist = 999.0
        
        in_zone = False
        
        for z in zones:
            # Distance to Zone edge
            dist = 0.0
            if current_price < z.min: dist = z.min - current_price
            elif current_price > z.max: dist = current_price - z.max
            else: 
                dist = 0.0
                in_zone = True
                
            if dist < min_dist:
                min_dist = dist
                nearest_zone = z
                
        return {
            'in_zone': in_zone,
            'nearest_zone_score': nearest_zone.score if nearest_zone else 0,
            'distance': min_dist
        }
