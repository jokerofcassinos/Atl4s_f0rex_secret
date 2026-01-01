import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("Atl4s-SupplyChain")

class SupplyChainGraph:
    """
    Knowledge Graph for XAUUSD (Gold).
    Models the relationship between macro variables and gold.
    """
    def __init__(self):
        # Nodes and their correlation weights to Gold
        self.nodes = {
            "USD": -0.8,       # Inverse correlation
            "RATES": -0.7,     # High rates = lower gold (opportunity cost)
            "OIL": 0.4,        # Inflation hedge link
            "VIX": 0.6,        # Safe haven demand
            "GEOPOLITICS": 0.9 # Direct escalation link
        }
        self.active_shocks = {}

    def inject_shock(self, source, magnitude, duration_bars=12):
        """
        Injects a shock from a macro source (e.g., Fed news -> RATES).
        source: Key in self.nodes
        magnitude: -1.0 to 1.0 (Dovish/Bearish to Hawkish/Bullish)
        """
        if source in self.nodes:
            self.active_shocks[source] = {
                "magnitude": magnitude,
                "remaining": duration_bars,
                "weight": self.nodes[source]
            }
            logger.info(f"Shock Injected: {source} | Mag: {magnitude:.2f} | Impact on Gold: {magnitude * self.nodes[source]:.2f}")

    def get_impact(self):
        """
        Calculates the aggregate impact on Gold from all active shocks.
        Returns: impact_score (-1.0 to 1.0)
        """
        total_impact = 0.0
        expired = []
        
        for source, shock in self.active_shocks.items():
            # Impact = Mag * SourceWeight * Decay?
            # Decay could be linear
            decay = shock['remaining'] / 12.0
            impact = shock['magnitude'] * shock['weight'] * decay
            total_impact += impact
            
            shock['remaining'] -= 1
            if shock['remaining'] <= 0:
                expired.append(source)
                
        for e in expired:
            del self.active_shocks[e]
            
        return max(-1.0, min(1.0, total_impact))

    def process_news(self, news_events):
        """
        Analyzes news titles for key shock triggers.
        """
        for event in news_events:
            title = event.get('event', '').upper()
            # Simple heuristic mapping
            if "FED" in title or "INTEREST RATE" in title or "FOMC" in title:
                # Hawkish (positive mag) -> Negative for Gold
                self.inject_shock("RATES", 0.8) # Scenario: Hawkish hint
            elif "CPI" in title or "INFLATION" in title:
                self.inject_shock("OIL", 0.6) # Inflation proxy
            elif "WAR" in title or "CONFLICT" in title or "SANCTIONS" in title:
                self.inject_shock("GEOPOLITICS", 1.0)
            elif "NONFARM PAYROLLS" in title or "NFP" in title:
                self.inject_shock("USD", 0.7)
