import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("CausalInferenceEngine")

class CausalInferenceEngine:
    """
    Sistema D-3: Causal Inference Engine
    Determina a causalidade dos movimentos de mercado (O 'Porquê').
    Utiliza um Grafo Causal (DAG) e Testes de Causalidade de Granger Simplificados.
    """
    def __init__(self):
        # Directed Acyclic Graph: Node -> [Children]
        self.causal_graph = {
            "Liquidity": ["Volume", "Spread"],
            "Volume": ["Price_Impulse", "Volatility"],
            "News_Sentiment": ["Liquidity", "Volatility"],
            "Market_Regime": ["Liquidity", "Price_Impulse"]
        }
        self.history = {
            "price": [],
            "volume": [],
            "sentiment": []
        }
        self.max_history = 50
        
    def infer_cause(self, market_data: Dict[str, Any], events: Dict[str, Any], chronos_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Infer multidimensional causal relationships.
        Integrates Time (Chronos), Sentiment, and Volume.
        """
        # Update history
        price = market_data.get('bid', 0)
        vol = market_data.get('volume', 0)
        sent = events.get('sentiment_score', 0.5)
        
        self.history['price'].append(price)
        self.history['volume'].append(vol)
        self.history['sentiment'].append(sent)
        
        # Trim
        if len(self.history['price']) > self.max_history:
            self.history['price'].pop(0)
            self.history['volume'].pop(0)
            self.history['sentiment'].pop(0)
            
        root_cause = "UNDEFINED"
        confidence = 0.5
        
        # 1. Granger/Lag Tests
        vol_leads_price = self._check_lag_correlation(self.history['volume'], self.history['price'])
        sent_leads_vol = self._check_lag_correlation(self.history['sentiment'], self.history['volume'])
        
        # 2. Chronos/IPDA Logic (Deep Reason)
        ipda_cause = None
        if chronos_context:
            cycle_phase = chronos_context.get('quarter_phase', 'WAITING')
            chronos_nar = chronos_context.get('chronos_narrative', '')
            
            # A. Manipulation Detection (Q2)
            if cycle_phase == "MANIPULATION":
                # If price is moving fast (high vol) -> Likely Judas
                if vol > numpy_mean(self.history['volume']) * 1.5:
                     ipda_cause = "INSTITUTIONAL_JUDAS_SWING"
                     confidence = 0.90
            
            # B. Distribution Detection (Q3)
            elif cycle_phase == "DISTRIBUTION":
                ipda_cause = "IPDA_ALGORITHMIC_DELIVERY"
                confidence = 0.95
                
            # C. Expansion Narrative
            elif "Expansion" in chronos_nar:
                ipda_cause = "SESSION_EXPANSION_FLOW"
                confidence = 0.88

        # 3. Synthesis
        if ipda_cause:
            root_cause = ipda_cause
        elif sent_leads_vol and vol_leads_price:
            root_cause = "SENTIMENT_DRIVEN_FLOW" 
            confidence = 0.85
        elif vol_leads_price:
            root_cause = "FLOW_DRIVEN_MOMENTUM"
            confidence = 0.8
        elif events.get('impact', 0) > 0.8:
            root_cause = "EXOGENOUS_SHOCK"
            confidence = 0.95
        else:
            root_cause = "NOISE_OR_MEAN_REVERSION"
            confidence = 0.6
            
        return {
            "root_cause": root_cause,
            "confidence": confidence,
            "causal_chain": self._trace_graph(root_cause),
            "lead_lag_metrics": {"vol_leads_price": vol_leads_price},
            "ontological_layer": ipda_cause if ipda_cause else "STANDARD_MECHANICS"
        }

    def _calculate_mean(self, data):
        if not data: return 0
        return sum(data) / len(data)

        
    def _check_lag_correlation(self, leader: List[float], follower: List[float], lag: int = 2) -> bool:
        if len(leader) < 10: return False
        
        # Simple directional check: Does sign(dX_t-lag) == sign(dY_t)?
        matches = 0
        total = 0
        
        for i in range(lag, len(leader)):
            dx_prev = leader[i-lag] - leader[i-lag-1]
            dy_now = follower[i] - follower[i-1]
            
            if (dx_prev > 0 and dy_now > 0) or (dx_prev < 0 and dy_now < 0):
                matches += 1
            total += 1
            
        if total == 0: return False
        return (matches / total) > 0.6 # >60% directional match
        
    def _trace_graph(self, root_node: str) -> List[str]:
        # Simple mockup of graph traversal
        if root_node == "SENTIMENT_DRIVEN_FLOW":
            return ["News_Sentiment", "Liquidity", "Volume", "Price_Impulse"]
        elif root_node == "FLOW_DRIVEN_MOMENTUM":
            return ["Liquidity", "Volume", "Price_Impulse"]
        return ["Unknown"]
