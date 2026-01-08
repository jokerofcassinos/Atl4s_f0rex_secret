
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import networkx as nx

logger = logging.getLogger("Ontology")

class CausalInferenceEngine:
    """
    System 2: Causal Inference Engine.
    Uses Judea Pearl's Causal Hierarchy (Association -> Intervention -> Counterfactuals)
    to understand market dynamics beyond correlation.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_initial_causal_model()
        
    def _build_initial_causal_model(self):
        """
        Defines the A Priori Causal Graph of the Market.
        Nodes: Variables
        Edges: Direct Causal Links (Parent -> Child)
        """
        # Causal Assumptions (Simplified for Trading)
        # News -> Sentiment
        self.graph.add_edge("News", "Sentiment", weight=0.8)
        
        # Sentiment -> OrderFlow
        self.graph.add_edge("Sentiment", "OrderFlow", weight=0.6)
        
        # OrderFlow -> Price
        self.graph.add_edge("OrderFlow", "Price", weight=0.9)
        
        # Volatility -> Spread
        self.graph.add_edge("Volatility", "Spread", weight=0.9)
        
        # Time -> Volatility (e.g. Session Open)
        self.graph.add_edge("Time", "Volatility", weight=0.7)
        
        # Price -> Indicator_RSI (Indicators are downstream effects, not causes)
        self.graph.add_edge("Price", "RSI", weight=1.0)
        
        logger.info("ONTOLOGY: Initial Causal Model Built (Nodes: %d, Edges: %d)", 
                    self.graph.number_of_nodes(), self.graph.number_of_edges())

    def update_with_observation(self, observation: Dict[str, Any]):
        """
        Updates node states with real-time data.
        """
        pass # In a full implementation, we'd update probabilistic states here.

    def infer_causality(self, cause: str, effect: str) -> float:
        """
        Returns the Causal Strength from Cause -> Effect.
        If no path exists, returns 0.0 (Independence).
        """
        if not self.graph.has_node(cause) or not self.graph.has_node(effect):
            return 0.0
            
        try:
            # Find all paths
            paths = list(nx.all_simple_paths(self.graph, source=cause, target=effect))
            if not paths: return 0.0
            
            # Simple aggregation of path weights (Product of weights along path)
            total_strength = 0.0
            for path in paths:
                path_strength = 1.0
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    path_strength *= self.graph[u][v]['weight']
                total_strength = max(total_strength, path_strength) # Max flow logic
            
            return total_strength
            
        except Exception as e:
            logger.error(f"Causal Inference Error: {e}")
            return 0.0

    def intervention(self, target_node: str, forced_value: float) -> Dict[str, float]:
        """
        The 'do(X)' operator.
        "What happens to the system if we FORCE 'target_node' to 'forced_value'?"
        """
        # This requires a structural equation model (SEM). 
        # For this phase, we use a simplified propagation.
        impacts = {}
        
        # Propagate downstream
        descendants = nx.descendants(self.graph, target_node)
        
        for node in descendants:
            strength = self.infer_causality(target_node, node)
            impacts[node] = strength * forced_value
            
        return impacts

    def distinguish_correlation(self, var_a: str, var_b: str) -> str:
        """
        Determines if relationship is Causal, Confounded, or Reverse.
        """
        a_cause_b = self.infer_causality(var_a, var_b)
        b_cause_a = self.infer_causality(var_b, var_a)
        
        if a_cause_b > 0 and b_cause_a == 0:
            return f"CAUSAL: {var_a} -> {var_b} ({a_cause_b:.2f})"
        elif b_cause_a > 0 and a_cause_b == 0:
            return f"REVERSE: {var_b} -> {var_a} ({b_cause_a:.2f})"
        elif a_cause_b > 0 and b_cause_a > 0:
            return "FEEDBACK LOOP"
        else:
            # Check for common ancestor (Confounder)
            # Z -> A and Z -> B
            preds_a = set(self.graph.predecessors(var_a)) if self.graph.has_node(var_a) else set()
            preds_b = set(self.graph.predecessors(var_b)) if self.graph.has_node(var_b) else set()
            common = preds_a.intersection(preds_b)
            if common:
                return f"CONFOUNDED (Common Cause: {list(common)})"
            
            return "INDEPENDENT"

class OntologyEngine:
    """
    System 1: Ontology Engine.
    Defines abstract concepts as graph nodes and maps data to Meaning.
    """
    def __init__(self):
        self.concepts = {
            'FEAR': {'volatility': 'HIGH', 'trend': 'DOWN', 'sentiment': 'PANIC'},
            'GREED': {'volatility': 'MODERATE', 'trend': 'UP', 'sentiment': 'EUPHORIA'},
            'STAGNATION': {'volatility': 'LOW', 'trend': 'FLAT', 'volume': 'LOW'},
            'LIQUIDITY_TRAP': {'volatility': 'LOW', 'trend': 'FLAT', 'volume': 'HIGH'} # High volume but no move
        }
        
    def classify_market_state(self, market_data: Dict[str, Any]) -> str:
        """
        Maps raw market data (Vol, Trend) to an Abstract Concept (e.g. FEAR).
        Returns the dominant concept.
        """
        best_match = "UNCERTAIN"
        best_score = 0
        
        # Normalize inputs (simplified)
        vol = "HIGH" if market_data.get('volatility', 0) > 1.0 else "LOW"
        trend_val = market_data.get('trend_score', 0)
        trend = "UP" if trend_val > 0.5 else ("DOWN" if trend_val < -0.5 else "FLAT")
        volume = "HIGH" if market_data.get('volume_score', 0) > 0.8 else "LOW"
        
        current_features = {'volatility': vol, 'trend': trend, 'volume': volume}
        
        for concept, definition in self.concepts.items():
            score = 0
            matches = 0
            for key, val in definition.items():
                if key in current_features and current_features[key] == val:
                    matches += 1
            
            # Semantic Match Score
            score = matches / len(definition)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = concept
                
        return best_match
