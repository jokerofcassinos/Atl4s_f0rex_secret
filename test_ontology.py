
import logging
import networkx as nx
from core.agi.ontology import CausalInferenceEngine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestOntology")

def test_causality():
    logger.info("--- Testing Causal Inference Engine ---")
    
    ontology = CausalInferenceEngine()
    
    # 1. Test Direct Causality (News -> Sentiment)
    strength = ontology.infer_causality("News", "Sentiment")
    logger.info(f"News -> Sentiment Strength: {strength:.2f} (Expected > 0)")
    
    if strength > 0:
        logger.info("SUCCESS: Direct causality confirmed.")
    else:
        logger.warning("FAILURE: Direct causality not found.")
        
    # 2. Test Transitive Causality (News -> Price)
    # News -> Sentiment -> OrderFlow -> Price
    strength_chain = ontology.infer_causality("News", "Price")
    logger.info(f"News -> Price Strength: {strength_chain:.2f} (Expected > 0, slightly degraded)")
    
    if strength_chain > 0:
         logger.info("SUCCESS: Transitive causality confirmed.")
    else:
         logger.warning("FAILURE: Transitive causality not found.")
         
    # 3. Test Independence (Time -> Price)
    # Time -> Volatility -> Spread. Time shouldn't cause Price direct (in this model).
    # Wait, Time -> Volatility -> Spread... no path to Price.
    # Actually, is there a path?
    # No path defined from Spread to Price in _build_initial_causal_model.
    
    strength_indep = ontology.infer_causality("Time", "Price")
    logger.info(f"Time -> Price Strength: {strength_indep:.2f} (Expected 0.0)")
    
    if strength_indep == 0.0:
        logger.info("SUCCESS: Independence confirmed.")
    else:
        logger.warning(f"FAILURE: Spurious causality detected ({strength_indep}).")
        
    # 4. Distinguish Correlation
    # Sentiment and Price are correlated.
    # Sentiment -> OrderFlow -> Price.
    # Causal check.
    rel = ontology.distinguish_correlation("Sentiment", "Price")
    logger.info(f"Sentiment vs Price: {rel}")
    
    if "CAUSAL" in rel:
         logger.info("SUCCESS: Defined as Causal.")
    else:
         logger.warning(f"FAILURE: Relationship misidentified as {rel}")

if __name__ == "__main__":
    test_causality()
