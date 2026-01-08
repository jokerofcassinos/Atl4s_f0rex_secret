
import logging
import numpy as np
from core.hyper_dimensional import HyperDimensionalEngine, HyperVector
from core.holographic_memory import HolographicMemory
from core.agi.active_inference import GenerativeModel, FreeEnergyMinimizer

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestActiveInference")

def test_holographic_substrate():
    logger.info("--- Testing Holographic Substrate ---")
    
    # 1. HD Engine
    hd = HyperDimensionalEngine()
    
    # Encode State A (Bullish)
    state_a = {'close_pct': 80, 'vol_pct': 20, 'rsi': 70}
    vec_a = hd.encode(state_a)
    
    # Encode State B (Bearish)
    state_b = {'close_pct': 20, 'vol_pct': 80, 'rsi': 30}
    vec_b = hd.encode(state_b)
    
    # Similarity Check
    sim = vec_a.similarity(vec_b)
    logger.info(f"Similarity (A vs B): {sim:.4f} (Expected Low)")
    
    # Temporal Binding
    # T1 -> T2
    # Sequence = T2 + Permute(T1)
    vec_t1 = vec_a
    vec_t2 = vec_b
    seq = vec_t2.bundle(vec_t1.permute())
    
    logger.info("Temporal Binding Successful")
    return hd

def test_memory_recall():
    logger.info("--- Testing Holographic Memory ---")
    mem = HolographicMemory("brain/test_memory.json")
    hd = HyperDimensionalEngine()
    
    # Store some fake experiences
    # Case 1: Bullish vector -> High Return
    vec_bull = hd.encode({'close_pct': 90, 'vol_pct': 40, 'rsi': 60})
    mem.store_experience(vec_bull.values, outcome=1.0, meta={'regime': 'bull'})
    
    # Case 2: Bearish vector -> Negative Return
    vec_bear = hd.encode({'close_pct': 10, 'vol_pct': 90, 'rsi': 20})
    mem.store_experience(vec_bear.values, outcome=-1.0, meta={'regime': 'bear'})
    
    # Recall
    # Test query similar to Bull
    query = hd.encode({'close_pct': 85, 'vol_pct': 45, 'rsi': 65})
    outcome, conf = mem.recall(query.values, k=1)
    
    logger.info(f"Recall Bullish Query -> Outcome: {outcome:.2f} (Expected > 0), Conf: {conf:.2f}")
    
    # Associative Recall
    assocs = mem.recall_associative(query.values, k=1)
    logger.info(f"Associative Recall Matches: {len(assocs)}")
    for a in assocs:
        logger.info(f" - Match: Sim={a['similarity']:.2f}, Outcome={a['outcome']}")

    return mem

def test_active_inference(mem, hd):
    logger.info("--- Testing Active Inference Engine ---")
    
    gen_model = GenerativeModel(mem, hd)
    fep = FreeEnergyMinimizer(gen_model)
    
    # Establish Belief State (Bullish Context)
    obs = hd.encode({'close_pct': 88, 'vol_pct': 42, 'rsi': 62})
    gen_model.update_belief(obs, prediction_error=0.0)
    
    # Evaluate Policies
    policies = ["BUY", "SELL", "HOLD"]
    result = fep.select_best_policy(policies, {})
    
    logger.info(f"Selected Policy: {result['selected_policy']}")
    logger.info(f"G Scores: {result['G_scores']}")
    
    # We expect BUY to have lowest G (Highest Profit = Lowest Risk) 
    # because memory associated this state with outcome +1.0
    
    if result['selected_policy'] == "BUY":
        logger.info("SUCCESS: Engine correctly inferred BUY based on Bullish Memory.")
    else:
        logger.warning(f"FAILURE: Engine chose {result['selected_policy']} instead of BUY.")

if __name__ == "__main__":
    hd = test_holographic_substrate()
    mem = test_memory_recall()
    test_active_inference(mem, hd)
