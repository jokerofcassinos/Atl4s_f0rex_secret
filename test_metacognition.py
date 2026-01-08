
import logging
from core.agi.metacognition import RecursiveReflectionLoop
from core.agi.symbiosis import UserIntentModeler, ExplanabilityGenerator

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMetacognition")

def test_metacognition():
    logger.info("--- Testing Metacognition ---")
    meta = RecursiveReflectionLoop()
    
    # 1. Test Overconfidence Damping
    # Scenario: 99% Confidence but last trade was a loss (simulated)
    decision = "BUY"
    confidence = 99.0
    context = {'config': {'mode': 'SNIPER'}}
    
    # For test, we need to mock recent history in BiasDetector or inject it
    # Currently history is empty, so let's mock the check_biases call or just rely on logic
    # The current implementation checks recent_outcomes passed in args.
    
    # Mock specific outcome in reflect? No, reflect passes [] currently in swarm.
    # We will test the class directly.
    
    biases = meta.bias_detector.check_biases(decision, confidence, recent_outcomes=[-1.0, -1.0])
    logger.info(f"Detected Biases (Expect Overconfidence > 0): {biases}")
    
    if biases.get('overconfidence', 0) > 0:
        logger.info("SUCCESS: Overconfidence detected.")
    else:
        logger.warning("FAILURE: Overconfidence NOT detected.")

def test_symbiosis():
    logger.info("--- Testing Symbiosis ---")
    
    # 1. User Intent
    user = UserIntentModeler()
    user.analyze_command("I want max risk aggressive mode")
    logger.info(f"Inferred Profile (Expect AGGRESSIVE): {user.profile['risk_tolerance']}")
    
    if user.profile['risk_tolerance'] == 'AGGRESSIVE':
        logger.info("SUCCESS: Aggressive intent inferred.")
    else:
        logger.warning(f"FAILURE: Inferred {user.profile['risk_tolerance']}")
        
    # 2. Explanability
    explain = ExplanabilityGenerator()
    meta_data = {
        'reflection_notes': ['Damped Overconfidence', 'Vetoed by Logic'],
        'active_inference_G': 1.5
    }
    narrative = explain.generate_narrative("BUY", 80.0, meta_data)
    logger.info(f"Narrative: {narrative}")
    
    if "Damped" in narrative or "damped" in narrative:
        if "G-Score" in narrative:
            logger.info("SUCCESS: Narrative contains Metacognitive and Active Inference context.")
        else:
            logger.warning("FAILURE: Narrative missing G-Score.")
    else:
         logger.warning("FAILURE: Narrative missing Metacognitive context (Damped).")

if __name__ == "__main__":
    test_metacognition()
    test_symbiosis()
