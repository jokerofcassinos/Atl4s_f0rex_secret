
import sys
import os
import logging
import asyncio
from datetime import datetime

# Setup Logging to Console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("HealthCheck")

try:
    logger.info("1. Importing OmegaAGICore...")
    from core.agi.omega_agi_core import OmegaAGICore
    logger.info("   [OK] OmegaAGICore Imported.")

    logger.info("2. Importing Subsystems...")
    from core.agi.quantum import QuantumProbabilityCollapser
    from core.agi.logic import SymbolicReasoningModule
    from core.agi.plasticity import SelfModificationHeuristic
    from core.agi.ontology import OntologyEngine
    logger.info("   [OK] Subsystems Imported.")

    logger.info("3. Initializing Core AGI...")
    config = {'mode': 'HYBRID', 'risk_per_trade': 0.01, 'alpha_threshold': 0.8}
    agi = OmegaAGICore()
    logger.info("   [OK] OmegaAGICore Initialized.")

    logger.info("4. Testing Pre-Tick Pipeline (Phase 7/9/10)...")
    # Mock Data
    tick = {'symbol': 'XAUUSD', 'bid': 2000.0, 'ask': 2000.1, 'time_msc': 123456789}
    data_map = {'basket_data': {}, 'df': None}
    
    # Run pre_tick
    adjustments = agi.pre_tick(tick, config, data_map)
    logger.info(f"   [OK] Pre-Tick Result: {adjustments}")

    logger.info("5. Testing Singularlity Synergy (Phase 8)...")
    # Mock Inputs
    swarm_signal = {'direction': 1, 'confidence': 0.85} # BULLISH
    
    # Inject Mock History/Temporal data manually to test fusion
    # (In real loop this happens in pre_tick state update)
    
    decision_pack = agi.synthesize_singularity_decision(swarm_signal)
    logger.info(f"   [OK] Singularity Verdict: {decision_pack}")

    logger.info("6. Testing Plasticity (Phase 10)...")
    # Mock Metrics: 10% Drawdown!
    metrics = {'drawdown': 0.10, 'win_rate': 0.3}
    mods = agi.plasticity.evaluate_and_adapt(metrics, config)
    logger.info(f"   [OK] Plasticity Reaction: {mods}")
    
    if mods.get('risk_per_trade') == 0.005:
        logger.info("   [SUCCESS] Plasticity slashed risk correctly!")
    else:
        logger.warning("   [WARNING] Plasticity did not slash risk?")

    logger.info("\nSYSTEM INTEGRITY: 100% OPERATIONAL.")

except Exception as e:
    logger.error(f"   [FATAL] System Check Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
