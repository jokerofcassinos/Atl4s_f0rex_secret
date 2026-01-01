
import logging
import os
import config
import json
from data_loader import DataLoader
from analysis.consensus import ConsensusEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_system():
    logger.info("--- Starting System Verification (Matrix 4.0 - Omniscience) ---")
    
    # 1. Test Data Loader
    data_loader = DataLoader()
    data_map = data_loader.get_data()
    
    # 2. Test Consensus (Omniscience)
    logger.info("2. Testing Consensus Engine (Wavelet/Topology)...")
    consensus = ConsensusEngine()
    
    # Run a single deliberation
    decision, score, details = consensus.deliberate(data_map)
    logger.info(f"Consensus Result: {decision} (Score: {score})")
    
    # Validation Checks
    if 'Wavelet' in details and 'coherence' in details['Wavelet']:
        w = details['Wavelet']
        logger.info(f"SUCCESS: Wavelet Active. Coherence: {w['coherence']:.4f}, Energy: {w['energy_fast']:.4f}")
    else:
        logger.error("FAILURE: Wavelet metrics missing.")
        
    if 'Topology' in details and 'loop_score' in details['Topology']:
        t = details['Topology']
        logger.info(f"SUCCESS: Topology Active. Loop Score: {t['loop_score']:.4f}, Betti-1: {t['betti_1']}")
    # Verify Hyper-Complexity
    game = details.get('Game')
    chaos = details.get('Chaos')
    
    if game:
        logger.info(f"SUCCESS: Game Theory Active. Nash Eq: {game.get('equilibrium_price', 0):.2f}, BullDom: {game.get('dominance_score', 0):.2f}")
    else:
        logger.error("FAILURE: Game Theory metrics missing.")
        
    if chaos:
        logger.info(f"SUCCESS: Chaos Engine Active. Lyapunov: {chaos.get('lyapunov', 0):.4f}")
    else:
        logger.error("FAILURE: Chaos metrics missing.")

    # 3. Test Neural Risk (Geometric Stop)
    logger.info("3. Testing Neural Risk (Geometric Stop)...")
    from analysis.risk_neural import NeuralRiskManager
    risk_manager = NeuralRiskManager()
    
    # Mock inputs
    entry = 2000.0
    direction = 1 # Buy
    atr = 2.0 # Base stop 3.0
    
    # Scenario A: Low Energy (Sniper)
    stop_a = risk_manager.get_geometric_stop(entry, direction, atr, kinematics_energy=0.2, wavelet_power=0.9, uncertainty=0.5)
    dist_a = entry - stop_a
    logger.info(f"Risk Test A (Low Energy, High Coherence): Dist {dist_a:.2f} (Expected < 3.0)")
    
    # Scenario B: High Energy (Chaos)
    stop_b = risk_manager.get_geometric_stop(entry, direction, atr, kinematics_energy=1.5, wavelet_power=0.2, uncertainty=0.5)
    dist_b = entry - stop_b
    logger.info(f"Risk Test B (High Energy, Low Coherence): Dist {dist_b:.2f} (Expected > 3.0)")
    
    # Scenario C: Heisenberg Limit
    stop_c = risk_manager.get_geometric_stop(entry, direction, atr, kinematics_energy=0, wavelet_power=1.0, uncertainty=5.0)
    dist_c = entry - stop_c
    logger.info(f"Risk Test C (Heisenberg Limit 5.0): Dist {dist_c:.2f} (Expected 5.0)")

    # 4. Test Quantum Sizing
    logger.info("4. Testing Quantum Position Sizing...")
    # Base Case
    lots_base = risk_manager.calculate_quantum_size(entry, entry-2.0, 1000, consensus_score=50)
    
    # Boost Case (Clean Trend, Fast Move, Memory Support)
    lots_boost = risk_manager.calculate_quantum_size(entry, entry-2.0, 1000, consensus_score=80, 
                                                   wavelet_coherence=0.9, kinematics_score=80, cortex_conf=0.9)
                                                   
    # Noise Case (Choppy)
    lots_noise = risk_manager.calculate_quantum_size(entry, entry-2.0, 1000, consensus_score=40,
                                                   wavelet_coherence=0.2, kinematics_score=20)
                                                   
    logger.info(f"Sizing Results: Base={lots_base} | Boost={lots_boost} | Noise={lots_noise}")
    if lots_boost > lots_base and lots_noise < lots_base:
        logger.info("SUCCESS: Dynamic Sizing Logic Verified.")
    else:
        logger.error("FAILURE: Sizing logic incoherent.")

    logger.info("--- Verification Complete ---")

if __name__ == "__main__":
    verify_system()
