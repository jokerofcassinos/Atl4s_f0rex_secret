
import logging
from core.agi.evolution import GeneticAlgoOptimizer, NeuralArchitectureSearch, AdversarialDreamer
from core.agi.execution import AdaptiveExecutionMatrix, GameTheoreticArbiter

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase5")

def test_evolution_pillar():
    logger.info("--- Testing Pillar IV: Evolution ---")
    
    # 1. Genetic Algorithm
    ga = GeneticAlgoOptimizer(population_size=6)
    ga.evolve() # Run one generation
    best = ga.population[0]
    logger.info(f"Best Genome Fitness: {best.fitness:.2f} | Alpha: {best.genes['alpha_threshold']:.1f}")
    
    if best.fitness > 0:
        logger.info("SUCCESS: Genetic Evolution functional.")
    else:
        logger.warning("FAILURE: Fitness zero.")
        
    # 2. NAS
    nas = NeuralArchitectureSearch()
    arch = nas.search_topology()
    logger.info(f"NAS Architecture Found: {arch}")
    
    # 3. Dreamer
    dreamer = AdversarialDreamer()
    nightmare = dreamer.generate_nightmare()
    logger.info(f"Nightmare Generated: {len(nightmare)} candles. Start: {nightmare[0]} End: {nightmare[-1]:.2f}")
    
def test_execution_pillar():
    logger.info("--- Testing Pillar V: Execution ---")
    
    # 1. Adaptive Matrix
    matrix = AdaptiveExecutionMatrix()
    
    # Case A: Panic Mode
    order_a = matrix.resolve_order_type("SELL", {'spread': 2.0, 'atr': 80})
    logger.info(f"Scenario Panic (High Spread+ATR) -> Order: {order_a} (Expect MARKET)")
    
    # Case B: Calm Limit
    order_b = matrix.resolve_order_type("BUY", {'spread': 2.0, 'atr': 10})
    logger.info(f"Scenario Calm (High Spread+Low ATR) -> Order: {order_b} (Expect LIMIT)")
    
    if order_a == "MARKET" and order_b == "LIMIT":
         logger.info("SUCCESS: Adaptive Routing functional.")
    else:
         logger.warning(f"FAILURE: Routing logic mismatch. A={order_a}, B={order_b}")

    # 2. Game Theory
    arbiter = GameTheoreticArbiter()
    sat = arbiter.check_market_saturation(0.9, 0.05) # Crowded Long
    logger.info(f"Saturation Check: {sat} (Expect CROWDED_LONG)")
    
    exit_signal = arbiter.suggest_nash_exit(sat)
    
    if exit_signal:
        logger.info("SUCCESS: Nash Exit triggered on Crowded Trade.")
    else:
        logger.warning("FAILURE: Nash Exit failed.")

if __name__ == "__main__":
    test_evolution_pillar()
    test_execution_pillar()
