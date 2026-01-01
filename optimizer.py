import random
import logging
import pandas as pd
import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from data_loader import DataLoader
from backtest_engine import BacktestEngine
import config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Atl4s-Optimizer")

# Genetic Algorithm Parameters
POPULATION_SIZE = 6
GENERATIONS = 3
MUTATION_RATE = 0.1
ELITISM_COUNT = 2

# Gene Definition (Min, Max)
GENE_BOUNDS = {
    'w_trend': (0.05, 0.40),
    'w_sniper': (0.05, 0.30),
    'w_quant': (0.05, 0.30),
    'w_pattern': (0.05, 0.25),
    'w_cycle': (0.05, 0.25),
    'w_sd': (0.05, 0.25),
    'w_div': (0.05, 0.20),
    'w_kin': (0.0, 0.20),
    'threshold': (30, 70),
    'chaos_threshold': (2.5, 4.0)
}

def create_individual():
    """Generate a random individual (genome)"""
    return {k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()}

def evaluate_fitness(individual, data_map):
    """
    Run backtest for a single individual and return fitness score.
    Fitness = Profit Factor * Win Rate * (1 - Drawdown/100)
    """
    # Create isolated engine instance
    engine = BacktestEngine()
    # Inject genes
    engine.consensus.update_parameters(individual)
    
    # Run Backtest (Silent mode ideally)
    # We use the existing run_preflight_check logic
    passed, metrics = engine.run_preflight_check(data_map)
    
    pf = metrics.get('pf', 0)
    wr = metrics.get('wr', 0)
    bal = metrics.get('balance', config.INITIAL_CAPITAL)
    
    # Simple Fitness: Profit * WinRate
    profit_pct = (bal - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL
    
    if profit_pct <= 0: return 0
    
    fitness = profit_pct * wr * 100
    return fitness

def crossover(parent1, parent2):
    """Uniform Crossover"""
    child = {}
    for k in GENE_BOUNDS.keys():
        child[k] = parent1[k] if random.random() > 0.5 else parent2[k]
    return child

def mutate(individual):
    """Random Mutation"""
    for k in GENE_BOUNDS.keys():
        if random.random() < MUTATION_RATE:
            # Mutate by +/- 10% or pick new random
            if random.random() > 0.5:
                change = individual[k] * 0.1 * (1 if random.random() > 0.5 else -1)
                individual[k] += change
            else:
                individual[k] = random.uniform(GENE_BOUNDS[k][0], GENE_BOUNDS[k][1])
            
            # Clamp bounds
            individual[k] = max(GENE_BOUNDS[k][0], min(GENE_BOUNDS[k][1], individual[k]))
    return individual

def run_optimization():
    logger.info("--- Starting Evolutionary Optimization ---")
    
    # Suppress noisy logs from ALL sub-modules dynamically
    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if name.startswith("Atl4s-") and name != "Atl4s-Optimizer":
            if isinstance(logger_obj, logging.PlaceHolder):
                continue
            logger_obj.setLevel(logging.ERROR)
            
    # Also explicitly silence known ones just in case they aren't initialized yet
    logging.getLogger("Atl4s-Consensus").setLevel(logging.ERROR)
    logging.getLogger("Atl4s-Backtest").setLevel(logging.ERROR)
    logging.getLogger("Atl4s-Strategy").setLevel(logging.ERROR)
    logging.getLogger("Atl4s-Quant").setLevel(logging.ERROR)
    logging.getLogger("Atl4s-Trend").setLevel(logging.ERROR)
    logging.getLogger("Atl4s-Patterns").setLevel(logging.ERROR)
    logging.getLogger("Atl4s-Cycle").setLevel(logging.ERROR)
    
    # Load Data Once
    loader = DataLoader()
    data_map = loader.get_data()
    if data_map['M5'] is None or len(data_map['M5']) < 500:
        logger.error("Insufficient data for optimization.")
        return

    # Optimization Speed Hack: Slice data to last 2000 candles (approx 1 week)
    # This makes the genetic algorithm run 10x faster while still testing recent market conditions.
    LOOKBACK = 2000
    if len(data_map['M5']) > LOOKBACK:
        logger.info(f"Slicing data to last {LOOKBACK} candles for speed...")
        data_map['M5'] = data_map['M5'].iloc[-LOOKBACK:]
        # Slice H1 accordingly
        start_time = data_map['M5'].index[0]
        if data_map['H1'] is not None:
            data_map['H1'] = data_map['H1'][data_map['H1'].index >= start_time]

    # Initialize Population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        logger.info(f"Generation {gen+1}/{GENERATIONS}")
        
        # Evaluate Fitness
        # Note: Parallelize this in production. For now, sequential to avoid pickling issues with complex objects.
        fitness_scores = []
        for i, ind in enumerate(population):
            logger.info(f"  > Evaluating Individual {i+1}/{len(population)}...")
            score = evaluate_fitness(ind, data_map)
            fitness_scores.append((ind, score))
            
        # Sort by Fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_ind, best_score = fitness_scores[0]
        logger.info(f"Best Fitness Gen {gen+1}: {best_score:.2f}")
        logger.info(f"Top Genes: {best_ind}")
        
        # Selection (Elitism + Tournament)
        new_pop = [x[0] for x in fitness_scores[:ELITISM_COUNT]]
        
        while len(new_pop) < POPULATION_SIZE:
            # Tournament Selection
            p1 = random.choice(fitness_scores[:10])[0]
            p2 = random.choice(fitness_scores[:10])[0]
            
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
            
        population = new_pop
        
    # Save Best Result
    final_best = population[0]
    logger.info("Optimization Complete.")
    logger.info(f"Optimal Parameters: {final_best}")
    
    # Write to config override or JSON
    import json
    with open("optimal_params.json", "w") as f:
        json.dump(final_best, f, indent=4)

if __name__ == "__main__":
    run_optimization()
