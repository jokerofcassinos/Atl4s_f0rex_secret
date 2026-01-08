
import logging
import os
from core.agi.learning import HistoryLearningEngine

# Mock Memory
class MockMemory:
    def __init__(self): pass

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLearning")

def test_history_learning():
    logger.info("--- Testing Phase 6: Empirical Learning ---")
    
    # 1. Init
    engine = HistoryLearningEngine(MockMemory())
    
    # Clear DB for test
    if os.path.exists(engine.history_file):
        os.remove(engine.history_file)
    engine.closed_trades = []
        
    # 2. Record Trades
    logger.info("Recording simulated trades...")
    engine.record_trade_closure(1001, 50.0, None, {'strategy': 'RSI'})
    engine.record_trade_closure(1002, -20.0, None, {'strategy': 'MACD'})
    engine.record_trade_closure(1003, 100.0, None, {'strategy': 'Wolf'})
    
    # 3. Analyze
    stats = engine.analyze_patterns()
    logger.info(f"Analysis: {stats}")
    
    if stats['win_rate'] > 0.6 and stats['total_profit'] == 130.0:
        logger.info("SUCCESS: Pattern Analysis correct.")
    else:
        logger.warning(f"FAILURE: Win Rate {stats['win_rate']} | Profit {stats['total_profit']}")
        
    # 4. Ingest External
    external = [{'ticket': 1004, 'profit': 10.0, 'meta': {}}]
    engine.ingest_mt5_history(external)
    
    if len(engine.closed_trades) == 4:
         logger.info("SUCCESS: Ingestion correct.")
    else:
         logger.warning("FAILURE: Ingestion count mismatch.")

if __name__ == "__main__":
    test_history_learning()
