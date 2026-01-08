import logging
import pandas as pd
import numpy as np
from core.agi.learning import HistoryLearningEngine
from core.memory.holographic import HolographicMemory

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase6")

def test_dream_cycle():
    print("\n--- TEST: HISTORY LEARNING (DREAM CYCLE) ---")
    
    # 1. Setup Memory and Engine
    memory = HolographicMemory(dimensions=1024)
    engine = HistoryLearningEngine(memory)
    
    # 2. Create Synthetic History (Sine Wave)
    # 500 candles. Sine wave makes prediction easy.
    # Top of sine -> Sell (Price goes down).
    # Bottom of sine -> Buy (Price goes up).
    
    indices = np.arange(500)
    prices = 100 + 10 * np.sin(indices * 0.1) # Period is approx 60
    
    # Pandas DF
    df = pd.DataFrame({'close': prices})
    
    print(f"Generated {len(df)} candles of historical data.")
    
    # 3. Dream!
    print("Starting Dream Cycle...")
    engine.dream_cycle(df, batch_size=50) # Sample 50 points
    
    print(engine.get_status())
    
    if engine.experiences_learned > 0:
        print(f"PASS: Learned {engine.experiences_learned} experiences.")
    else:
        print("FAIL: No experiences learned.")
        
    # 4. Verify Memory Changes
    # Pick a 'Top' context (high value, flat)
    # In sine wave, sin(x) ~ 1.
    top_price = 110.0
    ctx = {
        'last_price': top_price,
        'volatility': 0.0,
        'prices': [top_price]*50 # Implies flat top
    }
    
    # Note: Our simple dream engine encodes context simply.
    # Real test depends on how 'store_experience' works.
    # We just check if intuition is non-zero.
    
    intuition = memory.retrieve_intuition(ctx)
    print(f"Memory Intuition after Dreaming: {intuition}")
    
    if intuition != 0.0:
        print("PASS: Memory now has intuition about this context.")
    else:
        print("FAIL: Memory is empty/flat.")

if __name__ == "__main__":
    test_dream_cycle()
