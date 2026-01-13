
import os
import sys
import asyncio
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.execution_engine import ExecutionEngine

async def test_persistence():
    print("🧪 Testing ExecutionEngine Persistence...")
    
    # 1. Setup
    bridge = MagicMock()
    notifier = MagicMock()
    engine = ExecutionEngine(bridge, notifier)
    
    # 2. Add some fake trades
    test_ticket = 12345
    test_data = {
        'source': 'MOMENTUM_BREAKOUT',
        'confidence': 99.0,
        'direction': 'BUY',
        'entry_price': 1.2500,
        'lots': 0.1,
        'symbol': 'GBPUSD'
    }
    engine.trade_sources[test_ticket] = test_data
    
    # 3. Save
    engine._save_context()
    print(f"✅ Context saved to {engine.context_path}")
    
    # 4. Clear and Load new engine
    new_engine = ExecutionEngine(bridge, notifier)
    # Check if loaded automatically in __init__
    
    if test_ticket in new_engine.trade_sources:
        stored = new_engine.trade_sources[test_ticket]
        if stored['source'] == 'MOMENTUM_BREAKOUT' and stored['confidence'] == 99.0:
            print(f"✅ Context recovered successfully! Ticket {test_ticket} found.")
            return True
        else:
            print(f"❌ Context recovered but data mismatch: {stored}")
    else:
        print(f"❌ Ticket {test_ticket} not found in recovered context.")
    
    return False

if __name__ == "__main__":
    success = asyncio.run(test_persistence())
    if success:
        print("\n✨ PERSISTENCE VERIFIED!")
        sys.exit(0)
    else:
        print("\n🛑 PERSISTENCE FAILED!")
        sys.exit(1)
