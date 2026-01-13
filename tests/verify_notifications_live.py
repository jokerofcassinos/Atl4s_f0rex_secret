
import os
import sys
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.execution_engine import ExecutionEngine

async def test_closure_notification():
    print("🧪 Testing ExecutionEngine Closure Notification...")
    
    # 1. Setup mocks
    bridge = MagicMock()
    notifier = MagicMock()
    notifier.notify_trade_exit = AsyncMock() # Must be async
    
    engine = ExecutionEngine(bridge, notifier)
    
    # 2. Add an active trade to context
    test_ticket = 99999
    test_data = {
        'source': 'SNIPER_GOLDEN_GATE',
        'confidence': 85.0,
        'direction': 'SELL',
        'entry_price': 1.2600,
        'lots': 0.05,
        'symbol': 'GBPUSD'
    }
    engine.trade_sources[test_ticket] = test_data
    
    # 3. Simulate a tick where this trade is MISSING
    # Current tick data
    tick = {
        'symbol': 'GBPUSD',
        'bid': 1.2550,
        'ask': 1.2552,
        'trades_json': [] # Empty trades list = closure detection
    }
    
    # 4. Run monitor
    await engine.monitor_positions(tick)
    
    # 5. Verify Notification
    if notifier.notify_trade_exit.called:
        args, kwargs = notifier.notify_trade_exit.call_args
        print(f"✅ Notification TRIGGERED!")
        print(f"📊 Symbol: {kwargs.get('symbol')} | Reason: {kwargs.get('reason')}")
        print(f"💰 PnL Dollars: ${kwargs.get('pnl_dollars'):.2f} | Pips: {kwargs.get('pnl_pips'):.1f}")
        
        # Check if approx pnl is positive (Sell from 1.2600 to 1.2550 is a win)
        if kwargs.get('pnl_dollars') > 0:
            print("✅ Profit calculation looks correct.")
            return True
        else:
            print(f"❌ Profit calculation mismatch: {kwargs.get('pnl_dollars')}")
    else:
        print("❌ Notification was NOT triggered.")
    
    return False

if __name__ == "__main__":
    success = asyncio.run(test_closure_notification())
    if success:
        print("\n✨ NOTIFICATION VERIFIED!")
        sys.exit(0)
    else:
        print("\n🛑 NOTIFICATION FAILED!")
        sys.exit(1)
