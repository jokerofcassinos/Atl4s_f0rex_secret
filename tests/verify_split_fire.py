
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_laplace import LaplaceTradingSystem
from core.laplace_demon import LaplacePrediction, SignalStrength

# Configure Logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

class TestSplitFire(unittest.IsolatedAsyncioTestCase):
    async def test_split_fire_execution(self):
        print("\n🧪 TESTING SPLIT FIRE LOGIC 🧪")
        
        # 1. Initialize System (Mocking ZMQ port to avoid real connection)
        system = LaplaceTradingSystem(zmq_port=9999, symbol="TEST_PAIR")
        
        # 2. Mock Executor
        system.executor = AsyncMock()
        system.executor.execute_trade.return_value = 12345 # Fake Ticket ID
        
        # 3. Create Prediction with Multiplier 5.0
        prediction = LaplacePrediction(
            direction="BUY",
            confidence=95.0,
            lot_multiplier=5.0, # <--- THE KEY
            sl_pips=20,
            tp_pips=40,
            sl_price=1.2480,
            tp_price=1.2540,
            reasons=["Test Reason"],
            strength=SignalStrength.DIVINE,
            execute=True
        )
        
        # 4. Create Fake Tick
        tick = {
            'symbol': 'TEST_PAIR',
            'ask': 1.2500,
            'bid': 1.2498,
            'time': 1234567890
        }
        
        # 5. Execute Signal
        print("► Dispatching Signal with Multiplier 5.0...")
        await system._execute_signal(prediction, tick)
        
        # 6. Verify Call Count
        call_count = system.executor.execute_trade.call_count
        print(f"► Execute Trade Called: {call_count} times")
        
        self.assertEqual(call_count, 5, "Should have executed 5 separate orders")
        
        # 7. Verify Comments
        calls = system.executor.execute_trade.call_args_list
        for i, call in enumerate(calls):
            args, kwargs = call
            comment = kwargs.get('comment', '')
            print(f"  Order {i+1}: {comment}")
            self.assertIn(f"SPLIT_{i+1}", comment, f"Comment should contain SPLIT_{i+1}")

    async def test_standard_execution(self):
        print("\n🧪 TESTING STANDARD EXECUTION 🧪")
        system = LaplaceTradingSystem(zmq_port=9998, symbol="TEST_PAIR")
        system.executor = AsyncMock()
        system.executor.execute_trade.return_value = 12345
        
        prediction = LaplacePrediction(
            direction="SELL",
            confidence=85.0,
            lot_multiplier=1.0, # Standard
            sl_pips=20,
            tp_pips=40,
            sl_price=1.2520,
            tp_price=1.2460,
            reasons=["Standard Test"],
            strength=SignalStrength.STRONG,
            execute=True
        )
        
        tick = {'symbol': 'TEST_PAIR', 'ask': 1.2500, 'bid': 1.2498}
        
        print("► Dispatching Signal with Multiplier 1.0...")
        await system._execute_signal(prediction, tick)
        
        call_count = system.executor.execute_trade.call_count
        print(f"► Execute Trade Called: {call_count} times")
        self.assertEqual(call_count, 1, "Should have executed 1 order")

if __name__ == '__main__':
    unittest.main()
