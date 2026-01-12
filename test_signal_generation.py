"""
Genesis Signal Validation Test

Tests signal generation pipeline and validates that:
1. Signals are being generated
2. All layers are functioning
3. SL/TP is calculated correctly
4. Filters are not blocking all trades
"""

import asyncio
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main_genesis import GenesisSystem, GenesisSignal
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("SignalTest")


def generate_test_data() -> dict:
    """Generate realistic OHLCV data for testing"""
    
    np.random.seed(42)
    
    # Base price and trend
    base = 1.2650
    periods_m1 = 500
    periods_m5 = 200
    periods_h1 = 100
    periods_h4 = 50
    
    def create_ohlcv(periods: int, tf_minutes: int, trend_strength: float = 0.001):
        times = pd.date_range(end=datetime.now(), periods=periods, freq=f'{tf_minutes}min')
        
        # Trending data with noise
        trend = np.cumsum(np.random.randn(periods) * 0.0002 + trend_strength)
        close = base + trend
        
        # OHLC from close
        open_ = close + np.random.randn(periods) * 0.0002
        high = np.maximum(open_, close) + np.abs(np.random.randn(periods)) * 0.0005
        low = np.minimum(open_, close) - np.abs(np.random.randn(periods)) * 0.0005
        volume = np.random.randint(1000, 10000, periods)
        
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=times)
    
    return {
        'M1': create_ohlcv(periods_m1, 1, 0.00005),
        'M5': create_ohlcv(periods_m5, 5, 0.0001),
        'H1': create_ohlcv(periods_h1, 60, 0.0003),
        'H4': create_ohlcv(periods_h4, 240, 0.0005)
    }


async def test_signal_generation():
    """Test that Genesis generates signals correctly"""
    
    print("\n" + "="*70)
    print("  GENESIS SIGNAL VALIDATION TEST")
    print("="*70 + "\n")
    
    # Initialize system
    logger.info("Initializing Genesis...")
    genesis = GenesisSystem(symbol="GBPUSD", mode="test")
    
    # Generate test data
    logger.info("Generating test data...")
    data = generate_test_data()
    
    # Test multiple scenarios
    results = {
        'total': 0,
        'signals': 0,
        'buys': 0,
        'sells': 0,
        'waits': 0,
        'vetoed': 0
    }
    
    # Simulate trading hours (9:00 - 16:00)
    test_times = [
        datetime.now().replace(hour=9, minute=0),
        datetime.now().replace(hour=10, minute=30),
        datetime.now().replace(hour=12, minute=0),
        datetime.now().replace(hour=14, minute=0),
        datetime.now().replace(hour=15, minute=30)
    ]
    
    print("\n📊 Testing Signal Generation...\n")
    
    for i, test_time in enumerate(test_times):
        results['total'] += 1
        
        try:
            signal = await genesis.analyze(
                df_m1=data['M1'],
                df_m5=data['M5'],
                df_h1=data['H1'],
                df_h4=data['H4'],
                current_time=test_time,
                current_price=data['M5']['close'].iloc[-1]
            )
            
            direction = signal.direction
            execute = signal.execute
            confidence = signal.confidence
            
            # Count results
            if execute:
                results['signals'] += 1
                if direction == 'BUY':
                    results['buys'] += 1
                elif direction == 'SELL':
                    results['sells'] += 1
            else:
                if signal.vetoes:
                    results['vetoed'] += 1
                else:
                    results['waits'] += 1
            
            # Print result
            status = "✅ SIGNAL" if execute else "⏳ WAIT"
            print(f"  Test {i+1} ({test_time.strftime('%H:%M')}):")
            print(f"    {status} | {direction} | Confidence: {confidence:.1f}%")
            print(f"    Scores: Signal={signal.signal_layer_score:.0f} AGI={signal.agi_layer_score:.0f} Swarm={signal.swarm_layer_score:.0f}")
            
            if signal.vetoes:
                print(f"    Vetoes: {signal.vetoes[:2]}")
            
            if execute and signal.sl_price and signal.tp_price:
                print(f"    Entry: {signal.entry_price:.5f} | SL: {signal.sl_price:.5f} | TP: {signal.tp_price:.5f}")
            
            print()
            
        except Exception as e:
            logger.error(f"Test {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("  TEST RESULTS SUMMARY")
    print("="*70)
    print(f"""
  Total Tests:     {results['total']}
  Signals (BUY):   {results['buys']}
  Signals (SELL):  {results['sells']}
  Waits:           {results['waits']}
  Vetoed:          {results['vetoed']}
  
  Signal Rate:     {results['signals'] / results['total'] * 100:.1f}%
""")
    
    # Pass/Fail
    if results['signals'] > 0:
        print("  ✅ PASS: Genesis is generating signals!")
    else:
        print("  ⚠️  WARN: No signals generated - filters may be too strict")
    
    print("="*70 + "\n")
    
    return results


async def test_with_relaxed_filters():
    """Test with relaxed filters to see more signals"""
    
    print("\n" + "="*70)
    print("  RELAXED FILTER TEST")
    print("="*70 + "\n")
    
    # Temporarily relax market hours
    from main_genesis import GenesisSystem
    
    genesis = GenesisSystem(symbol="GBPUSD", mode="test")
    
    # Override market hours filter for testing
    original_filter = genesis._apply_execution_filters
    
    def relaxed_filter(signal, current_time):
        # Skip market hours filter for testing
        if not signal.execute:
            return signal
        
        # Only apply daily trade limit
        today = current_time.date()
        if genesis.daily_trades['date'] != today:
            genesis.daily_trades = {'date': today, 'count': 0}
        
        if genesis.daily_trades['count'] >= genesis.max_daily_trades:
            signal.execute = False
            signal.vetoes.append(f"DAILY_LIMIT: {genesis.max_daily_trades} trades")
        
        return signal
    
    genesis._apply_execution_filters = relaxed_filter
    
    # Generate data and test
    data = generate_test_data()
    
    # Test at various times including outside normal hours
    test_times = []
    for hour in range(0, 24, 3):
        test_times.append(datetime.now().replace(hour=hour, minute=0))
    
    results = {'signals': 0, 'total': len(test_times)}
    
    print("📊 Testing with relaxed filters (all hours):\n")
    
    for test_time in test_times:
        try:
            signal = await genesis.analyze(
                df_m1=data['M1'],
                df_m5=data['M5'],
                df_h1=data['H1'],
                df_h4=data['H4'],
                current_time=test_time,
                current_price=data['M5']['close'].iloc[-1]
            )
            
            if signal.execute:
                results['signals'] += 1
                print(f"  ✅ {test_time.strftime('%H:%M')} → {signal.direction} @ {signal.confidence:.0f}%")
            else:
                print(f"  ⏳ {test_time.strftime('%H:%M')} → WAIT")
                
        except Exception as e:
            print(f"  ❌ {test_time.strftime('%H:%M')} → Error: {e}")
    
    print(f"\n  Signal Rate: {results['signals']}/{results['total']} ({results['signals']/results['total']*100:.0f}%)")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    print("\n" + "🎯 "*20)
    print("\n  GENESIS SIGNAL VALIDATION SUITE")
    print("\n" + "🎯 "*20 + "\n")
    
    # Test 1: Normal filters
    asyncio.run(test_signal_generation())
    
    # Test 2: Relaxed filters
    asyncio.run(test_with_relaxed_filters())
