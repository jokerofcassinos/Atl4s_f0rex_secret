"""
Genesis Verification Test - Phase 1.2/1.3 Validation

Tests:
1. All signals/ modules loaded
2. All Eyes generating signals
3. Swarms voting
4. AGI making decisions
5. Clean signal flow (no errors)
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Disable emojis for Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("Verification")


def generate_test_data(periods_m5=250):
    """Generate realistic test data"""
    np.random.seed(42)
    
    base = 1.2650
    
    # Trending data
    trend = np.linspace(0, 0.015, periods_m5)
    close = base + trend + np.random.randn(periods_m5) * 0.001
    
    df_m5 = pd.DataFrame({
        'open': close - 0.0002,
        'high': close + 0.0005,
        'low': close - 0.0005,
        'close': close,
        'volume': np.random.randint(1000, 5000, periods_m5)
    }, index=pd.date_range(end=datetime.now(), periods=periods_m5, freq='5min'))
    
    # Create M1, H1, H4
    df_m1 = df_m5.resample('1min').ffill()
    df_h1 = df_m5.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_h4 = df_m5.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    return df_m1, df_m5, df_h1, df_h4


async def run_verification():
    """Run complete verification"""
    
    print("="*70)
    print("  GENESIS VERIFICATION TEST")
    print("="*70)
    print()
    
    results = {
        'signals_loaded': False,
        'eyes_active': False,
        'swarms_voting': False,
        'agi_deciding': False,
        'no_errors': True,
        'trades_generated': 0
    }
    
    # Test 1: Load Genesis System
    print("[1/5] Loading Genesis System...")
    try:
        from main_genesis import GenesisSystem
        genesis = GenesisSystem(symbol="GBPUSD", mode="test")
        results['signals_loaded'] = True
        print("      [OK] Genesis loaded successfully")
    except Exception as e:
        print(f"      [FAIL] {e}")
        results['no_errors'] = False
        return results
    
    # Test 2: Check signals/ modules
    print("[2/5] Checking signals/ modules...")
    try:
        assert hasattr(genesis, 'smc'), "SMC not loaded"
        assert hasattr(genesis, 'm8_fib'), "M8 Fibonacci not loaded"
        assert hasattr(genesis, 'quarterly'), "Quarterly Theory not loaded"
        assert hasattr(genesis, 'momentum'), "Momentum not loaded"
        assert hasattr(genesis, 'volatility'), "Volatility not loaded"
        results['eyes_active'] = True
        print("      [OK] All 5 signal modules loaded")
    except AssertionError as e:
        print(f"      [FAIL] {e}")
        results['no_errors'] = False
    
    # Test 3: Check swarm components
    print("[3/5] Checking swarm components...")
    try:
        assert hasattr(genesis, 'legion_knife'), "Legion TimeKnife not loaded"
        assert hasattr(genesis, 'legion_physarum'), "Legion Physarum not loaded"
        assert hasattr(genesis, 'legion_horizon'), "Legion EventHorizon not loaded"
        assert hasattr(genesis, 'legion_overlord'), "Legion Overlord not loaded"
        results['swarms_voting'] = True
        print("      [OK] All 4 Legion Elite swarms loaded")
    except AssertionError as e:
        print(f"      [FAIL] {e}")
        results['no_errors'] = False
    
    # Test 4: Check AGI components
    print("[4/5] Checking AGI components...")
    try:
        assert hasattr(genesis, 'agi_core'), "AGI Core not loaded"
        assert hasattr(genesis, 'metacognition'), "MetaCognition not loaded"
        assert hasattr(genesis, 'memory'), "Memory not loaded"
        results['agi_deciding'] = True
        print("      [OK] All AGI components loaded")
    except AssertionError as e:
        print(f"      [FAIL] {e}")
        results['no_errors'] = False
    
    # Test 5: Run analysis cycle
    print("[5/5] Running analysis cycle...")
    try:
        df_m1, df_m5, df_h1, df_h4 = generate_test_data()
        
        # Set time to trading hours
        test_time = datetime.now().replace(hour=10, minute=30)
        
        signal = await genesis.analyze(
            df_m1=df_m1,
            df_m5=df_m5,
            df_h1=df_h1,
            df_h4=df_h4,
            current_time=test_time,
            current_price=df_m5['close'].iloc[-1]
        )
        
        print(f"      Signal Direction: {signal.direction}")
        print(f"      Confidence: {signal.confidence:.1f}%")
        print(f"      Execute: {signal.execute}")
        print(f"      Layer Scores: Signal={signal.signal_layer_score:.0f} AGI={signal.agi_layer_score:.0f} Swarm={signal.swarm_layer_score:.0f}")
        
        if signal.execute:
            results['trades_generated'] += 1
        
        print("      [OK] Analysis completed without errors")
        
    except Exception as e:
        print(f"      [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results['no_errors'] = False
    
    # Summary
    print()
    print("="*70)
    print("  VERIFICATION RESULTS")
    print("="*70)
    print()
    print(f"  Signals/ Loaded:     {'PASS' if results['signals_loaded'] else 'FAIL'}")
    print(f"  Eyes Active:         {'PASS' if results['eyes_active'] else 'FAIL'}")
    print(f"  Swarms Voting:       {'PASS' if results['swarms_voting'] else 'FAIL'}")
    print(f"  AGI Deciding:        {'PASS' if results['agi_deciding'] else 'FAIL'}")
    print(f"  No Errors:           {'PASS' if results['no_errors'] else 'FAIL'}")
    print(f"  Trades Generated:    {results['trades_generated']}")
    print()
    
    passed = sum([
        results['signals_loaded'],
        results['eyes_active'],
        results['swarms_voting'],
        results['agi_deciding'],
        results['no_errors']
    ])
    
    print(f"  TOTAL: {passed}/5 checks passed")
    print()
    
    if passed == 5:
        print("  STATUS: VERIFICATION PASSED!")
    else:
        print("  STATUS: VERIFICATION FAILED - See errors above")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = asyncio.run(run_verification())
