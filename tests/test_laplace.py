"""
LAPLACE DEMON - Test Suite
═══════════════════════════

Comprehensive tests for all signal modules.
Run with: python -m pytest tests/test_laplace.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_test_ohlcv(n_bars: int = 200, 
                        start_price: float = 1.2500,
                        volatility: float = 0.0010,
                        trend: float = 0.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)  # Reproducible
    
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    prices = [start_price]
    for i in range(1, n_bars):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] + change)
    
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close + abs(np.random.normal(0, volatility/2))
        low = close - abs(np.random.normal(0, volatility/2))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(100, 1000)
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def generate_bullish_data(n_bars: int = 100) -> pd.DataFrame:
    """Generate clearly bullish trending data."""
    return generate_test_ohlcv(n_bars, trend=0.0005, volatility=0.0008)


def generate_bearish_data(n_bars: int = 100) -> pd.DataFrame:
    """Generate clearly bearish trending data."""
    return generate_test_ohlcv(n_bars, trend=-0.0005, volatility=0.0008)


def generate_ranging_data(n_bars: int = 100) -> pd.DataFrame:
    """Generate ranging/consolidating data."""
    return generate_test_ohlcv(n_bars, trend=0.0, volatility=0.0005)


# ============================================================================
# TIMING MODULE TESTS
# ============================================================================

class TestQuarterlyTheory:
    """Tests for Quarterly Theory (90-minute cycles)."""
    
    def test_quarter_detection(self):
        """Test that quarters are correctly identified."""
        from signals.timing import QuarterlyTheory
        
        qt = QuarterlyTheory()
        
        # Q1 (0-22.5 min) - should be accumulation
        time_q1 = datetime(2024, 1, 1, 10, 10)  # 10 minutes into hour
        quarter, minutes = qt.get_current_quarter(time_q1)
        assert quarter.value == "Q1"
        
        # Q2 (22.5-45 min) - should be manipulation
        time_q2 = datetime(2024, 1, 1, 10, 30)
        quarter, minutes = qt.get_current_quarter(time_q2)
        assert quarter.value == "Q2"
        
        # Q3 (45-67.5 min) - should be distribution
        time_q3 = datetime(2024, 1, 1, 10, 50)
        quarter, minutes = qt.get_current_quarter(time_q3)
        assert quarter.value == "Q3"
        
        # Q4 (67.5-90 min) - should be continuation
        time_q4 = datetime(2024, 1, 1, 11, 20)
        quarter, minutes = qt.get_current_quarter(time_q4)
        assert quarter.value == "Q4"
    
    def test_golden_zone_identification(self):
        """Test that Q3 is identified as golden zone."""
        from signals.timing import QuarterlyTheory
        
        qt = QuarterlyTheory()
        df = generate_test_ohlcv(50)
        
        # Q3 time
        time_q3 = datetime(2024, 1, 1, 10, 50)
        result = qt.analyze(time_q3, df)
        
        assert result.is_golden_zone == True
        assert result.tradeable == True


class TestM8FibonacciSystem:
    """Tests for M8 Fibonacci timing system."""
    
    def test_gate_detection(self):
        """Test M8 gate identification."""
        from signals.timing import M8FibonacciSystem
        
        m8 = M8FibonacciSystem()
        
        # Q1 gate (0-2 min) - dead zone
        time_q1 = datetime(2024, 1, 1, 10, 0, 30)  # 30 seconds in
        result = m8.get_m8_position(time_q1)
        assert result['gate'] == 'Q1'
        assert result['tradeable'] == False
        
        # Q3 gate (4-6 min) - golden zone
        time_q3 = datetime(2024, 1, 1, 10, 4, 30)  # 4.5 minutes in
        result = m8.get_m8_position(time_q3)
        assert result['gate'] == 'Q3'
        assert result['tradeable'] == True
        assert result['score'] > 0


# ============================================================================
# STRUCTURE MODULE TESTS
# ============================================================================

class TestSMCAnalyzer:
    """Tests for SMC (Smart Money Concepts) analyzer."""
    
    def test_trend_detection(self):
        """Test that trends are correctly identified."""
        from signals.structure import SMCAnalyzer
        
        smc = SMCAnalyzer()
        
        # Bullish data
        df_bull = generate_bullish_data(100)
        result = smc.analyze(df_bull)
        assert result['trend'] == 'BULLISH'
        
        # Bearish data
        df_bear = generate_bearish_data(100)
        smc2 = SMCAnalyzer()
        result = smc2.analyze(df_bear)
        assert result['trend'] == 'BEARISH'
    
    def test_order_block_detection(self):
        """Test order block detection."""
        from signals.structure import SMCAnalyzer
        
        smc = SMCAnalyzer()
        df = generate_test_ohlcv(100)
        
        result = smc.analyze(df)
        
        # Should detect some order blocks
        assert 'active_order_blocks' in result


class TestBlackRockPatterns:
    """Tests for BlackRock/Aladdin pattern detection."""
    
    def test_month_end_detection(self):
        """Test month-end rebalancing detection."""
        from signals.structure import BlackRockPatterns
        
        br = BlackRockPatterns()
        
        # Last day of month
        end_of_month = datetime(2024, 1, 31, 16, 0)
        result = br.is_month_end_rebalancing(end_of_month)
        assert result['in_rebalancing_window'] == True
        
        # Middle of month
        mid_month = datetime(2024, 1, 15, 10, 0)
        result = br.is_month_end_rebalancing(mid_month)
        assert result['in_rebalancing_window'] == False


class TestGannGeometry:
    """Tests for Gann geometric patterns."""
    
    def test_gann_grid_calculation(self):
        """Test Gann level calculation."""
        from signals.structure import GannGeometry
        
        gann = GannGeometry()
        
        swing_low = 1.2500
        levels = gann.calculate_gann_grid(swing_low)
        
        assert levels['anchor'] == swing_low
        assert 'L36' in levels['levels']
        assert 'L72' in levels['levels']
        assert 'L144' in levels['levels']
        
        # Check correct calculation
        assert abs(levels['levels']['L36'] - 1.2536) < 0.0001
        assert abs(levels['levels']['L144'] - 1.2644) < 0.0001


class TestTeslaVortex:
    """Tests for Tesla 3-6-9 pattern detection."""
    
    def test_consecutive_candle_count(self):
        """Test consecutive candle counting."""
        from signals.structure import TeslaVortex
        
        tesla = TeslaVortex()
        
        # Create data with 5 consecutive bullish candles
        df = generate_bullish_data(20)
        
        result = tesla.count_consecutive_candles(df)
        
        assert 'count' in result
        assert 'direction' in result


# ============================================================================
# CORRELATION MODULE TESTS
# ============================================================================

class TestSMTDivergence:
    """Tests for SMT divergence detection."""
    
    def test_correlation_calculation(self):
        """Test correlation calculation between pairs."""
        from signals.correlation import SMTDivergence
        
        smt = SMTDivergence()
        
        # Generate two correlated datasets
        df1 = generate_test_ohlcv(100)
        df2 = generate_test_ohlcv(100)
        
        correlation = smt.calculate_correlation(df1, df2)
        
        # Should return a correlation value
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1


class TestPowerOfOne:
    """Tests for Power of One (SD bands)."""
    
    def test_band_calculation(self):
        """Test standard deviation band calculation."""
        from signals.correlation import PowerOfOne
        
        po = PowerOfOne()
        
        df_daily = generate_test_ohlcv(30)
        bands = po.calculate_bands(df_daily, session_open=1.2500)
        
        assert 'open' in bands
        assert '+1SD' in bands
        assert '-1SD' in bands
        assert '+2.5SD' in bands
        assert '-2.5SD' in bands


# ============================================================================
# MOMENTUM MODULE TESTS
# ============================================================================

class TestMomentumAnalyzer:
    """Tests for momentum analysis."""
    
    def test_full_analysis(self):
        """Test complete momentum analysis."""
        from signals.momentum import MomentumAnalyzer
        
        ma = MomentumAnalyzer()
        df = generate_test_ohlcv(100)
        
        result = ma.analyze(df)
        
        assert 'rsi' in result
        assert 'macd' in result
        assert 'stochastic' in result
        assert 'composite' in result


class TestToxicFlowDetector:
    """Tests for toxic flow detection."""
    
    def test_compression_detection(self):
        """Test compression pattern detection."""
        from signals.momentum import ToxicFlowDetector
        
        tfd = ToxicFlowDetector()
        
        # Ranging data should show compression
        df = generate_ranging_data(50)
        
        result = tfd.detect_compression(df)
        
        assert 'detected' in result


# ============================================================================
# VOLATILITY MODULE TESTS
# ============================================================================

class TestVolatilityAnalyzer:
    """Tests for volatility analysis."""
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        from signals.volatility import VolatilityAnalyzer
        
        va = VolatilityAnalyzer()
        df = generate_test_ohlcv(100)
        
        result = va.analyze(df)
        
        assert 'atr' in result
        assert result['atr']['value'] > 0
    
    def test_regime_classification(self):
        """Test volatility regime classification."""
        from signals.volatility import VolatilityAnalyzer
        
        va = VolatilityAnalyzer()
        df = generate_test_ohlcv(100)
        
        result = va.analyze(df)
        
        assert result['regime'] is not None
        assert result['regime'].regime in ['LOW', 'NORMAL', 'HIGH', 'EXTREME']


# ============================================================================
# LAPLACE DEMON CORE TESTS
# ============================================================================

class TestLaplaceDemonCore:
    """Tests for the main Laplace Demon core."""
    
    def test_initialization(self):
        """Test core initialization."""
        from core.laplace_demon import LaplaceDemonCore
        
        laplace = LaplaceDemonCore(symbol="GBPUSD")
        
        assert laplace.symbol == "GBPUSD"
        assert laplace.quarterly is not None
        assert laplace.m8_fib is not None
        assert laplace.smc is not None
    
    def test_simple_signal(self):
        """Test simple signal generation."""
        from core.laplace_demon import LaplaceDemonCore
        
        laplace = LaplaceDemonCore(symbol="GBPUSD")
        df = generate_test_ohlcv(200)
        
        direction, sl, tp, conf, source = laplace.get_simple_signal(df)
        
        # Should return valid structure even if no signal
        assert source is not None
    
    def test_full_analysis(self):
        """Test full analysis pipeline."""
        from core.laplace_demon import LaplaceDemonCore
        
        laplace = LaplaceDemonCore(symbol="GBPUSD")
        
        df_m5 = generate_test_ohlcv(200)
        df_h1 = df_m5.resample('1h').agg({
            'open': 'first', 'high': 'max', 
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_h4 = df_m5.resample('4h').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        prediction = laplace.analyze(
            df_m1=None,
            df_m5=df_m5,
            df_h1=df_h1,
            df_h4=df_h4,
            current_time=datetime.now()
        )
        
        assert prediction is not None
        assert prediction.direction in ['BUY', 'SELL', 'WAIT']
        assert 0 <= prediction.confidence <= 100


# ============================================================================
# BACKTEST ENGINE TESTS
# ============================================================================

class TestBacktestEngine:
    """Tests for backtest engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        from backtest.engine import BacktestEngine, BacktestConfig
        
        config = BacktestConfig(
            initial_capital=30.0,
            leverage=3000.0,
            symbol="GBPUSD"
        )
        
        engine = BacktestEngine(config)
        
        assert engine.balance == 30.0
        assert engine.config.symbol == "GBPUSD"
    
    def test_pip_calculation(self):
        """Test pip calculation."""
        from backtest.engine import BacktestEngine, BacktestConfig
        
        config = BacktestConfig(initial_capital=30.0)
        engine = BacktestEngine(config)
        
        pips = engine.calculate_pips(1.2500, 1.2550)
        assert abs(pips - 50) < 0.1


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🔮 LAPLACE DEMON TEST SUITE")
    print("═" * 60 + "\n")
    
    # Run with pytest
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above.")
