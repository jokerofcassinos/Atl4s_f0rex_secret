"""
LAPLACE DEMON - Signal Generation Module v2.0
══════════════════════════════════════════════

Consolidated signal generation with advanced institutional theories:
- Quarterly Theory (90-minute cycles)
- M8 Fibonacci (8-minute timing)
- SMC/ICT Structure (Order Blocks, FVG)
- BlackRock Patterns (Seek & Destroy, Iceberg)
- Cross-Asset Correlation (SMT Divergence)
- Momentum & Volatility Analysis
"""

# Timing Analysis
from .timing import (
    QuarterlyTheory,
    M8FibonacciSystem,
    TimeMacroFilter,
    InitialBalanceFilter
)

# Structure Analysis (SMC & BlackRock)
from .structure import (
    SMCAnalyzer,
    InstitutionalLevels,
    BlackRockPatterns,
    VectorCandleTheory,
    GannGeometry,
    TeslaVortex
)

# Correlation Analysis
from .correlation import (
    SMTDivergence,
    PowerOfOne,
    InversionFVG,
    MeanThreshold,
    AMDPowerOfThree
)

# Momentum Analysis
from .momentum import (
    MomentumAnalyzer,
    ToxicFlowDetector
)

# Volatility Analysis
from .volatility import (
    VolatilityAnalyzer,
    DisplacementCandle,
    VolatilityFilter,
    BalancedPriceRange
)

__all__ = [
    # Timing
    'QuarterlyTheory',
    'M8FibonacciSystem',
    'TimeMacroFilter',
    'InitialBalanceFilter',
    
    # Structure
    'SMCAnalyzer',
    'InstitutionalLevels',
    'BlackRockPatterns',
    'VectorCandleTheory',
    'GannGeometry',
    'TeslaVortex',
    
    # Correlation
    'SMTDivergence',
    'PowerOfOne',
    'InversionFVG',
    'MeanThreshold',
    'AMDPowerOfThree',
    
    # Momentum
    'MomentumAnalyzer',
    'ToxicFlowDetector',
    
    # Volatility
    'VolatilityAnalyzer',
    'DisplacementCandle',
    'VolatilityFilter',
    'BalancedPriceRange'
]
