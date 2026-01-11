# File Analysis #13: Signals Folder (Laplace Demon Modules)

## Summary

| File | Lines | Classes | Status |
|------|-------|---------|--------|
| `structure.py` | 1085 | SMCAnalyzer, InstitutionalLevels, BlackRockPatterns | ✅ |
| `momentum.py` | 433 | MomentumAnalyzer, ToxicFlowDetector | ✅ |
| `timing.py` | 661 | QuarterlyTheory, M8FibonacciSystem, TimeMacroFilter | ✅ |
| `correlation.py` | 569 | SMTDivergence, PowerOfOne, InversionFVG, AMDPowerOfThree | ✅ |
| `volatility.py` | 413 | VolatilityAnalyzer, DisplacementCandle, BalancedPriceRange | ✅ |
| **Total** | **3161** | **15 classes** | **✅ All OK** |

---

## Key Concepts Implemented

### structure.py (SMC Core)
- **Order Blocks**: BOS-validated institutional zones
- **Fair Value Gaps**: Vectorized detection
- **Liquidity Pools**: Fractal-based with SFP validation
- **IPDA Ranges**: 20/40/60 day institutional levels
- **BlackRock Patterns**: Seek & Destroy, Iceberg Detection

### momentum.py
- **RSI/MACD/Stochastic**: Standard with divergence
- **ToxicFlowDetector**: Compression/Expansion patterns
- **Exhaustion Vector**: Parabolic move detection

### timing.py
- **Quarterly Theory**: 90-min cycles with Q1-Q4
- **M8 Fibonacci**: 8-min cycles with triple validation
- **Time Macros**: xx:50-xx:10 window detection
- **Initial Balance**: London/NY session filters

### correlation.py
- **SMT Divergence**: Cross-pair weakness detection
- **Power of One**: Standard deviation bands
- **Inversion FVG**: Support/resistance flip
- **AMD Power of Three**: Judas Swing detection

### volatility.py
- **Volatility Regimes**: LOW/NORMAL/HIGH/EXTREME
- **Displacement Candles**: >2x ATR + 70% body
- **Balanced Price Range**: Bullish+Bearish FVG overlap

---

## Code Quality: ✅ PROFESSIONAL GRADE

All files feature:
- Proper error handling
- Safe default returns
- Vectorized calculations
- Dataclass returns
- Comprehensive logging
