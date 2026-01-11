# File Analysis #15: src/ Utility Files

## Summary

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `strategy_core.py` | 125 | Legacy strategy | ⚠️ Bug |
| `quantum_math.py` | 110 | Advanced math | ✅ |
| `macro_math.py` | 181 | Macro analysis | ✅ |
| Others | ~300 | Utilities | - |

---

## strategy_core.py

### ⚠️ BUG FOUND (line 55)
```python
# Line 55:
if undersold and prob_bullish > 0.6:  # TYPO: 'undersold' -> 'oversold'
```

**Issue:** Variable `undersold` is undefined. Should be `oversold`.

### Otherwise OK
- Simple strategy with Z-Score, RSI, EMA
- ATR-based SL/TP (1.5x / 3x)
- Fixed 0.01 lot for $30 account

---

## quantum_math.py ✅

### Advanced Functions
| Function | Purpose |
|----------|---------|
| `calculate_entropy()` | Shannon entropy for chaos |
| `calculate_hurst_exponent()` | Trend vs mean reversion |
| `fisher_information_curvature()` | Manifold geometry |
| `kalman_filter()` | Price smoothing |
| `z_score()` | Statistical deviation |

All vectorized with proper error handling.

---

## macro_math.py ✅

### Advanced Functions
| Function | Purpose |
|----------|---------|
| `garch_11_forecast()` | Volatility prediction |
| `wavelet_haar_mra()` | Multi-resolution coherence |
| `calculate_cointegration()` | Engle-Granger test |
| `bayesian_regime_detect()` | Expansion/Contraction |

All have proper error handling and edge case guards.

---

## Verdict

| File | Status |
|------|--------|
| strategy_core.py | ⚠️ Fix typo |
| quantum_math.py | ✅ OK |
| macro_math.py | ✅ OK |
