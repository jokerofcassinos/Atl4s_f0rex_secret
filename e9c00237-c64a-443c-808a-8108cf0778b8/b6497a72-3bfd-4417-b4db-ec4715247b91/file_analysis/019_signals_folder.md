# Analysis 019: signals/ Folder Audit

## Overview

The `signals/` folder contains **Laplace Demon** signal generation modules.
Total: **5 core modules** + `__init__.py`

---

## Module Breakdown

### 1. `structure.py` (1085 lines, 40KB)
**Purpose:** Smart Money Concepts (SMC) and institutional patterns.

**Classes:**
- `SMCAnalyzer` - Order Blocks, FVG, BOS, CHoCH detection
- `InstitutionalLevels` - .00/.20/.50/.80 grid, IPDA ranges
- `StructureAnalyzer` - Main structure analysis

**Key Functions:**
- `_detect_order_blocks()` - Validated OB detection
- `_detect_fvgs()` - Fair Value Gap detection
- `_detect_liquidity_pools()` - Stop hunt targets

**Status:** ✅ Well implemented, follows ICT methodology

---

### 2. `momentum.py` (433 lines, 15KB)
**Purpose:** Momentum indicators and divergence detection.

**Classes:**
- `MomentumAnalyzer` - RSI, MACD, Stochastic analysis
- `ToxicFlowDetector` - Compression/Expansion detection

**Key Functions:**
- `_detect_rsi_divergence()` - Bullish/Bearish divergence
- `detect_compression()` - Staircasing detection
- `detect_exhaustion()` - Parabolic move detection

**Status:** ✅ Good implementation

---

### 3. `timing.py` (661 lines, 23KB)
**Purpose:** Institutional timing theories.

**Classes:**
- `QuarterlyTheory` - 90-minute Q1-Q4 cycles
- `M8FibonacciSystem` - 8-minute Fibonacci gates

**Key Features:**
- Q1 Accumulation → Q2 Manipulation → Q3 Distribution → Q4 Continuation
- M8 Golden Zone detection (minutes 4-6)
- Session-aware (London, NY, Asian)

**Status:** ✅ Advanced implementation

---

### 4. `correlation.py` (21KB)
**Purpose:** Multi-asset correlation analysis.

**Need to analyze:** Check if correlations affect signal generation.

---

### 5. `volatility.py` (14KB)
**Purpose:** Volatility regimes and ATR analysis.

**Need to analyze:** Check integration with execution.

---

## Legacy Comparison

| Module | Legacy Exists? | Differences |
|--------|---------------|-------------|
| `structure.py` | ❌ NEW | SMC not in legacy |
| `momentum.py` | ❌ NEW | Advanced momentum |
| `timing.py` | ❌ NEW | M8 Fibonacci new |
| `correlation.py` | ❌ NEW | Multi-asset new |
| `volatility.py` | ❌ NEW | ATR analysis new |

> **Note:** These are ALL NEW modules not present in legacy!
> They need to be properly integrated with execution path.

---

## Potential Issues

### 🟡 Issue 1: Not Connected to Main Execution
These modules may not be used in the main execution flow (check `main.py`).

### 🟡 Issue 2: Added Complexity
5 new signal modules = more latency and potential conflicts.

---

## Recommendation

**Verify Integration:** Check if `signals/` modules are called in main.py or consensus.py.
If not, they're "dead code" and not contributing to decisions.
