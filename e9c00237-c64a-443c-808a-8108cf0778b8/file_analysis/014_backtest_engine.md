# File Analysis #14: Backtest Engine

## Summary

| Metric | Value |
|--------|-------|
| **File** | `backtest/engine.py` |
| **Lines** | 704 |
| **Classes** | Trade, BacktestConfig, BacktestResult, BacktestEngine |
| **Status** | ✅ Professional Grade |

---

## Key Features

### Trade Class
- Full metadata tracking (MFE, MAE, duration)
- R-Multiple calculation
- Signal source attribution

### BacktestConfig
- Realistic defaults ($30, 3000x leverage)
- Spread/slippage modeling
- Session hour filters

### BacktestEngine

#### Anti-Ruin Protection (lines 247-273)
```python
# 1. FreeMargin Check
MIN_FREE_MARGIN = 15.0
if free_margin < MIN_FREE_MARGIN:
    return None  # Block trade

# 2. Max Risk Check
MAX_RISK_PCT = 5.0
if potential_loss > max_risk_dollars:
    # Reduce trade size
```

#### Execution Realism
- Spread application (Buy at Ask)
- Random slippage (0-0.5 pips)
- Position sizing based on SL distance

#### Metrics Calculated
| Metric | Calculation |
|--------|-------------|
| Win Rate | Winners / Total |
| Profit Factor | Gross Profit / Gross Loss |
| Sharpe Ratio | Annualized Mean / Std |
| Sortino Ratio | Mean / Downside Std |
| Calmar Ratio | Return / Max DD |
| Expectancy | Net Profit / Total Trades |

---

## Verdict: ✅ BACKTEST ENGINE OK

Well-designed with:
- Realistic execution modeling
- Comprehensive metrics
- Anti-Ruin safety checks
- Export functionality
