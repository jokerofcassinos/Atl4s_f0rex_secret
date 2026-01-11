# File Analysis #7: `analysis/swarm/veto_swarm.py`

## Summary

| Metric | Value |
|--------|-------|
| **Lines** | 162 |
| **Veto Conditions** | 4 |
| **Meta-Critic** | ✅ Can suppress |

---

## Veto Conditions

### 1. Extreme Volatility (Lines 52-86)
```python
dynamic_threshold = max(0.5, mean_atr_pct * 3.0)  # 3x normal ATR
if atr_pct > dynamic_threshold:
    votes.append("VETO: Extreme Volatility")
```
**Status:** ✅ Reasonable - allows 3x normal volatility.

### 2. RSI Extremes (Lines 88-96)
```python
if rsi > 85: votes.append("VETO: RSI > 85 Overbought")
if rsi < 15: votes.append("VETO: RSI < 15 Oversold")
```
**Status:** ✅ Reasonable thresholds (85/15).

### 3. Volume Disconnect (Lines 98-107)
```python
if price_trend > 0 and vol_trend < 0:
    pass  # Warning only, not hard veto
```
**Status:** ✅ Smart - only warns, doesn't block.

### 4. Weekend Guard (Lines 109-126)
```python
if day_of_week >= 5 and not is_crypto_profile:
    votes.append("VETO: Market Closed (Weekend)")
```
**Status:** ✅ Good - respects crypto profile.

---

## Meta-Critic Safety Valve (Lines 128-145)

```python
reflection = self.meta_critic.reflect(decision="VETO", ...)
if reflection['adjusted_confidence'] < 80.0:
    logger.info("VETO SUPPRESSED by Meta-Critic")
    return None  # No veto!
```

> [!TIP]
> **Good design:** The Meta-Critic can override paranoid vetoes!

---

## Verdict: ✅ WELL DESIGNED

This file is reasonable. The issue is **stacking**:
- VetoSwarm veto → +25 penalty in orchestrator
- Multiple VETOs compound rapidly

---

## Next: main.py differences
