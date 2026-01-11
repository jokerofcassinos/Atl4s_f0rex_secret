# File Analysis #6: `core/swarm_orchestrator.py`

## Summary

| Metric | Legacy | New | Delta |
|--------|--------|-----|-------|
| **Exists in Legacy?** | ❌ | ✅ | NEW FILE |
| **Lines** | - | 1440 | 🔴 MASSIVE |
| **Active Agents** | - | 28 | Multiple clusters |
| **Penalty Layers** | - | 5+ | Over-filtering! |

---

## 🔴 CRITICAL: Decision Paralysis Architecture

This file shows the **core problem** - multiple layers of filtering:

### Layer 1: VETO Penalty (Lines 398-404)
```python
for t in thoughts:
    if t.signal_type == "VETO":
        penalty_score += 25.0  # Heavy penalty
```

### Layer 2: Gate 3 Tier 1 Conflict (Lines 441-455)
```python
if fractal_vote != liquidity_vote:
    return ("WAIT", 0.0, {"reason": "GATE3_TIER1_CONFLICT"})  # FORCE WAIT!
```

### Layer 3: Physics Conflict (Lines 459-466)
```python
if physics_decision action not in allowed_actions:
    penalty_score += 15.0
```

### Layer 4: Holographic Danger (Lines 468-474)
```python
if intuition < -30.0:
    penalty_score += 20.0
```

### Layer 5: Trend Aikido (Lines 709-741)
```python
if trend_strength > 80.0 and not is_exhausted:
    final_decision = trend_direction  # INVERTS DECISION!
```

### Layer 6: Causal Truth Filter (Lines 779-791)
```python
if causal_analysis['truth_score'] < 0.4:
    final_decision = "WAIT"  # BLOCKS TRADE
```

### Layer 7: Metacognition Threshold (Lines 798-799)
```python
if final_score < 38.0 and final_decision != "EXIT_ALL":
    # Attempt deep reasoning rescue...
```

---

## Agent Hierarchy

28 agents loaded in `initialize_swarm()`:

| Category | Agents | Count |
|----------|--------|-------|
| Safety/Veto | VetoSwarm, BlackSwanSwarm, RedTeamSwarm | 3 |
| Technical | TrendingSwarm, SniperSwarm, QuantSwarm, TechnicalSwarm, FractalVisionSwarm | 5 |
| Institutional | WhaleSwarm, OrderFlowSwarm, LiquidityMapSwarm, SmartMoneySwarm, NewsSwarm | 5 |
| Physics | ChaosSwarm, QuantumGridSwarm, SingularitySwarm, GravitySwarm, ThermodynamicSwarm | 5 |
| Meta | OracleSwarm, CouncilSwarm, ActiveInferenceSwarm, DreamSwarm, ReflectionSwarm | 5 |
| Time | ChronosSwarm, TimeKnifeSwarm, EventHorizonSwarm | 3 |
| Causal | CausalSwarm, BayesianSwarm | 2 |

---

## Problems Identified

| # | Problem | Severity | Line |
|---|---------|----------|------|
| 1 | Gate 3 forces WAIT on any Fractal/Liquidity conflict | 🔴 CRITICAL | 454 |
| 2 | 25.0 penalty per VETO is harsh | 🔴 HIGH | 403 |
| 3 | Trend Aikido can INVERT signals | 🔴 HIGH | 716 |
| 4 | Causal Truth 0.4 threshold too strict | 🟡 MED | 787 |
| 5 | Metacognition 38.0 threshold | 🟡 MED | 799 |
| 6 | 28 agents running per tick = latency | 🟡 MED | - |

---

## Proposed Fixes

### Fix 1: Relax Gate 3 (Fractal/Liquidity Conflict)
```python
# CURRENT: Force WAIT
if fractal_vote != liquidity_vote:
    return ("WAIT", 0.0, ...)

# PROPOSED: Penalty instead of hard block
if fractal_vote != liquidity_vote:
    penalty_score += 10.0
    penalty_reasons.append("TIER1_CONFLICT")
```

### Fix 2: Reduce Penalty per VETO
```python
# CURRENT
penalty_score += 25.0

# PROPOSED  
penalty_score += 15.0
```

### Fix 3: Disable Trend Aikido during testing
```python
# Comment out lines 709-722
# if trend_strength > 80.0:
#     final_decision = trend_direction
```

---

## Next File

→ `analysis/swarm/veto_swarm.py` (Critical gatekeeper)
