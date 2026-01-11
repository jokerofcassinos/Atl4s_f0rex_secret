# File Analysis #2: `analysis/scalper_swarm.py`

## Summary

| Metric | Legacy | New | Delta |
|--------|--------|-----|-------|
| **Lines** | 119 | 304 | +185 (+155%) |
| **Algorithm** | 5-Eye Voting | Unified Field | Complete rewrite! |
| **Complexity** | Simple | Physics Simulation | 🔴 OVER-ENGINEERED |

---

## 🚨 CRITICAL FINDING: SIGNAL INVERSION

**Line 142 in New Version:**

```python
# --- SIGNAL INVERSION (User Request) ---
S = -S  # <-- ALL SIGNALS ARE INVERTED!
```

> [!CAUTION]
> **This single line inverts every signal the Swarm produces!** If the Unified Field says BUY, it becomes SELL. This is catastrophic for performance.

---

## Architecture Comparison

### Legacy: Simple 5-Eye Voting (Clear Logic)

```python
votes = {
    'hybrid': 0,        # Eye 1
    'pullback': 0,      # Eye 2
    'momentum': 0,      # Eye 3
    'ofi': 0,           # Eye 4
    'hurst_climax': 0   # Eye 5
}

# Simple weighted sum
for eye, weight in self.weights.items():
    final_vector += votes[eye] * weight

# Simple threshold
if final_vector >= self.threshold: action = "BUY"
elif final_vector <= -self.threshold: action = "SELL"
```

### New: Unified Field Physics (Over-Complex)

```python
# 1. Particle Velocity (v) - normalized from micro_stats
v = np.clip(raw_vel * 5, -1.0, 1.0)

# 2. Field Pressure (P) - OFI + Alpha blend
P = (norm_ofi * 0.6) + (norm_alpha * 0.4)

# 3. Strange Attractor (A) - Chaos/Hurst based
A = attractor_dir * min(abs(phy_score)/5.0, 1.0)

# 4. Entropy-Dynamic Weights
if entropy < 0.3: w_v=0.4, w_p=0.4, w_a=0.2  # LAMINAR
elif entropy > 0.7: w_v=0.1, w_p=0.5, w_a=0.4  # TURBULENT
else: w_v=0.33, w_p=0.33, w_a=0.33  # TRANSITION

# 5. Unified Vector
S = (w_v * v) + (w_p * P) + (w_a * A)

# THEN INVERTED
S = -S  # <-- CATASTROPHIC!
```

---

## Problems Identified

| # | Problem | Severity | Line |
|---|---------|----------|------|
| 1 | **Signal Inversion (`S = -S`)** | 🔴 CRITICAL | 142 |
| 2 | Lost 5-Eye voting simplicity | 🔴 HIGH | - |
| 3 | df_m1 required (wasn't in legacy) | 🟡 MED | 48 |
| 4 | Complex "Wick Rejection" logic | 🟡 MED | 199-222 |
| 5 | Resonance check adds complexity | 🟢 LOW | 27-46 |

---

## Root Cause Analysis

### Why Legacy Worked ($8K/day):

1. **Simple voting** - 5 clear eyes, each with specific purpose
2. **Fast execution** - No physics simulation
3. **Direct signals** - No inversions or complex transforms
4. **Entropy gate** - Simple filter (e_min < entropy < e_max)

### Why New Fails ($30/day):

1. **Signal is inverted** - BUY becomes SELL, SELL becomes BUY
2. **Physics model is speculative** - "Strange Attractors" and "Field Pressure" are theoretical
3. **df_m1 dependency** - May not be available or properly synced
4. **Multiple veto gates** - Wick rejection, body density, velocity guard, trend alignment

---

## Proposed Fixes

### Fix 1: REMOVE Signal Inversion (IMMEDIATE)

```python
# Line 142: Comment out or delete
# S = -S  # REMOVE THIS LINE
```

### Fix 2: Restore 5-Eye Voting (RESTORE LEGACY)

The simplest fix is to completely replace the new `process_tick` with the legacy version:

```python
def process_tick(self, tick, df_m5, alpha_score, tech_score, phy_score, micro_stats):
    # ... (Copy legacy implementation)
```

### Fix 3: Hybrid Approach (Merge Best of Both)

Keep Unified Field calculation but use it as ONE eye in a voting system:

```python
votes = {
    'unified_field': S,      # New physics model
    'pullback': pullback_vote,
    'momentum': momentum_vote,
    'ofi': ofi_vote,
    'legacy_hybrid': hybrid_vote
}
```

---

## Comparison: Key Method Signatures

**Legacy:**
```python
def process_tick(self, tick, df_m5, alpha_score, tech_score, phy_score, micro_stats):
```

**New:**
```python
def process_tick(self, tick, df_m5, df_m1, alpha_score, tech_score, phy_score, micro_stats, forced_lots=None):
```

> [!WARNING]
> New version requires `df_m1` parameter that legacy doesn't have. This breaks compatibility and requires M1 data to be passed from main loop.

---

## Verification Checklist

- [ ] Remove `S = -S` inversion line
- [ ] Test with original signal direction
- [ ] Compare execution count before/after
- [ ] If still poor, restore legacy 5-eye voting
- [ ] Verify win rate improvement

---

## Next File

→ `analysis/deep_cognition.py` (Legacy: 7.2KB vs New: 8.1KB)
