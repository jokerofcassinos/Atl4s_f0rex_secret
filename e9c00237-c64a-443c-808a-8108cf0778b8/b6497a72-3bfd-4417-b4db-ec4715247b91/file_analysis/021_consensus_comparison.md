# Analysis 021: Legacy vs New consensus.py Comparison

## File Comparison

| Metric | Legacy | New | Diff |
|--------|--------|-----|------|
| **Lines** | 728 | 974 | +246 (+34%) |
| **Bytes** | 34KB | 46KB | +12KB |
| **Eyes Used** | 10 | 14 | +4 |
| **VETOs** | 2 | 5+ | +3 |
| **Complexity** | Medium | VERY HIGH | ⚠️ |

---

## Added Complexity in New Version

### 1. ThoughtTree Integration (Lines 194-300)
```python
# For EACH module result, creates thought trees and decision memories
tree = self.thought_orchestrator.get_or_create_tree(module_name)
memory = self.decision_memory.get_or_create_memory(module_name)
```
**Impact:** Adds 100-200ms latency per module per tick.

### 2. Meta-Thinking Layer (Lines 267-300)
```python
# Cross-module thought connections
meta_insights = {}
for module_name in results.keys():
    for other_module in results.keys():
        insights = self.decision_memory.get_cross_module_insights(...)
```
**Impact:** O(n²) complexity for n modules.

### 3. Holographic Vector Logic (Lines 735-809)
```python
# NEW thresholds that may be too strict
if abs(v_momentum) > 50:  # ISSUE: Too high?
if abs(v_reversion) > 40:  # ISSUE: Too high?
if abs(v_structure) > 60:  # ISSUE: Too high?
```
**Impact:** Blocks signals that would have passed in legacy.

### 4. Unified Reasoning (Lines 926-959)
```python
if unified_decision.agreement_score < 0.5:
    final_score = final_score * 0.8  # Confidence reduction
```
**Impact:** Reduces confidence even for valid signals.

### 5. Health Monitor (Lines 961-967)
```python
if health_status == HealthStatus.CRITICAL:
    decision = "WAIT"
    final_score = 0
```
**Impact:** Additional VETO potential.

---

## Legacy Modules Still Working (OK)

| Module | Lines | Function |
|--------|-------|----------|
| TrendArchitect | 35-45 | Trend detection |
| Sniper | 46-50 | FVG/OB detection |
| Quant | 51-55 | Technical analysis |
| Patterns | 60-65 | Pattern recognition |
| Divergence | 70-75 | Divergence detection |

---

## New VETO Sources (Potential Over-Filtering)

| VETO | Line | Trigger |
|------|------|---------|
| Architect Audit | 478-488 | Low coherence |
| Chaos Threshold | 486-488 | High entropy |
| Global Regime Lock | 490-505 | H4/H1 conflict |
| VPIN Toxicity | 708-717 | Toxic order flow |
| Black-Swan | 898-912 | Low survival prob |
| Health Monitor | 961-967 | Critical status |

---

## Recommendation

**Reduce VETO Layers:**
1. Disable Meta-Thinking (100-200ms latency for minimal benefit)
2. Lower Holographic thresholds (50→30, 40→25, 60→40)
3. Relax Black-Swan survival threshold (45%→35%)
4. Make Health Monitor warning-only (not VETO)
