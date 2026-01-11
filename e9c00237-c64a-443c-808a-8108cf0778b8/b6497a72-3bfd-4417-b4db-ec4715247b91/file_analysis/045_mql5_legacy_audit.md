# Analysis 045: MQL5 Bridge & Legacy Codebase

## MQL5/ Folder (4 files, 86KB)

**Purpose:** MetaTrader 5 Expert Advisor integration

| File | Size | Purpose |
|------|------|---------|
| `Atl4sBridge.mq5` | 42KB | Main EA with ZMQ |
| `Atl4sBridge.ex5` | 35KB | Compiled EA |
| `Atl4sBridge_Tester.mq5` | 6KB | Strategy Tester |
| `Atl4sDataExporter.mq5` | 3KB | Data export |

### Atl4sBridge.mq5 (42KB)
- ZMQ communication with Python
- Order execution functions
- Account info retrieval
- Position management
- Tick data streaming

---

## Legacy Codebase (Atl4s_f0rex_secret-main)

**Files:** 40+ root files, 8 subdirectories
**Size:** ~170KB (excluding data)

### Root Files Comparison

| File | Legacy | New | Notes |
|------|--------|-----|-------|
| main.py | 27KB | 46KB | +19KB growth |
| backtest_engine.py | 12KB | 52KB | +40KB growth |
| config.py | 2KB | 3KB | Minor changes |
| risk_manager.py | 4KB | 8KB | +4KB growth |

### Directory Structure

```
Atl4s_f0rex_secret-main/
├── analysis/           # 8 files
├── data/              # Historical data
├── mql5/              # Legacy EA
├── src/               # Utilities
├── static/            # Web assets
├── templates/         # HTML templates
└── reports/           # Output reports
```

---

## Legacy vs New Size Comparison

| Component | Legacy | New | Growth |
|-----------|--------|-----|--------|
| Python Files | ~50 | ~350+ | 7x |
| Total Size | ~170KB | ~1.5MB | 9x |
| Swarm Agents | 0 | 88 | ∞ |
| AGI Modules | 0 | 40+ | ∞ |
| C++ Backend | 0 | ~3MB | ∞ |

---

## Key Differences

### Legacy (Simple)
- Basic consensus logic
- 8 analysis Eyes
- Direct execution
- No swarm intelligence
- No AGI features

### New (Complex)
- 88 swarm agents
- 14+ Eyes
- C++ acceleration
- Full AGI stack
- Dual architecture issue

---

## Summary

| Component | Files | Size | Status |
|-----------|-------|------|--------|
| MQL5/ | 4 | 86KB | ✅ Production |
| Legacy | 40+ | 170KB | ✅ Reference |
