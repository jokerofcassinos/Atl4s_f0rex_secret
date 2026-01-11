# Analysis 037: Root Utilities Deep Dive

## 1. risk_manager.py (223 lines, 8KB)

**Class:** `RiskManager`

### Key Methods

| Method | Purpose |
|--------|---------|
| `calculate_position_size()` | Kelly + Drawdown sizing |
| `calculate_quantum_lots()` | Sigmoid power-law scaling |
| `get_stop_loss()` | ATR-based SL |
| `get_take_profit()` | R:R based TP |
| `check_margin_survival()` | Margin safety check |
| `calculate_safe_margin_lots()` | Max safe lots |

### Quantum Lots Formula
```python
# Base Power Law + Confidence + Entropy + Sigmoid Cap
base_lots = pow(current_equity, 0.65) / 350.0
conf_factor = 1.0 + (confidence_score / 200.0)
entropy_factor = 1.2 - (entropy * 0.4)
raw_lots = base_lots * conf_factor * entropy_factor
final_lots = MAX_CEILING * tanh(raw_lots / MAX_CEILING)
```

**Status:** ✅ Sophisticated, integrated

---

## 2. optimizer.py (171 lines, 6KB)

**Purpose:** Genetic Algorithm for parameter optimization

### Parameters
```python
POPULATION_SIZE = 6
GENERATIONS = 3
MUTATION_RATE = 0.1
ELITISM_COUNT = 2
```

### Gene Bounds
```python
GENE_BOUNDS = {
    'w_trend': (0.05, 0.40),
    'w_sniper': (0.05, 0.30),
    'threshold': (30, 70),
    'chaos_threshold': (2.5, 4.0),
    ...
}
```

### Fitness Function
```
Fitness = profit_pct * win_rate * 100
```

**Status:** ✅ Basic GA, needs scaling

---

## 3. Other Root Utilities

| File | Size | Purpose |
|------|------|---------|
| `report_generator.py` | 18KB | HTML/PDF reports |
| `data_loader.py` | 18KB | Multi-TF data loading |
| `bridge.py` | 6KB | ZMQ bridge wrapper |
| `orchestrator.py` | 5KB | Main orchestration |
| `download_data.py` | 4KB | Data download |
| `simulation_system.py` | 4KB | Simulation engine |
| `news_scraper.py` | 3KB | News analysis |
| `build_cpp.py` | 2KB | C++ build script |

---

## Summary

Root utilities total: **~80KB of infrastructure code**
All well-integrated and functional.
