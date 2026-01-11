# File Analysis #12: main.py (OmegaSystem Entry Point)

## Summary

| Metric | Value |
|--------|-------|
| **Lines** | 895 |
| **Class** | OmegaSystem |
| **Methods** | 4 (init, smart_startup, interactive_startup, run) |
| **Status** | ✅ Well Designed |

---

## Core Structure

```
OmegaSystem
├── __init__ (lines 70-142)
│   ├── Bridge (ZMQ)
│   ├── DataLoader
│   ├── SwarmOrchestrator (Cortex)
│   ├── OmegaAGICore
│   ├── ExecutionEngine
│   └── HistoryLearningEngine
│
├── smart_startup (lines 144-270)
│   └── AGI-driven startup (market scanner, profiler)
│
├── interactive_startup (lines 272-371)
│   └── Manual mode selection
│
└── run (lines 384-890)
    └── Main tick processing loop
```

---

## Run Loop Flow (lines 384-600+)

```mermaid
graph TD
    A[Get Tick] --> B[AGI Schedule Check]
    B --> C{Should Trade?}
    C -->|No| D[Sleep 60s]
    C -->|Yes| E[Update Active Trades]
    E --> F[VTP/VSL Guards]
    F --> G[Dynamic Stops]
    G --> H[Predictive Exits]
    H --> I[Catastrophe Guard]
    I --> J[Load Data - 60s throttle]
    J --> K[AGI pre_tick]
    K --> L[M8 Fibonacci Validation]
    L --> M{M8 Pass?}
    M -->|No| N[WAIT]
    M -->|Yes| O[Cortex Decision]
    O --> P[Execute Signal]
```

---

## Key Components in Run Loop

### 1. AGI Schedule Enforcement (line 393-398)
```python
should_trade, reason = self.agi.should_trade_now(self.symbol)
if not should_trade:
    await asyncio.sleep(60)
    continue
```

### 2. VTP/VSL Guards (lines 466-478)
- Individual guards per trade
- Dynamic stops (Event Horizon)
- Predictive exits (The Magnet)

### 3. Catastrophe Guard (lines 483-491)
- 50% drawdown triggers emergency exit
- Closes all positions

### 4. AGI Pre-Tick (lines 548-556)
```python
agi_adjustments = self.agi.pre_tick(tick, self.config, data_map)
```
- Brain reasons before Body moves

### 5. M8 Fibonacci Triple Validation (lines 573-600)
- Gates execution based on M8 timeframe
- If fails, returns WAIT

### 6. Cortex Decision (line 600)
```python
decision, confidence, metadata = await self.cortex.process_tick(...)
```
- Final decision from SwarmOrchestrator

---

## Verdict: ✅ MAIN.PY IS WELL DESIGNED

The entry point and main loop are properly structured with:
- Proper startup sequence
- Risk guards at the top of loop
- AGI integration without blocking
- Throttled data loading

**No critical issues found in main.py.**
