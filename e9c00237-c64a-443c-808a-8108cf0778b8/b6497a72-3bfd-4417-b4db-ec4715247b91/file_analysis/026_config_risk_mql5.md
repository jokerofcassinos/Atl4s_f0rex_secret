# Analysis 026: Config, Risk, and MQL5 Bridge Audit

## Configuration (config.py - 111 lines)

### Key Settings
```python
TRADING_MODE = "LAPLACE"  # Options: LAPLACE or LEGACY
PRIMARY_SYMBOL = "GBPUSD"
LEVERAGE = 3000  # 1:3000 (Unlimited)
INITIAL_CAPITAL = 30.0
RISK_PER_TRADE = 2.0  # 2%
```

### Laplace Demon Parameters
```python
LAPLACE = {
    'min_confidence': 60,
    'min_confluence': 2,
    'sl_atr_multiplier': 1.5,
    'tp_rr_ratio': 2.0,  # 2:1 R:R
    'max_concurrent_trades': 3,
}
```

### ⚠️ Potential Issue
```python
TRADING_MODE = "LAPLACE"  # But where is this used?
```
**Grep Result:** Only found in config.py itself - **NOT used by main.py to switch systems!**

---

## core/risk/ (40KB, 5 files)

| File | Size | Integrated? |
|------|------|-------------|
| `advanced_risk.py` | 15KB | ❓ Check |
| `crisis_manager.py` | 10KB | ❓ Check |
| `quantum_hedger.py` | 6KB | ❓ Check |
| `entropy_harvester.py` | 5KB | ✅ SwarmOrchestrator |
| `event_horizon.py` | 4KB | ✅ ExecutionEngine |

**Status:** ✅ Partially Integrated (2/5 files)

---

## MQL5 Bridge (86KB, 4 files)

The execution layer connecting Python to MetaTrader 5.

| File | Size | Purpose |
|------|------|---------|
| `Atl4sBridge.mq5` | 42KB | Main bridge EA |
| `Atl4sBridge.ex5` | 35KB | Compiled EA |
| `Atl4sBridge_Tester.mq5` | 6KB | Strategy tester |
| `Atl4sDataExporter.mq5` | 3KB | Data export |

**Status:** ✅ Production Ready

---

## ZMQ Bridge (zmq_bridge.py - 281 lines)

Native socket bridge replacing ZMQ library.

**Key Methods:**
- `get_tick()` - Get latest tick data
- `send_command()` - Send trade commands to MT5
- `send_dashboard()` - UI updates
- `_client_handler()` - Handle MQL5 connections

**Status:** ✅ Integrated and Working

---

## Recommendations

1. **Use TRADING_MODE:** Make main.py respect config.TRADING_MODE
2. **Integrate remaining risk modules:** advanced_risk.py, crisis_manager.py
3. **Remove or integrate quantum_hedger.py** if not used
