# 🔮 LAPLACE DEMON (Atl4s-Forex v3.0)
> **The Deterministic Trading Intelligence**
>
> *"An intellect which at a certain moment would know all forces that set nature in motion... nothing would be uncertain and the future just like the past would be present before its eyes."* — Pierre-Simon Laplace

![Laplace](https://img.shields.io/badge/System-Laplace%20Demon-8A2BE2)
![Intelligence](https://img.shields.io/badge/Intelligence-Deterministic-ff00ff)
![Target](https://img.shields.io/badge/Target-70%25%20Win%20Rate-00ff88)
![Capital](https://img.shields.io/badge/Starting%20Capital-$30-gold)

***

## 🌌 The Vision

**LAPLACE DEMON** is named after Pierre-Simon Laplace's famous thought experiment: a hypothetical "demon" with perfect knowledge of all positions and forces in the universe could predict the future with absolute certainty.

While we cannot achieve omniscience, this system implements **advanced institutional trading theories** to approach deterministic market prediction:

- **Quarterly Theory** — 90-minute institutional cycles (Q1-Q4)
- **M8 Fibonacci** — 8-minute micro-timing for precision entries
- **BlackRock Patterns** — Seek & Destroy, Iceberg Detection, Month-End Rebalancing
- **SMC/ICT Structure** — Order Blocks, Fair Value Gaps, Break of Structure
- **Gann Geometry** — Sacred number intervals (36/72/144)
- **Tesla Vortex** — 3-6-9 cycle exhaustion patterns
- **SMT Divergence** — Cross-pair correlation analysis

**Target**: 70%+ Win Rate | GBPUSD | $30 Starting Capital

---

## 🧠 Core Systems

### 1. 📊 Quarterly Theory (90-Minute Cycles)
The institutional day is divided into quarters:
- **Q1 (Accumulation)**: Price ranges, positions build - **NO TRADES**
- **Q2 (Manipulation)**: The "Judas Swing" - fake breakouts trap retail
- **Q3 (Distribution)**: The REAL move - **GOLDEN ZONE**
- **Q4 (Continuation)**: Trend extension or reversal

### 2. ⏱️ M8 Fibonacci (8-Minute Gates)
Precision timing within each cycle:
- **Q1 (0-2min)**: Dead Zone - no entries allowed
- **Q2 (2-4min)**: Penalty zone - high conviction only
- **Q3 (4-6min)**: Golden zone - optimal entries (+2 bonus)
- **Q4 (6-8min)**: Decay zone - reduced confidence

### 3. 🏦 BlackRock/Aladdin Patterns
Institutional execution footprints:
- **Seek & Destroy**: Liquidity sweeps (Outside Bars)
- **Iceberg Detection**: Absorption blocks (repeated rejections)
- **Month-End Rebalancing**: Portfolio flow at London Fix

### 4. 📐 Gann Geometry
The market respects mathematical intervals:
- **36 pips**: Minor support/resistance
- **72 pips**: Moderate reaction zone
- **144 pips**: Major reversal level (Square of 12)

### 5. ⚡ Tesla 3-6-9 Vortex
Cycle exhaustion detection:
- **3 candles**: Normal impulse
- **6 candles**: Momentum slowing
- **9 candles**: HIGH reversal probability (~70%)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAPLACE DEMON v3.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   TIMING    │  │  STRUCTURE  │  │  MOMENTUM   │             │
│  │             │  │             │  │             │             │
│  │ • Quarterly │  │ • SMC/OB    │  │ • RSI Div   │             │
│  │ • M8 Fib    │  │ • BlackRock │  │ • MACD      │             │
│  │ • Macros    │  │ • Gann      │  │ • Toxic     │             │
│  │ • IB Filter │  │ • Tesla     │  │   Flow      │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │              LAPLACE DEMON CORE                  │           │
│  │                                                  │           │
│  │    [Score] = Timing + Structure + Momentum       │           │
│  │                      + Volatility + Correlation  │           │
│  │                                                  │           │
│  │    Execute if: Confluence >= 2 AND Conf >= 60%  │           │
│  └──────────────────────────────────────────────────┘           │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │              EXECUTION ENGINE                    │           │
│  │                                                  │           │
│  │    ZMQ Bridge → MT5 → Market                    │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Atl4s-Forex/
├── main_laplace.py          # 🔮 NEW: Laplace Demon trading system
├── main.py                  # Legacy Omega system
├── run_laplace_backtest.py  # Professional backtest runner
├── config.py                # Configuration
│
├── core/
│   ├── laplace_demon.py     # 🧠 Central intelligence
│   ├── zmq_bridge.py        # MT5 communication
│   └── execution_engine.py  # Order execution
│
├── signals/                 # 📊 Signal generation modules
│   ├── timing.py            # Quarterly, M8, Macros
│   ├── structure.py         # SMC, BlackRock, Gann, Tesla
│   ├── correlation.py       # SMT, AMD, Power of One
│   ├── momentum.py          # RSI, MACD, Toxic Flow
│   └── volatility.py        # ATR, Bollinger, BPR
│
├── backtest/                # 📈 Backtesting suite
│   ├── engine.py            # Simulation engine
│   ├── charts.py            # Visualization
│   └── metrics.py           # Performance analysis
│
└── reports/                 # 📊 Generated reports
```

---

## ⚡ Quick Start

### Requirements
- Python 3.9+
- MetaTrader 5 (Windows)
- pandas, numpy, matplotlib

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backtest
```bash
python run_laplace_backtest.py
```

### 3. Run Live Trading
```bash
python main_laplace.py --symbol GBPUSD --port 5558
```

---

## � Backtest Features

- **Realistic Simulation**: Spread, slippage, commission modeling
- **Multiple Timeframes**: M1, M5, H1, H4, D1 analysis
- **Visual Reports**: Equity curve, drawdown, trade distribution
- **Monte Carlo**: 1000-iteration confidence analysis
- **Walk-Forward**: K-fold out-of-sample validation
- **Statistical Edge**: T-test for significance

---

## 🛡️ Risk Management

The Laplace Demon is paranoid by design:

- **Virtual SL/TP**: Broker-invisible stop management
- **ATR-Based Stops**: Dynamic sizing based on volatility
- **Confluence Filter**: Minimum 2+ signals required
- **Time Gates**: Only trades in killzones
- **Veto System**: Any module can block a trade

---

## 📜 Credits

**Concept & Architecture**: Laplace Demon
**Logic Engine**: Deepmind Advanced Agentic Coding (Antigravity)
**Philosophy**: *"The market is deterministic. We just need to know its rules."*

> *"Give me the positions of all atoms in the universe, and I will predict the future."*
> — The Laplace Demon

---

## 🔄 Version History

- **v3.0 (2026)**: Laplace Demon - Complete rewrite with institutional theories
- **v2.0**: Omega Protocol - AGI/Swarm architecture
- **v1.0**: Initial XAUUSD scalping bot
