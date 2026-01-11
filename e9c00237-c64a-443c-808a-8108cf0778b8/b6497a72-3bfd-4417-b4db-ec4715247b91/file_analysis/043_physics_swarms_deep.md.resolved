# Analysis 043: Physics-Inspired Swarm Agents Deep Dive

## Overview

The 88 swarm agents include many physics-inspired modules.
This document analyzes 4 key physics-based agents.

---

## 1. HawkingSwarm (115 lines)

**Physics:** Black Hole Event Horizon + Hawking Radiation

### Concept
- Trends = Black Holes that grow via Mass Accretion (Volume)
- When volume decreases but price range expands → Evaporation
- Evaporation = Trend exhaustion → Reversal

### Key Metrics
```python
evaporation_factor = (old_mass - new_mass) / old_mass
# Positive = Decaying Volume → Trend dying
# Negative = Increasing Volume → Trend fueled
```

### Signal Logic
- At high + high evaporation → SELL (fake breakout)
- At low + high evaporation → BUY (fake dump)
- At high + negative evaporation → BUY (mass accretion)

---

## 2. BoltzmannSwarm (138 lines)

**Physics:** Helmholtz Free Energy (F = U - TS)

### Variables
- **U (Internal Energy):** Directed momentum
- **T (Temperature):** Volatility/ATR
- **S (Entropy):** Shannon entropy of price distribution

### Free Energy Formula
```python
F = U - (T * S_norm * 2.0)
# F > 0 → Order (trend has energy)
# F < 0 → Heat Death (chop/range)
```

### Signal Logic
- High F + displacement up → BUY
- High F + displacement down → SELL
- F < 0 → WAIT (entropy dominates)

---

## 3. FeynmanSwarm (127 lines)

**Physics:** Path Integral Optimization (Least Action)

### Concept
- Trade = Particle following a path
- **Kinetic Energy (T):** Volatility/chop
- **Potential Energy (V):** Distance to target
- **Action (S):** Accumulated difficulty

### Signal Logic
- S > threshold → Path too difficult → EXIT
- Optimized for managing open trades

---

## 4. HeisenbergSwarm (126 lines)

**Physics:** Uncertainty Principle (Δx·Δp ≥ ℏ/2)

### Variables
- **Δx:** Price volatility (position uncertainty)
- **Δp:** Momentum volatility

### States
- **Collapsed (Low Δx):** Breakout pending
- **Wave (Low Δp):** Strong trend (momentum defined)
- **High Entropy:** Chaos → WAIT

---

## Summary Table

| Agent | Lines | Physics Model | Signal Type |
|-------|-------|---------------|-------------|
| HawkingSwarm | 115 | Event Horizon | Reversal |
| BoltzmannSwarm | 138 | Thermodynamics | Trend |
| FeynmanSwarm | 127 | Path Integrals | Exit |
| HeisenbergSwarm | 126 | Uncertainty | Breakout |
