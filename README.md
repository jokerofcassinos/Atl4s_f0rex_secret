# Atl4s-Forex 2.0: Technical System Architecture & Cognitive Manual

## 🌌 Executive Summary
**Atl4s-Forex 2.0** represents a paradigm shift in automated trading. Unlike traditional systems relying on lagged indicators (MA, MACD), Atl4s operates as a **Cognitive Entity**. It utilizes a "Multi-Brain" architecture that mimics human intuition but is grounded in **Quantum Mechanics, Newtonian Physics, and Institutional Order Flow**.

The system does not "guess". It calculates the probability of future states using aggressive simulation (Monte Carlo), assesses the stability of the present (Chaos Theory), and validates the physical momentum of price (Kinematics).

---

## 🧠 Cognitive Architecture: The 7 Brains

The bot's decision-making process is distributed across 7 distinct analytical engines, each specialized in a dimension of market reality.

### 1. Deep Cognition (The Pre-Frontal Cortex)
* **Role:** Orchestrator & Final Judge.
* **Mechanism:** It receives normalized signals (-1.0 to 1.0) from all other brains. It applies a **Weighted Consensus Algorithm** to determine the final "Alpha" score.
* **Key Logic:**
    * **Cognitive Dissonance:** If the *Technical Trend* says BUY but *Physics* says CRASH, the Deep Cognition penalizes the score, enforcing a "Wait" state.
    * **Normalization:** Uses `np.tanh(score / 25.0)` to convert raw metrics into a sigmoid probability curve.

### 2. Cortex Memory (The Hippocampus)
* **Role:** Experience Recall & Intuition.
* **Algorithm:** **Holographic Associative Memory**.
* **Process:**
    1.  **Vectorization:** Every 5-minute candle is converted into a high-dimensional feature vector (RSI, Volatility, Range, Body_Size).
    2.  **Cosine Similarity Search:** When a new candle forms, the system searches its database for the *k-Nearest Neighbors* (past moments that look like "now").
    3.  **Outcome Projection:** If 8 out of 10 similar past moments resulted in a price drop, the Cortex injects a bearish bias into the active decision.

### 3. Smart Money Engine 2.0 (The Institutional Tracker)
* **Role:** Structure & Liquidity Detection.
* **Concepts:** Based on ICT (Inner Circle Trader) concepts.
    * **FVG (Fair Value Gap):** Identifies inefficiencies where price moved too fast, leaving unfilled orders. The bot expects price to magnetize back to these zones.
    * **Order Blocks (OB):** Detects the specific candles where institutions accumulated positions before an impulsive move. These act as "Concrete Walls" of support/resistance.
    * **Liquidity Grabs:** Distinguishes between a "Breakout" and a "Fakeout" (Stop Hunt) by analyzing wick behavior relative to key highs/lows.

### 4. Kinematics Engine (The Physics Solver)
* **Role:** Momentum & Energy Analysis.
* **Theory:** Markets follow laws similar to Newtonian mechanics.
* **Metrics:**
    * **Velocity ($v$):** First derivative of price ($dp/dt$).
    * **Acceleration ($a$):** Second derivative of price ($d^2p/dt^2$).
    * **Phase Space:** Plots $v$ vs $a$. The resulting "Orbit" reveals the system's energy.
    * **Orbit Energy ($E$):** calculated as $E = \sqrt{v^2 + a^2}$. High Energy = Breakout or Crash. Low Energy = Chop.
    * **State Classification:** "ACCELERATING UP", "DECELERATING DOWN" (Gravity Pull), "CRASH" (High neg acceleration).

### 5. Quantum Math (The Chaos Analyzer)
* **Role:** Stability & True Value Estimation.
* **Algorithms:**
    * **Shannon Entropy:** Measures information content. High Entropy ($>2.0$) = Maximum Disorder (Random Walk). The bot reduces risk.
    * **Kalman Filter:** A recursive algorithm used by GPS systems. It filters out "noise" to estimate the "True Position" of price. If Price > Kalman, it's overextended.
    * **Hurst Exponent ($H$):** 
        * $H = 0.5$: Random Walk.
        * $H > 0.5$: Trending (Fractal Persistency).
        * $H < 0.5$: Mean Reverting (Anti-persistence).

### 6. Hyper Dimension / Third Eye (The Anomaly Detector)
* **Role:** Statistical Extremes.
* **Logic:** Combines **Bollinger Bands** (Statistical Deviation) with **RSI** (Momentum) and **Candle Geometry**.
* **Example Signal:** `DIMENSIONAL_SELL_REVERSAL` triggers ONLY if:
    * Price is above Upper Bollinger Band ($2\sigma$).
    * RSI is Overbought ($>70$).
    * Candle forms a "Wick Rejection" (Shooting Star).
    * This confirms a statistical probability of mean reversion.

### 7. The Oracle (Pre-Cognition)
* **Role:** Future Simulation.
* **Algorithm:** **Monte Carlo Simulation (Geometric Brownian Motion)**.
* **Process:** Uses current Drift ($\mu$) and Volatility ($\sigma$) to generate 1,000 distinct future price paths for the next 50 candles.
* **Output:** A Probability Map. "In 75% of parallel universes, price is higher than now." This gives the bot a "Win Rate Prediction" before the trade is even taken.

---

## ⚙️ Operational Workflow (Main Loop)

1.  **Ingestion:** Python Bridge receives live ticks from MT5 via ZMQ (ZeroMQ).
2.  **Clock Alignment:** A smart timer waits for the exact 5-minute mark (Sao Paulo Time).
3.  **Synthesis:**
    *   `DataMap` updates.
    *   All 7 Brains run in parallel.
    *   `DeepCognition` gathers the votes.
4.  **Action:**
    *   **Threshold Check:** If Alpha Score > `0.50`, signal is VALID.
    *   **Risk Calculation:** Bot calculates position size based on Account Balance (1% Risk) and ATR (Volatility).
5.  **Notification:**
    *   Sends a Windows Balloon Tip via `plyer` / PowerShell.
    *   Format: `BUY SIGNAL (Alpha: 0.78) | FutureProb: 82% | State: HIGH_ENERGY`.

---

## 📁 File Structure Map
* `main.py`: The Central Nervous System. Orchestrates the loop.
* `analysis/deep_cognition.py`: The consensus engine.
* `analysis/smart_money.py`: The institutional logic.
* `analysis/kinematics.py`: The physics engine.
* `analysis/cortex_memory.py`: The database of past experiences.
* `src/quantum_math.py`: The library of advanced math functions.
* `src/notifications.py`: The interface to the OS Windows system.
* `update_github.py`: The automation script for cloud backups.

---
*System Architected by Antigravity for Atl4s-Forex Project.*
