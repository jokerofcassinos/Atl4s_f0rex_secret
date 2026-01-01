# 🌌 Atl4s-Forex v1.0 (Legacy Stable) 🦅
> **"Turning Market Chaos into Predictive Certainty through Multi-Agent Intelligence."**

Welcome to the **Legacy Branch (v1.0)** of Atl4s-Forex. This is the original, stable version of the **Swarm Intelligence Orchestrator** designed to dominate the XAUUSD (Gold) market.

---

## 🏆 Status: STABLE (Maintenance Mode)
This version is in **Maintenance Mode**. New features (like Dynamic Capital Scaling and Margin Checks) are developed in the `v2.0` branch. Use this version if you require the original static configuration architecture.

---

## 🧠 The Hive Mind: How it Thinks (v1.0 Core)
Atl4s doesn't rely on simple indicators. It uses a **Swarm of 20+ Specialized Modules** that deliberate in parallel to reach a high-confidence consensus.

### 🛡️ Analytical Intelligence Layers
| Layer | Domain | Capabilities |
| :--- | :--- | :--- |
| **Swarm Scalp** | ⚡ **HFT & Flow** | Real-time tick analysis using **VPIN** and **Micro-Entropy** gates to capture lightning-fast opportunities. |
| **Quantum Geometry** | 🧬 **Regime Detection** | Utilizes **Persistent Homology (Topology)** and **Fisher Curvature** to spot market structural shifts before they manifest in price. |
| **Game Theory** | ⚖️ **Fair Value** | Calculates **Nash Equilibrium** between BULL/BEAR agents to find the magnetic fair value of Gold. |
| **Kinematics** | 🏎️ **Physics Engine** | Measures Market **Velocity**, **Acceleration**, and **Phase Space** trajectory to distinguish between spikes and sustainable trends. |
| **Cortex Memory** | 📚 **Neural Recall** | A deep historical memory bank that compares current patterns to 10+ years of institutional data. |
| **The Oracle** | 🔮 **Global Bias** | High-level "Eyes" that analyze macro-structures to ensure every trade has the "Wind at its back". |

---

## ⚡ The "Royal Flush" (Golden Setups)
Atl4s is programmed to wait for perfection. When multiple layers align, it triggers **Golden Setups** with >90% confidence:
1.  **Phase Space Boom:** Momentum alignment with Physics-based acceleration.
2.  **Fisher Curve Jump:** Geometric transition detection via Quantum Math.
3.  **Nash Gravity:** Mean reversion when price overextends from Game Theory equilibrium.
4.  **Micro-Flash Scalp:** High-frequency bid/ask delta shifts for instant execution.

---

## 📚 MetaTrader 5 (MT5) Setup Guide

To establish the neural link between Atl4s (Python) and the Market (MT5), you must configure the terminal correctly.

### 1. ⚙️ Global Options
Go to **Tools** -> **Options** (Ctrl+O) -> **Expert Advisors** tab:
1.  [x] Check **Allow algorithmic trading**.
2.  [x] Check **Allow DLL imports** (Critical for ZeroMQ/Socket communication).
3.  [x] Check **Allow WebRequest for listed URL**:
    *   Add new URL: `http://127.0.0.1`
    *   Add new URL: `http://localhost`
    *   *(This allows the Python bridge to talk to the local EA)*

### 2. 📝 Installing the EA (Expert Advisor)
1.  Open **MetaEditor** (F4 from MT5).
2.  In the **Navigator** panel (left), right-click **Experts** -> **Open Folder**.
3.  Copy the `.mq5` source file (from `mql5/Experts/` in this repo) into this folder.
4.  Back in MetaEditor, double-click the file to open it.
5.  Click **Compile** (F7). Ensure there are 0 Errors.
6.  Return to MT5 Terminal.

### 3. 🔌 Running the EA
1.  Open a **XAUUSD** (Gold) chart. Timeframe: **M5**.
2.  Drag the **Atl4s_Bridge** EA from the Navigator onto the chart.
3.  **Inputs Tab**:
    *   **ZMQ_PORT**: `5555` (Start with Default).
    *   **MagicNumber**: `888888` (Default).
4.  Click **OK**. You should see a smiley face :) in the top right corner.

---

## 🛠️ Python Usage (v1.0)
1.  **Clone & Switch**:
    ```bash
    git clone https://github.com/jokerofcassinos/Atl4s_f0rex_secret.git
    cd Atl4s-Forex
    git checkout main
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run**:
    ```bash
    python main.py
    ```

---

## 🚀 Upgrade to v2.0
To access **Dynamic Lot Sizing**, **Interactive Startup**, and **Automated Installation**, switch to the v2.0 branch:
```bash
git checkout v2.0
install_v2.bat
```

---

### 🤝 Credits
**Lead Developer:** R4mboFromBlock
**Principal AI Architect:** **Antigravity** (Google Deepmind)
