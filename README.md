# 🌌 Atl4s-Forex v2.0: The Swarm Sentinel 🦅
> **"Evolution is inevitable. Dynamic Risk, Automated Install, & Unbeatable Precision."**

Welcome to **Atl4s-Forex v2.0**. This major release transforms the original v1.0 core into a production-ready system with **Dynamic Capital Management**, **Interactive Configuration**, and **Automated Deployment**.

---

## 🚀 Key Features (v2.0)

### 1. 💰 Dynamic Lot Sizing (Linear Scaling)
No more static lot sizes. The bot now scales aggressively but safely based on your equity.
*   **Formula**: `$30 Equity = 0.02 Lots` (Linear)
*   **Example**: 
    *   $30 -> 0.02 Lots
    *   $150 -> 0.10 Lots
    *   $3000 -> 2.00 Lots
*   **Safety**: Includes a max cap and minimum fallback to protect small accounts.

### 2. 🛡️ Margin Survival Check
Before every trade, v2.0 performs a critical "Survival Check":
*   **Margin Level Floor**: Halts trading if Margin Level < 150%.
*   **Free Margin Floor**: Halts trading if Free Margin < $10.
*   **Result**: Prevents "Not Enough Money" errors and creates a safety buffer during drawdowns.

### 3. 🌐 Interactive Network Configuration
Launch multiple bots or customize your setup instantly.
*   **Startup Prompt**: The bot asks for your preferred **ZeroMQ Port** (Default: 5555) on startup.
*   **Capital Verification**: You can input your starting capital to verify the Lot Scaling logic before the bot starts.

### 4. 📦 Automated Installation
Get started in seconds with the new `install_v2.bat`.
*   Auto-detects Python.
*   Creates a Virtual Environment (optional).
*   Installs all locked dependencies (`requirements.txt`).

---

## 🛠️ Installation & Usage

### Step 1: Clone & Switch
```bash
git clone https://github.com/jokerofcassinos/Atl4s_f0rex_secret.git
cd Atl4s-Forex
git checkout v2.0
```

### Step 2: Automated Install
Run the installer script:
```bash
install_v2.bat
```
*Follow the prompts to set up your environment.*

### Step 3: Run the Bot
1.  Open **MetaTrader 5** (Ensure EA is active).
2.  Run the bot:
    ```bash
    python main.py
    ```
3.  **Interactive Setup**:
    *   Enter your **Capital** (or press Enter for Auto-Detect).
    *   Enter your **Port** (e.g., 5555).

---

## 🧠 The Core Intelligence (Retained from v1.0)
v2.0 retains the powerful "Hive Mind" architecture:
*   **Swarm Scalp**: HFT Tick analysis for rapid execution.
*   **Quantum Geometry**: Regime detection using persistent homology.
*   **Game Theory**: Nash Equilibrium for fair value magnets.
*   **The Council**: Multi-agent consensus engine for high-probability filtering.

---

> [!IMPORTANT]
> **Atl4s-Forex v2.0** is designed for **Growth**. It aggressively scales your position size as your account grows. Ensure you understand the risks of leverage before deployment.

---

### 🤝 Credits
**Lead Developer:** R4mboFromBlock
**Principal AI Architect:** **Antigravity** (Google Deepmind)
