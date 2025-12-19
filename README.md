# Atl4s-Forex 2.0: Deep Awakening Architecture

## 🌌 Visão Geral do Sistema
O Atl4s-Forex 2.0 não é apenas um bot de trading baseado em indicadores. Ele foi reestruturado para funcionar como uma **Entidade Cognitiva**, dividida em múltiplas camadas de consciência, análise quântica e percepção psicológica do mercado. O objetivo é simular a intuição de um trader profissional amplificada por matemática de alta precisão.

---

## 🧠 O "Cérebro" Central: Deep Cognition
O módulo `DeepCognition` atua como o **Córtex Pré-Frontal** do sistema. Ele não decide sozinho, mas sim **orquestra o consenso** entre todas as sub-partes do bot.

### Como Funciona:
1. **Coleta de Inputs:** Recebe sinais de instinto (técnico), estrutura (Smart Money), física (Kinematics) e probabilidade (Oracle).
2. **Consultoria Subconsciente:** Acessa o `CortexMemory` para lembrar situações passadas similares.
3. **Ponderação Dinâmica:** Se o "Futuro" (Oracle) discorda do "Agora" (Técnico), ele reduz drasticamente a confiança (Cognitive Dissonance).
4. **Normatização (Alpha):** O resultado é uma pontuação Alpha entre -1.0 e 1.0.

---

## 🔮 Os Módulos de Análise (Sub-Sistemas)

### 1. Smart Money 2.0 (`smart_money.py`)
Focado na **Estrutura Institucional**.
- **O que faz:** Detecta onde os grandes players ("Smart Money") deixaram rastros.
- **Tecnologia:**
  - **Impulsive FVG Detection:** Identifica desequilíbrios de preço (Fair Value Gaps) criados por movimentos explosivos.
  - **Order Blocks (OB):** Localiza zonas de oferta/demanda baseadas em velas institucionais antes de quebra de estrutura, filtrando por tamanho do corpo vs pavio.

### 2. Deep Cognition & Cortex Memory (`deep_cognition.py`, `cortex_memory.py`)
Focado na **Experiência e Aprendizado**.
- **O que faz:** "Lembra" do que aconteceu em cenários parecidos.
- **Tecnologia:**
  - **Memória Vetorial (Holographic Recall):** Armazena o estado do mercado (RSI, Volatilidade, ROC) como vetores.
  - **Similaridade de Cosseno:** Quando um novo candle fecha, ele busca no banco de dados os 10 vetores mais próximos do passado para ver se o resultado foi Bullish ou Bearish.

### 3. Hyper Dimension / Third Eye (`hyper_dimension.py`)
Focado na **Realidade Multidimensional**.
- **O que faz:** Cruza dados de diferentes dimensões (Volatilidade vs Momentum vs Preço) para encontrar anomalias.
- **Tecnologia:** Verifica o "Estado da Realidade" (Ex: Consolidação, Expansão, Manipulação de Pavio). Identifica se o preço está "fora da realidade" (Bandas de Bollinger) mas com energia para continuar.

### 4. Quantum Math (`quantum_math.py`)
Focado no **Caos e Entropia**.
- **O que faz:** Mede a desordem do mercado para saber se é operável.
- **Tecnologia:**
  - **Entropia de Shannon:** Se alta, o mercado está em Caos (aleatório/ruído) -> Bot reduz a mão ou fica em WAIT.
  - **Filtro de Kalman:** Estima o "Valor Real" do preço, ignorando o ruído momentâneo das velas.

### 5. Kinematics (`kinematics.py`)
Focado na **Física do Preço**.
- **O que faz:** Trata o preço como um objeto físico com massa e velocidade.
- **Tecnologia:**
  - **Phase Space Analysis:** Plota Velocidade vs Aceleração.
  - **Detecção de Energia:** Se a órbita no espaço de fase é grande, há alta energia (Tendência Forte ou Crash Iminente).

### 6. Prediction Engine / Oracle (`prediction_engine.py`)
Focado na **Pre-Cognição**.
- **O que faz:** Simula o futuro milhares de vezes.
- **Tecnologia:**
  - **Simulação de Monte Carlo:** Roda 1000 caminhos aleatórios baseados na volatilidade e drift atuais (Geometric Brownian Motion).
  - **Probabilidade Futura:** Calcula a % de chance do preço estar acima ou abaixo do atual daqui a 50 candles.

### 7. Microstructure (`microstructure.py`)
Focado no **Fluxo em Tempo Real**.
- **O que faz:** Analisa cada tick que chega do MT5.
- **Tecnologia:**
  - **Tick Velocity:** Quão rápido as ordens estão chegando?
  - **Order Flow Delta:** A agressão é de compra ou venda? Usado para "Reflexo Rápido" na decisão final.

---

## 🔔 Sistema de Notificação Inteligente (`main.py`)
O bot opera em um ciclo estrito de **5 minutos** (alinhado com o horário de São Paulo).
- **Equilibrium:** Quando não há sinal claro, o bot entra em estado de `EQUILIBRIUM` (Neutralidade).
- **Wait:** Se há um sinal forte mas uma contradição perigosa (Ex: Tendência de Alta mas Crash Físico iminente), ele envia um alerta de `WAIT`.
- **Sinal:** Se o consenso (Alpha) supera `0.60`, ele envia COMPRA ou VENDA, calculando automaticamente o lote sugerido baseado no seu saldo e risco.

## 💾 Automação
O arquivo `update_github.py` permite que o bot faça backup de sua própria "mente" (código e memória) para a nuvem automaticamente.
