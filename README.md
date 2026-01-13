# 🔮 LAPLACE DEMON (Atl4s-Forex v3.0)
### "A Inteligência Determinística para o Mercado de Câmbio"

O **Laplace Demon** não é apenas um bot de trading; é um ecossistema de inteligência quantitativa projetado para prever a direção do mercado através da síntese de teorias institucionais, algoritmos de AGI (Inteligência Artificial Geral) e uma arquitetura de execução de ultra-baixa latência.

---

## 🏛️ Arquitetura do Sistema

O sistema é dividido em camadas de processamento que imitam a cogitação humana e institucional:

### 1. O Olho da Providência (Camada Analítica)
O bot utiliza múltiplos "Olhos" (módulos) para escanear o mercado em busca de confluências:
- **Scalp Swarm**: Enxame de agentes que buscam micro-ineficiências no fluxo de ordens.
- **The Sniper**: Módulo de precisão baseado em zonas de oferta e demanda institucionais.
- **The Whale**: Rastreador de liquidez de grandes players (Smart Money).
- **Quantum Grid**: Algoritmo de posicionamento dinâmico em zonas de exaustão.

### 2. Teorias Institucionais Integradas
Diferente de indicadores comuns (RSI, Médias), o Laplace foca em fundamentos de tempo e preço:
- **Quarterly Theory**: Ciclos de 90 minutos (Acumulação, Manipulação, Distribuição).
- **M8 Fibonacci**: Ciclos de 8 minutos sincronizados com o pulso do mercado.
- **SMC (Smart Money Concepts)**: Estruturas de BOS (Break of Structure) e CHoCH.
- **SMT Divergence**: Correlações inter-mercado (DXY, EURUSD, GBPUSD).
- **BlackRock Patterns**: Padrões de rebalanceamento de grandes fundos.

---

## 🧠 Inteligência Artificial (Cortex & Nexus)

O Laplace Demon utiliza uma rede neural de última geração para filtrar sinais falsos:
- **Neural Oracle**: Um classificador treinado com milhares de trades para prever a probabilidade de vitória antes da execução. (Precisão atual: ~83%).
- **Cortex Memory**: Memória de curto prazo que aprende com os erros recentes da sessão.
- **Akashic Records**: Base de dados de longo prazo que armazena "DNA" de setups vitoriosos.

---

## 🛡️ Gestão de Risco (Santo Graal)

A segurança do capital é a prioridade absoluta:
- **Drawdown Oracle**: Ajusta o tamanho do lote dinamicamente com base na volatilidade e histórico recente.
- **Omega Sniper**: Protocolo de alta convicção que escala a alavancagem apenas em setups de >90% de confiança.
- **Time-Decayed Take Profit**: Escada descendente de saída que garante o lucro conforme o tempo de trade aumenta.
- **Persistence Layer**: Memória física (`trade_context.json`) que permite ao bot retomar ordens após reinicializações do sistema.

---

## 🚀 Instalação e Configuração

### Requisitos Mínimos
- **Python**: 3.9.x ou superior.
- **SO**: Windows (devido à integração com MetaTrader 5).
- **Conta**: MetaTrader 5 (Preferencialmente IC Markets ou Pepperstone para baixos spreads).

### Configuração Automática
Executar o script de instalação no PowerShell:
```powershell
.\setup_genesis.ps1
```

### Configuração Manual
1. Instalar dependências: `pip install -r requirements.txt`
2. Configurar o MetaTrader 5:
   - Ativar "Algo Trading".
   - Adicionar `http://localhost` e URLs de API no MT5.
3. Configurar o `config.py` com suas preferências de risco.

---

## 🛠️ Como Executar

Para iniciar o sistema em modo real (ou paper trading):
```powershell
.\venv\Scripts\python.exe main_laplace.py
```

Para rodar o simulador de backtest ultra-rápido:
```powershell
.\venv\Scripts\python.exe run_laplace_backtest.py
```

---

## 📊 Notificações
O bot utiliza o **Telegram** para enviar relatórios em tempo real:
- **Entrada**: Setup, Confiança, Preço, SL e TP.
- **Execução**: Alerta imediato de transmissão.
- **Saída**: Lucro/Prejuízo em Dólares e Pips.

---

## ⚠️ Disclaimer
O trading em Forex envolve riscos substanciais. O **Laplace Demon** é uma ferramenta de auxílio à decisão. Resultados passados não garantem lucros futuros. Utilize sempre em conta Demo antes de ir para o mercado real.

---
*Developed by the Atl4s-Forex Team | Laplace Version 3.0*
