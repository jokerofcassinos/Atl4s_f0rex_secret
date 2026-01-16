---
name: novos setups
description: pensar em novos setups para serem implementados
version: 1.2
---

Pensar em novos setups para serem implementados, sempre que possível, pensar em setups que sejam diferentes dos existentes e que sejam capazes de serem implementados no sistema. é importante analisar tambem que os setups novos abrem oportunidades de lucro explorando tambem os pontos fracos dos setups existentes, ou seja, os setups novos devem ser capazes de explorar oportunidades de lucro que os setups existentes não exploram. além disso, voce deve pensar já de forma antecipada os vetos nescessarios, ja prevenindo possiveis problemas e falhas que possam ocorrer em cenarios diversos. e tambem ja pense nos split fires.

# 🚀 PROPOSTAS DE NOVOS SETUPS (ARQUITETURA AGI)

Análise de Gaps Atuais:
- Já temos: Breakouts (LION/MOMENTUM), Reversão de Extensão (SNIPER/SINGULARITY), Estrutura (SMC RIDER).
- **O que falta?**
    1.  **Exploração do Ciclo de 8 Minutos:** Não temos um setup nativo para o "Sweet Spot" de Fibonacci.
    2.  **Preenchimento de Vácuo (FVG):** O bot ignora "Fair Value Gaps" (FVG) como gatilho primário.
    3.  **Compressão de Volatilidade:** O bot muitas vezes perde o *início* da expansão pós-squeeze.

---

## 1. THE GOLDEN COIL (A Espiral Dourada - 8min)
**Conceito:** Utiliza o timeframe místico de 8 minutos (Fibonacci) para operar retrações precisas na tendência. Diferente do Breakout (que compra topos), este setup compra *fundos* dentro de uma tendência de alta confirmada.

*   **Lógica Principal:**
    *   Tendência Primária (H1) = TENDÊNCIA (Hurst > 0.6).
    *   Vela M8 anterior fechou a favor da tendência (Impulso).
    *   Preço atual recua (pullback) até a **Zona de Ouro (50% - 61.8%)** da vela M8 anterior.
    *   Gatilho: Toque na zona + Micro-Rejeição em M1.

*   **Vantagem (Edge):** Stop Loss curtíssimo (logo abaixo da vela M8), R:R insano (3:1 ou 5:1).
*   **Vetos Preventivos (Safety):**
    *   **Veto de Quebra de Momentum:** Se o pullback for muito forte (Volume > 150% da média), cancela (não é pullback, é reversão).
    *   **Veto de Estrutura:** Se o recuo quebrar a mínima da M8 anterior, invalida.

---

## 2. THE VOID FILLER (O Preenchedor de Vácuo)
**Conceito:** O mercado odeia ineficiência. Grandes movimentos deixam "vácuos" (FVGs - Fair Value Gaps). Este setup busca o fechamento desses gaps quando o preço retorna para testá-los e rejeita.

*   **Lógica Principal:**
    *   Identificar Grande Vela de Deslocamento (Displacement) em M5.
    *   Detectar FVG (Espaço entre Pavio A e Pavio C).
    *   Aguardar retorno do preço ao *início* do FVG.
    *   Gatilho: Rejeição imediata (Wick) na zona do FVG a favor do movimento original.

*   **Vantagem (Edge):** Captura movimentos rápidos de continuação ou reversão técnica que outros bots ignoram por não ser "suporte/resistência" clássico.
*   **Vetos Preventivos (Safety):**
    *   **Veto de Inversão:** Se o corpo da vela fechar *além* do FVG (engolfando o gap), o setup morre (o suporte virou resistência).
    *   **Veto de Notícias:** Bloquear 5min após notícias, pois FVGs são frequentemente violados violentamente.

---

## 3. VOLATILITY SQUEEZE HUNTER (Caçador de Compressão)
**Conceito:** O mercado alterna entre compressão e expansão. O bot atual sofre em "Choppy Markets". Este setup *identifica* o Choppy (BB Width mínima + Volume Baixo) e posiciona ordens *somente* na explosão.

*   **Lógica Principal:**
    *   Bandas de Bollinger (20, 2.0) extremamente estreitas (Squeeze).
    *   ADX < 20 (Tendência morta).
    *   Gatilho: Abertura das Bandas (Boca de Jacaré) + Pico de Volume + Rompimento do canal de Keltner.

*   **Vantagem (Edge):** Evita operar o ruído lateral (onde perdemos dinheiro) e entra apenas quando a inércia é quebrada.
*   **Vetos Preventivos (Safety):**
    *   **Veto de Fakeout (Armadilha):** Se romper e o volume for baixo (< Média 20), é armadilha. Bloquear.
    *   **Veto de Caos:** Se Lyapunov estiver alto *antes* do rompimento, ignorar (falso sinal errático).

---

## 4. THE QUANTUM HARPOON (O Arpão Quântico)
**Conceito:** Baseado em Mean Reversion Extrema. Quando o preço estica demais (elástico), ele tende a voltar com violência para a média. O Arpão identifica esse ponto de exaustão matemática.

*   **Lógica Principal:**
    *   **Z-Score > 3.0 (ou < -3.0):** Preço está a 3 desvios padrão da média (evento estatisticamente raro, < 0.3%).
    *   **Kinematics Warning:** Aceleração começa a cair (Derivada Segunda inverte) ou Ângulo de Ataque > 80 graus (Insustentável).
    *   **Gatilho:** Fechamento de vela M1 revertendo a direção (Candle de Rejeição) após tocar a Banda de Bollinger 3.0.

*   **Vantagem (Edge):** Win Rate altíssimo para Scalps curtos (retorno à média).
*   **Vetos Preventivos (Safety):**
    *   **Veto de Tendência Absoluta (Trem-Bala):** Se o Consenso Global estiver **EXTREMO (> 80)**, não operar contra, mesmo com Z-Score alto. O mercado pode ficar irracional por mais tempo que nós temos de margem.
    *   **Veto de Notícia:** Bloquear em Payroll/CPI, onde 3-Sigma é rompido facilmente.

---

## 5. THE FRACTAL ECHO (O Eco Fractal)
**Conceito:** O mercado repete padrões em escalas diferentes. Se um padrão de reversão acontece em M1, M5 e M15 *simultaneamente* (alinhamento fractal), a probabilidade de sucesso é multiplicada.

*   **Lógica Principal:**
    *   Detectar Fractal de Alta/Baixa (Padrão de 5 velas: High no meio, 2 lower highs de cada lado) em M1.
    *   Verificar se existe Fractal correspondente em M5 na mesma zona.
    *   **Gatilho:** Rompimento da máxima/mínima do Fractal M1 alinhado.

*   **Vantagem (Edge):** Confirmação multi-tempo elimina ruído de M1.
*   **Vetos Preventivos:**
    *   **Veto de Divergência:** Se M1 diz Compra mas M15 diz Venda, silencia o Eco.

---

## Próximos Passos (Implementação)
1. Criar módulo `squeezes.py` e `gaps.py` para detecção matemática.
2. Integrar lógica do Ciclo M8 no `laplace_demon.py` (já temos slices M8, falta a lógica de retração).