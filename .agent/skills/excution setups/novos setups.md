---
name: novos setups
description: pensar em novos setups para serem implementados
version: 1.1
---

Pensar em novos setups para serem implementados, sempre que possível, pensar em setups que sejam diferentes dos existentes e que sejam capazes de serem implementados no sistema. é importante analisar tambem que os setups novos abrem oportunidades de lucro explorando tambem os pontos fracos dos setups existentes, ou seja, os setups novos devem ser capazes de explorar oportunidades de lucro que os setups existentes não exploram. além disso, voce deve pensar já de forma antecipada os vetos nescessarios, ja prevenindo possiveis problemas e falhas que possam ocorrer em cenarios diversos.

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

## Próximos Passos (Implementação)
1. Criar módulo `squeezes.py` e `gaps.py` para detecção matemática.
2. Integrar lógica do Ciclo M8 no `laplace_demon.py` (já temos slices M8, falta a lógica de retração).