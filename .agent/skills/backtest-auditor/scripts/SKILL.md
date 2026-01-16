---
name: backtest-auditor
description: Skill crítica para a fase de transição. Analisa logs de backtest, isola losses (vsl_hit) e propõe correções de lógica para converter loss em lucro/wait.
version: 1.0.0
---

# Auditor de Backtest (Transition Phase)

**Objetivo:** Atingir $1k de lucro com $30 em 10 dias.
**Foco:** Análise cirúrgica de perdas (`vsl_hit`) para adaptação a caos de mercado.

## Procedimento de Execução

1. **Localização do Log**:
   - Pergunte ao usuário onde está o arquivo de log do backtest (ex: `logs/backtest_run.log`) se ele não tiver fornecido.

2. **Extração de Dados (Via Script)**:
   - Não tente ler o arquivo inteiro de uma vez. Use o script auxiliar.
   - Execute: `python .agent/skills/backtest-auditor/scripts/extract_failures.py <caminho_do_log>`
   - *Nota:* Certifique-se de estar rodando no `venv` se necessário, embora o script use bibliotecas padrão.

3. **Análise Cirúrgica (Cognitiva)**:
   Para CADA trade retornado pelo script, você deve processar a seguinte análise mental e gerar um relatório:

   * **Cenário de Mercado:** O que estava acontecendo? (Alta volatilidade, consolidação, notícia, fim de pregão?)
   * **O Erro Lógico:**
        * *Confusão de Sistemas:* Indicadores conflitantes foram ignorados?
        * *Falta de Visão:* O bot ignorou uma estrutura maior (macro)?
        * *Timing:* A entrada foi cedo ou tarde demais?
   * **A Solução (Code-Level):**
        * O que deve ser alterado no código `if/else` ou na lógica de pesos?
        * O objetivo é: Converter este Loss em **Profit** ou **Wait** (não entrar).

## Formato de Saída Obrigatório

Para cada falha crítica identificada, forneça:

> **🔴 Trade #<ID>**
> * **Causa Raiz:** [Explicação técnica breve]
> * **Falha de Adaptação:** [Por que o bot não se adaptou ao cenário?]
> * **Ação Corretiva:** [Sugestão de código ou lógica específica para implementar]

## Diretrizes Finais
* Lembre-se: Precisamos de **volume** (200 trades/10 dias). Não crie regras que matem trades legítimos. Seja cirúrgico.
* Priorize a detecção de padrões. Se 5 trades falharam pelo mesmo motivo, agrupe-os.