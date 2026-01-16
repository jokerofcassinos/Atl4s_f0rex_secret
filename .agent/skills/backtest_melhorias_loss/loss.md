## 🕵️ SKILL: AUTONOMOUS FORENSIC AUDITOR (AUTO-EXECUTE)

quando o ceo pedir para que analise os erros do backtest, execute imediatamente o seguinte python para capturar o contexto das falhas recentes:
    `inspect_losers.py` para extrair o numero de trade das falhas e depois execute algo parecido com: `PS D:\Atl4s-Forex> Get-Content laplace_backtest.log -Tail 30000 | Select-String "TRADE #(numero do trade):" -Context 50,0 | Select-Object -First 1` para extrair o contexto do trade falho

    voce deve analisar criteriosamente o contexto do trade falho, de forma cirurgica, identificando as falhas e as oportunidades de melhoria. é importante que analise totalmente como o contexto foi formado e o que ocasionou na decisão errada, identificando os indicadores, padrões e etc. Além disso é importante identificar se tem ordens entre as falhas que foram acertadas, com isso voce pode ver e comparar as diferenças de estratégias entre as falhas e as acertos, identificando as oportunidades de melhoria. por exemplo, o trade 110 foi errado, mas o trade 111 que foi quase ao mesmo tempo deu certo, voce pode ver as diferenças entre os dois e identificar as oportunidades de melhoria.
