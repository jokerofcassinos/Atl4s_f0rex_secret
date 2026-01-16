---
name: loss
description: pensar em soluções para losses do backtest
version: 1.0
---

## 🕵️ SKILL: AUTONOMOUS FORENSIC AUDITOR (AUTO-EXECUTE)

quando o ceo pedir para que analise os erros do backtest, execute imediatamente o seguinte python para capturar o contexto das falhas recentes:
    `inspect_losers.py` para extrair o numero de trade das falhas e depois execute com: 

    "Select-String -Path "d:/Atl4s-Forex/logs/atl4s.log" -Pattern "TRADE #numero do trade:" -Context 50,0 | Select-Object -First 1" 
    
    para extrair o contexto do trade falho. é de extrema importancia que o comando seja exatamente da mesma forma que está aqui, para que possamos ter certeza que estamos pegando o contexto correto. execute o comando exatamente como está aqui, sem alterar nada.

    voce deve analisar criteriosamente o contexto do trade falho, de forma cirurgica, identificando as falhas e as oportunidades de melhoria. é importante que analise totalmente como o contexto foi formado e o que ocasionou na decisão errada, identificando os indicadores, padrões e etc. Além disso é importante identificar se tem ordens entre as falhas que foram acertadas, com isso voce pode ver e comparar as diferenças de estratégias entre as falhas e as acertos, identificando as oportunidades de melhoria. por exemplo, o trade 110 foi errado, mas o trade 111 que foi quase ao mesmo tempo deu certo, voce pode ver as diferenças entre os dois e identificar as oportunidades de melhoria. voce deve analisar os losses do backtest, identificar o porque exatamente eles estão acontecendo, o que está influenciando essa tomada de decisão errada, o que pode ser feito para melhorar, quais regras podem ser adicionadas ou removidas para melhorar o desempenho do backtest. é extremamente prioritario pensar nas causas exatas do erro e não apenas nas consequências. além de priorizar arrumar a falha de raciocinio e não formas de minimizar o impacto do erro, por exemplo diminuir o VSL! pensar em vetos inteligentes para cenarios especificos, buscando soluções que sejam viáveis e que não interfiram no desempenho do backtest. além de diminuir os losses e aumentar win rate. 
    precisamos entender que tipo de analise errada o bot está fazendo e em qual cenario especifico ele esta com dificuldades de adaptação, para que ele identifique o cenario e aja de acordo com ele, é importante nao mexer no sl e nem nessas questões, e sim identificar qual conflito de decisão ele esta sofrendo, por exemplo preço subindo e ele vendendo. o intuito é adaptar o bot em todos os tipos de cenarios diversos, sejam eles de alta, baixa ou laterais. para que possamos realmente ter um bot que se adapta a qualquer cenario e que tenha uma alta win rate e um baixo loss rate. para isso realmente é necessario varios testes e analises para que possamos chegar em um resultado satisfatorio. testes esses que sejam em contextos diversos, explorando cenarios gerais que o bot ainda nao tem conhecimento, nós devemos fazer essa adaptação para que ele "amplie" a sua visão e seja totalmente profit com 100% de certeza quando for operar no mt5. 
    gere sempre um relatorio final para o ceo puxar o contexto do loss no grafico real, para que em situações super complicadas, seja possivel ter uma visão mais ampla do que aconteceu de acordo com a imagem do grafico, o ceo irá buscar o contexto historico no grafico real e enviar de volta para voce, para voce efetuar a analise final e identificar as oportunidades de melhoria.

    voce deve relatar os clusters dos losses.

    exemplo de relatorio final:

🔴 Cluster 1: The "Void Filler" Trap
Date: 2026-01-09 (Friday) Time: 18:10 (Server Time/Backtest Time) Setup: VOID_FILLER_FVG (SELL) Price Level: ~1.3402 Outcome: Stop Loss Hit (Shifted trend?) Question: Was this a valid Gap Fill, or was the market launching a strong reversal/trend extension that the bot tried to fade?

🔴 Cluster 2: Volatility Squeeze Failure
Date: 2026-01-09 (Friday) Time: 21:25 Setup: VOLATILITY_SQUEEZE (SELL) Price Level: ~1.3405 Outcome: Stop Loss Hit Question: Did the squeeze break upward violently? The bot signaled SELL implies it expected a breakdown. Was there a specific news event or candle pattern?

🔴 Cluster 3: Relentless Selling (Void Filler again)
Date: 2026-01-09 (Friday) Time: 22:20 Setup: VOID_FILLER_FVG (SELL) Price Level: ~1.3404 Outcome: Stop Loss Hit Question: The bot tried to sell again just an hour later. Was the market grinding up slowly (Drift) or spiking?

🔴 Cluster 4: The Midnight Squeeze
Date: 2026-01-13 (Tuesday) Time: 00:25 Setup: VOLATILITY_SQUEEZE (SELL) Price Level: ~1.3462 Outcome: Stop Loss Hit Question: This occurred right after the new week start (Jan 12/13). Was this a gap opening or a breakout from Asian Range?
 