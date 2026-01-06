"""
Teste da Fase 6: Sistema de Votação com Consciência
Verifica se o sistema de pensamento recursivo e meta-pensamento está funcionando.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.agi.thought_tree import GlobalThoughtOrchestrator, ThoughtTree
from core.agi.decision_memory import GlobalDecisionMemory, ModuleDecisionMemory

def test_thought_tree():
    print("="*60)
    print("   TESTE: Thought Tree (Árvore de Pensamento)")
    print("="*60)
    
    # 1. Teste de criação de árvore
    print("\n1. Criando árvore de pensamento para módulo 'Trend':")
    tree = ThoughtTree("Trend", max_depth=5)
    
    # Cria nó raiz
    root_id = tree.create_node(
        question="Why did I decide BUY?",
        context={'score': 75.0, 'direction': 1},
        confidence=0.75
    )
    print(f"   Nó raiz criado: {root_id}")
    
    # Responde o nó
    tree.answer_node(root_id, "Trend is strongly bullish with high momentum", confidence=0.8)
    print(f"   Resposta adicionada ao nó raiz")
    
    # Cria nós filhos (perguntas recursivas)
    child1_id = tree.create_node(
        question="Was this decision correct?",
        parent_id=root_id,
        context={'parent_decision': 'BUY'}
    )
    tree.answer_node(child1_id, "Similar past decisions had 70% success rate", confidence=0.7)
    print(f"   Nó filho 1 criado: {child1_id}")
    
    child2_id = tree.create_node(
        question="How should I act now?",
        parent_id=root_id,
        context={'current_decision': 'BUY'}
    )
    tree.answer_node(child2_id, "Continue with BUY signal, monitor for reversal", confidence=0.75)
    print(f"   Nó filho 2 criado: {child2_id}")
    
    # 2. Teste de cadeia de pensamento
    print("\n2. Cadeia de pensamento:")
    chain = tree.get_thought_chain(child1_id)
    for i, node in enumerate(chain):
        print(f"   Nível {i}: {node.question}")
        if node.answer:
            print(f"            Resposta: {node.answer}")
    
    # 3. Teste de orquestrador global
    print("\n3. Orquestrador Global:")
    orchestrator = GlobalThoughtOrchestrator()
    
    tree1 = orchestrator.get_or_create_tree("Trend")
    tree2 = orchestrator.get_or_create_tree("Sniper")
    
    node1_id = tree1.create_node("Why BUY?", context={'score': 75})
    node2_id = tree2.create_node("Why BUY?", context={'score': 80})
    
    # Conecta nós de diferentes módulos
    orchestrator.connect_cross_module("Trend", node1_id, "Sniper", node2_id)
    print(f"   Árvores criadas: {len(orchestrator.trees)}")
    print(f"   Conexões cross-module: {len(orchestrator.cross_module_connections)}")
    
    # 4. Resumo global
    summary = orchestrator.get_global_thought_summary()
    print(f"\n4. Resumo Global:")
    print(f"   Total de módulos: {summary['total_modules']}")
    print(f"   Total de nós: {summary['total_nodes']}")
    print(f"   Conexões cross-module: {summary['total_cross_module_connections']}")
    print(f"   Perguntas sem resposta: {summary['unanswered_questions']}")

def test_decision_memory():
    print("\n" + "="*60)
    print("   TESTE: Decision Memory (Memória de Decisões)")
    print("="*60)
    
    # 1. Cria memória para um módulo
    print("\n1. Criando memória para módulo 'Trend':")
    memory = ModuleDecisionMemory("Trend", max_memory=100)
    
    # Registra algumas decisões
    decision1_id = memory.record_decision(
        decision="BUY",
        score=75.0,
        context={'trend_score': 75, 'regime': 'TRENDING'},
        reasoning="Strong bullish trend detected",
        confidence=0.75
    )
    print(f"   Decisão 1 registrada: {decision1_id}")
    
    decision2_id = memory.record_decision(
        decision="SELL",
        score=-60.0,
        context={'trend_score': -60, 'regime': 'TRENDING'},
        reasoning="Bearish reversal detected",
        confidence=0.6
    )
    print(f"   Decisão 2 registrada: {decision2_id}")
    
    # 2. Registra resultados
    print("\n2. Registrando resultados:")
    memory.record_result(decision1_id, "WIN", pnl=50.0)
    memory.record_result(decision2_id, "LOSS", pnl=-30.0)
    print(f"   Resultado da decisão 1: WIN (+50.0)")
    print(f"   Resultado da decisão 2: LOSS (-30.0)")
    
    # 3. Adiciona perguntas recursivas
    print("\n3. Adicionando perguntas recursivas:")
    memory.add_recursive_question(decision1_id, "Why did I decide BUY?", "Strong trend")
    memory.add_recursive_question(decision1_id, "Was this correct?", "Yes, resulted in WIN")
    memory.add_recursive_question(decision1_id, "How should I act now?", "Continue with similar pattern")
    
    decision = memory.decisions[0]
    print(f"   Perguntas recursivas: {len(decision.recursive_questions)}")
    for i, q in enumerate(decision.recursive_questions):
        print(f"      {i+1}. {q}")
        if i < len(decision.recursive_answers):
            print(f"         Resposta: {decision.recursive_answers[i]}")
    
    # 4. Análise de padrões
    print("\n4. Análise de padrões:")
    patterns = memory.analyze_decision_patterns(lookback_days=30)
    print(f"   Total de decisões: {patterns['total_decisions']}")
    print(f"   Win rate: {patterns['win_rate']:.1%}")
    print(f"   Score médio: {patterns['avg_score']:.2f}")
    print(f"   Confiança média: {patterns['avg_confidence']:.2f}")
    print(f"   Padrões de sucesso: {len(patterns['success_patterns'])}")
    
    # 5. Recomendação baseada em histórico
    print("\n5. Recomendação baseada em histórico:")
    recommendation = memory.get_recommendation({
        'trend_score': 70,
        'regime': 'TRENDING'
    })
    print(f"   Recomendação: {recommendation['recommendation']}")
    print(f"   Confiança: {recommendation['confidence']:.2f}")
    print(f"   Razão: {recommendation['reason']}")
    
    # 6. Teste de memória global
    print("\n6. Memória Global:")
    global_memory = GlobalDecisionMemory()
    
    memory1 = global_memory.get_or_create_memory("Trend")
    memory2 = global_memory.get_or_create_memory("Sniper")
    
    memory1.record_decision("BUY", 75.0, {'score': 75}, "Trend bullish")
    memory2.record_decision("BUY", 80.0, {'score': 80}, "Sniper level hit")
    
    insights = global_memory.get_cross_module_insights("Trend", "Sniper")
    print(f"   Insights cross-module:")
    print(f"      Módulo: {insights.get('module', 'N/A')}")
    print(f"      Win rate: {insights.get('win_rate', 0):.1%}")
    print(f"      Total decisões: {insights.get('total_decisions', 0)}")

def main():
    test_thought_tree()
    test_decision_memory()
    
    print("\n" + "="*60)
    print("   TESTE FASE 6 CONCLUÍDO")
    print("="*60)

if __name__ == "__main__":
    main()
