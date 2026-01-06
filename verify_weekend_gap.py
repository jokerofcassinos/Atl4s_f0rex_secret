"""
Teste do Weekend Gap Predictor (Fase 5)
Verifica se o sistema de predição de gaps de fim de semana está funcionando corretamente.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from analysis.agi.weekend_gap_predictor import WeekendGapPredictor

def test_weekend_gap_predictor():
    print("="*60)
    print("   TESTE: Weekend Gap Predictor (Fase 5)")
    print("="*60)
    
    # Inicializa o preditor
    predictor = WeekendGapPredictor(history_days=365)
    
    # 1. Teste de detecção de fim de semana
    print("\n1. Teste de Detecção de Fim de Semana:")
    now = datetime.now()
    is_weekend = predictor.is_weekend(now)
    is_friday = predictor.is_friday_close(now)
    is_monday = predictor.is_monday_open(now)
    print(f"   Hoje: {now.strftime('%A, %Y-%m-%d %H:%M')}")
    print(f"   É fim de semana: {is_weekend}")
    print(f"   É sexta-feira (fechamento): {is_friday}")
    print(f"   É segunda-feira (abertura): {is_monday}")
    
    # 2. Cria dados mock para teste
    print("\n2. Criando dados históricos mock...")
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Simula preços com alguns gaps de fim de semana
    np.random.seed(42)
    base_price = 2000.0
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # Simula gaps de fim de semana
        if dates[i].weekday() == 0:  # Segunda-feira
            prev_friday_idx = i - 3 if i >= 3 else 0
            # Adiciona gap aleatório
            gap = np.random.normal(0, 10)  # Gap médio de ±10 pontos
            prices.append(prices[prev_friday_idx] + gap)
        else:
            # Movimento normal durante a semana
            change = np.random.normal(0, 5)
            prices.append(prices[-1] + change)
    
    df_d1 = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 2)) for p in prices],
        'low': [p - abs(np.random.normal(0, 2)) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(prices))
    }, index=dates)
    
    print(f"   Dados criados: {len(df_d1)} períodos")
    print(f"   Período: {df_d1.index[0]} a {df_d1.index[-1]}")
    
    # 3. Teste de cálculo de gaps históricos
    print("\n3. Calculando gaps históricos...")
    historical_gaps = predictor.calculate_historical_gaps(df_d1)
    print(f"   Gaps encontrados: {len(historical_gaps)}")
    
    if historical_gaps:
        print("\n   Detalhes dos gaps:")
        for i, gap in enumerate(historical_gaps[:5]):  # Mostra apenas os 5 primeiros
            print(f"   Gap {i+1}:")
            print(f"     Sexta: {gap['friday_date'].strftime('%Y-%m-%d')} @ {gap['friday_close']:.2f}")
            print(f"     Segunda: {gap['monday_date'].strftime('%Y-%m-%d')} @ {gap['monday_open']:.2f}")
            print(f"     Gap: {gap['gap_pct']:.2f}% ({gap['gap_points']:.2f} pontos)")
            print(f"     Direção: {gap['direction']}")
            print(f"     Tendência criada: {gap['trend_created'].get('trend', 'NONE')}")
            print(f"     Liquidez: {gap['liquidity_analysis'].get('liquidity', 'UNKNOWN')}")
    
    # 4. Teste de análise recursiva
    print("\n4. Análise Recursiva de Gaps:")
    if historical_gaps:
        recursive_analysis = predictor.recursive_question_analysis(historical_gaps, depth=3)
        print(f"   Confiança: {recursive_analysis.get('confidence', 0):.2%}")
        
        level_1 = recursive_analysis.get('level_1_direction', {})
        print(f"   Nível 1 - Direção:")
        print(f"     UP: {level_1.get('up_pct', 0):.2%}")
        print(f"     DOWN: {level_1.get('down_pct', 0):.2%}")
        print(f"     Dominante: {level_1.get('dominant', 'MIXED')}")
        
        level_2 = recursive_analysis.get('level_2_frequency', {})
        print(f"   Nível 2 - Frequência:")
        print(f"     Total de gaps: {level_2.get('total_gaps', 0)}")
        print(f"     Gap médio: {level_2.get('avg_gap_size_pct', 0):.2f}%")
        print(f"     Gap máximo: {level_2.get('max_gap_size_pct', 0):.2f}%")
        
        level_3 = recursive_analysis.get('level_3_trend_creation', {})
        print(f"   Nível 3 - Criação de Tendência:")
        print(f"     Taxa de criação: {level_3.get('trend_creation_rate', 0):.2%}")
        print(f"     Tendências fortes: {level_3.get('strong_trends', 0)}")
    else:
        print("   Nenhum gap histórico encontrado para análise recursiva")
    
    # 5. Teste de predição
    print("\n5. Predição de Gap de Fim de Semana:")
    data_map = {
        'D1': df_d1,
        'H4': df_d1.resample('4H').last().fillna(method='ffill'),
        'M5': df_d1.resample('5T').last().fillna(method='ffill')
    }
    
    prediction_result = predictor.predict_weekend_gap(df_d1, data_map)
    
    print(f"   Predição: {prediction_result.get('prediction', {}).get('direction', 'UNKNOWN')}")
    print(f"   Gap esperado: {prediction_result.get('prediction', {}).get('expected_gap_pct', 0):.2f}%")
    print(f"   Probabilidade: {prediction_result.get('prediction', {}).get('probability', 0):.2%}")
    print(f"   Confiança: {prediction_result.get('confidence', 0):.2%}")
    print(f"   Recomendação: {prediction_result.get('recommendation', 'WAIT')}")
    
    # 6. Teste de deliberação (interface padrão)
    print("\n6. Teste de Deliberação (Interface Padrão):")
    deliberation_result = predictor.deliberate(data_map)
    
    print(f"   Decisão: {deliberation_result.get('decision', 'WAIT')}")
    print(f"   Score: {deliberation_result.get('score', 0):.2f}")
    print(f"   Razão: {deliberation_result.get('reason', 'N/A')}")
    
    print("\n" + "="*60)
    print("   TESTE CONCLUÍDO")
    print("="*60)

if __name__ == "__main__":
    test_weekend_gap_predictor()
