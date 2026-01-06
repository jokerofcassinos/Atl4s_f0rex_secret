"""
Fase 5: Sistema de Gap de Fim de Semana
Weekend Gap Predictor - Sistema de Predição de Gaps de Fim de Semana

Este módulo implementa o sistema de análise recursiva de gaps de fim de semana,
simulando movimentos possíveis entre sexta-feira e segunda-feira e fazendo
perguntas recursivas sobre padrões históricos.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger("WeekendGapPredictor")

class WeekendGapPredictor:
    """
    O Preditor de Gaps de Fim de Semana.
    
    Responsabilidades:
    1. Simulação de movimentos possíveis entre sexta e segunda
    2. Perguntas recursivas: "Gráfico desceu/subiu?", "Quantas vezes?", "Criou tendência?"
    3. Análise de liquidez, bullish/bearish, possíveis análises
    4. Integração com todos os sistemas de análise
    """
    
    def __init__(self, history_days: int = 365):
        """
        Inicializa o preditor de gaps.
        
        Args:
            history_days: Quantos dias de histórico analisar para padrões
        """
        self.history_days = history_days
        self.gap_history = []  # Armazena histórico de gaps analisados
        self.pattern_cache = {}  # Cache de padrões similares
        
    def is_weekend(self, timestamp: datetime = None) -> bool:
        """
        Verifica se estamos em fim de semana.
        
        Args:
            timestamp: Timestamp a verificar (default: agora)
            
        Returns:
            True se for sábado ou domingo
        """
        if timestamp is None:
            timestamp = datetime.now()
        return timestamp.weekday() >= 5  # 5 = Saturday, 6 = Sunday
    
    def is_friday_close(self, timestamp: datetime = None) -> bool:
        """
        Verifica se estamos próximo do fechamento de sexta-feira.
        Mercados Forex/Crypto fecham às 22:00 GMT na sexta.
        
        Args:
            timestamp: Timestamp a verificar (default: agora)
            
        Returns:
            True se for sexta-feira após 20:00
        """
        if timestamp is None:
            timestamp = datetime.now()
        return timestamp.weekday() == 4 and timestamp.hour >= 20  # Friday >= 20:00
    
    def is_monday_open(self, timestamp: datetime = None) -> bool:
        """
        Verifica se estamos na abertura de segunda-feira.
        Mercados abrem às 00:00 GMT na segunda.
        
        Args:
            timestamp: Timestamp a verificar (default: agora)
            
        Returns:
            True se for segunda-feira antes das 10:00
        """
        if timestamp is None:
            timestamp = datetime.now()
        return timestamp.weekday() == 0 and timestamp.hour < 10  # Monday < 10:00
    
    def get_friday_close_price(self, df: pd.DataFrame) -> Optional[float]:
        """
        Obtém o preço de fechamento da última sexta-feira.
        
        Args:
            df: DataFrame com dados históricos
            
        Returns:
            Preço de fechamento da última sexta ou None
        """
        if df is None or len(df) < 1:
            return None
            
        # Procura pela última sexta-feira no histórico
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            return df_copy['close'].iloc[-1]  # Fallback: último preço
            
        # Filtra apenas sextas-feiras
        fridays = df_copy[df_copy.index.weekday == 4]
        if len(fridays) > 0:
            return fridays['close'].iloc[-1]
        
        # Fallback: último preço disponível
        return df_copy['close'].iloc[-1]
    
    def get_monday_open_price(self, df: pd.DataFrame) -> Optional[float]:
        """
        Obtém o preço de abertura da última segunda-feira.
        
        Args:
            df: DataFrame com dados históricos
            
        Returns:
            Preço de abertura da última segunda ou None
        """
        if df is None or len(df) < 1:
            return None
            
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            return df_copy['open'].iloc[0]  # Fallback: primeiro preço
            
        # Filtra apenas segundas-feiras
        mondays = df_copy[df_copy.index.weekday == 0]
        if len(mondays) > 0:
            return mondays['open'].iloc[-1]
        
        # Fallback: primeiro preço disponível
        return df_copy['open'].iloc[0]
    
    def calculate_historical_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calcula todos os gaps históricos de fim de semana.
        
        Pergunta recursiva: "Quantas vezes o gráfico desceu/subiu no fim de semana?"
        
        Args:
            df: DataFrame com dados históricos (preferencialmente D1 ou H4)
            
        Returns:
            Lista de dicionários com informações sobre cada gap
        """
        if df is None or len(df) < 7:
            return []
        
        gaps = []
        df_copy = df.copy()
        
        # Garante que o índice é DatetimeIndex
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            logger.warning("DataFrame index não é DatetimeIndex. Usando fallback.")
            return []
        
        # Itera sobre o histórico procurando por sextas e segundas consecutivas
        for i in range(len(df_copy) - 1):
            current_date = df_copy.index[i]
            next_date = df_copy.index[i + 1]
            
            # Verifica se há um gap de fim de semana (sexta -> segunda)
            if current_date.weekday() == 4 and next_date.weekday() == 0:
                friday_close = df_copy.iloc[i]['close']
                monday_open = df_copy.iloc[i + 1]['open']
                
                gap_pct = ((monday_open - friday_close) / friday_close) * 100
                gap_points = monday_open - friday_close
                
                # Determina direção
                direction = "UP" if gap_pct > 0 else "DOWN" if gap_pct < 0 else "FLAT"
                
                # Analisa o movimento subsequente (criou tendência?)
                trend_created = self._analyze_post_gap_trend(df_copy, i + 1)
                
                gap_info = {
                    'friday_date': current_date,
                    'monday_date': next_date,
                    'friday_close': friday_close,
                    'monday_open': monday_open,
                    'gap_pct': gap_pct,
                    'gap_points': gap_points,
                    'direction': direction,
                    'trend_created': trend_created,
                    'liquidity_analysis': self._analyze_liquidity(df_copy, i, i + 1)
                }
                
                gaps.append(gap_info)
        
        return gaps
    
    def _analyze_post_gap_trend(self, df: pd.DataFrame, monday_idx: int, lookahead: int = 5) -> Dict[str, Any]:
        """
        Analisa se o gap criou uma tendência.
        
        Pergunta recursiva: "Criou tendência?"
        
        Args:
            df: DataFrame com dados
            monday_idx: Índice da segunda-feira após o gap
            lookahead: Quantos períodos à frente analisar
            
        Returns:
            Dicionário com análise de tendência
        """
        if monday_idx + lookahead >= len(df):
            return {'trend': 'INSUFFICIENT_DATA', 'strength': 0.0}
        
        monday_price = df.iloc[monday_idx]['close']
        future_price = df.iloc[monday_idx + lookahead]['close']
        
        change_pct = ((future_price - monday_price) / monday_price) * 100
        
        # Determina força da tendência
        if abs(change_pct) > 2.0:
            trend = "STRONG_" + ("UP" if change_pct > 0 else "DOWN")
            strength = min(abs(change_pct) / 5.0, 1.0)  # Normaliza até 1.0
        elif abs(change_pct) > 0.5:
            trend = "WEAK_" + ("UP" if change_pct > 0 else "DOWN")
            strength = abs(change_pct) / 2.0
        else:
            trend = "NONE"
            strength = 0.0
        
        return {
            'trend': trend,
            'strength': strength,
            'change_pct': change_pct
        }
    
    def _analyze_liquidity(self, df: pd.DataFrame, friday_idx: int, monday_idx: int) -> Dict[str, Any]:
        """
        Analisa a liquidez antes e depois do gap.
        
        Args:
            df: DataFrame com dados
            friday_idx: Índice da sexta-feira
            monday_idx: Índice da segunda-feira
            
        Returns:
            Dicionário com análise de liquidez
        """
        if 'volume' not in df.columns:
            return {'liquidity': 'UNKNOWN', 'friday_volume': 0, 'monday_volume': 0}
        
        friday_volume = df.iloc[friday_idx]['volume'] if friday_idx < len(df) else 0
        monday_volume = df.iloc[monday_idx]['volume'] if monday_idx < len(df) else 0
        
        # Compara com média de volume
        avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else monday_volume
        
        liquidity = "HIGH" if monday_volume > avg_volume * 1.5 else "NORMAL" if monday_volume > avg_volume * 0.5 else "LOW"
        
        return {
            'liquidity': liquidity,
            'friday_volume': float(friday_volume),
            'monday_volume': float(monday_volume),
            'avg_volume': float(avg_volume)
        }
    
    def recursive_question_analysis(self, gaps: List[Dict[str, Any]], depth: int = 3) -> Dict[str, Any]:
        """
        Faz perguntas recursivas sobre os gaps históricos.
        
        Perguntas recursivas:
        - Nível 1: "Gráfico desceu/subiu?"
        - Nível 2: "Quantas vezes?"
        - Nível 3: "Criou tendência?"
        - Nível N: "Por que criou tendência?", "Qual contexto?"
        
        Args:
            gaps: Lista de gaps históricos
            depth: Profundidade da recursão (limite para evitar loops infinitos)
            
        Returns:
            Dicionário com análise recursiva
        """
        if not gaps or depth <= 0:
            return {'analysis': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        # Nível 1: Pergunta básica - "Gráfico desceu/subiu?"
        up_gaps = [g for g in gaps if g['direction'] == 'UP']
        down_gaps = [g for g in gaps if g['direction'] == 'DOWN']
        flat_gaps = [g for g in gaps if g['direction'] == 'FLAT']
        
        total_gaps = len(gaps)
        up_pct = len(up_gaps) / total_gaps if total_gaps > 0 else 0
        down_pct = len(down_gaps) / total_gaps if total_gaps > 0 else 0
        
        # Nível 2: "Quantas vezes?"
        avg_gap_size = np.mean([abs(g['gap_pct']) for g in gaps]) if gaps else 0
        max_gap = max([abs(g['gap_pct']) for g in gaps]) if gaps else 0
        
        # Nível 3: "Criou tendência?"
        trend_analysis = self._recursive_trend_analysis(gaps, depth - 1)
        
        # Análise de contexto (profundidade adicional)
        context_analysis = self._analyze_gap_context(gaps, depth - 1) if depth > 1 else {}
        
        return {
            'level_1_direction': {
                'up_pct': up_pct,
                'down_pct': down_pct,
                'flat_pct': len(flat_gaps) / total_gaps if total_gaps > 0 else 0,
                'dominant': 'UP' if up_pct > 0.6 else 'DOWN' if down_pct > 0.6 else 'MIXED'
            },
            'level_2_frequency': {
                'total_gaps': total_gaps,
                'avg_gap_size_pct': avg_gap_size,
                'max_gap_size_pct': max_gap,
                'up_count': len(up_gaps),
                'down_count': len(down_gaps)
            },
            'level_3_trend_creation': trend_analysis,
            'context_analysis': context_analysis,
            'confidence': min(len(gaps) / 20.0, 1.0)  # Mais gaps = mais confiança (até 1.0)
        }
    
    def _recursive_trend_analysis(self, gaps: List[Dict[str, Any]], depth: int) -> Dict[str, Any]:
        """
        Análise recursiva de tendências criadas por gaps.
        
        Args:
            gaps: Lista de gaps
            depth: Profundidade restante
            
        Returns:
            Análise de tendências
        """
        if depth <= 0 or not gaps:
            return {'trend_creation_rate': 0.0, 'strong_trends': 0}
        
        trends_created = [g for g in gaps if g.get('trend_created', {}).get('trend') != 'NONE']
        strong_trends = [g for g in trends_created if g.get('trend_created', {}).get('strength', 0) > 0.5]
        
        trend_rate = len(trends_created) / len(gaps) if gaps else 0
        strong_trend_rate = len(strong_trends) / len(gaps) if gaps else 0
        
        return {
            'trend_creation_rate': trend_rate,
            'strong_trend_rate': strong_trend_rate,
            'trends_created': len(trends_created),
            'strong_trends': len(strong_trends),
            'avg_trend_strength': np.mean([g.get('trend_created', {}).get('strength', 0) for g in trends_created]) if trends_created else 0
        }
    
    def _analyze_gap_context(self, gaps: List[Dict[str, Any]], depth: int) -> Dict[str, Any]:
        """
        Análise recursiva do contexto dos gaps.
        
        Pergunta: "Por que criou tendência?", "Qual contexto?"
        
        Args:
            gaps: Lista de gaps
            depth: Profundidade restante
            
        Returns:
            Análise de contexto
        """
        if depth <= 0 or not gaps:
            return {}
        
        # Analisa padrões de liquidez
        high_liquidity_gaps = [g for g in gaps if g.get('liquidity_analysis', {}).get('liquidity') == 'HIGH']
        low_liquidity_gaps = [g for g in gaps if g.get('liquidity_analysis', {}).get('liquidity') == 'LOW']
        
        # Correlação entre liquidez e criação de tendência
        high_liq_trends = [g for g in high_liquidity_gaps if g.get('trend_created', {}).get('trend') != 'NONE']
        low_liq_trends = [g for g in low_liquidity_gaps if g.get('trend_created', {}).get('trend') != 'NONE']
        
        return {
            'liquidity_correlation': {
                'high_liquidity_gaps': len(high_liquidity_gaps),
                'low_liquidity_gaps': len(low_liquidity_gaps),
                'high_liq_trend_rate': len(high_liq_trends) / len(high_liquidity_gaps) if high_liquidity_gaps else 0,
                'low_liq_trend_rate': len(low_liq_trends) / len(low_liquidity_gaps) if low_liquidity_gaps else 0
            },
            'gap_size_correlation': {
                'large_gaps': len([g for g in gaps if abs(g['gap_pct']) > 1.0]),
                'small_gaps': len([g for g in gaps if abs(g['gap_pct']) < 0.5])
            }
        }
    
    def predict_weekend_gap(self, df: pd.DataFrame, data_map: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Prediz o gap de fim de semana baseado em análise histórica e contexto atual.
        
        Integração com todos os sistemas de análise.
        
        Args:
            df: DataFrame com dados atuais (preferencialmente M5 ou H1)
            data_map: Mapa de dados multi-timeframe (H4, D1, W1)
            
        Returns:
            Dicionário com predição e análise
        """
        if df is None or len(df) < 7:
            return {
                'prediction': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'recommendation': 'WAIT'
            }
        
        # 1. Calcula gaps históricos
        historical_gaps = self.calculate_historical_gaps(df)
        
        if not historical_gaps:
            # Tenta com timeframe maior se disponível
            if data_map and 'D1' in data_map:
                historical_gaps = self.calculate_historical_gaps(data_map['D1'])
            elif data_map and 'H4' in data_map:
                historical_gaps = self.calculate_historical_gaps(data_map['H4'])
        
        # 2. Análise recursiva
        recursive_analysis = self.recursive_question_analysis(historical_gaps, depth=3)
        
        # 3. Contexto atual
        current_context = self._analyze_current_context(df, data_map)
        
        # 4. Predição baseada em padrões
        prediction = self._generate_prediction(recursive_analysis, current_context, historical_gaps)
        
        # 5. Recomendação
        recommendation = self._generate_recommendation(prediction, recursive_analysis, current_context)
        
        return {
            'prediction': prediction,
            'recursive_analysis': recursive_analysis,
            'current_context': current_context,
            'historical_gaps': len(historical_gaps),
            'recommendation': recommendation,
            'confidence': recursive_analysis.get('confidence', 0.0)
        }
    
    def _analyze_current_context(self, df: pd.DataFrame, data_map: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analisa o contexto atual do mercado antes do fim de semana.
        
        Args:
            df: DataFrame atual
            data_map: Mapa de dados multi-timeframe
            
        Returns:
            Análise do contexto atual
        """
        if df is None or len(df) < 1:
            return {}
        
        current_price = df['close'].iloc[-1]
        recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0
        
        # Análise de volatilidade
        if len(df) >= 20:
            volatility = df['close'].pct_change().std() * 100
        else:
            volatility = 0
        
        # Análise multi-timeframe se disponível
        mtf_analysis = {}
        if data_map:
            for tf_name, tf_df in data_map.items():
                if tf_df is not None and len(tf_df) > 0:
                    tf_price = tf_df['close'].iloc[-1]
                    tf_ma = tf_df['close'].rolling(20).mean().iloc[-1] if len(tf_df) >= 20 else tf_price
                    mtf_analysis[tf_name] = {
                        'price': float(tf_price),
                        'above_ma': tf_price > tf_ma,
                        'trend': 'BULLISH' if tf_price > tf_ma else 'BEARISH'
                    }
        
        return {
            'current_price': float(current_price),
            'recent_trend_pct': recent_trend,
            'volatility': volatility,
            'mtf_analysis': mtf_analysis,
            'is_friday': self.is_friday_close(),
            'is_weekend': self.is_weekend()
        }
    
    def _generate_prediction(self, recursive_analysis: Dict, current_context: Dict, historical_gaps: List[Dict]) -> Dict[str, Any]:
        """
        Gera predição baseada em análise recursiva e contexto.
        
        Args:
            recursive_analysis: Análise recursiva dos gaps
            current_context: Contexto atual
            historical_gaps: Lista de gaps históricos
            
        Returns:
            Predição do gap
        """
        if not recursive_analysis or recursive_analysis.get('confidence', 0) < 0.3:
            return {
                'direction': 'UNKNOWN',
                'expected_gap_pct': 0.0,
                'probability': 0.5
            }
        
        level_1 = recursive_analysis.get('level_1_direction', {})
        dominant = level_1.get('dominant', 'MIXED')
        
        # Calcula gap esperado baseado na média histórica
        if historical_gaps:
            avg_gap = np.mean([g['gap_pct'] for g in historical_gaps])
            std_gap = np.std([g['gap_pct'] for g in historical_gaps])
        else:
            avg_gap = 0.0
            std_gap = 0.0
        
        # Ajusta baseado no contexto atual
        context_adjustment = 0.0
        if current_context.get('recent_trend_pct', 0) > 1.0:
            context_adjustment = 0.2  # Tendência de alta recente aumenta probabilidade de gap up
        elif current_context.get('recent_trend_pct', 0) < -1.0:
            context_adjustment = -0.2  # Tendência de baixa recente aumenta probabilidade de gap down
        
        expected_gap = avg_gap + context_adjustment
        
        # Determina direção
        if dominant == 'UP' or expected_gap > 0.3:
            direction = 'UP'
            probability = level_1.get('up_pct', 0.5)
        elif dominant == 'DOWN' or expected_gap < -0.3:
            direction = 'DOWN'
            probability = level_1.get('down_pct', 0.5)
        else:
            direction = 'FLAT'
            probability = 0.5
        
        return {
            'direction': direction,
            'expected_gap_pct': expected_gap,
            'expected_gap_range': (expected_gap - std_gap, expected_gap + std_gap),
            'probability': probability,
            'confidence': recursive_analysis.get('confidence', 0.0)
        }
    
    def _generate_recommendation(self, prediction: Dict, recursive_analysis: Dict, current_context: Dict) -> str:
        """
        Gera recomendação baseada na predição.
        
        Args:
            prediction: Predição do gap
            recursive_analysis: Análise recursiva
            current_context: Contexto atual
            
        Returns:
            Recomendação: 'BUY', 'SELL', 'WAIT', 'CLOSE_POSITIONS'
        """
        if prediction.get('confidence', 0) < 0.3:
            return 'WAIT'
        
        direction = prediction.get('direction', 'UNKNOWN')
        expected_gap = prediction.get('expected_gap_pct', 0)
        probability = prediction.get('probability', 0.5)
        
        # Se estamos em sexta-feira e há alta probabilidade de gap
        if current_context.get('is_friday', False):
            if direction == 'UP' and probability > 0.6 and abs(expected_gap) > 0.5:
                return 'BUY'  # Comprar antes do gap up
            elif direction == 'DOWN' and probability > 0.6 and abs(expected_gap) > 0.5:
                return 'SELL'  # Vender antes do gap down
            elif abs(expected_gap) > 1.0:
                return 'CLOSE_POSITIONS'  # Gap muito grande, fechar posições
        
        # Se estamos em segunda-feira e o gap já aconteceu
        if current_context.get('is_monday', False):
            trend_analysis = recursive_analysis.get('level_3_trend_creation', {})
            if trend_analysis.get('trend_creation_rate', 0) > 0.6:
                # Gaps frequentemente criam tendências
                if direction == 'UP':
                    return 'BUY'  # Seguir a tendência do gap up
                elif direction == 'DOWN':
                    return 'SELL'  # Seguir a tendência do gap down
        
        return 'WAIT'
    
    def deliberate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Método principal de deliberação (interface padrão do sistema).
        
        Args:
            data_map: Mapa de dados multi-timeframe
            
        Returns:
            Resultado da deliberação
        """
        # Prefere D1 para análise de gaps, fallback para H4 ou M5
        df = data_map.get('D1') or data_map.get('H4') or data_map.get('M5')
        
        if df is None:
            return {
                'decision': 'WAIT',
                'score': 0.0,
                'reason': 'No data available for weekend gap analysis'
            }
        
        result = self.predict_weekend_gap(df, data_map)
        
        # Converte para formato padrão do sistema
        recommendation = result.get('recommendation', 'WAIT')
        confidence = result.get('confidence', 0.0)
        
        decision = recommendation
        score = confidence * 100 if recommendation != 'WAIT' else 0.0
        
        reason = f"Weekend Gap Analysis: {result.get('prediction', {}).get('direction', 'UNKNOWN')} gap expected ({result.get('prediction', {}).get('expected_gap_pct', 0):.2f}%)"
        
        return {
            'decision': decision,
            'score': score,
            'reason': reason,
            'details': result
        }
