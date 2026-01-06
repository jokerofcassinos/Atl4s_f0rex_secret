"""
Fase 6: Memória de Decisões dos Módulos
Decision Memory - Armazena e analisa decisões passadas de cada módulo

Cada módulo tem sua memória de decisões, permitindo aprendizado e reflexão.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger("DecisionMemory")

@dataclass
class ModuleDecision:
    """
    Uma decisão tomada por um módulo.
    """
    decision_id: str
    module_name: str
    timestamp: datetime
    decision: str  # "BUY", "SELL", "WAIT", "VETO", etc.
    score: float
    context: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    
    # Resultado (preenchido posteriormente)
    result: Optional[str] = None  # "WIN", "LOSS", "BREAKEVEN", "PENDING"
    result_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    time_to_result: Optional[timedelta] = None
    
    # Análise recursiva
    recursive_questions: List[str] = field(default_factory=list)
    recursive_answers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário (para serialização)."""
        return {
            'decision_id': self.decision_id,
            'module_name': self.module_name,
            'timestamp': self.timestamp.isoformat(),
            'decision': self.decision,
            'score': self.score,
            'context': self.context,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'result': self.result,
            'result_timestamp': self.result_timestamp.isoformat() if self.result_timestamp else None,
            'pnl': self.pnl,
            'time_to_result': str(self.time_to_result) if self.time_to_result else None,
            'recursive_questions': self.recursive_questions,
            'recursive_answers': self.recursive_answers
        }

class ModuleDecisionMemory:
    """
    Memória de decisões de um módulo específico.
    """
    
    def __init__(self, module_name: str, max_memory: int = 1000):
        """
        Inicializa a memória de decisões.
        
        Args:
            module_name: Nome do módulo
            max_memory: Número máximo de decisões a armazenar
        """
        self.module_name = module_name
        self.max_memory = max_memory
        self.decisions: List[ModuleDecision] = []
        self.pending_decisions: Dict[str, ModuleDecision] = {}  # decision_id -> decision
        
    def record_decision(self, decision: str, score: float, context: Dict[str, Any],
                       reasoning: str = "", confidence: float = 0.0) -> str:
        """
        Registra uma nova decisão.
        
        Args:
            decision: Decisão tomada
            score: Score da decisão
            context: Contexto completo (dados de mercado, etc.)
            reasoning: Raciocínio por trás da decisão
            confidence: Confiança na decisão
            
        Returns:
            ID da decisão registrada
        """
        decision_id = f"{self.module_name}_{datetime.now().timestamp()}_{len(self.decisions)}"
        
        module_decision = ModuleDecision(
            decision_id=decision_id,
            module_name=self.module_name,
            timestamp=datetime.now(),
            decision=decision,
            score=score,
            context=context,
            reasoning=reasoning,
            confidence=confidence
        )
        
        self.decisions.append(module_decision)
        self.pending_decisions[decision_id] = module_decision
        
        # Limita memória
        if len(self.decisions) > self.max_memory:
            self.decisions = self.decisions[-self.max_memory:]
        
        return decision_id
    
    def record_result(self, decision_id: str, result: str, pnl: Optional[float] = None):
        """
        Registra o resultado de uma decisão.
        
        Args:
            decision_id: ID da decisão
            result: Resultado ("WIN", "LOSS", "BREAKEVEN")
            pnl: Lucro/Prejuízo (opcional)
        """
        # Procura na lista de decisões
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                decision.result = result
                decision.result_timestamp = datetime.now()
                decision.pnl = pnl
                if decision.timestamp:
                    decision.time_to_result = decision.result_timestamp - decision.timestamp
                
                # Remove dos pendentes
                if decision_id in self.pending_decisions:
                    del self.pending_decisions[decision_id]
                
                return
        
        logger.warning(f"DecisionMemory {self.module_name}: Decision {decision_id} not found.")
    
    def add_recursive_question(self, decision_id: str, question: str, answer: str = ""):
        """
        Adiciona uma pergunta recursiva a uma decisão.
        
        Args:
            decision_id: ID da decisão
            question: Pergunta recursiva
            answer: Resposta (opcional, pode ser preenchida depois)
        """
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                decision.recursive_questions.append(question)
                if answer:
                    decision.recursive_answers.append(answer)
                else:
                    decision.recursive_answers.append("")
                return
        
        logger.warning(f"DecisionMemory {self.module_name}: Decision {decision_id} not found.")
    
    def analyze_decision_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analisa padrões nas decisões passadas.
        
        Perguntas recursivas:
        - "Por que fiz isso?"
        - "Foi correto?"
        - "O que aconteceria se...?"
        
        Args:
            lookback_days: Quantos dias olhar para trás
            
        Returns:
            Análise de padrões
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_decisions = [d for d in self.decisions 
                          if d.timestamp >= cutoff_date and d.result is not None]
        
        if not recent_decisions:
            return {
                'total_decisions': 0,
                'win_rate': 0.0,
                'avg_score': 0.0,
                'avg_confidence': 0.0,
                'success_patterns': [],
                'failure_patterns': []
            }
        
        # Estatísticas básicas
        total = len(recent_decisions)
        wins = len([d for d in recent_decisions if d.result == "WIN"])
        losses = len([d for d in recent_decisions if d.result == "LOSS"])
        win_rate = wins / total if total > 0 else 0.0
        
        avg_score = np.mean([d.score for d in recent_decisions])
        avg_confidence = np.mean([d.confidence for d in recent_decisions])
        
        # Análise de padrões de sucesso
        successful_decisions = [d for d in recent_decisions if d.result == "WIN"]
        failed_decisions = [d for d in recent_decisions if d.result == "LOSS"]
        
        # Padrões comuns em decisões bem-sucedidas
        success_patterns = self._extract_patterns(successful_decisions)
        failure_patterns = self._extract_patterns(failed_decisions)
        
        return {
            'total_decisions': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_confidence': avg_confidence,
            'success_patterns': success_patterns,
            'failure_patterns': failure_patterns,
            'avg_pnl': np.mean([d.pnl for d in recent_decisions if d.pnl is not None]) if any(d.pnl for d in recent_decisions) else 0.0
        }
    
    def _extract_patterns(self, decisions: List[ModuleDecision]) -> List[Dict[str, Any]]:
        """
        Extrai padrões comuns de um conjunto de decisões.
        
        Args:
            decisions: Lista de decisões
            
        Returns:
            Lista de padrões identificados
        """
        if not decisions:
            return []
        
        patterns = []
        
        # Padrão 1: Decisões com alta confiança
        high_conf = [d for d in decisions if d.confidence > 0.7]
        if len(high_conf) > len(decisions) * 0.5:
            patterns.append({
                'type': 'HIGH_CONFIDENCE',
                'frequency': len(high_conf) / len(decisions),
                'description': 'Decisões com alta confiança (>70%)'
            })
        
        # Padrão 2: Decisões com score alto
        high_score = [d for d in decisions if abs(d.score) > 50]
        if len(high_score) > len(decisions) * 0.5:
            patterns.append({
                'type': 'HIGH_SCORE',
                'frequency': len(high_score) / len(decisions),
                'description': 'Decisões com score alto (>50)'
            })
        
        # Padrão 3: Contextos similares (simplificado)
        # Em produção, usar análise vetorial mais sofisticada
        common_context_keys = set()
        for d in decisions[:10]:  # Amostra
            common_context_keys.update(d.context.keys())
        
        if common_context_keys:
            patterns.append({
                'type': 'COMMON_CONTEXT',
                'frequency': 1.0,
                'description': f'Contextos com chaves: {", ".join(list(common_context_keys)[:5])}'
            })
        
        return patterns
    
    def find_similar_decisions(self, current_context: Dict[str, Any], 
                              limit: int = 5) -> List[ModuleDecision]:
        """
        Encontra decisões similares no passado.
        
        Args:
            current_context: Contexto atual
            limit: Número máximo de decisões similares
            
        Returns:
            Lista de decisões similares
        """
        # Implementação simplificada: compara chaves do contexto
        # Em produção, usar embedding vetorial
        current_keys = set(current_context.keys())
        
        similarities = []
        for decision in self.decisions:
            if decision.result is None:
                continue  # Pula decisões sem resultado
            
            decision_keys = set(decision.context.keys())
            intersection = len(current_keys & decision_keys)
            union = len(current_keys | decision_keys)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.3:  # Limiar mínimo
                similarities.append((decision, similarity))
        
        # Ordena por similaridade
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [d[0] for d in similarities[:limit]]
    
    def get_recommendation(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtém recomendação baseada em decisões passadas similares.
        
        Pergunta recursiva: "Como agir agora?"
        
        Args:
            current_context: Contexto atual
            
        Returns:
            Recomendação com base em histórico
        """
        similar_decisions = self.find_similar_decisions(current_context, limit=10)
        
        if not similar_decisions:
            return {
                'recommendation': 'WAIT',
                'confidence': 0.0,
                'reason': 'No similar past decisions found'
            }
        
        # Analisa resultados das decisões similares
        successful = [d for d in similar_decisions if d.result == "WIN"]
        failed = [d for d in similar_decisions if d.result == "LOSS"]
        
        success_rate = len(successful) / len(similar_decisions) if similar_decisions else 0.0
        
        # Determina recomendação baseada em sucesso
        if success_rate > 0.6:
            # Decisões similares foram bem-sucedidas
            # Recomenda seguir o padrão
            avg_decision = np.mean([1 if d.decision == "BUY" else -1 if d.decision == "SELL" else 0 
                                   for d in successful])
            
            if avg_decision > 0.3:
                recommendation = "BUY"
            elif avg_decision < -0.3:
                recommendation = "SELL"
            else:
                recommendation = "WAIT"
            
            return {
                'recommendation': recommendation,
                'confidence': success_rate,
                'reason': f'Similar decisions had {success_rate:.1%} success rate',
                'similar_decisions_count': len(similar_decisions),
                'successful_count': len(successful)
            }
        else:
            # Decisões similares falharam
            return {
                'recommendation': 'WAIT',
                'confidence': 1.0 - success_rate,
                'reason': f'Similar decisions had low success rate ({success_rate:.1%})',
                'similar_decisions_count': len(similar_decisions),
                'failed_count': len(failed)
            }

class GlobalDecisionMemory:
    """
    Memória global de decisões de todos os módulos.
    """
    
    def __init__(self):
        self.module_memories: Dict[str, ModuleDecisionMemory] = {}
    
    def get_or_create_memory(self, module_name: str) -> ModuleDecisionMemory:
        """
        Obtém ou cria memória para um módulo.
        
        Args:
            module_name: Nome do módulo
            
        Returns:
            ModuleDecisionMemory do módulo
        """
        if module_name not in self.module_memories:
            self.module_memories[module_name] = ModuleDecisionMemory(module_name)
        return self.module_memories[module_name]
    
    def get_cross_module_insights(self, module_name: str, 
                                 other_module_name: str) -> Dict[str, Any]:
        """
        Obtém insights sobre como outro módulo pensou em contextos similares.
        
        Pergunta: "O que módulo X pensou quando módulo Y decidiu Y?"
        
        Args:
            module_name: Módulo que está perguntando
            other_module_name: Módulo a analisar
            
        Returns:
            Insights sobre o outro módulo
        """
        if module_name not in self.module_memories or other_module_name not in self.module_memories:
            return {'error': 'One or both modules not found'}
        
        other_memory = self.module_memories[other_module_name]
        analysis = other_memory.analyze_decision_patterns()
        
        return {
            'module': other_module_name,
            'win_rate': analysis.get('win_rate', 0.0),
            'avg_confidence': analysis.get('avg_confidence', 0.0),
            'success_patterns': analysis.get('success_patterns', []),
            'total_decisions': analysis.get('total_decisions', 0)
        }
