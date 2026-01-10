import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np

from core.memory.holographic import HolographicMemory
from core.agi.decision_memory import GlobalDecisionMemory, ModuleDecisionMemory
from core.agi.thought_tree import GlobalThoughtOrchestrator, ThoughtTree

logger = logging.getLogger("InfiniteWhyEngine")


@dataclass
class MemoryEvent:
    """
    Módulo 1: Memória Holográfica e Decisão Recursiva

    Representa um evento atômico observado pelo sistema:
    - um candle/tick
    - um conjunto de análises
    - uma decisão (ou ausência de decisão)
    - o resultado observado posteriormente
    """
    event_id: str
    timestamp: float

    symbol: str
    timeframe: str

    market_state: Dict[str, Any]          # snapshot completo: preço, candle, indicadores, sessão, notícias, etc
    analysis_state: Dict[str, Any]        # saídas dos módulos de análise naquele instante

    decision: Optional[str] = None        # "BUY", "SELL", "WAIT", "CLOSE", etc
    decision_score: float = 0.0
    decision_meta: Dict[str, Any] = field(default_factory=dict)

    # Preenchido após o desfecho da ordem / sequência
    outcome: Optional[Dict[str, Any]] = None   # {pnl, max_drawdown, max_favor, duration, regime_final, ...}


@dataclass
class WhyNode:
    """
    Nó lógico da Árvore de Pensamento causal para um único evento.
    """
    node_id: str
    depth: int
    question: str
    context: Dict[str, Any]
    answer: Optional[str] = None
    confidence: float = 0.0
    children: List["WhyNode"] = field(default_factory=list)


@dataclass
class ScenarioBranch:
    """
    Representa um ramo contrafactual: "e se eu tivesse feito X ao invés de Y".
    """
    branch_id: str
    counterfactual_decision: str              # decisão hipotética
    parameters: Dict[str, Any]                # slots, vsl, vtp, modo de execução, etc
    estimated_outcome: Dict[str, Any]         # métricas estimadas (pnl esperado, drawdown, risco, probabilidade)
    supporting_events: List[Tuple[str, float]] = field(default_factory=list)
    # lista de (event_id_similar, similarity_score)


class InfiniteWhyEngine:
    """
    ALGORITMO DE RECURSIVIDADE INFINITA (The Infinite Why Engine)

    Responsabilidades centrais do Módulo 1:
    - Capturar cada evento (tick/candle + análises + decisão).
    - Armazenar vetor holográfico + contexto bruto para cada evento.
    - Buscar eventos passados similares via pattern matching vetorial.
    - Executar Deep_Scan_Recursive:
        - "Por que eu fiz isso?" (causa imediata)
        - "Por que a causa existia?" (meta-causa)
        - "O que teria acontecido se...?" (ramificação de cenários)
    """

    def __init__(
        self,
        max_depth: int = 4,   # Phase 3 Optimization: Reduced from 32 (AGI Ultra) to 4
        max_branches: int = 12, # Phase 3 Optimization: Reduced from 128 (AGI Ultra) to 12
        vector_top_k: int = 20, # Phase 3 Optimization: Reduced from 512 to 20
        parallel_workers: int = 2, # Phase 3 Optimization: Reduced from 8 to 2
        enable_meta_reasoning: bool = True,  # AGI Ultra: Meta-cognitive layer
        adaptive_depth: bool = True,  # AGI Ultra: Dynamic depth based on importance
    ):
        # Memória holográfica contínua (todos os ticks, todos os contextos)
        self.holographic_memory = HolographicMemory()

        # Memória de decisões por módulo (Trend, Sniper, Consensus, etc.)
        self.global_decision_memory = GlobalDecisionMemory()

        # Orquestrador de árvores de pensamento globais
        self.thought_orchestrator = GlobalThoughtOrchestrator()

        # Log explícito de eventos + índices vetoriais para pattern matching direto
        self.events: List[MemoryEvent] = []
        self.event_vectors: List[np.ndarray] = []

        # AGI Ultra: Core parameters
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.vector_top_k = vector_top_k
        
        # AGI Ultra: Parallel processing
        self.parallel_workers = parallel_workers
        self._executor = ThreadPoolExecutor(max_workers=parallel_workers)
        
        # AGI Ultra: Meta-reasoning configuration
        self.enable_meta_reasoning = enable_meta_reasoning
        self.meta_reasoning_depth = 3  # Levels of "why am I thinking this?"
        
        # AGI Ultra: Adaptive depth control
        self.adaptive_depth = adaptive_depth
        self.min_adaptive_depth = 4  # Minimum depth for low-importance decisions
        self.max_adaptive_depth = 32  # Maximum depth for critical decisions
        self.importance_thresholds = {
            'critical': 0.9,  # Use full 32 depth
            'high': 0.7,      # Use 24 depth
            'medium': 0.5,    # Use 16 depth
            'low': 0.3,       # Use 8 depth
        }
        
        # AGI Ultra: Multi-scale causal tracking
        self.causal_scales = ['immediate', 'short_term', 'medium_term', 'fundamental']
        self.causal_chains: Dict[str, List[Dict[str, Any]]] = {scale: [] for scale in self.causal_scales}
        
        # AGI Ultra: Contextual question templates (dynamic generation)
        self.question_generators: List[Callable] = []
        self._init_question_generators()
        
        logger.info(f"InfiniteWhyEngine initialized: depth={max_depth}, branches={max_branches}, "
                    f"parallel_workers={parallel_workers}, meta_reasoning={enable_meta_reasoning}")
    
    def _init_question_generators(self):
        """AGI Ultra: Initialize dynamic question generators for contextual questioning."""
        self.question_generators = [
            # Market structure questions
            lambda ctx: f"Why is the {ctx.get('structure', 'market')} suggesting {ctx.get('signal', 'this')}?",
            # Temporal questions
            lambda ctx: f"Why is this {ctx.get('timeframe', 'moment')} significant for {ctx.get('decision', 'action')}?",
            # Causal questions
            lambda ctx: f"What underlying force caused {ctx.get('observation', 'this pattern')}?",
            # Counterfactual questions
            lambda ctx: f"What would happen if {ctx.get('alternative', 'the opposite')} occurred?",
            # Meta-questions
            lambda ctx: f"Why am I considering {ctx.get('factor', 'this factor')} important?",
            # Risk questions
            lambda ctx: f"What risk does {ctx.get('scenario', 'this scenario')} pose?",
            # Confidence questions
            lambda ctx: f"Why is my confidence {ctx.get('confidence_level', 'at this level')}?",
        ]
    
    # -------------------------------------------------------------------------
    # AGI ULTRA: ADAPTIVE DEPTH CONTROL
    # -------------------------------------------------------------------------
    def adaptive_depth_control(self, decision_importance: float, context: Dict[str, Any]) -> int:
        """
        AGI Ultra: Dynamically adjusts reasoning depth based on decision importance.
        
        Args:
            decision_importance: 0.0 to 1.0 score of decision significance
            context: Additional context for depth determination
            
        Returns:
            Optimal depth for this decision
        """
        if not self.adaptive_depth:
            return self.max_depth
        
        # Base depth from importance
        if decision_importance >= self.importance_thresholds['critical']:
            base_depth = 32
        elif decision_importance >= self.importance_thresholds['high']:
            base_depth = 24
        elif decision_importance >= self.importance_thresholds['medium']:
            base_depth = 16
        elif decision_importance >= self.importance_thresholds['low']:
            base_depth = 8
        else:
            base_depth = self.min_adaptive_depth
        
        # Modifiers from context
        volatility_mod = min(8, int(context.get('volatility_percentile', 50) / 12.5))
        uncertainty_mod = min(4, int(context.get('uncertainty', 0) * 4))
        
        final_depth = min(self.max_adaptive_depth, base_depth + volatility_mod + uncertainty_mod)
        
        logger.debug(f"Adaptive depth: importance={decision_importance:.2f}, "
                     f"base={base_depth}, final={final_depth}")
        
        return final_depth
    
    # -------------------------------------------------------------------------
    # AGI ULTRA: META-REASONING LAYER
    # -------------------------------------------------------------------------
    def meta_reasoning_layer(
        self,
        module_name: str,
        original_reasoning: Dict[str, Any],
        decision: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """
        AGI Ultra: Implements reasoning about the reasoning process itself.
        
        "Why am I thinking this way?" - Critical for self-improvement.
        
        Args:
            module_name: Which module is reasoning
            original_reasoning: The reasoning that was produced
            decision: The decision made
            confidence: Confidence in the decision
            
        Returns:
            Meta-reasoning analysis with insights
        """
        if not self.enable_meta_reasoning:
            return {"enabled": False}
        
        meta_analysis = {
            "meta_depth": 0,
            "reasoning_quality": 0.0,
            "cognitive_biases_detected": [],
            "alternative_framings": [],
            "confidence_calibration": 0.0,
            "improvement_suggestions": [],
        }
        
        tree = self.thought_orchestrator.get_or_create_tree(f"{module_name}_meta")
        
        # Level 1: Why did I reach this conclusion?
        meta_q1 = f"Why did {module_name} conclude {decision} with confidence {confidence:.1%}?"
        meta_node1 = tree.create_node(question=meta_q1, context=original_reasoning)
        
        # Analyze reasoning patterns
        patterns = self.global_decision_memory.get_or_create_memory(module_name).analyze_decision_patterns(30)
        consistency = patterns.get('decision_consistency', 0.5)
        
        tree.answer_node(
            meta_node1,
            answer=f"Based on {len(original_reasoning)} reasoning factors with {consistency:.1%} historical consistency",
            confidence=consistency
        )
        meta_analysis["meta_depth"] = 1
        meta_analysis["reasoning_quality"] = consistency
        
        # Level 2: Am I biased?
        if self.meta_reasoning_depth >= 2:
            meta_q2 = "What cognitive biases might affect this reasoning?"
            meta_node2 = tree.create_node(question=meta_q2, parent_id=meta_node1, context={"level": 2})
            
            # Check for common biases
            biases = self._detect_cognitive_biases(original_reasoning, decision, patterns)
            meta_analysis["cognitive_biases_detected"] = biases
            
            tree.answer_node(
                meta_node2,
                answer=f"Detected {len(biases)} potential biases: {', '.join(b['name'] for b in biases[:3])}",
                confidence=0.6
            )
            meta_analysis["meta_depth"] = 2
        
        # Level 3: How could I think differently?
        if self.meta_reasoning_depth >= 3:
            meta_q3 = "What alternative framings of this situation exist?"
            meta_node3 = tree.create_node(question=meta_q3, parent_id=meta_node1, context={"level": 3})
            
            alternatives = self._generate_alternative_framings(original_reasoning, decision)
            meta_analysis["alternative_framings"] = alternatives
            
            tree.answer_node(
                meta_node3,
                answer=f"Found {len(alternatives)} alternative framings to consider",
                confidence=0.5
            )
            meta_analysis["meta_depth"] = 3
        
        # Confidence calibration: How accurate have my confidence estimates been?
        meta_analysis["confidence_calibration"] = self._calibrate_confidence(
            module_name, confidence, patterns
        )
        
        # Improvement suggestions based on meta-analysis
        meta_analysis["improvement_suggestions"] = self._generate_improvement_suggestions(
            meta_analysis, patterns
        )
        
        return meta_analysis
    
    def _detect_cognitive_biases(
        self,
        reasoning: Dict[str, Any],
        decision: str,
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect potential cognitive biases in reasoning."""
        biases = []
        
        # Recency bias: Over-weighting recent events
        if patterns.get('recency_weight', 0) > 0.7:
            biases.append({
                'name': 'recency_bias',
                'severity': patterns.get('recency_weight', 0),
                'description': 'Over-weighting recent events in decision making'
            })
        
        # Confirmation bias: Only seeing supporting evidence
        if reasoning.get('contradicting_signals', 0) == 0 and reasoning.get('supporting_signals', 0) > 3:
            biases.append({
                'name': 'confirmation_bias',
                'severity': 0.6,
                'description': 'No contradicting signals considered'
            })
        
        # Overconfidence bias
        win_rate = patterns.get('win_rate', 0.5)
        avg_confidence = patterns.get('avg_confidence', 0.5)
        if avg_confidence > win_rate + 0.2:
            biases.append({
                'name': 'overconfidence_bias',
                'severity': avg_confidence - win_rate,
                'description': f'Historical confidence ({avg_confidence:.1%}) exceeds win rate ({win_rate:.1%})'
            })
        
        # Anchoring bias: Stuck on initial analysis
        if reasoning.get('revision_count', 0) == 0:
            biases.append({
                'name': 'anchoring_bias',
                'severity': 0.4,
                'description': 'No revision of initial analysis'
            })
        
        return biases
    
    def _generate_alternative_framings(
        self,
        reasoning: Dict[str, Any],
        decision: str
    ) -> List[Dict[str, Any]]:
        """Generate alternative ways to frame the situation."""
        alternatives = []
        
        # Opposite decision framing
        opposite = {"BUY": "SELL", "SELL": "BUY", "WAIT": "TRADE", "CLOSE": "HOLD"}.get(decision, "WAIT")
        alternatives.append({
            'framing': 'contrarian',
            'decision': opposite,
            'rationale': f"What if the market is pricing in the opposite of {decision}?"
        })
        
        # Time horizon framing
        alternatives.append({
            'framing': 'longer_horizon',
            'decision': decision,
            'rationale': "How would this look on a longer timeframe?"
        })
        
        # Risk-first framing
        alternatives.append({
            'framing': 'risk_minimization',
            'decision': 'WAIT' if decision in ['BUY', 'SELL'] else decision,
            'rationale': "What if minimizing risk is more important than capturing opportunity?"
        })
        
        return alternatives
    
    def _calibrate_confidence(
        self,
        module_name: str,
        current_confidence: float,
        patterns: Dict[str, Any]
    ) -> float:
        """Calculate how well-calibrated historical confidence estimates have been."""
        calibration = patterns.get('confidence_calibration', 0.5)
        return calibration
    
    def _generate_improvement_suggestions(
        self,
        meta_analysis: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable improvement suggestions from meta-analysis."""
        suggestions = []
        
        biases = meta_analysis.get('cognitive_biases_detected', [])
        for bias in biases:
            if bias['name'] == 'recency_bias':
                suggestions.append("Consider longer historical lookback periods")
            elif bias['name'] == 'confirmation_bias':
                suggestions.append("Actively seek contradicting evidence")
            elif bias['name'] == 'overconfidence_bias':
                suggestions.append("Reduce position sizes until confidence calibrates")
            elif bias['name'] == 'anchoring_bias':
                suggestions.append("Implement mandatory re-analysis checkpoints")
        
        if meta_analysis.get('reasoning_quality', 0) < 0.5:
            suggestions.append("Increase reasoning depth for low-quality assessments")
        
        return suggestions
    
    # -------------------------------------------------------------------------
    # AGI ULTRA: MULTI-SCALE CAUSAL CHAINS
    # -------------------------------------------------------------------------
    def multi_scale_causal_chains(
        self,
        event: MemoryEvent,
        module_name: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        AGI Ultra: Track causes at multiple temporal scales simultaneously.
        
        Scales:
        - immediate: What caused this tick/candle?
        - short_term: What trend is forming (5-15 minutes)?
        - medium_term: What session/hour pattern exists?
        - fundamental: What macro force is driving the market?
        """
        causal_chains = {}
        
        for scale in self.causal_scales:
            chain = self._extract_causal_chain(event, scale, module_name)
            causal_chains[scale] = chain
            
            # Store for future reference
            self.causal_chains[scale].append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'chain': chain
            })
            
            # Limit storage
            if len(self.causal_chains[scale]) > 10000:
                self.causal_chains[scale] = self.causal_chains[scale][-5000:]
        
        return causal_chains
    
    def _extract_causal_chain(
        self,
        event: MemoryEvent,
        scale: str,
        module_name: str
    ) -> List[Dict[str, Any]]:
        """Extract causal chain at a specific temporal scale."""
        chain = []
        
        if scale == 'immediate':
            # Immediate causes from market state
            if 'last_candle' in event.market_state:
                chain.append({
                    'cause': 'price_movement',
                    'details': event.market_state.get('last_candle', {}),
                    'confidence': 0.9
                })
            if 'spread' in event.market_state:
                chain.append({
                    'cause': 'spread_condition',
                    'details': {'spread': event.market_state.get('spread')},
                    'confidence': 0.8
                })
                
        elif scale == 'short_term':
            # Short-term trend causes
            if 'trend_direction' in event.analysis_state:
                chain.append({
                    'cause': 'trend_momentum',
                    'details': {'direction': event.analysis_state.get('trend_direction')},
                    'confidence': 0.7
                })
            if 'rsi' in event.analysis_state:
                chain.append({
                    'cause': 'oscillator_signal',
                    'details': {'rsi': event.analysis_state.get('rsi')},
                    'confidence': 0.6
                })
                
        elif scale == 'medium_term':
            # Session/hour patterns
            if 'session' in event.market_state:
                chain.append({
                    'cause': 'session_behavior',
                    'details': {'session': event.market_state.get('session')},
                    'confidence': 0.6
                })
            if 'volatility_regime' in event.analysis_state:
                chain.append({
                    'cause': 'volatility_regime',
                    'details': {'regime': event.analysis_state.get('volatility_regime')},
                    'confidence': 0.7
                })
                
        elif scale == 'fundamental':
            # Macro forces
            if 'macro_bias' in event.analysis_state:
                chain.append({
                    'cause': 'macro_sentiment',
                    'details': {'bias': event.analysis_state.get('macro_bias')},
                    'confidence': 0.5
                })
            if 'news_score' in event.market_state:
                chain.append({
                    'cause': 'news_impact',
                    'details': {'score': event.market_state.get('news_score')},
                    'confidence': 0.4
                })
        
        return chain
    
    # -------------------------------------------------------------------------
    # AGI ULTRA: PARALLEL COUNTERFACTUAL SIMULATOR
    # -------------------------------------------------------------------------
    def parallel_counterfactual_simulator(
        self,
        query_event: MemoryEvent,
        similar_events: List[Tuple[MemoryEvent, float]],
        num_scenarios: int = 5, # Phase 3 Optimization: Reduced from 100 to 5
    ) -> List[ScenarioBranch]:
        """
        AGI Ultra: Simulate 100+ counterfactual scenarios in parallel.
        
        Uses ThreadPoolExecutor for parallel simulation of alternative decisions.
        """
        # Generate scenario parameter combinations
        scenarios_to_simulate = []
        
        possible_decisions = ["BUY", "SELL", "WAIT", "CLOSE"]
        slot_choices = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        vsl_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        vtp_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        # Generate diverse scenario combinations
        for decision in possible_decisions:
            for slot in slot_choices:
                for vsl in vsl_multipliers:
                    for vtp in vtp_multipliers:
                        if len(scenarios_to_simulate) >= num_scenarios:
                            break
                        scenarios_to_simulate.append({
                            'decision': decision,
                            'slot_multiplier': slot,
                            'vsl_multiplier': vsl,
                            'vtp_multiplier': vtp,
                        })
        
        # Parallel simulation
        branches: List[ScenarioBranch] = []
        futures = []
        
        for params in scenarios_to_simulate[:num_scenarios]:
            future = self._executor.submit(
                self._simulate_single_scenario,
                query_event,
                similar_events,
                params
            )
            futures.append((future, params))
        
        # Collect results
        for future, params in futures:
            try:
                est_outcome, support = future.result(timeout=1.0)
                branches.append(ScenarioBranch(
                    branch_id=str(uuid.uuid4()),
                    counterfactual_decision=params['decision'],
                    parameters=params,
                    estimated_outcome=est_outcome,
                    supporting_events=support,
                ))
            except Exception as e:
                logger.warning(f"Scenario simulation failed: {e}")
        
        # Sort by expected PnL
        branches.sort(key=lambda b: b.estimated_outcome.get('expected_pnl', 0), reverse=True)
        
        logger.debug(f"Parallel simulation completed: {len(branches)} scenarios analyzed")
        
        return branches
    
    def _simulate_single_scenario(
        self,
        query_event: MemoryEvent,
        similar_events: List[Tuple[MemoryEvent, float]],
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        """Simulate a single counterfactual scenario (thread-safe)."""
        return self._estimate_counterfactual_outcome(query_event, similar_events, params)
    
    # -------------------------------------------------------------------------
    # AGI ULTRA: CONTEXTUAL QUESTION GENERATOR
    # -------------------------------------------------------------------------
    def contextual_question_generator(
        self,
        context: Dict[str, Any],
        question_type: str = 'all'
    ) -> List[str]:
        """
        AGI Ultra: Generate contextually relevant questions dynamically.
        
        Instead of fixed templates, generates questions based on actual context.
        """
        questions = []
        
        # Use registered question generators
        for generator in self.question_generators:
            try:
                question = generator(context)
                questions.append(question)
            except Exception:
                pass
        
        # Dynamic questions based on context keys
        if 'volatility' in context:
            vol = context['volatility']
            if vol > 0.8:
                questions.append(f"Why is volatility extremely high ({vol:.1%})? Is this sustainable?")
            elif vol < 0.2:
                questions.append(f"Why is volatility so low ({vol:.1%})? What could trigger expansion?")
        
        if 'consensus_score' in context:
            score = context['consensus_score']
            if abs(score) > 0.8:
                questions.append(f"What could invalidate this strong consensus ({score:.1%})?")
            elif abs(score) < 0.2:
                questions.append(f"Why is there no clear consensus? What information is missing?")
        
        if 'similar_events_count' in context:
            count = context['similar_events_count']
            if count < 5:
                questions.append("Why are there few similar historical events? Is this a novel situation?")
            elif count > 100:
                questions.append("With many similar events, what makes this one unique?")
        
        # Filter by type if specified
        if question_type != 'all':
            type_keywords = {
                'causal': ['why', 'cause', 'force'],
                'counterfactual': ['what if', 'would', 'alternative'],
                'meta': ['am I', 'thinking', 'consider'],
                'risk': ['risk', 'invalidate', 'danger'],
            }
            keywords = type_keywords.get(question_type, [])
            questions = [q for q in questions if any(kw in q.lower() for kw in keywords)]
        
        return questions

    # -------------------------------------------------------------------------
    # CAPTURA DO EVENTO ATUAL
    # -------------------------------------------------------------------------
    def capture_event(
        self,
        symbol: str,
        timeframe: str,
        market_state: Dict[str, Any],
        analysis_state: Dict[str, Any],
        decision: Optional[str],
        decision_score: float,
        decision_meta: Dict[str, Any],
        module_name: str,
    ) -> MemoryEvent:
        """
        Captura um snapshot completo do instante de decisão/análise.

        - Constrói o objeto MemoryEvent.
        - Codifica estado em vetor holográfico.
        - Alimenta HolographicMemory (para "intuição" futura).
        - Registra decisão na GlobalDecisionMemory do módulo.
        """
        event_id = str(uuid.uuid4())
        now_ts = time.time()

        event = MemoryEvent(
            event_id=event_id,
            timestamp=now_ts,
            symbol=symbol,
            timeframe=timeframe,
            market_state=market_state,
            analysis_state=analysis_state,
            decision=decision,
            decision_score=decision_score,
            decision_meta=decision_meta or {},
        )

        # 1) Persistência vetorial na memória holográfica
        encoded_context = self._encode_event_to_vector(event)
        self.events.append(event)
        self.event_vectors.append(encoded_context)

        # Outcome é desconhecido neste momento, treated como 0.0 (neutro) na placa
        self.holographic_memory.plate.learn(encoded_context, outcome_score=0.0)

        # 2) Integra com memória de decisões por módulo, se houver decisão efetiva
        if decision is not None:
            module_memory: ModuleDecisionMemory = self.global_decision_memory.get_or_create_memory(module_name)
            module_memory.record_decision(
                decision=decision,
                score=decision_score,
                context={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "market_state": market_state,
                    "analysis_state": analysis_state,
                    "decision_meta": decision_meta,
                    "event_id": event_id,
                },
                reasoning=decision_meta.get("reason", ""),
                confidence=decision_meta.get("confidence", abs(decision_score) / 100.0),
            )

        return event

    def update_event_outcome(
        self,
        event_id: str,
        outcome: Dict[str, Any],
        outcome_score: float,
        module_name: str,
    ) -> None:
        """
        Posteriormente, após o fechamento da ordem / sequência,
        atualiza o resultado associado ao evento e alimenta a placa holográfica
        com o "sinal" de sucesso/fracasso.
        """
        for idx, ev in enumerate(self.events):
            if ev.event_id == event_id:
                ev.outcome = outcome

                # Atualiza a placa com outcome real (reforço)
                if idx < len(self.event_vectors):
                    vec = self.event_vectors[idx]
                    self.holographic_memory.plate.learn(vec, outcome_score=outcome_score)

                # Espelha o resultado na DecisionMemory do módulo
                module_memory: ModuleDecisionMemory = self.global_decision_memory.get_or_create_memory(module_name)
                module_memory.record_result(
                    decision_id=self._infer_decision_id_for_event(ev, module_name),
                    result=outcome.get("result_label", "PENDING"),
                    pnl=outcome.get("pnl"),
                )
                break

    # -------------------------------------------------------------------------
    # PATTERN MATCHING VETORIAL
    # -------------------------------------------------------------------------
    def pattern_match_recursive(
        self,
        query_event: MemoryEvent,
        top_k: Optional[int] = None,
    ) -> List[Tuple[MemoryEvent, float]]:
        """
        Busca no banco de memória eventos passados similares via similaridade
        vetorial (cosine).
        """
        if not self.events:
            return []

        if top_k is None:
            top_k = self.vector_top_k

        q_vec = self._encode_event_to_vector(query_event)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        sims: List[Tuple[int, float]] = []
        for idx, v in enumerate(self.event_vectors):
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
            sims.append((idx, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[: min(top_k, len(sims))]

        return [(self.events[idx], score) for idx, score in sims]

    # -------------------------------------------------------------------------
    # DEEP SCAN RECURSIVO (INFINITE WHY LOOP)
    # -------------------------------------------------------------------------
    def deep_scan_recursive(
        self,
        module_name: str,
        query_event: MemoryEvent,
        max_depth: Optional[int] = None,
        max_branches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Núcleo do "The Infinite Why Engine" para um evento.

        - Abre um nó raiz na ThoughtTree do módulo.
        - Gera recursivamente camadas de "por que", "o que causou",
          e "o que teria acontecido se".
        - Usa DecisionMemory + HolographicMemory para estimar respostas.
        """
        if max_depth is None:
            max_depth = self.max_depth
        if max_branches is None:
            max_branches = self.max_branches

        tree: ThoughtTree = self.thought_orchestrator.get_or_create_tree(module_name)
        module_memory: ModuleDecisionMemory = self.global_decision_memory.get_or_create_memory(module_name)

        # 1) EVENTO ATUAL COMO NÓ RAIZ
        root_question = f"Why did I {query_event.decision or 'ANALYZE'} at {query_event.symbol} {query_event.timeframe}?"
        root_node_id = tree.create_node(
            question=root_question,
            context={
                "event_id": query_event.event_id,
                "market_state": query_event.market_state,
                "analysis_state": query_event.analysis_state,
                "decision": query_event.decision,
                "decision_score": query_event.decision_score,
            },
            confidence=abs(query_event.decision_score) / 100.0 if query_event.decision_score != 0 else 0.0,
        )

        # Resposta imediata baseada no meta da decisão + intuição holográfica
        intuition = self.holographic_memory.retrieve_intuition(
            {
                "symbol": query_event.symbol,
                "timeframe": query_event.timeframe,
                **query_event.market_state,
                **query_event.analysis_state,
            }
        )
        immediate_reason = query_event.decision_meta.get("reason", "")
        tree.answer_node(
            root_node_id,
            answer=immediate_reason or f"Based on multi-module analysis and holographic intuition={intuition:.3f}",
            confidence=min(1.0, abs(intuition)),
        )

        # 2) BUSCA DE EVENTOS SIMILARES
        similar_events = self.pattern_match_recursive(query_event, top_k=max_branches)

        # 3) EXPANSÃO RECURSIVA (WHY-CHAIN)
        why_root = WhyNode(
            node_id=root_node_id,
            depth=0,
            question=root_question,
            context={"intuition": intuition, "similar_events_count": len(similar_events)},
        )

        self._expand_why_chain(
            module_name=module_name,
            tree=tree,
            module_memory=module_memory,
            parent_node=why_root,
            parent_tree_node_id=root_node_id,
            similar_events=similar_events,
            depth=1,
            max_depth=max_depth,
        )

        # 4) RAMIFICAÇÃO DE CENÁRIOS CONTRAFACTUAIS
        scenario_branches = self._scenario_branching(
            module_name=module_name,
            query_event=query_event,
            similar_events=similar_events,
            max_branches=max_branches,
        )

        return {
            "root_node_id": root_node_id,
            "why_tree": why_root,
            "scenario_branches": scenario_branches,
        }

    def _expand_why_chain(
        self,
        module_name: str,
        tree: ThoughtTree,
        module_memory: ModuleDecisionMemory,
        parent_node: WhyNode,
        parent_tree_node_id: str,
        similar_events: List[Tuple[MemoryEvent, float]],
        depth: int,
        max_depth: int,
    ) -> None:
        """
        Expansão recursiva da cadeia de "porquês".
        Estrutura genérica que pode ser estendida para milhões de combinações
        de parâmetros de pensamento sem precisar codificar manualmente cada pergunta.
        """
        if depth >= max_depth:
            return

        # Constrói um conjunto de "dimensões de porquê" genéricas que
        # podem ser combinadas dinamicamente gerando trilhões de perguntas:
        why_dimensions = [
            "market_structure",      # por que a estrutura do mercado sugeria isso?
            "microstructure",        # por que o fluxo de ordens empurrou nessa direção?
            "regime_context",        # por que o regime (tendência/range) justificava a decisão?
            "risk_profile",          # por que o perfil de risco/slots estava adequado?
            "temporal_context",      # por que este horário/sessão era relevante?
            "memory_alignment",      # por que a memória histórica concordava?
            "counterparty_behavior", # por que o comportamento do contraparte parecia X?
        ]

        # PARA CADA DIMENSÃO, criamos nós-filho + respondemos usando DecisionMemory + estatísticas
        for dim in why_dimensions:
            child_question = f"Why was the {dim} configuration supporting this decision?"
            child_tree_id = tree.create_node(
                question=child_question,
                parent_id=parent_tree_node_id,
                context={"dimension": dim},
            )
            
            # Check if node was actually created (create_node returns parent_id if max depth reached)
            if child_tree_id == parent_tree_node_id:
                # Max depth reached in ThoughtTree, stop recursion
                continue

            # Exemplo: consulta padrões de sucesso/fracasso recentes do módulo
            patterns = module_memory.analyze_decision_patterns(lookback_days=30)
            win_rate = patterns.get("win_rate", 0.0)

            answer = f"Historical pattern for {dim} shows win_rate={win_rate:.1%} over last 30 days"
            tree.answer_node(child_tree_id, answer=answer, confidence=win_rate)

            child_node = WhyNode(
                node_id=child_tree_id,
                depth=depth,
                question=child_question,
                context={"dimension": dim, "win_rate": win_rate},
                answer=answer,
                confidence=win_rate,
            )
            parent_node.children.append(child_node)

            # RECURSÃO: meta-porquê ("por que esse win_rate existe?") em nível mais profundo
            self._expand_why_chain(
                module_name=module_name,
                tree=tree,
                module_memory=module_memory,
                parent_node=child_node,
                parent_tree_node_id=child_tree_id,
                similar_events=similar_events,
                depth=depth + 1,
                max_depth=max_depth,
            )

    # -------------------------------------------------------------------------
    # RAMIFICAÇÃO DE CENÁRIOS "O QUE TERIA ACONTECIDO SE..."
    # -------------------------------------------------------------------------
    def _scenario_branching(
        self,
        module_name: str,
        query_event: MemoryEvent,
        similar_events: List[Tuple[MemoryEvent, float]],
        max_branches: int,
    ) -> List[ScenarioBranch]:
        """
        Constrói uma árvore de cenários contrafactuais baseada em:
        - decisões alternativas (BUY/SELL/WAIT/CLOSE)
        - parâmetros alternativos (slots, VSL, VTP, modo)
        - evidência histórica (eventos similares bem-sucedidos/fracassados)
        """
        branches: List[ScenarioBranch] = []

        # Espaço conceitual de decisões e parâmetros (pode ser expandido
        # dinamicamente por camada AGI superior / QuestionGenerator).
        possible_decisions = ["BUY", "SELL", "WAIT", "CLOSE"]
        slot_choices = [0.25, 0.5, 1.0, 2.0]  # fração dos slots padrão
        vsl_multipliers = [0.5, 1.0, 1.5, 2.0]
        vtp_multipliers = [0.5, 1.0, 1.5, 2.0]

        # Limita combinatória inicial; em produção, isto seria controlado
        # por heurísticas de importância / custo.
        for decision_alt in possible_decisions:
            for slot_mul in slot_choices:
                if len(branches) >= max_branches:
                    return branches

                params = {
                    "decision": decision_alt,
                    "slot_multiplier": slot_mul,
                    "vsl_multiplier": 1.0,
                    "vtp_multiplier": 1.0,
                }
                est_outcome, support = self._estimate_counterfactual_outcome(
                    query_event=query_event,
                    similar_events=similar_events,
                    params=params,
                )
                branches.append(
                    ScenarioBranch(
                        branch_id=str(uuid.uuid4()),
                        counterfactual_decision=decision_alt,
                        parameters=params,
                        estimated_outcome=est_outcome,
                        supporting_events=support,
                    )
                )

        return branches

    def _estimate_counterfactual_outcome(
        self,
        query_event: MemoryEvent,
        similar_events: List[Tuple[MemoryEvent, float]],
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        """
        Núcleo de estimativa contrafactual simplificado:
        - filtra eventos similares que tomaram a decisão hipotética
        - combina outcomes ponderados pela similaridade
        """
        decision_alt = params["decision"]
        weighted_pnl = 0.0
        total_weight = 0.0
        max_favor = 0.0
        max_adverse = 0.0
        support_ids: List[Tuple[str, float]] = []

        for ev, sim in similar_events:
            if ev.decision != decision_alt or not ev.outcome:
                continue
            weight = max(sim, 0.0)
            if weight <= 0:
                continue

            pnl = float(ev.outcome.get("pnl", 0.0))
            fav = float(ev.outcome.get("max_favorable_excursion", 0.0))
            adv = float(ev.outcome.get("max_adverse_excursion", 0.0))

            weighted_pnl += pnl * weight
            total_weight += weight
            max_favor = max(max_favor, fav)
            max_adverse = max(max_adverse, adv)
            support_ids.append((ev.event_id, sim))

        if total_weight == 0:
            # HEURISTIC FALLBACK (For cold start)
            # If we have no history, we infer outcome based on simple logic relative to the current decision score.
            # E.g. If current score is +0.8 (Strong Buy), then decision="BUY" should have positive expected PnL.
            base_score = query_event.decision_score
            
            # Simple heuristic mapping
            heuristic_pnl = 0.0
            if decision_alt == "BUY":
                heuristic_pnl = base_score * 10.0 # Arbitrary scaling
            elif decision_alt == "SELL":
                heuristic_pnl = -base_score * 10.0
                
            # Apply multipliers
            heuristic_pnl *= params.get('slot_multiplier', 1.0)
            
            return (
                {
                    "expected_pnl": heuristic_pnl,
                    "expected_max_favor": abs(heuristic_pnl) * 1.5,
                    "expected_max_adverse": abs(heuristic_pnl) * 0.5,
                    "confidence": 0.1, # Low confidence purely heuristic
                    "is_heuristic": True
                },
                [],
            )


        expected_pnl = weighted_pnl / total_weight

        return (
            {
                "expected_pnl": expected_pnl,
                "expected_max_favor": max_favor,
                "expected_max_adverse": max_adverse,
                "confidence": min(1.0, total_weight),
            },
            support_ids,
        )

    # -------------------------------------------------------------------------
    # ENCODER UTILITÁRIO
    # -------------------------------------------------------------------------
    def _encode_event_to_vector(self, ev: MemoryEvent) -> np.ndarray:
        """
        Converte o evento inteiro em um único vetor de alto nível
        (tick + análises + decisão).
        """
        context = {
            "symbol": ev.symbol,
            "timeframe": ev.timeframe,
            "decision": ev.decision or "NONE",
            "decision_score": ev.decision_score,
            **ev.market_state,
            **ev.analysis_state,
        }
        return self.holographic_memory.plate.encode_state(context)

    def _infer_decision_id_for_event(self, ev: MemoryEvent, module_name: str) -> str:
        """
        Em uma implementação completa, o event_id seria mapeado diretamente
        ao decision_id ao registrar a decisão. Aqui mantemos um placeholder
        para integração futura explícita.
        """
        # Pseudocódigo / placeholder: em produção, manter um índice explícito:
        # self.event_to_decision_id[(module_name, ev.event_id)]
        return f"{module_name}_UNKNOWN_DECISION_ID_FOR_{ev.event_id}"

