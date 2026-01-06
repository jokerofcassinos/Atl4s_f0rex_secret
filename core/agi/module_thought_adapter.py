import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from core.agi.infinite_why_engine import InfiniteWhyEngine, MemoryEvent

logger = logging.getLogger("AGIModuleAdapter")


@dataclass
class ModuleThoughtResult:
    """
    Resultado padronizado de um módulo de análise com camada AGI:
    - decisão discreta ("BUY"/"SELL"/"WAIT")
    - score numérico (0-100, ou -100..100 conforme módulo)
    - root_node_id na ThoughtTree
    - stats de cenários contrafactuais (se disponíveis)
    """

    decision: str
    score: float
    thought_root_id: Optional[str]
    meta: Dict[str, Any]


class AGIModuleAdapter:
    """
    Adaptador genérico para plugar qualquer módulo de `analysis/`
    no InfiniteWhyEngine, sem reescrever toda a lógica original.

    Fluxo:
      1. Módulo calcula sua análise normal (score, direção, etc.).
      2. Adapter converte isso em MemoryEvent + chama InfiniteWhyEngine.capture_event().
      3. Executa deep_scan_recursive() para gerar árvore de pensamento local.
      4. Opcionalmente retorna IDs / insights para uso em camadas superiores.
    """

    def __init__(self, module_name: str, why_engine: Optional[InfiniteWhyEngine] = None):
        self.module_name = module_name
        self.why_engine = why_engine or InfiniteWhyEngine()

    # ------------------------------------------------------------------
    # Core Entry Point
    # ------------------------------------------------------------------
    def think_on_analysis(
        self,
        symbol: str,
        timeframe: str,
        market_state: Dict[str, Any],
        raw_module_output: Dict[str, Any],
        decision_mapping: Optional[Dict[str, Any]] = None,
    ) -> ModuleThoughtResult:
        """
        Converte a saída bruta do módulo em um evento de memória,
        gera a árvore de pensamento recursivo e devolve o resultado enriquecido.
        """
        # 1) Converter saída em decisão discreta + score
        decision, score = self._infer_decision_and_score(raw_module_output, decision_mapping)

        decision_meta = {
            "source_module": self.module_name,
            "raw_output": raw_module_output,
        }

        # 2) Construir estado de análise (normalizado) que ficará salvo na memória
        analysis_state = {
            "module": self.module_name,
            "raw_score": score,
            "raw_decision": decision,
        }
        analysis_state.update(raw_module_output or {})

        # 3) Capturar evento completo na memória holográfica
        event: MemoryEvent = self.why_engine.capture_event(
            symbol=symbol,
            timeframe=timeframe,
            market_state=market_state,
            analysis_state=analysis_state,
            decision=decision,
            decision_score=float(score),
            decision_meta=decision_meta,
            module_name=self.module_name,
        )

        # 4) Abrir loop de "porquês" para esse módulo/evento
        deep = self.why_engine.deep_scan_recursive(
            module_name=self.module_name,
            query_event=event,
        )

        root_id = deep.get("root_node_id")
        scenario_branches = deep.get("scenario_branches", [])

        # 5) Pequena síntese numérica dos cenários contrafactuais (apenas o essencial)
        scenario_summary = self._summarize_branches(scenario_branches)

        meta = {
            "thought_root_id": root_id,
            "scenario_summary": scenario_summary,
        }

        return ModuleThoughtResult(
            decision=decision,
            score=score,
            thought_root_id=root_id,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_decision_and_score(
        self,
        raw_output: Dict[str, Any],
        decision_mapping: Optional[Dict[str, Any]],
    ) -> Tuple[str, float]:
        """
        Traduz a saída do módulo (score/direction/etc.) para uma decisão padronizada.
        """
        if decision_mapping and "custom" in decision_mapping:
            fn = decision_mapping["custom"]
            return fn(raw_output)

        # Convenção default:
        # - se houver 'direction': 1->BUY, -1->SELL, 0->WAIT
        # - senão, usa apenas sinal de 'score'
        score = float(raw_output.get("score", raw_output.get("raw_score", 0.0)))
        direction = raw_output.get("direction")

        if direction is None:
            if score > 0:
                direction = 1
            elif score < 0:
                direction = -1
            else:
                direction = 0

        if direction > 0:
            decision = "BUY"
        elif direction < 0:
            decision = "SELL"
        else:
            decision = "WAIT"

        return decision, score

    def _summarize_branches(self, scenario_branches):
        """
        Retorna apenas um resumo leve dos cenários:
          - total_branches
          - prob_positive (expected_pnl > 0)
        """
        try:
            if not scenario_branches:
                return {
                    "total_branches": 0,
                    "prob_positive": 0.0,
                }

            total = len(scenario_branches)
            positive = 0
            for b in scenario_branches:
                outcome = getattr(b, "estimated_outcome", None) or {}
                if float(outcome.get("expected_pnl", 0.0)) > 0.0:
                    positive += 1

            prob_positive = positive / float(total) if total > 0 else 0.0

            return {
                "total_branches": total,
                "prob_positive": prob_positive,
            }
        except Exception as e:
            logger.error("AGIModuleAdapter._summarize_branches error: %s", e)
            return {
                "total_branches": 0,
                "prob_positive": 0.0,
            }

