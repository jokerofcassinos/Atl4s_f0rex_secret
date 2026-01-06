import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from core.agi.infinite_why_engine import InfiniteWhyEngine, MemoryEvent

logger = logging.getLogger("AGISwarmAdapter")


@dataclass
class SwarmThoughtResult:
    """
    Resultado padronizado para módulos swarm:
    - aggregated_signal: qualquer métrica de consenso (score, vote, etc.)
    - thought_root_id: nó raiz na árvore de pensamento do swarm
    - meta: resumo de cenários / conexões cross-module
    """

    aggregated_signal: Any
    thought_root_id: Optional[str]
    meta: Dict[str, Any]


class AGISwarmAdapter:
    """
    Adaptador genérico para `analysis/swarm/*`.

    Ideia:
      - Cada swarm já produz algum tipo de score / voto / distribuição.
      - Aqui convertemos esse estado em MemoryEvent específico de swarm e
        abrimos um deep_scan_recursive dedicado a esse "enxame".
    """

    def __init__(self, swarm_name: str, why_engine: Optional[InfiniteWhyEngine] = None):
        self.swarm_name = swarm_name
        self.why_engine = why_engine or InfiniteWhyEngine()

    def think_on_swarm_output(
        self,
        symbol: str,
        timeframe: str,
        market_state: Dict[str, Any],
        swarm_output: Dict[str, Any],
    ) -> SwarmThoughtResult:
        """
        swarm_output pode conter:
          - 'score' ou 'prob_buy'/'prob_sell'
          - 'votes' por direção
          - qualquer contexto interno relevante para o enxame
        """
        decision = swarm_output.get("decision", "WAIT")
        score = float(swarm_output.get("score", 0.0))

        analysis_state = {
            "swarm": self.swarm_name,
            **swarm_output,
        }

        ev: MemoryEvent = self.why_engine.capture_event(
            symbol=symbol,
            timeframe=timeframe,
            market_state=market_state,
            analysis_state=analysis_state,
            decision=decision,
            decision_score=score,
            decision_meta={"source_swarm": self.swarm_name},
            module_name=f"swarm::{self.swarm_name}",
        )

        # OPTIMIZED: Re-enabled with max_depth=3 to prevent exponential explosion
        deep = self.why_engine.deep_scan_recursive(
            module_name=f"swarm::{self.swarm_name}",
            query_event=ev,
            max_depth=3,  # Limit recursion to 3 levels instead of 32
        )
        root_id = deep.get("root_node_id")
        scenario_branches = deep.get("scenario_branches", [])

        meta = {
            "thought_root_id": root_id,
            "scenario_count": len(scenario_branches),
        }

        return SwarmThoughtResult(
            aggregated_signal=swarm_output.get("aggregated_signal", swarm_output),
            thought_root_id=root_id,
            meta=meta,
        )

