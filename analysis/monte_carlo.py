import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from core.agi.infinite_why_engine import InfiniteWhyEngine, MemoryEvent

logger = logging.getLogger("Atl4s-MonteCarlo")


class MonteCarloSimulator:
    """
    Fase 7.0 (legado): Simulador básico de Monte Carlo para curva de capital.
    Mantido para compatibilidade e métricas puras de risco.
    """

    def __init__(self, num_simulations: int = 1000, initial_capital: float = 100.0):
        self.num_simulations = num_simulations
        self.initial_capital = initial_capital

    def run(self, trades_pnl_percent: List[float]) -> Dict[str, float]:
        """
        Runs Monte Carlo simulation on a list of trade PnL percentages.

        trades_pnl_percent: List of floats (e.g., [0.01, -0.005, 0.02])
                            representing 1% gain, 0.5% loss, etc.

        Returns:
            metrics (dict): Risk of Ruin, Median Drawdown, Worst Drawdown, Safety Score.
        """
        if not trades_pnl_percent or len(trades_pnl_percent) < 10:
            logger.warning("Insufficient trades for Monte Carlo simulation.")
            return {
                "risk_of_ruin": 0.0,
                "median_drawdown": 0.0,
                "worst_case_drawdown": 0.0,
                "safety_score": 50.0,  # Neutral
            }

        ruin_count = 0
        max_drawdowns: List[float] = []
        final_balances: List[float] = []

        # Convert to numpy array for speed
        trades = np.array(trades_pnl_percent, dtype=float)

        for _ in range(self.num_simulations):
            # Shuffle trades to create one hypothetical equity path
            np.random.shuffle(trades)

            equity = [self.initial_capital]
            peak = self.initial_capital
            max_dd = 0.0
            ruined = False

            for pnl in trades:
                # Simple compounding: New Balance = Old Balance * (1 + PnL)
                current_balance = equity[-1] * (1.0 + pnl)

                if current_balance < (self.initial_capital * 0.5):  # 50% drawdown threshold
                    ruined = True

                equity.append(current_balance)

                # Drawdown calc
                if current_balance > peak:
                    peak = current_balance
                dd = (peak - current_balance) / peak
                if dd > max_dd:
                    max_dd = dd

            if ruined:
                ruin_count += 1

            max_drawdowns.append(max_dd)
            final_balances.append(equity[-1])

        # Metrics
        risk_of_ruin = (ruin_count / float(self.num_simulations)) * 100.0
        median_dd = float(np.median(max_drawdowns) * 100.0)
        worst_case_dd = float(np.percentile(max_drawdowns, 95) * 100.0)  # 95th percentile

        logger.info(
            "Monte Carlo Results: RoR=%.1f%%, Median DD=%.1f%%, Worst DD=%.1f%%",
            risk_of_ruin,
            median_dd,
            worst_case_dd,
        )

        # Safety Score (0-100)
        # 100 = RoR 0% and Low DD
        score = 100.0 - (risk_of_ruin * 10.0) - (median_dd * 2.0)
        score = max(0.0, min(100.0, score))

        return {
            "risk_of_ruin": risk_of_ruin,
            "median_drawdown": median_dd,
            "worst_case_drawdown": worst_case_dd,
            "safety_score": score,
        }


@dataclass
class ThoughtScenarioSummary:
    """
    Resumo estatístico de todos os ramos de pensamento simulados
    para um único evento (Memória Holográfica + contrafactuais).
    """

    total_branches: int
    success_branches: int
    failure_branches: int
    success_probability: float
    avg_expected_pnl: float
    best_expected_pnl: float
    worst_expected_pnl: float


class AGIMonteCarloSimulator(MonteCarloSimulator):
    """
    Fase 7.1: Enhanced Monte Carlo

    - Simula milhões de cenários conceituais através dos ramos contrafactuais
      do InfiniteWhyEngine (cenários "e se eu tivesse feito X?").
    - Conecta curva de capital (Monte Carlo clássico) com árvore de pensamento.
    - Mantém memória de simulações passadas para acelerar futuros cálculos similares.
    """

    def __init__(
        self,
        why_engine: InfiniteWhyEngine,
        module_name: str,
        num_simulations: int = 1000,
        initial_capital: float = 100.0,
        cache_enabled: bool = True,
    ):
        super().__init__(num_simulations=num_simulations, initial_capital=initial_capital)
        self.why_engine = why_engine
        self.module_name = module_name
        self.cache_enabled = cache_enabled

        # Memória de simulações passadas (hash_de_contexto -> resultado)
        self._simulation_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # CHAVE DE CACHE (MEMÓRIA DE SIMULAÇÕES)
    # ------------------------------------------------------------------
    def _make_cache_key(
        self,
        query_event: MemoryEvent,
        trades_pnl_percent: List[float],
    ) -> str:
        """
        Cria uma chave determinística para identificar simulações equivalentes.
        Usa apenas estatísticas agregadas para não vazar dados crus.
        """
        if not trades_pnl_percent:
            trades_pnl_percent = [0.0]

        arr = np.array(trades_pnl_percent, dtype=float)
        stats = (
            round(float(arr.mean()), 8),
            round(float(arr.std()), 8),
            len(arr),
        )

        # Contexto mínimo do evento que afeta o espaço de cenários
        ctx = (
            query_event.symbol,
            query_event.timeframe,
            query_event.decision or "NONE",
        )

        return f"{ctx[0]}|{ctx[1]}|{ctx[2]}|{stats[0]}|{stats[1]}|{stats[2]}"

    # ------------------------------------------------------------------
    # INTERFACE PRINCIPAL AGI-AWARE
    # ------------------------------------------------------------------
    def run_with_thought(
        self,
        query_event: MemoryEvent,
        trades_pnl_percent: List[float],
        max_depth: Optional[int] = None,
        max_branches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Núcleo da Fase 7:
        - Executa Monte Carlo clássico sobre PnLs históricos.
        - Abre Deep_Scan_Recursive no InfiniteWhyEngine para o mesmo evento.
        - Extraí estatísticas de todos os cenários contrafactuais.
        - Retorna um bloco unificado de risco + árvore de pensamento.
        """
        if self.cache_enabled:
            cache_key = self._make_cache_key(query_event, trades_pnl_percent)
            if cache_key in self._simulation_cache:
                logger.debug("AGIMonteCarloSimulator: using cached result for key=%s", cache_key)
                return self._simulation_cache[cache_key]
        else:
            cache_key = ""

        # 1) Monte Carlo clássico para curva de capital
        base_metrics = self.run(trades_pnl_percent)

        # 2) Deep Scan Recursivo no motor AGI para este evento de memória
        deep_scan = self.why_engine.deep_scan_recursive(
            module_name=self.module_name,
            query_event=query_event,
            max_depth=max_depth,
            max_branches=max_branches,
        )
        scenario_branches = deep_scan.get("scenario_branches", [])

        # 3) Estatísticas de todos os cenários contrafactuais (bilhões virtuais)
        scenario_summary = self._summarize_scenarios(scenario_branches)

        result: Dict[str, Any] = {
            "base_risk_metrics": base_metrics,
            "root_thought_node_id": deep_scan.get("root_node_id"),
            "scenario_summary": {
                "total_branches": scenario_summary.total_branches,
                "success_branches": scenario_summary.success_branches,
                "failure_branches": scenario_summary.failure_branches,
                "success_probability": scenario_summary.success_probability,
                "avg_expected_pnl": scenario_summary.avg_expected_pnl,
                "best_expected_pnl": scenario_summary.best_expected_pnl,
                "worst_expected_pnl": scenario_summary.worst_expected_pnl,
            },
            # Para debug profundo / auditoria:
            "raw_scenario_branches": scenario_branches,
        }

        if self.cache_enabled and cache_key:
            self._simulation_cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    # ANÁLISE ESTATÍSTICA DOS CENÁRIOS CONTRAFACTUAIS
    # ------------------------------------------------------------------
    def _summarize_scenarios(self, scenario_branches: List[Any]) -> ThoughtScenarioSummary:
        """
        Recebe a lista de cenários contrafactuais produzidos pelo InfiniteWhyEngine
        e condensa em métricas agregadas.
        """
        if not scenario_branches:
            return ThoughtScenarioSummary(
                total_branches=0,
                success_branches=0,
                failure_branches=0,
                success_probability=0.0,
                avg_expected_pnl=0.0,
                best_expected_pnl=0.0,
                worst_expected_pnl=0.0,
            )

        total = len(scenario_branches)
        success = 0
        failure = 0

        expected_pnls: List[float] = []

        for branch in scenario_branches:
            outcome = getattr(branch, "estimated_outcome", None) or {}
            exp_pnl = float(outcome.get("expected_pnl", 0.0))
            expected_pnls.append(exp_pnl)
            if exp_pnl > 0.0:
                success += 1
            elif exp_pnl < 0.0:
                failure += 1

        if not expected_pnls:
            return ThoughtScenarioSummary(
                total_branches=total,
                success_branches=success,
                failure_branches=failure,
                success_probability=0.0,
                avg_expected_pnl=0.0,
                best_expected_pnl=0.0,
                worst_expected_pnl=0.0,
            )

        arr = np.array(expected_pnls, dtype=float)
        success_prob = success / float(total) if total > 0 else 0.0

        return ThoughtScenarioSummary(
            total_branches=total,
            success_branches=success,
            failure_branches=failure,
            success_probability=success_prob,
            avg_expected_pnl=float(arr.mean()),
            best_expected_pnl=float(arr.max()),
            worst_expected_pnl=float(arr.min()),
        )
