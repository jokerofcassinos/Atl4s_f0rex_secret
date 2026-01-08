"""
AGI Ultra-Complete: Simulation System AGI

Sistema de Simulação Avançada:
- ParallelScenarioSimulator: Simulações paralelas
- ScenarioClusteringEngine: Agrupamento de cenários
- SensitivityAnalyzer: Análise de sensibilidade
- RiskScenarioGenerator: Geração de cenários de risco
"""

import logging
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("SimulationSystemAGI")


class ScenarioType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    VOLATILE = "volatile"
    STABLE = "stable"
    CRISIS = "crisis"
    RANDOM = "random"


@dataclass
class Scenario:
    """Simulation scenario."""
    id: str
    type: ScenarioType
    parameters: Dict[str, float]
    returns: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of simulation."""
    scenario_id: str
    final_equity: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    trades: int


class ParallelScenarioSimulator:
    """Runs parallel scenario simulations."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results: List[SimulationResult] = []
        
        logger.info("ParallelScenarioSimulator initialized")
    
    def simulate(self, scenario: Scenario, strategy_func: Callable = None) -> SimulationResult:
        """Simulate a scenario."""
        capital = self.initial_capital
        peak = capital
        trades = 0
        wins = 0
        
        returns = scenario.returns or self._generate_returns(scenario)
        
        equity_curve = [capital]
        for r in returns:
            trade_result = r * (capital * 0.02)
            capital += trade_result
            trades += 1
            if trade_result > 0:
                wins += 1
            
            peak = max(peak, capital)
            equity_curve.append(capital)
        
        if len(equity_curve) > 1:
            daily_returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] 
                            for i in range(1, len(equity_curve))]
            sharpe = (np.mean(daily_returns) / max(0.001, np.std(daily_returns))) * np.sqrt(252) if daily_returns else 0
        else:
            sharpe = 0
        
        max_dd = (peak - min(equity_curve)) / peak if peak > 0 else 0
        
        result = SimulationResult(
            scenario_id=scenario.id,
            final_equity=capital,
            max_drawdown=max_dd,
            sharpe_ratio=float(sharpe),
            win_rate=wins / trades if trades > 0 else 0,
            trades=trades
        )
        
        self.results.append(result)
        return result
    
    def _generate_returns(self, scenario: Scenario, days: int = 252) -> List[float]:
        """Generate returns based on scenario."""
        params = scenario.parameters
        
        base_return = params.get('mean_return', 0.0005)
        volatility = params.get('volatility', 0.02)
        
        if scenario.type == ScenarioType.BULLISH:
            base_return = 0.001
        elif scenario.type == ScenarioType.BEARISH:
            base_return = -0.0005
        elif scenario.type == ScenarioType.VOLATILE:
            volatility = 0.04
        elif scenario.type == ScenarioType.CRISIS:
            base_return = -0.002
            volatility = 0.05
        
        returns = np.random.normal(base_return, volatility, days)
        return list(returns)
    
    def run_parallel(self, scenarios: List[Scenario]) -> List[SimulationResult]:
        """Run scenarios in parallel."""
        results = []
        for scenario in scenarios:
            result = self.simulate(scenario)
            results.append(result)
        return results


class ScenarioClusteringEngine:
    """Clusters similar scenarios."""
    
    def __init__(self):
        self.clusters: Dict[str, List[Scenario]] = defaultdict(list)
        
        logger.info("ScenarioClusteringEngine initialized")
    
    def cluster(self, scenarios: List[Scenario], n_clusters: int = 5) -> Dict[str, List[Scenario]]:
        """Cluster scenarios."""
        for scenario in scenarios:
            cluster_key = scenario.type.value
            self.clusters[cluster_key].append(scenario)
        
        return dict(self.clusters)
    
    def get_representative(self, cluster_key: str) -> Optional[Scenario]:
        """Get representative scenario from cluster."""
        if cluster_key not in self.clusters:
            return None
        
        scenarios = self.clusters[cluster_key]
        if not scenarios:
            return None
        
        return scenarios[len(scenarios) // 2]


class SensitivityAnalyzer:
    """Analyzes parameter sensitivity."""
    
    def __init__(self):
        self.sensitivity_results: Dict[str, Dict] = {}
        
        logger.info("SensitivityAnalyzer initialized")
    
    def analyze(self, base_params: Dict[str, float], 
                param_name: str, 
                values: List[float],
                evaluate_func: Callable) -> Dict[str, Any]:
        """Analyze sensitivity to a parameter."""
        results = []
        
        for value in values:
            params = base_params.copy()
            params[param_name] = value
            
            performance = evaluate_func(params)
            results.append({
                'value': value,
                'performance': performance
            })
        
        performances = [r['performance'] for r in results]
        sensitivity = (max(performances) - min(performances)) / max(0.001, np.mean(performances))
        
        analysis = {
            'parameter': param_name,
            'results': results,
            'sensitivity': sensitivity,
            'optimal_value': values[np.argmax(performances)]
        }
        
        self.sensitivity_results[param_name] = analysis
        return analysis
    
    def get_most_sensitive(self, top_n: int = 3) -> List[str]:
        """Get most sensitive parameters."""
        sorted_params = sorted(
            self.sensitivity_results.items(),
            key=lambda x: x[1]['sensitivity'],
            reverse=True
        )
        return [p[0] for p in sorted_params[:top_n]]


class RiskScenarioGenerator:
    """Generates risk scenarios."""
    
    def __init__(self):
        self.historical_crises = [
            {"name": "Flash Crash", "drop": -0.09, "duration": 1},
            {"name": "Brexit Vote", "drop": -0.08, "duration": 3},
            {"name": "COVID Crash", "drop": -0.35, "duration": 30},
            {"name": "2008 Crisis", "drop": -0.50, "duration": 180}
        ]
        
        self._scenario_counter = 0
        
        logger.info("RiskScenarioGenerator initialized")
    
    def generate_extreme(self, count: int = 10) -> List[Scenario]:
        """Generate extreme risk scenarios."""
        scenarios = []
        
        for crisis in self.historical_crises[:count]:
            self._scenario_counter += 1
            
            returns = []
            daily_drop = crisis['drop'] / crisis['duration']
            for _ in range(crisis['duration']):
                returns.append(daily_drop + random.gauss(0, abs(daily_drop) * 0.5))
            
            scenarios.append(Scenario(
                id=f"risk_{self._scenario_counter}",
                type=ScenarioType.CRISIS,
                parameters={'mean_return': daily_drop, 'volatility': 0.05},
                returns=returns
            ))
        
        while len(scenarios) < count:
            self._scenario_counter += 1
            drop = random.uniform(-0.3, -0.05)
            duration = random.randint(1, 30)
            
            returns = [drop / duration + random.gauss(0, 0.02) for _ in range(duration)]
            
            scenarios.append(Scenario(
                id=f"risk_{self._scenario_counter}",
                type=ScenarioType.CRISIS,
                parameters={'mean_return': drop / duration, 'volatility': 0.03},
                returns=returns
            ))
        
        return scenarios[:count]


class SimulationSystemAGI:
    """Main Simulation System AGI."""
    
    def __init__(self):
        self.simulator = ParallelScenarioSimulator()
        self.clustering = ScenarioClusteringEngine()
        self.sensitivity = SensitivityAnalyzer()
        self.risk_generator = RiskScenarioGenerator()
        
        self._scenario_counter = 0
        
        logger.info("SimulationSystemAGI initialized")
    
    def create_scenario(self, scenario_type: ScenarioType, 
                       parameters: Dict[str, float] = None) -> Scenario:
        """Create a simulation scenario."""
        self._scenario_counter += 1
        
        return Scenario(
            id=f"scenario_{self._scenario_counter}",
            type=scenario_type,
            parameters=parameters or {}
        )
    
    def run_stress_test(self, strategy_func: Callable = None) -> Dict[str, Any]:
        """Run stress test with extreme scenarios."""
        scenarios = self.risk_generator.generate_extreme(10)
        results = self.simulator.run_parallel(scenarios)
        
        return {
            'scenarios_tested': len(scenarios),
            'results': [
                {
                    'id': r.scenario_id,
                    'final_equity': r.final_equity,
                    'max_drawdown': r.max_drawdown,
                    'survived': r.final_equity > 0
                }
                for r in results
            ],
            'survival_rate': sum(1 for r in results if r.final_equity > 0) / len(results)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGI status."""
        return {
            'simulations_run': len(self.simulator.results),
            'clusters': len(self.clustering.clusters),
            'sensitivity_analyzed': len(self.sensitivity.sensitivity_results)
        }

    def full_mental_simulation(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        API Compatibility Wrapper for OmegaAGICore.
        Runs a quick Monte Carlo simulation based on current state.
        """
        # 1. Create Scenarios
        scenarios = []
        
        # Optimistic Scenario (Trend Continuation)
        scenarios.append(self.create_scenario(
            ScenarioType.BULLISH,
            {'mean_return': 0.001, 'volatility': 0.02}
        ))
        
        # Pessimistic Scenario (Reversal)
        scenarios.append(self.create_scenario(
            ScenarioType.BEARISH,
            {'mean_return': -0.001, 'volatility': 0.03}
        ))
        
        # Volatile Scenario (Noise)
        scenarios.append(self.create_scenario(
            ScenarioType.VOLATILE,
            {'mean_return': 0.0, 'volatility': 0.05}
        ))
        
        # 2. Run Simulation
        results = self.simulator.run_parallel(scenarios)
        
        
        # 3. Analyze Results
        friction_cost = 0.00015 # Hardcoded estimation or import from FrictionModel
        
        bull_wins = sum(1 for r in results if (r.final_equity - self.simulator.initial_capital) > friction_cost)
        total_runs = len(results)
        prob_up = bull_wins / total_runs if total_runs > 0 else 0.5
        
        # Adjust expected moves by friction cost (Spread + Comm)
        # Assuming profit target is hit, we lose spread.
        expected_moves = [r.final_equity - self.simulator.initial_capital - friction_cost for r in results]
        
        optimistic = max(expected_moves)
        pessimistic = min(expected_moves)
        
        recommendation = "WAIT"
        # Increased threshold for Buy/Sell to account for friction validation
        if prob_up > 0.75: recommendation = "BUY"
        elif prob_up < 0.25: recommendation = "SELL"
        
        return {
            'recommendation': recommendation,
            'scenarios': {
                'optimistic': optimistic,
                'pessimistic': pessimistic,
                'mean': float(np.mean(expected_moves))
            },
            'monte_carlo': {
                'positive_probability': prob_up,
                'iterations': total_runs
            }
        }
