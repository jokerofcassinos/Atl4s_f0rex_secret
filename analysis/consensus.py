import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .trend_architect import TrendArchitect
from .sniper import Sniper
from .quant import Quant
from .volatility import VolatilityGuard
from .patterns import PatternRecon
from .market_cycle import MarketCycle
from .supply_demand import SupplyDemand
from .divergence import DivergenceHunter
from .kinematics import Kinematics
from .microstructure import MicroStructure
from .fractal_vision import FractalVision
from .math_core import MathCore
from .quantum_core import QuantumCore
from .cortex_memory import CortexMemory
from .prediction_engine import PredictionEngine
from .wavelet_core import WaveletCore
from .topology_engine import TopologyEngine
from .game_theory import GameTheoryCore
from .chaos_engine import ChaosEngine
from .supply_chain import SupplyChainGraph
from .fifth_eye import FifthEye
from .sixth_eye import SixthEye
from .seventh_eye import SeventhEye
from .eighth_eye import EighthEye
from .ninth_eye import NinthEye
from .tenth_eye import TenthEye
from .recursive_reasoner import RecursiveReasoner
from .black_swan_adversary import BlackSwanAdversary
from .agi.weekend_gap_predictor import WeekendGapPredictor
from core.agi.thought_tree import GlobalThoughtOrchestrator, ThoughtTree
from core.agi.decision_memory import GlobalDecisionMemory, ModuleDecisionMemory
# AGI Ultra Imports (Phase 7)
from core.agi.unified_reasoning import UnifiedReasoningLayer, ModuleInsight
from core.agi.health_monitor import HealthMonitor, HealthStatus
from core.agi.memory_integration import MemoryIntegrationLayer
from analysis.agi.pattern_hunter import PatternHunter

logger = logging.getLogger("Atl4s-Consensus")

class ConsensusEngine:
    def __init__(self):
        self.trend = TrendArchitect()
        self.sniper = Sniper()
        self.quant = Quant()
        self.volatility = VolatilityGuard()
        self.patterns = PatternRecon()
        self.cycle = MarketCycle()
        self.supply_demand = SupplyDemand()
        self.divergence = DivergenceHunter()
        self.kinematics = Kinematics()
        self.micro = MicroStructure() # Initialize here, but needs ticks from main
        self.fractal = FractalVision()
        self.math = MathCore()
        self.quantum = QuantumCore()
        self.cortex = CortexMemory()
        self.prediction = PredictionEngine()
        self.wavelet = WaveletCore()
        self.topology = TopologyEngine()
        self.game = GameTheoryCore()
        self.chaos = ChaosEngine()
        self.sc_graph = SupplyChainGraph()
        self.oracle = FifthEye()
        self.council = SixthEye()
        self.overlord = SeventhEye() # The High Synthesis
        self.sovereign = EighthEye() # The Universal Alignment
        self.singularity = NinthEye() # The High Geometry
        self.architect = TenthEye() # The Meta Manager
        self.debater = RecursiveReasoner() # The Internal Critic
        self.adversary = BlackSwanAdversary() # The Stress Tester
        self.weekend_gap = WeekendGapPredictor() # Phase 5: Weekend Gap Predictor
        
        # Akashic AGI Integration
        try:
            self.pattern_hunter = PatternHunter()
            self.agi_active = True
        except Exception as e:
            logger.warning(f"Failed to load PatternHunter: {e}")
            self.pattern_hunter = None
            self.agi_active = False
        
        # Phase 6: Thought Tree and Decision Memory
        self.thought_orchestrator = GlobalThoughtOrchestrator()
        self.decision_memory = GlobalDecisionMemory()
        
        # Phase 7: AGI Ultra Integration
        self.unified_reasoning = UnifiedReasoningLayer(conflict_resolution_strategy="weighted_vote")
        self.health_monitor = HealthMonitor(latency_threshold_ms=500, loop_detection_window=50)
        self.memory_integration = MemoryIntegrationLayer()
        
        # Register core modules for health monitoring
        for module_name in ['Trend', 'Sniper', 'Quant', 'Oracle', 'Council', 'Overlord']:
            self.health_monitor.register_component(module_name)
        
        # Persistent Thread Pool for fast execution
        self.executor = ThreadPoolExecutor(max_workers=14)
        
        # Default Genome (Weights & Thresholds)
        self.params = {
            'w_trend': 0.20,
            'w_sniper': 0.25, 
            'w_quant': 0.15,
            'w_pattern': 0.10,
            'w_cycle': 0.10,
            'w_sd': 0.10,
            'w_div': 0.05,
            'w_kin': 0.15,    # Increased from 0.05 (Physics Priority)
            'w_fractal': 0.10, # Reduced from 0.15
            'threshold': 15,  # Aggressive Base Threshold (was 25)
            'chaos_threshold': 3.5 
        }
        
        # Phase 6 Performance Optimization: Semantic De-Duplication
        self.last_module_states = {} # module_name -> state_hash

    def update_parameters(self, new_params):
        """Update weights and thresholds for optimization"""
        self.params.update(new_params)

    def update_ticks(self, tick):
        """Pass live tick to micro module"""
        self.micro.on_tick(tick)

    def _consult_akashic(self, df):
        """Helper to consult the Akashic Records"""
        if not self.agi_active or df.empty: return None
        try:
            row = df.iloc[-1]
            return self.pattern_hunter.analyze_live_market(
                row['open'], row['high'], row['low'], row['close'], 
                row.get('tick_volume', 0), row.get('RSI_14', 50), row.get('ATR_14', 0.0001), 
                row.name
            )
        except Exception as e:
            logger.error(f"Akashic Error: {e}")
            return None

    def deliberate(self, data_map, parallel=True, verbose=True):
        """
        Aggregates votes from all sub-modules using Dynamic Regime Weights.
        data_map: {'M5': df_m5, 'H1': df_h1, 'M8': df_m8 (Optional)}
        """
        df_m5 = data_map['M5']
        df_h1 = data_map.get('H1')
        df_m8 = data_map.get('M8') # Fibonacci Timeframe
        
        details = {}
        
        # Micro-Structure Analysis (Instant)
        micro_metrics = self.micro.analyze()
        m_vel = micro_metrics['velocity']
        m_freq = micro_metrics['frequency']
        m_delta = micro_metrics['delta']
        m_rej = micro_metrics['rejection']
        m_imb = micro_metrics.get('imbalance', 0)
        m_ent = micro_metrics.get('entropy', 1.0)
        details['Micro'] = micro_metrics
        
        # Define analysis tasks
        # CRITICAL: If M8 is available, we use it for Physics (Kinematics) and Quantum Analysis
        # to align with the "8-Minute Execution" rhythm.
        
        tasks = {
            'Trend': lambda: self.trend.analyze(df_m5, df_h1),
            'Sniper': lambda: self.sniper.analyze(df_m5),
            'Quant': lambda: self.quant.analyze(df_m5),
            'Patterns': lambda: self.patterns.analyze(df_m5),
            'Cycle': lambda: self.cycle.analyze(df_m5),
            'SupplyDemand': lambda: self.supply_demand.analyze(df_m5),
            'Divergence': lambda: self.divergence.analyze(df_m5),
            'Volatility': lambda: self.volatility.analyze(df_m5),
            # Kinematics uses M8 if available for the 8-minute execution pulse
            'Kinematics': lambda: self.kinematics.analyze(df_m8 if df_m8 is not None else df_m5),
            'Fractal': lambda: self.fractal.analyze(df_h1, data_map.get('H4')),
            'Math': lambda: self.math.analyze(df_m5),
            # Quantum Coherence check on the execution timeframe (M8)
            'Quantum': lambda: self.quantum.analyze(df_m8 if df_m8 is not None else df_m5, []), 
            'Cortex': lambda: self.cortex.recall(self.cortex.extract_features(df_m5)),
            'Prediction': lambda: self.prediction.analyze(df_m5),
            'Wavelet': lambda: self.wavelet.decompose(df_m5['close'].values),
            'Topology': lambda: self.topology.analyze_persistence(df_m5['close'].values),
            'Game': lambda: self.game.calculate_nash_equilibrium(df_m5),
            'Chaos': lambda: self.chaos.calculate_lyapunov(df_m5),
            'SupplyChain': lambda: self.sc_graph.get_impact(),
            'Oracle': lambda: self.oracle.deliberate(data_map),
            'Council': lambda: self.council.deliberate(data_map),
            'Overlord': lambda: self.overlord.deliberate(data_map),
            'Sovereign': lambda: self.sovereign.deliberate(data_map),
            'Singularity': lambda: self.singularity.deliberate(data_map),
            'WeekendGap': lambda: self.weekend_gap.deliberate(data_map),
            'Akashic': lambda: self._consult_akashic(df_m5) 
        }
        
        results = {}
        
        if parallel:
            # Execute in parallel using persistent executor
            future_to_task = {self.executor.submit(func): name for name, func in tasks.items()}
            for future in as_completed(future_to_task):
                name = future_to_task[future]
                try:
                    start_t = time.time()
                    results[name] = future.result()
                    latency_ms = (time.time() - start_t) * 1000
                    self.health_monitor.update_component(name, latency_ms=latency_ms)
                except Exception as e:
                    logger.error(f"Analysis Error in {name}: {e}")
                    self.health_monitor.record_error(name, str(e))
        else:
            # Execute sequentially for backtest stabilization
            for name, func in tasks.items():
                try:
                    results[name] = func()
                except Exception as e:
                    logger.error(f"Sequential Analysis Error in {name}: {e}")

        if not parallel:
             pass # Optimization: return early? No, we need holographic logic.

        # Phase 6: Recursive Thinking - Each module thinks about its decision
        # Pergunta recursiva: "Por que pensei assim?", "Foi correto?", "Como agir agora?"
        for module_name, result in results.items():
            if result is None:
                continue
            
            # Obtém ou cria árvore de pensamento para o módulo
            tree = self.thought_orchestrator.get_or_create_tree(module_name)
            
            # Obtém ou cria memória de decisões
            memory = self.decision_memory.get_or_create_memory(module_name)
            
            # Extrai decisão e score
            decision = result.get('decision', 'WAIT') if isinstance(result, dict) else 'WAIT'
            score = result.get('score', 0.0) if isinstance(result, dict) else 0.0
            reasoning = result.get('reason', '') if isinstance(result, dict) else ''
            
            # Phase 6 Optimization: Semantic De-Duplication
            # If the module's decision state (Decision + Score) hasn't changed,
            # we skip the expensive Thought Tree generation.
            # We include the first 50 chars of reasoning to catch subtle changes.
            state_key = f"{decision}:{score:.2f}:{str(reasoning)[:50]}"
            last_state = self.last_module_states.get(module_name)
            
            if state_key == last_state:
                # Nothing changed, skip expensive Thought Tree operations
                continue
                
            self.last_module_states[module_name] = state_key
            
            # Cria nó de pensamento raiz
            root_node_id = tree.create_node(
                question=f"Why did I decide {decision}?",
                context={'result': result, 'data_map_keys': list(data_map.keys())},
                confidence=abs(score) / 100.0 if score != 0 else 0.0
            )
            
            # Responde o nó raiz
            tree.answer_node(root_node_id, reasoning or f"Based on analysis: {decision}")
            
            # Perguntas recursivas adicionais
            if decision != 'WAIT':
                # Pergunta 1: "Foi correto?"
                child1_id = tree.create_node(
                    question="Was this decision correct?",
                    parent_id=root_node_id,
                    context={'parent_decision': decision}
                )
                # Busca decisões similares no passado
                similar_decisions = memory.find_similar_decisions(
                    {'decision': decision, 'score': score}, limit=5
                )
                if similar_decisions:
                    success_rate = len([d for d in similar_decisions if d.result == "WIN"]) / len(similar_decisions)
                    answer1 = f"Similar past decisions had {success_rate:.1%} success rate"
                    tree.answer_node(child1_id, answer1, confidence=success_rate)
                else:
                    tree.answer_node(child1_id, "No similar past decisions found", confidence=0.0)
                
                # Pergunta 2: "Como agir agora?"
                child2_id = tree.create_node(
                    question="How should I act now?",
                    parent_id=root_node_id,
                    context={'current_decision': decision}
                )
                recommendation = memory.get_recommendation({'decision': decision, 'score': score})
                tree.answer_node(child2_id, recommendation.get('reason', ''), 
                               confidence=recommendation.get('confidence', 0.0))
            
            # Registra decisão na memória
            decision_id = memory.record_decision(
                decision=decision,
                score=score,
                context={'result': result, 'data_map_keys': list(data_map.keys())},
                reasoning=reasoning,
                confidence=abs(score) / 100.0 if score != 0 else 0.0
            )
            
            # Adiciona perguntas recursivas à memória
            if decision != 'WAIT':
                memory.add_recursive_question(decision_id, "Why did I decide this?")
                memory.add_recursive_question(decision_id, "Was this correct?", 
                                            "Based on similar past decisions")
                memory.add_recursive_question(decision_id, "How should I act now?", 
                                            recommendation.get('reason', '') if decision != 'WAIT' else '')

        # Phase 6: Meta-Thinking - Analyze other modules' thoughts
        # Sistema de meta-pensamento: análise do pensamento de outros módulos
        meta_insights = {}
        for module_name in results.keys():
            if module_name not in meta_insights:
                meta_insights[module_name] = {}
            
            # Analisa o que outros módulos pensaram
            for other_module in results.keys():
                if other_module == module_name:
                    continue
                
                insights = self.decision_memory.get_cross_module_insights(module_name, other_module)
                meta_insights[module_name][other_module] = insights
        
        # Conecta pensamentos similares entre módulos
        for module1 in results.keys():
            tree1 = self.thought_orchestrator.get_or_create_tree(module1)
            for module2 in results.keys():
                if module1 == module2:
                    continue
                
                # Busca pensamentos similares
                similar = self.thought_orchestrator.find_similar_thoughts(
                    module1, f"Decision analysis for {module1}", threshold=0.5
                )
                for mod_name, node_id, similarity in similar[:3]:  # Top 3
                    if mod_name == module2:
                        tree2 = self.thought_orchestrator.get_or_create_tree(module2)
                        # Conecta nós similares
                        self.thought_orchestrator.connect_cross_module(
                            module1, tree1.root_nodes[-1] if tree1.root_nodes else "",
                            module2, node_id
                        )

        # Unpack Results
        # Trend
        trend_res = results.get('Trend')
        if trend_res is None: 
            trend_res = {'score': 0, 'direction': 0, 'regime': 'RANGING', 'river': 0}
        elif isinstance(trend_res, tuple):
             # Legacy tuple support: (score, direction) or (score, dir, regime, river)
             t_score = trend_res[0] if len(trend_res) > 0 else 0
             t_dir = trend_res[1] if len(trend_res) > 1 else 0
             trend_res = {'score': t_score, 'direction': t_dir, 'regime': 'RANGING', 'river': 0}

        t_score = trend_res.get('score', 0)
        t_dir = trend_res.get('direction', 0)
        regime = trend_res.get('regime', 'RANGING')
        river = trend_res.get('river', 0)
        ocean_dir = trend_res.get('ocean', 0) # H4 Trend (Previously undefined in local scope)
        details['Trend'] = {'score': t_score, 'dir': t_dir, 'regime': regime, 'river': river, 'ocean': ocean_dir}
        
        # Sniper
        sniper_res = results.get('Sniper')
        if sniper_res is None: sniper_res = (0, 0)
        s_score = sniper_res[0] if isinstance(sniper_res, tuple) else sniper_res.get('score', 0)
        s_dir = sniper_res[1] if isinstance(sniper_res, tuple) else sniper_res.get('dir', 0)
        details['Sniper'] = {'score': s_score, 'dir': s_dir}
        
        # Quant
        quant_res = results.get('Quant')
        if quant_res is None: quant_res = (0, 0, "NEUTRAL")
        if isinstance(quant_res, tuple):
             q_score = quant_res[0]
             q_dir = quant_res[1]
             q_type = quant_res[2] if len(quant_res) > 2 else "NEUTRAL"
        else:
             q_score = quant_res.get('score', 0)
             q_dir = quant_res.get('dir', 0)
             q_type = quant_res.get('type', "NEUTRAL")
        
        details['Quant'] = {'score': q_score, 'dir': q_dir, 'type': q_type}
        
        # Patterns
        patterns_res = results.get('Patterns')
        if patterns_res is None: patterns_res = (0, 0, "None")
        if isinstance(patterns_res, tuple):
            p_score, p_dir, p_name = patterns_res
        else:
            p_score = patterns_res.get('score', 0)
            p_dir = patterns_res.get('dir', 0)
            p_name = patterns_res.get('name', "None")

        details['Patterns'] = {'score': p_score, 'dir': p_dir, 'name': p_name}
        
        # Cycle
        cycle_res = results.get('Cycle')
        if cycle_res is None: cycle_res = ("NEUTRAL", 0)
        if isinstance(cycle_res, tuple):
            c_phase, c_score = cycle_res
        else:
            c_phase = cycle_res.get('phase', 'NEUTRAL')
            c_score = cycle_res.get('score', 0)

        c_dir = 0
        if c_phase == "MANIPULATION_BUY": c_dir = 1
        elif c_phase == "MANIPULATION_SELL": c_dir = -1
        details['Cycle'] = {'phase': c_phase, 'score': c_score, 'dir': c_dir}
        
        # SupplyDemand
        sd_res = results.get('SupplyDemand')
        if sd_res is None: sd_res = (0, 0, {})
        sd_score = sd_res[0] if isinstance(sd_res, tuple) else sd_res.get('score', 0)
        sd_dir = sd_res[1] if isinstance(sd_res, tuple) else sd_res.get('dir', 0)
        sd_info = sd_res[2] if isinstance(sd_res, tuple) and len(sd_res)>2 else sd_res.get('info', {})
        details['SupplyDemand'] = {'score': sd_score, 'dir': sd_dir, 'info': sd_info}
        
        # Divergence
        div_res = results.get('Divergence')
        if div_res is None: div_res = (0, 0, "None")
        d_score = div_res[0] if isinstance(div_res, tuple) else div_res.get('score', 0)
        d_dir = div_res[1] if isinstance(div_res, tuple) else div_res.get('dir', 0)
        d_type = div_res[2] if isinstance(div_res, tuple) and len(div_res)>2 else div_res.get('type', "None")
        details['Divergence'] = {'score': d_score, 'dir': d_dir, 'type': d_type}
        
        # Volatility
        vol_res = results.get('Volatility')
        if vol_res is None: vol_res = (0, 0)
        v_score = vol_res[0] if isinstance(vol_res, tuple) else vol_res.get('score', 0)
        v_dir = vol_res[1] if isinstance(vol_res, tuple) else vol_res.get('dir', 0)
        details['Volatility'] = {'score': v_score, 'dir': v_dir}
        
        # Kinematics
        kin_res = results.get('Kinematics')
        if kin_res is None: kin_res = (0, 0, 0, 0, 0)
        if isinstance(kin_res, tuple):
             k_vel, k_acc, k_score, k_angle, k_energy = kin_res
        else:
             k_vel = kin_res.get('vel', 0)
             k_acc = kin_res.get('acc', 0)
             k_score = kin_res.get('score', 0)
             k_angle = kin_res.get('angle', 0)
             k_energy = kin_res.get('energy', 0)

        k_dir = 1 if k_score > 0 else -1 if k_score < 0 else 0
        details['Kinematics'] = {'vel': k_vel, 'acc': k_acc, 'score': abs(k_score), 'angle': k_angle, 'energy': k_energy}
        
        # Fractal Vision
        fractal_res = results.get('Fractal')
        if fractal_res is None: fractal_res = {'score': 0, 'h4_structure': 'NEUTRAL'}
        # FIXED: Check if it's a tuple
        if isinstance(fractal_res, tuple):
             # Legacy fractal tuple? Let's assume (score, structure)
             f_score = fractal_res[0] if len(fractal_res) > 0 else 0
             h4_struc = "NEUTRAL" # Default
             fractal_res = {'score': f_score, 'h4_structure': h4_struc}
        
        f_score = fractal_res.get('score', 0)
        f_dir = 1 if f_score > 0 else -1 if f_score < 0 else 0
        details['Fractal'] = fractal_res
        
        # Math Core
        math_res = results.get('Math')
        if math_res is None: math_res = {'regime_prob': 0.5, 'hurst': 0.5, 'entropy': 0, 'kalman_diff': 0}
        if isinstance(math_res, tuple):
             math_res = {'regime_prob': 0.5, 'hurst': 0.5, 'entropy': 0, 'kalman_diff': 0}

        regime_prob = math_res.get('regime_prob', 0.5)
        hurst = math_res.get('hurst', 0.5)
        entropy = math_res.get('entropy', 0)
        kalman_diff = math_res.get('kalman_diff', 0)
        
        # New Quantum Math Metrics
        # New Quantum Math Metrics (Optimized: Last window only)
        from src.quantum_math import QuantumMath
        fisher_curv = QuantumMath.fisher_information_curvature(df_m5['close'], last_only=True)
        robust_hurst = QuantumMath.calculate_hurst_exponent(df_m5['close'], last_only=True)
        
        details['Math'] = math_res
        details['Math']['fisher_curvature'] = fisher_curv
        details['Math']['robust_hurst'] = robust_hurst
        
        # Quantum Core
        quantum_res = results.get('Quantum')
        if quantum_res is None: quantum_res = {'tunneling_prob': 0.0, 'is_excited': False}
        if isinstance(quantum_res, tuple):
            q_tunnel = quantum_res[0]
            is_excited = False
            quantum_res = {'tunneling_prob': q_tunnel, 'is_excited': is_excited}
        else:
            q_tunnel = quantum_res.get('tunneling_prob', 0)
            is_excited = quantum_res.get('is_excited', False)

        details['Quantum'] = quantum_res
        
        # Cortex Memory
        cortex_prob = results.get('Cortex')
        if cortex_prob is None: cortex_prob = 0.5
        details['Cortex'] = {'bullish_prob': cortex_prob}
        
        # Prediction Engine (Monte Carlo)
        pred_res = results.get('Prediction')
        if pred_res is None: pred_res = {'prob_bullish': 0.5, 'skew': 0}
        if isinstance(pred_res, tuple):
             pred_res = {'prob_bullish': 0.5, 'skew': 0}
             
        mc_bullish = pred_res.get('prob_bullish', 0.5)
        mc_skew = pred_res.get('skew', 0)
        details['Prediction'] = pred_res
        
        # Omniscience Modules
        wavelet_res = results.get('Wavelet')
        if wavelet_res is None: wavelet_res = {'coherence': 0, 'energy_fast': 0}
        w_coherence = wavelet_res['coherence']
        details['Wavelet'] = wavelet_res
        
        topology_res = results.get('Topology')
        if topology_res is None: topology_res = (0, 0)
        topo_score, topo_loops = topology_res
        details['Topology'] = {'loop_score': topo_score, 'betti_1': topo_loops}
        
        # Hyper-Complexity Modules
        game_res = results.get('Game')
        if game_res is None: game_res = {'equilibrium_price': 0, 'dominance_score': 0}
        nash_price = game_res['equilibrium_price']
        bull_dom = game_res['dominance_score']
        details['Game'] = game_res
        
        chaos_res = results.get('Chaos')
        if chaos_res is None: chaos_res = 0.0
        lyapunov = chaos_res
        details['Chaos'] = {'lyapunov': lyapunov}
        
        # Supply Chain
        sc_impact = results.get('SupplyChain', 0)
        details['SupplyChain'] = {'impact': sc_impact}
        
        # Fifth Eye (Oracle)
        oracle_res = results.get('Oracle', {'decision': 'WAIT', 'score': 0})
        details['Oracle'] = oracle_res
        oracle_score = oracle_res['score']
        oracle_dir = 1 if oracle_res['decision'] == "BUY" else -1 if oracle_res['decision'] == "SELL" else 0
        
        # Sixth Eye (Council)
        council_res = results.get('Council', {'anchor': 'WAIT', 'score': 0})
        details['Council'] = council_res
        council_anchor = council_res['anchor']
        council_dir = 1 if "BUY" in council_anchor else -1 if "SELL" in council_anchor else 0
        
        # Seventh Eye (Overlord)
        overlord_res = results.get('Overlord', {'decision': 'WAIT', 'score': 0})
        details['Overlord'] = overlord_res
        overlord_score = overlord_res['score']
        overlord_dir = 1 if overlord_res['decision'] == "BUY" else -1 if overlord_res['decision'] == "SELL" else 0
        
        # Eighth Eye (Sovereign)
        sovereign_res = results.get('Sovereign', {'decision': 'WAIT', 'score': 0})
        details['Sovereign'] = sovereign_res
        sov_score = sovereign_res['score']
        sov_dir = 1 if "BUY" in sovereign_res['decision'] else -1 if "SELL" in sovereign_res['decision'] else 0
        
        # Ninth Eye (Singularity)
        singularity_res = results.get('Singularity', {'decision': 'WAIT', 'score': 0})
        details['Singularity'] = singularity_res
        is_singularity = singularity_res['decision'] == "SINGULARITY_REACHED"
        
        # Phase 5: Weekend Gap Predictor
        weekend_gap_res = results.get('WeekendGap')
        if weekend_gap_res is None: 
            weekend_gap_res = {'decision': 'WAIT', 'score': 0, 'reason': ''}
        elif isinstance(weekend_gap_res, tuple):
             # Legacy tuple support: (decision, score)
             wg_dec = weekend_gap_res[0] if len(weekend_gap_res) > 0 else 'WAIT'
             wg_score = weekend_gap_res[1] if len(weekend_gap_res) > 1 else 0
             weekend_gap_res = {'decision': wg_dec, 'score': wg_score, 'reason': ''}
             
        details['WeekendGap'] = weekend_gap_res
        weekend_gap_score = weekend_gap_res.get('score', 0)
        weekend_gap_decision = weekend_gap_res.get('decision', 'WAIT')
        weekend_gap_dir = 1 if weekend_gap_decision == "BUY" else -1 if weekend_gap_decision == "SELL" else 0
        
        # --- AUDIT FLAGS ---
        market_state = {
            'hurst': hurst,
            'lyapunov': lyapunov,
            'volatility': v_score
        }
        architect_audit = self.architect.deliberate(details, market_state)
        system_veto = False
        veto_msg = ""
        
        if architect_audit['veto']:
             system_veto = True
             veto_msg = f"ARCHITECT: Low Coherence ({architect_audit['coherence']:.2f})"
             
        if entropy > self.params['chaos_threshold']:
             system_veto = True
             veto_msg = f"CHAOS: High Entropy ({entropy:.2f})"

        # --- GLOBAL REGIME LOCK (New!) ---
        # If H4 Structure opposes H1 River, we are in a "Choppy/Transition" phase.
        # We should either VETO or severely penalize.
        h4_structure = fractal_res.get('h4_structure', 'NEUTRAL')
        
        # Map H4 Structure to Direction
        h4_dir = 0
        if h4_structure == "BULLISH": h4_dir = 1
        elif h4_structure == "BEARISH": h4_dir = -1
        
        # If H4 and H1 River are defined and opposite
        if h4_dir != 0 and river != 0 and h4_dir != river:
            logger.warning(f"GLOBAL REGIME LOCK: H4 ({h4_structure}) opposes H1 River ({river}). Market is Conflicted.")
            # We don't VETO completely, but we raise the bar significantly
            # Unless it's a "Golden Setup" (Reversal)
            pass # Logic handled in weighting/threshold below
            
        # --- GOLDEN SETUP OVERRIDES (The Royal Flush) ---

        # --- GOLDEN SETUP OVERRIDES (The Royal Flush) ---

        # 0. OMEGA SNIPER (Akashic Override)
        akashic_res = results.get('Akashic')
        details['Akashic'] = akashic_res
        if akashic_res and akashic_res['status'] == 'SUCCESS':
            if akashic_res['confidence'] > 0.90:
                logger.info(f"OMEGA SNIPER ACTIVATED: Akashic Match {akashic_res['confidence']*100:.1f}% -> {akashic_res['direction']}")
                # We override EVERYTHING. History repeats.
                decision = akashic_res['direction']
                # Score is 100 + confidence
                score = 100.0 + (akashic_res['confidence'] * 10)
                if decision == "SELL": score = -score
                
                # We assume the caller handles the 'limit_entry' logic based on details['Akashic']['avg_drawdown']
                return decision, score, details
        # 1. Sniper Level + Reversal Pattern + Divergence
        if s_score > 50 and p_name != "None" and d_type != "None":
            if s_dir == p_dir == d_dir:
                logger.info(f"GOLDEN SETUP DETECTED: Sniper + {p_name} + {d_type}. FULL SEND!")
                decision = "BUY" if s_dir == 1 else "SELL"
                # Dynamic Score: Base 95 + excess Sniper Score
                base_dynamic = 95.0 + (s_score - 50) * 0.1
                score = base_dynamic if decision == "BUY" else -base_dynamic
                
                # --- HYBRID ATTACK: OMEGA SNIPER UPGRADE (10x) ---
                # "The Tank": Slow, precise -> Heavily Leveraged
                # UPGRADED: 5x -> 10x with 97% WR
                details['lot_multiplier'] = 10.0
                details['tp_multiplier'] = 0.5
                details['mode'] = "OMEGA_REV_SNIPER"
                
                return decision, score, details
                
        # 2. Momentum Burst (Kinematics) + Trend Alignment
        if abs(k_score) > 70 and regime == "TRENDING":
            # Require alignment with BOTH M5 Trend and H1 River
            if k_dir == t_dir and (river == 0 or k_dir == river):
                # Phase Space Confirmation: Boom Phase (0-90 for Buy, 180-270 for Sell - wait, 180-270 is Accel Down)
                # Buy: Angle 0-90 (Acc Up)
                # Sell: Angle 180-270 (Acc Down)
                valid_phase = False
                if t_dir == 1 and 0 <= k_angle < 90: valid_phase = True
                elif t_dir == -1 and 180 <= k_angle < 270: valid_phase = True
                
                if valid_phase:
                    decision = "BUY" if k_dir == 1 else "SELL"
                    # Dynamic Score: Base 90 + Kinematics Intensity
                    intensity = min(10.0, (abs(k_score) - 70) * 0.2)
                    base_dynamic = 90.0 + intensity
                    score = base_dynamic if decision == "BUY" else -base_dynamic
                    logger.info(f"GOLDEN SETUP: Phase Space BOOM ({k_angle:.0f} deg). AGGRESSIVE ENTRY.")
                    return decision, score, details
                
        # 3. Micro-Structure Flash Scalp (New!)
        # High Frequency + High Delta + Wick Rejection + Quantum Cloud Confirmation
        if m_freq > 2.0: # > 2 ticks/sec (Active)
            if m_rej == "BULLISH_REJECTION" and m_delta > 0:
                 if mc_skew > -0.5: # Cloud doesn't heavily oppose (Neutral or Positive Skew is fine)
                     logger.info(f"GOLDEN SETUP: Micro-Structure Bullish Rejection + Delta. FLASH ENTRY. (Skew {mc_skew:.2f})")
                     return "BUY", 90.0, details
            elif m_rej == "BEARISH_REJECTION" and m_delta < 0:
                 if mc_skew < 0.5:
                     logger.info(f"GOLDEN SETUP: Micro-Structure Bearish Rejection + Delta. FLASH ENTRY. (Skew {mc_skew:.2f})")
                     # Dynamic Score based on Frequency
                     score = -(90.0 + min(5.0, m_freq))
                     return "SELL", score, details
                     
        # 4. Fisher Curvature Jump (Regime Transition)
        if fisher_curv > 2.0: # Threshold for "significant" geometric change
            # Curvature spike usually means imminent explosion. Align with Flow.
            if abs(m_delta) > 10:
                dir_flow = 1 if m_delta > 0 else -1
                decision = "BUY" if dir_flow == 1 else "SELL"
                # Dynamic Score based on Curvature magnitude
                metric = min(5.0, (fisher_curv - 2.0) * 2)
                base_dynamic = 92.0 + metric
                score = base_dynamic if decision == "BUY" else -base_dynamic
                logger.info(f"GOLDEN SETUP: Fisher Curvature Spike ({fisher_curv:.2f}). Transition imminent. Aligning with Flow.")
                return decision, score, details

        confluence_boost = 0
        # 4. Quantum Excited State (Mean Reversion)
        if is_excited:
             # If price is "Excited" (far from mean), prefer Mean Reversion trades
             # If Trend is UP but price is Overbought (Excited), maybe veto Buy or boost Sell?
             if t_dir == 1 and kalman_diff > 0: # Price above Kalman
                 logger.info("Quantum Excited State: Price Overextended. Reducing Buy Confidence.")
                 confluence_boost -= 20
             elif t_dir == -1 and kalman_diff < 0:
                 logger.info("Quantum Excited State: Price Overextended (Down). Reducing Sell Confidence.")
                 confluence_boost -= 20

        # --- 9. Dynamic Weighting 2.0 ---
        # Base Weights
        w_trend = self.params['w_trend']
        w_sniper = self.params['w_sniper']
        w_quant = self.params['w_quant']
        w_pattern = self.params['w_pattern']
        w_cycle = self.params['w_cycle']
        w_sd = self.params['w_sd']
        w_div = self.params['w_div']
        w_kin = self.params['w_kin']
        w_fractal = self.params.get('w_fractal', 0.15)
        
        # Adjust based on Regime & Hurst
        if regime == "TRENDING":
            if hurst > 0.6: # Strong Trend Persistence
                w_trend += 0.15 
                w_kin += 0.10 
                w_quant -= 0.05 
                w_div -= 0.05
                logger.info(f"Regime: STRONG TREND (Hurst {hurst:.2f})")
            else:
                w_trend += 0.05
                w_kin += 0.05
                logger.info(f"Regime: TRENDING (Hurst {hurst:.2f})")
        else:
            # Ranging
            if hurst < 0.4: # Strong Mean Reversion
                w_quant += 0.15 
                w_cycle += 0.05
                w_div += 0.05
                w_trend -= 0.15
                logger.info(f"Regime: MEAN REVERSION (Hurst {hurst:.2f})")
            else:
                w_quant += 0.10
                logger.info(f"Regime: RANGING (Hurst {hurst:.2f})")

        # --- CONFLUENCE LOGIC (Predator Mode) ---
        # confluence_boost already initialized above
        
        # 1. Pattern + Sniper
        if p_name != "None" and s_score > 50:
            if p_dir == s_dir:
                confluence_boost += 20
                logger.info(f"CONFLUENCE: {p_name} + Sniper Level.")
        
        # 2. Trend + Pullback (The "Dip Buy")
        # If Trend is UP and Quant says PULLBACK_BUY -> Massive Boost
        if t_dir == 1 and q_type == "PULLBACK_BUY":
            confluence_boost += 30
            logger.info("CONFLUENCE: Trend UP + Quant Pullback (Dip Buy).")
            
        # 3. Trend + Pullback (The "Rally Sell")
        if t_dir == -1 and q_type == "PULLBACK_SELL":
            confluence_boost += 30
            logger.info("CONFLUENCE: Trend DOWN + Quant Pullback (Rally Sell).")
            
        # 4. Sniper FVG is King
        if s_score > 50:
            confluence_boost += 10
            
        # 5. Fractal Confirmation
        if f_dir != 0:
            if f_dir == t_dir:
                confluence_boost += 15
                logger.info("CONFLUENCE: Fractal Structure confirms Trend.")
                
        # 6. Quantum Tunneling Confirmation
        if q_tunnel > 0.7:
            confluence_boost += 10
            logger.info(f"CONFLUENCE: Quantum Tunneling Probability High ({q_tunnel:.2f}).")
            
        # 7. Cortex Memory Confirmation
        if cortex_prob > 0.7: # Strong Bullish Memory
             if t_dir == 1:
                 confluence_boost += 10
                 logger.info("CONFLUENCE: Cortex Memory recalls Bullish Outcome.")
        elif cortex_prob < 0.3: # Strong Bearish Memory
             if t_dir == -1:
                 confluence_boost += 10
                 logger.info("CONFLUENCE: Cortex Memory recalls Bearish Outcome.")
                 
        # 8. Monte Carlo Confirmation
        if mc_bullish > 0.75:
            if t_dir == 1:
                confluence_boost += 15
                logger.info(f"CONFLUENCE: Monte Carlo predicts Bullish Future ({mc_bullish*100:.1f}%).")
        elif mc_bullish < 0.25:
            if t_dir == -1:
                confluence_boost += 15
        elif mc_bullish < 0.25:
            if t_dir == -1:
                confluence_boost += 15
                logger.info(f"CONFLUENCE: Monte Carlo predicts Bearish Future ({(1-mc_bullish)*100:.1f}%).")

        # 9. Wavelet Coherence (Time-Frequency Lock)
        if w_coherence > 0.8:
            confluence_boost += 10
            logger.info(f"CONFLUENCE: Market Coherence High ({w_coherence:.2f}). Signal is Clear.")
            
        # 10. Topological Loop (Inefficiency Detection)
        if topo_loops > 0:
            # Loop implies Mean Reversion or Cycle
            if regime == "RANGING":
                confluence_boost += 20
                logger.info(f"CONFLUENCE: Topological Loop Detected (Score {topo_score:.2f}). Arbitrage Opportunity.")
            elif regime == "TRENDING":
                # If trending, a loop might mean end of trend (reversal)
                # Be careful to align direction
                pass

        # 11. Nash Equilibrium Reversion (Game Theory)
        current_price = df_m5.iloc[-1]['close']
        if abs(current_price - nash_price) > (df_m5.iloc[-1].get('ATR', 1) * 3): # Far from Fair Value
            # If price is far above Nash, expect sell.
            if current_price > nash_price and t_dir == -1:
                confluence_boost += 25
                logger.info(f"CONFLUENCE: Price far above Nash Equilibrium. Gravity Pull Down.")
            elif current_price < nash_price and t_dir == 1:
                confluence_boost += 25
                logger.info(f"CONFLUENCE: Price far below Nash Equilibrium. Gravity Pull Up.")
                
        # 12. Lyapunov Stability (Chaos Theory)
        if lyapunov < 0: # Stable/Attracting
             confluence_boost += 10
             logger.info(f"CONFLUENCE: Negative Lyapunov ({lyapunov:.4f}). Market is Ordered/Predictable.")
        elif lyapunov > 0.05: # High Chaos
             confluence_boost -= 20 # Penalize
             logger.warning(f"CHAOS WARNING: High Lyapunov ({lyapunov:.4f}). Unpredictable.")
             
        # 13. VPIN Toxicity Veto
        vpin = micro_metrics.get('vpin', 0.5)
        if vpin > 0.8:
            # Toxic flow detected. Highly informed/HFT predatiry volume.
            # Only trade IF our direction aligns with the informed flow (aggressive follow)
            # Otherwise, VETO.
            direction = 1 if m_delta > 0 else -1
            mismatch = False # We'll check this later in the final vector
            logger.warning(f"VPIN AWARENESS: Toxic Flow Detected ({vpin:.2f}). Informed trading high.")

        # 14. Supply Chain Shock Bias
        if abs(sc_impact) > 0.5:
             # Supply chain bias is strong. Align with shock.
             confluence_boost += 15
             logger.info(f"CONFLUENCE: Supply Chain Shock Bias ({sc_impact:.2f}) provides macro tailwind.")
        
        # 15. Phase 5: Weekend Gap Predictor Confluence
        if weekend_gap_dir != 0 and abs(weekend_gap_score) > 30:
            # Weekend gap prediction is significant
            if weekend_gap_dir == t_dir or weekend_gap_dir == k_dir:
                confluence_boost += 20
                logger.info(f"CONFLUENCE: Weekend Gap Predictor ({weekend_gap_decision}) aligns with momentum.")
            elif weekend_gap_decision == "CLOSE_POSITIONS":
                # Gap muito grande previsto, reduzir exposição
                confluence_boost -= 30
                logger.warning(f"WEEKEND GAP WARNING: Large gap predicted. Reducing exposure.")

        # --- HOLOGRAPHIC VECTOR LOGIC (v3.0) ---
        
        # Phase X: Akashic Memory Soft Integration (The "Subconscious Nudge")
        # AGGRESSIVE MODE: Memory is barely positive (>55%)? Use it.
        akashic_boost_val = 0.0
        if akashic_res and akashic_res['status'] == 'SUCCESS':
             ak_conf = akashic_res.get('confidence', 0.0)
             if ak_conf > 0.55: # Lowered from 0.60
                 ak_dir_str = akashic_res.get('direction', 'NEUTRAL')
                 if ak_dir_str != 'NEUTRAL':
                     ak_dir_val = 1 if ak_dir_str == 'BUY' else -1
                     # Boost: (Conf - 0.5) * 100. (Double Aggression)
                     # 0.55 -> 5pts
                     # 0.60 -> 10pts
                     # 0.80 -> 30pts
                     # 0.90 -> 40pts
                     base_boost = (ak_conf - 0.5) * 100.0 
                     akashic_boost_val = base_boost * ak_dir_val
                     logger.info(f"AKASHIC AGGRESSIVE: Memory Nudge {ak_dir_str} (Conf {ak_conf:.2f}) -> Vector Boost {akashic_boost_val:.1f}")

        # 1. Momentum Vector (The "Push")
        # Components: Trend, Kinematics, Fractal, Volatility(if Expansion), SupplyChain, WeekendGap
        # Weights normalized for this vector
        v_momentum = (
            (t_score * t_dir * 1.0) +
            (abs(k_score) * k_dir * 0.8) +
            (abs(f_score) * f_dir * 0.5) +
            (sc_impact * 20) +
            (weekend_gap_score * weekend_gap_dir * 0.3)  # Phase 5: Weekend Gap contribution
        )
        
        # 2. Reversion Vector (The "Pull")
        # Components: Quant (Mean Rev), Cycle, Divergence, Nash, Topology (Loop)
        # Note: Reversion pulls AGAINST the current price extremity
        v_reversion = (
            (q_score * q_dir * 1.0) +
            (c_score * c_dir * 0.7) +
            (d_score * d_dir * 0.8) +
            (topo_score * (1 if regime=="RANGING" else 0) * 0.5) # Loop implies reversion in range
        )
        
        # 3. Structure Vector (The "Map")
        # Components: Sniper (FVG), SupplyDemand, Patterns, Fortress (Levels), Akashic(Memory)
        # Note: Structure dictates WHERE trade is valid
        v_structure = (
            (s_score * s_dir * 1.2) + # Sniper is King
            (sd_score * sd_dir * 0.8) +
            (p_score * p_dir * 0.6) +
            (akashic_boost_val * 2.0) # Memory Validated (2.0x Weight)
        )
        
        # --- HOLOGRAPHIC DECISION MATRIX ---
        
        final_decision = "WAIT"
        final_score = 0
        holographic_reason = "Neutral"

        # Logic A: MOMENTUM BREAKOUT
        # ✅ PHASE 15: Hour 13 Filter (NY Overlap Liquidity Grab Protection)
        veto_momentum = False
        
        # Get hour from DATA (not wall-clock!)
        try:
            data_hour = df_m5.index[-1].hour
        except:
            data_hour = -1
        
        # Hour 13 is deadly for Momentum (0% WR from forensic analysis)
        if data_hour == 13:
            logger.info(f"[MOMENTUM] SILENCED: Hour 13 Liquidity Grab Zone")
            veto_momentum = True
        
        # Chaos Veto (The "Regime Lock")
        if not veto_momentum and (lyapunov > 0.8 or entropy > 2.5):
             logger.info(f"[MOMENTUM] SILENCED by Extreme Chaos (Lyapunov {lyapunov:.2f}, Entropy {entropy:.2f})")
             veto_momentum = True

        if abs(v_momentum) > 30 and not veto_momentum:
             mom_dir = 1 if v_momentum > 0 else -1
             # Check alignment
             if (v_structure * mom_dir) >= 0: # Structure doesn't oppose
                 if (v_reversion * mom_dir) > -30: # Reversion doesn't excessively oppose
                     final_decision = "SELL" if mom_dir == 1 else "BUY"
                     final_score = abs(v_momentum) + abs(v_structure)
                     holographic_reason = "MOMENTUM_BREAKOUT"

        # Logic B: REVERSION TRADE (Sniper Entry)
        # Reversion is Strong AND Structure Supports
        # ✅ PHASE 11 FIX: Wick Filter (The "Knife Guard")
        # If Chaos is High, we MUST have a rejection wick to revert.
        if abs(v_reversion) > 25:
             rev_dir = 1 if v_reversion > 0 else -1
             
             # Micro-Pattern Validation (The Wick Filter)
             # rev_dir=1 means we want to SELL (Inverted Logic below). So we need BEARISH rejection.
             # rev_dir=-1 means we want to BUY. So we need BULLISH rejection.
             micro_data = details.get('Micro', {})
             rejection = micro_data.get('rejection', 'NEUTRAL')
             
             safe_to_revert = True
             safe_to_revert = True
             if lyapunov > 0.8 or entropy > 2.5: # In Extreme Chaos, require wicks
                 if rev_dir == 1 and rejection != "BEARISH_REJECTION":
                      safe_to_revert = False
                      logger.info("[REVERSION] SILENCED: Falling Knife (No Bearish Wick in Extreme Chaos)")
                 elif rev_dir == -1 and rejection != "BULLISH_REJECTION":
                      safe_to_revert = False
                      logger.info("[REVERSION] SILENCED: Falling Knife (No Bullish Wick in Extreme Chaos)")

             if (v_structure * rev_dir) > 20 and safe_to_revert: # Structure MUST support Reversion (Confluence)
                 
                  # --- KINEMATICS PHASE SPACE VETO (Deep Forensic Fix) ---
                  # User feedback: "Bot sells into accelerating uptrend".
                  # If Kinematics shows strong acceleration AGAINST our reversion direction, BLOCK.
                  # k_dir: +1 = Accelerating UP, -1 = Accelerating DOWN.
                  # rev_dir: +1 = BUY signal from reversion, -1 = SELL signal.
                  # Inverted logic: rev_dir=1 becomes SELL, rev_dir=-1 becomes BUY (line 857).
                  # So if final_decision will be SELL, we need rev_dir = +1.
                  # We want to BLOCK SELL if k_dir = +1 (accelerating UP).
                  # Simplified: If inverted trade direction opposes kinematics, BLOCK.
                  
                  # 'final_decision' for reversion: SELL if rev_dir==1, BUY if rev_dir==-1.
                  # If final_decision is SELL (rev_dir==1) and k_dir is +1 (UP ROCKET), this is BAD.
                  # If final_decision is BUY (rev_dir==-1) and k_dir is -1 (DOWN CRASH), this is BAD.
                  # So: BLOCK if (rev_dir * k_dir) > 0 (same sign means opposing the physics).
                  
                  kinematics_opposes = (rev_dir * k_dir) > 0
                  
                  # Phase Space Angle > 25 degrees (Weak Acceleration) is enough to kill Reversion.
                  # Reversion requires DECELERATION (Opposite K-Dir) or FLAT.
                  if kinematics_opposes and abs(k_score) > 25: 
                       logger.warning(f"KINEMATICS VETO: Blocking REVERSION (Accelerating against trade). k_dir={k_dir}, k_score={k_score:.1f}")
                  else:
                       # ✅ PHASE 15 FIX: Divergence Veto (Restored)
                       # Must not revert against Divergence.
                       div_vetoed = False
                       div_res = details.get('Divergence', {})
                       div_type = div_res.get('type', '')
                       
                       # If SELL (rev_dir=1), check Bullish Divergence
                       if rev_dir == 1 and 'bullish' in str(div_type).lower():
                           logger.info(f"[REVERSION] SILENCED by Divergence Conflict ({div_type}).")
                           div_vetoed = True
                       # If BUY (rev_dir=-1), check Bearish Divergence
                       elif rev_dir == -1 and 'bearish' in str(div_type).lower():
                           logger.info(f"[REVERSION] SILENCED by Divergence Conflict ({div_type}).")
                           div_vetoed = True
                           
                       if not div_vetoed:
                           # --- SNIPER CONFLICT VETO (Trade #97-122 Fix) ---
                           # If SMC/Sniper strongly says BUY (>75) but we're about to SELL, BLOCK.
                           # This prevents selling when there's a strong bullish OB/FVG confluence.
                           sniper_result = details.get('Sniper', {})
                           s_score_c = sniper_result.get('score', 0) if isinstance(sniper_result, dict) else 0
                           sniper_signal = sniper_result.get('signal') if isinstance(sniper_result, dict) else None
                           sniper_score = sniper_result.get('confidence', 0) if isinstance(sniper_result, dict) else 0
                           
                           planned_decision = "SELL" if rev_dir == 1 else "BUY"
                           
                           if planned_decision == "SELL" and sniper_signal == "BUY" and sniper_score > 75:
                               logger.warning(f"SNIPER CONFLICT VETO: Blocking SELL (Sniper says BUY {sniper_score:.1f})")
                               # Don't set final_decision, just skip this trade
                           elif planned_decision == "BUY" and sniper_signal == "SELL" and sniper_score > 75:
                               logger.warning(f"SNIPER CONFLICT VETO: Blocking BUY (Sniper says SELL {sniper_score:.1f})")
                               # Don't set final_decision, just skip this trade
                           else:
                               # ✅ SIGNAL INVERSION (User Request): Revert the Reversion
                               final_decision = "SELL" if rev_dir == 1 else "BUY"
                               final_score = abs(v_reversion) + abs(v_structure)
                               holographic_reason = "REVERSION_SNIPER"
                               
                               # --- OMEGA SNIPER UPGRADE (10x - 97% WR) ---
                               # "Reversion Sniper = Omega Sniper Concept (10x)"
                               details['lot_multiplier'] = 10.0
                               details['tp_multiplier'] = 0.5
                               details['mode'] = "OMEGA_REV_SNIPER"
                               logger.info(f"OMEGA SNIPER LOGIC ACTIVATED (Reversion Vector). Multiplier 10x.")
                 
        # Logic C: STRUCTURE BOUNCE (Laminar Flow)
        # Structure is Strong + Trend is Laminar Flow (Low Entropy)
        # ✅ PHASE 0 FIX #3: Relaxed threshold from 60 → 40
        if abs(v_structure) > 40 and entropy < 1.0:
             struc_dir = 1 if v_structure > 0 else -1
             if (v_momentum * struc_dir) >= 0:
                 final_decision = "BUY" if struc_dir == 1 else "SELL"
                 final_score = abs(v_structure) + abs(v_momentum)
                 holographic_reason = "STRUCTURE_FLOW"
                 
        # Logic D: PROTOCOL LION (Last Resort Breakout) - RELAXED
        # \"The Lion does not ask for permission.\"
        # If we are WAITING, but Structure suggests a clear path, FORCE the trade.
        # RELAXED: Removed Ocean alignment requirement, lowered threshold to 30
        if final_decision == "WAIT":
             # Requirement 1: Strong Structure (> 30) - Relaxed from 40
             if abs(v_structure) > 30:
                 struc_dir = 1 if v_structure > 0 else -1
                 
                 # REMOVED: Ocean alignment requirement (was blocking most trades)
                 # Check v_momentum isn't fighting us too hard (>-30) - Relaxed from -20
                 if (v_momentum * struc_dir) > -30:
                     # --- LION SAFEGUARDS (The "Right Veto") ---
                     # Ensure we don't Lion-force into a Sniper wall or Divergence
                     # Use sanitized 'details' populated earlier, not raw 'results'
                     sniper_res = details.get('Sniper', {})
                     div_res = details.get('Divergence', {})
                     pat_res = details.get('Patterns', {})

                     s_dir_check = sniper_res.get('direction', 0) if isinstance(sniper_res, dict) else 0
                     s_score_check = sniper_res.get('score', 0) if isinstance(sniper_res, dict) else 0
                     div_type = div_res.get('type', '') if isinstance(div_res, dict) else ''
                     pat_name = pat_res.get('pattern', '') if isinstance(pat_res, dict) else ''

                     # ✅ PHASE 11 FIX: Chaos Veto (The "Leonidas Gate") - CALIBRATED
                     # User Report: "LION executes in Choppy/Ranging markets (Lyapunov ~0.49-0.60)."
                     # FIX: Drastically lowered threshold. LION requires stability.
                     vetoed = False
                     
                     # Dynamic Threshold based on Regime
                     lion_chaos_limit = 0.50
                     if regime != "TRENDING":
                         lion_chaos_limit = 0.35 # Very strict in Range/MeanReversion
                     
                     if lyapunov > lion_chaos_limit or entropy > 2.0:
                          logger.info(f"[LION] SILENCED by Chaos (Lyapunov {lyapunov:.2f} > {lion_chaos_limit}, Entropy {entropy:.2f})")
                          vetoed = True
                     
                     # ✅ PHASE 15 FIX: Restore Divergence Veto (Critical for MC Conflict)
                     if not vetoed:
                         if (struc_dir == 1 and 'bearish' in str(div_type).lower()) or \
                            (struc_dir == -1 and 'bullish' in str(div_type).lower()):
                             logger.info(f"[LION] SILENCED by Divergence Conflict ({div_type}).")
                             vetoed = True
                     
                     # ✅ PHASE 15 FIX: Restore Sniper Conflict @70 (Strict threshold)
                     # Was 75, lowered to 70 to catch the "Sniper SELL 85%" cases.
                     if not vetoed:
                         if (struc_dir == 1 and s_dir_check == -1 and s_score_check > 70) or \
                            (struc_dir == -1 and s_dir_check == 1 and s_score_check > 70):
                             logger.info(f"[LION] SILENCED by Sniper Conflict ({s_score_check}).")
                             vetoed = True
                     
                     # ✅ PHASE 15 FIX: Kinematics Veto (use k_dir already calculated at line 405)
                     # Threshold lowered to 15° - Trade #84 had angle 24° which bypassed 30°
                     if not vetoed:
                         # SELL (struc_dir=-1) but Kinematics UP (k_dir=1) → block
                         # BUY (struc_dir=1) but Kinematics DOWN (k_dir=-1) → block
                         if (struc_dir == -1 and k_dir == 1 and abs(k_angle) > 15) or \
                            (struc_dir == 1 and k_dir == -1 and abs(k_angle) > 15):
                             logger.info(f"[LION] SILENCED by Kinematics Counter-Trend (k_dir={k_dir}, angle={k_angle}°).")
                             vetoed = True
                     
                     # ✅ PHASE 15 FIX: Smart Money Manipulation Veto (Replaces rigid Hour 13 filter)
                     # If Cycle detects Manipulation (Liquidity Grab), we don't trade AGAINST it.
                     cycle_res = results.get('Cycle', ('NEUTRAL', 0))
                     # cycle_res is a tuple: (phase, score) e.g. ("MANIPULATION_BUY", 90)
                     cycle_phase = cycle_res[0] if isinstance(cycle_res, tuple) else str(cycle_res)
                     
                     if not vetoed:
                         # Block SELL if:
                         # 1. Smart Money is Buying (MANIPULATION_BUY)
                         # 2. Market is in confirmed UP Trend (EXPANSION_BUY)
                         if struc_dir == -1:
                             if "MANIPULATION_BUY" in str(cycle_phase):
                                 logger.info(f"[LION] SILENCED by Smart Money: Bullish Manipulation Detected.")
                                 vetoed = True
                             elif "EXPANSION_BUY" in str(cycle_phase):
                                 logger.info(f"[LION] SILENCED by Cycle Phase: Expansion BUY (Trend against Trade).")
                                 vetoed = True

                         # Block BUY if:
                         # 1. Smart Money is Selling (MANIPULATION_SELL)
                         # 2. Market is in confirmed DOWN Trend (EXPANSION_SELL)
                         elif struc_dir == 1:
                             if "MANIPULATION_SELL" in str(cycle_phase):
                                 logger.info(f"[LION] SILENCED by Smart Money: Bearish Manipulation Detected.")
                                 vetoed = True
                             elif "EXPANSION_SELL" in str(cycle_phase):
                                 logger.info(f"[LION] SILENCED by Cycle Phase: Expansion SELL (Trend against Trade).")
                                 vetoed = True

                     if not vetoed:
                         logger.warning(f"[LION] PROTOCOL LION ACTIVATED: Structure ({v_structure:.1f}) override WAIT.")
                         final_decision = "BUY" if struc_dir == 1 else "SELL"
                         final_score = abs(v_structure) + 20 # Higher score boost
                         holographic_reason = "LION_BREAKOUT"
                         # BOOST: 25x for "1k Challenge"
                         lot_multiplier = 25.0
                         
                         # Lion trades get 10x leverage now
                         details['mode'] = "LION_PROTOCOL"
                         details['lot_multiplier'] = 10.0

        # Logic E: QUANTUM HARMONY (OVERRIDE MODE - Phase 15)
        # Very high tunnel_prob (>0.95) can OVERRIDE other setups
        quantum_prob = details.get('Quantum', {}).get('tunnel_prob', 0)
        quantum_can_override = quantum_prob > 0.95  # Only extreme tunneling overrides
        quantum_as_fallback = quantum_prob > 0.75 and final_decision == "WAIT"
        
        if quantum_can_override or quantum_as_fallback:
             struc_dir = 1 if v_structure > 0 else -1
             if abs(v_structure) > 5:  # Any structure signal
                 vetoed = False
                 
                 # Chaos Veto (Essential)
                 if lyapunov > 0.8 or entropy > 2.5:
                      logger.info(f"[QUANTUM] SILENCED by Extreme Chaos (Lyapunov {lyapunov:.2f}, Entropy {entropy:.2f})")
                      vetoed = True
                 
                 # Sniper Conflict @90 (Only extreme conflicts)
                 sniper_res = details.get('Sniper', {})
                 s_dir_check = sniper_res.get('direction', 0) if isinstance(sniper_res, dict) else 0
                 s_score_check = sniper_res.get('score', 0) if isinstance(sniper_res, dict) else 0
                 
                 if not vetoed:
                     if (struc_dir == 1 and s_dir_check == -1 and s_score_check > 90) or \
                        (struc_dir == -1 and s_dir_check == 1 and s_score_check > 90):
                         logger.info(f"[QUANTUM] SILENCED by Extreme Sniper Conflict ({s_score_check}).")
                         vetoed = True

                 if not vetoed:
                     mode_str = "OVERRIDE" if quantum_can_override else "FALLBACK"
                     logger.warning(f"🌌 QUANTUM HARMONY [{mode_str}]: Tunneling Prob ({quantum_prob:.2f}) + Structure.")
                     final_decision = "BUY" if struc_dir == 1 else "SELL"
                     final_score = abs(v_structure) + 40  # Higher score boost
                     holographic_reason = "QUANTUM_HARMONY"
                     
                     details['mode'] = "QUANTUM_HARMONY"
                     details['lot_multiplier'] = 5.0
                 
        # Override with Golden Setups (The Royal Flush) logic preserved below...
        total_vector = v_momentum + v_reversion + v_structure # Legacy compatibility
        
        # ✅ SIGNAL INVERSION (User Request): If we are in REVERSION_SNIPER or MOMENTUM_BREAKOUT, 
        # we must invert the numeric vector so LaplaceDemon (V2) sees the correct direction.
        if holographic_reason in ["REVERSION_SNIPER", "MOMENTUM_BREAKOUT"]:
             total_vector = -total_vector
             
        details['Vectors'] = {
            'Momentum': v_momentum,
            'Reversion': v_reversion,
            'Structure': v_structure,
            'Reason': holographic_reason
        }
        
        # Volatility Gatekeeper
        if v_score == 0:
            logger.info("Volatility Guard: Market too quiet. Veto.")
            return "WAIT", 0, details
        
        # --- PHYSICS OVERRIDE (Scenario Adaptation - The Missing Analysis) ---
        # User Feedback: "All modules fail in a specific scenario."
        # The Scenario: Kinematics UP + Monte Carlo UP, but bot SELLS.
        # The Fix: If Physics + Probability agree, but decision is opposite, BLOCK.
        
        # THRESHOLD LOWERED: Was 40/0.75, now 25/0.70 to catch more toxic scenarios.
        physics_bullish = (k_dir == 1 and abs(k_score) > 25)  # Lowered from 40
        physics_bearish = (k_dir == -1 and abs(k_score) > 25) # Lowered from 40
        
        mc_strongly_bullish = (mc_bullish > 0.70) # Lowered from 0.75
        mc_strongly_bearish = (mc_bullish < 0.30) # Lowered from 0.25
        
        # OVERRIDE CONDITION: Physics + MC Agree, but bot is going opposite
        if physics_bullish and mc_strongly_bullish and final_decision == "SELL":
            logger.warning(f"PHYSICS OVERRIDE: Blocking SELL (Kinematics UP k_score={k_score:.1f} + MC Bullish {mc_bullish*100:.1f}%)")
            final_decision = "WAIT"
            holographic_reason = "PHYSICS_OVERRIDE"
            
        elif physics_bearish and mc_strongly_bearish and final_decision == "BUY":
            logger.warning(f"PHYSICS OVERRIDE: Blocking BUY (Kinematics DOWN k_score={k_score:.1f} + MC Bearish {(1-mc_bullish)*100:.1f}%)")
            final_decision = "WAIT"
            holographic_reason = "PHYSICS_OVERRIDE"
        
        # VPIN Veto Check
        if vpin > 0.8:
             # Toxic flow. If our total vector direction is opposite to informed flow delta, VETO.
             flow_dir = 1 if m_delta > 0 else -1
             trade_dir = 1 if total_vector > 0 else -1 if total_vector < 0 else 0
             if trade_dir != 0 and trade_dir != flow_dir:
                 logger.warning(f"VPIN VETO: Trade direction {trade_dir} opposes Toxic Flow {flow_dir}. DANGER.")
                 return "WAIT", 0, details
        
        # Breakdown Log for Debugging Stagnation
        health_score = architect_audit.get('health_score', 100)
        # Use details['Vectors'] for breakdown
        
        # Normalize Score (Prevent 100.0 Saturation)
        # Total Vector can be ~300. We map it to 0-100 logic.
        raw_score = abs(total_vector)
        final_score = min(99.9, raw_score * 0.4) # Scaling Factor 0.4 (250 -> 100)
        
        # --- ADAPTIVE THRESHOLD (Scalping Mode) ---
        threshold = self.params['threshold']
        
        # Global Regime Lock Penalty
        if h4_dir != 0 and river != 0 and h4_dir != river:
             threshold += 15 # Raise the bar significantly
             logger.info("Global Regime Lock: Threshold raised by +15 due to H4/H1 Conflict.")
        
        # Bayesian Adjustment
        if regime_prob > 0.8: # High Confidence in Trend
            threshold -= 5
            logger.info(f"Bayesian Confidence High ({regime_prob:.2f}). Lowering threshold.")
        elif regime_prob < 0.3: # High Confidence in Range (Low Trend Prob)
            # If we are in range, maybe raise threshold for trend trades?
            # Or lower it for mean reversion if we had that logic separated.
            pass

        # Dynamic Aggression: Quantum Skew
        # If Skew favors Trend direction strongly, lower threshold aggressively
        if (total_vector > 0 and mc_skew > 0.5) or (total_vector < 0 and mc_skew < -0.5):
             threshold -= 10
             logger.info(f"Quantum Skew ({mc_skew:.2f}) favors trade. Threshold LOWERED to {threshold}.")
        elif (total_vector > 0 and mc_skew < -0.5) or (total_vector < 0 and mc_skew > 0.5):
             threshold += 20
             logger.info(f"Quantum Skew ({mc_skew:.2f}) opposes trade. Threshold RAISED to {threshold}.")

        # --- HYPER-AGGRESSIVE: FORCE SCALP ---
        # If Micro Frequency is High and Delta confirms, or Phase Space is Accelerating -> CRASH THE GATE
        if m_freq > 3.0:
            threshold -= 15
            logger.info(f"HYPER-ACTIVE: High Frequency ({m_freq} tps). Threshold LOWERED significantly to {threshold}.")
            
        if abs(k_score) >= 80: # Phase Space BOOM/CRASH
             threshold -= 10
             logger.info("HYPER-ACTIVE: Phase Space Acceleration. Threshold LOWERED by 10.")


        if regime == "TRENDING" and hurst > 0.6 and entropy < 2.5:
            threshold = max(10, threshold - 15) # Cap at 10 (Very Aggressive)
            logger.info(f"Adaptive Threshold: LOWERED to {threshold} (High Probability Regime)")
        
        decision = "WAIT"
        
        # 1. HOLOGRAPHIC PRIORITY
        if final_decision != "WAIT":
            # ✅ REFINED NORMALIZATION: Matching Laplace formula (50% base + edge over threshold)
            # This ensures the 88% and 99% filters correspond to the numbers the user sees.
            raw_h_score = final_score if final_score > 0 else abs(total_vector)
            # Formula: 50% + (raw - threshold)
            normalized_conf = 50 + (raw_h_score - threshold)
            final_score = min(99.9, max(1.0, normalized_conf))
            
            # Apply strict confidence filters (User Requests)
            # ✅ PHASE 15: Relaxed thresholds to allow more executions
            if holographic_reason == "Neutral" and final_score < 75.0:  # Relaxed from 88
                 logger.info(f"HOLOGRAPHIC VETO: Reason={holographic_reason} Conf={final_score:.1f}% < 75%. Silencing.")
                 decision = "WAIT"
                 final_score = 0
            elif holographic_reason == "REVERSION_SNIPER" and final_score < 70.0:  # Relaxed from 88
                 logger.info(f"HOLOGRAPHIC VETO: Reason={holographic_reason} Conf={final_score:.1f}% < 70%. Silencing.")
                 decision = "WAIT"
                 final_score = 0
            elif holographic_reason == "MOMENTUM_BREAKOUT" and final_score < 75.0:
                 logger.info(f"HOLOGRAPHIC VETO: Reason={holographic_reason} Conf={final_score:.1f}% < 75%. Silencing.")
                 decision = "WAIT"
                 final_score = 0
            else:
                 decision = final_decision
                 logger.info(f"HOLOGRAPHIC CONSENSUS: {decision} | Reason: {holographic_reason} | Conf: {final_score:.1f}%")
            
        # 2. LEGACY FALLBACK (CONSENSUS_VOTE) - ✅ SIGNAL INVERSION (User Request)
        elif final_score > threshold: 
            # ✅ REFINED NORMALIZATION: Matching Laplace formula
            normalized_conf = 50 + (abs(total_vector) - threshold)
            final_score = min(99.9, max(1.0, normalized_conf))
            
            # ✅ SIGNAL INVERSION: If logic fell back here, it's a CONSENSUS_VOTE.
            if total_vector > 0:
                decision = "SELL" # Inverted from BUY
            else:
                decision = "BUY"  # Inverted from SELL
            
            # CONSENSUS_VOTE inversion also applies to total_vector for Laplace (Phase 2 consistency)
            total_vector = -total_vector
            holographic_reason = "CONSENSUS_VOTE"
            logger.info(f"LEGACY CONSENSUS (Inverted): {decision} | Reason: {holographic_reason} | Conf: {final_score:.1f}%")
        
        # --- PHASE 20: NUCLEAR PROFIT BOOST ($1k Challenge) ---
        # If we have a finalized decision with HIGH confidence for Sniper/Breakout, engage 25x leverage.
        if decision != "WAIT" and final_score >= 98.0:
             # Check reasons
             is_sniper = "SNIPER" in holographic_reason or "TRAP" in holographic_reason
             is_breakout = "BREAKOUT" in holographic_reason
             if is_sniper or is_breakout:
                  logger.warning(f"NUCLEAR BOOST: Engaging 25x Multiplier for {holographic_reason} (Conf {final_score:.1f}%)")
                  lot_multiplier = 25.0
                  details['lot_multiplier'] = 25.0
                  details['mode'] = "NUCLEAR_SCALING"
                
        # --- PHASE 10: RECURSIVE DEBATE (Chain-of-Thought) ---
        if decision != "WAIT":
            decision, final_score, debate_log = self.debater.debate(decision, final_score, data_map)
            details['RecursiveDebate'] = debate_log
            logger.info(f"RECURSIVE DEBATE: {debate_log}")
            
        # --- PHASE 11: ADVERSARIAL BLACK-SWAN AUDIT ---
        if decision != "WAIT":
             current_price = df_m5.iloc[-1]['close']
             atr = df_m5.iloc[-20:]['close'].diff().abs().mean() # Quick ATR
             # Determine hypothetical SL based on config or logic
             sl_dist = atr * 2.0
             sl_price = current_price - sl_dist if decision == "BUY" else current_price + sl_dist
             
             is_safe, surv_prob = self.adversary.audit_trade(decision, current_price, sl_price, atr)
             details['BlackSwanAudit'] = {'safe': is_safe, 'survival_prob': surv_prob}
             
             if not is_safe:
                 logger.info(f"Black-Swan Audit: VETO entry. Survival probability {surv_prob:.2f} is too low.")
                 decision = "WAIT"
                 final_score = 0

        # --- FINAL SYSTEM VETO APPLY ---
        if system_veto and decision != "WAIT":
            logger.warning(f"VETO ACTIVE: {veto_msg}. Silencing Main Decision (Score Preserved for Swarm).")
            decision = "WAIT"

        if decision != "WAIT" or verbose:
            logger.info(f"Consensus Reached: {decision} (Score: {final_score:.2f})")
            
        # Phase 6: Add Thought Tree and Decision Memory details
        details['ThoughtTree'] = self.thought_orchestrator.get_global_thought_summary()
        details['MetaThinking'] = meta_insights
        
        # Phase 7: AGI Ultra - Unified Reasoning Synthesis
        if decision != "WAIT":
            # Gather insights from key modules for unified reasoning
            insights = []
            for mod_name in ['Trend', 'Sniper', 'Oracle', 'Overlord', 'Sovereign']:
                mod_res = results.get(mod_name)
                if mod_res and isinstance(mod_res, dict):
                    mod_decision = mod_res.get('decision', 'WAIT')
                    mod_score = abs(mod_res.get('score', 0)) / 100.0
                    mod_reason = mod_res.get('reason', '')
                    insights.append(ModuleInsight(
                        module_name=mod_name,
                        decision=mod_decision,
                        confidence=mod_score,
                        reasoning=mod_reason,
                        supporting_evidence=[]
                    ))
            
            # Get unified decision
            if insights:
                unified_decision = self.unified_reasoning.synthesize(
                    insights, 
                    context={'regime': regime, 'hurst': hurst}
                )
                details['UnifiedReasoning'] = {
                    'agreement': unified_decision.agreement_score,
                    'conflict_type': unified_decision.conflict_type.value,
                    'key_factors': unified_decision.key_factors[:3]
                }
                
                # If low agreement, reduce confidence
                if unified_decision.agreement_score < 0.5:
                    final_score = final_score * 0.8
                    logger.info(f"AGI: Low module agreement ({unified_decision.agreement_score:.1%}). Confidence reduced.")
        
        # Phase 7: Health Check
        health_status = self.health_monitor.get_overall_health()
        details['SystemHealth'] = health_status.value
        if health_status == HealthStatus.CRITICAL:
            logger.warning("AGI HEALTH: System in CRITICAL state. Forcing WAIT.")
            decision = "WAIT"
            final_score = 0
        
        # Heartbeat Log (Show activity even if WAIT)
        if decision == "WAIT":
            logger.info(f"Consensus Heartbeat: Trend={t_score:.0f} Sniper={s_score:.0f} Quant={q_score:.0f} Vector={total_vector:.2f}")
            # total_vector = 0 # REMOVED: We need the raw score for Veto logic in Laplace
            pass
        else:
            # ✅ Laplace V2 Integration: Return the normalized signed score
            direction_sign = 1 if decision == "BUY" else -1
            total_vector = final_score * direction_sign
            
        return decision, total_vector, details
