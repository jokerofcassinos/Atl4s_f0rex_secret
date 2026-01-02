import logging
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

    def update_parameters(self, new_params):
        """Update weights and thresholds for optimization"""
        self.params.update(new_params)

    def update_ticks(self, tick):
        """Pass live tick to micro module"""
        self.micro.on_tick(tick)

    def deliberate(self, data_map, verbose=True):
        """
        Aggregates votes from all sub-modules using Dynamic Regime Weights.
        Runs analysis in parallel threads for speed.
        data_map: {'M5': df_m5, 'H1': df_h1}
        """
        df_m5 = data_map['M5']
        df_h1 = data_map.get('H1')
        
        details = {}
        
        # Micro-Structure Analysis (Instant)
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
        tasks = {
            'Trend': lambda: self.trend.analyze(df_m5, df_h1),
            'Sniper': lambda: self.sniper.analyze(df_m5),
            'Quant': lambda: self.quant.analyze(df_m5),
            'Patterns': lambda: self.patterns.analyze(df_m5),
            'Cycle': lambda: self.cycle.analyze(df_m5),
            'SupplyDemand': lambda: self.supply_demand.analyze(df_m5),
            'Divergence': lambda: self.divergence.analyze(df_m5),
            'Volatility': lambda: self.volatility.analyze(df_m5),
            'Kinematics': lambda: self.kinematics.analyze(df_m5),
            'Fractal': lambda: self.fractal.analyze(df_h1, data_map.get('H4')),
            'Math': lambda: self.math.analyze(df_m5),
            'Quantum': lambda: self.quantum.analyze(df_m5, []), 
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
            'Singularity': lambda: self.singularity.deliberate(data_map)
        }
        
        results = {}
        
        # Execute in parallel
        # Execute in parallel using persistent executor
        future_to_task = {self.executor.submit(func): name for name, func in tasks.items()}
        for future in as_completed(future_to_task):
            name = future_to_task[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Analysis Error in {name}: {e}")
                # We don't set anything so results.get(name, default) works

        # Unpack Results
        # Trend
        trend_res = results.get('Trend')
        if trend_res is None: trend_res = {'score': 0, 'direction': 0, 'regime': 'RANGING', 'river': 0}
        t_score = trend_res['score']
        t_dir = trend_res['direction']
        regime = trend_res['regime']
        river = trend_res['river']
        details['Trend'] = {'score': t_score, 'dir': t_dir, 'regime': regime, 'river': river}
        
        # Sniper
        sniper_res = results.get('Sniper')
        if sniper_res is None: sniper_res = (0, 0)
        s_score, s_dir = sniper_res
        details['Sniper'] = {'score': s_score, 'dir': s_dir}
        
        # Quant
        quant_res = results.get('Quant')
        if quant_res is None: quant_res = (0, 0, "NEUTRAL")
        q_score, q_dir, q_type = quant_res
        
        details['Quant'] = {'score': q_score, 'dir': q_dir}
        
        # Patterns
        patterns_res = results.get('Patterns')
        if patterns_res is None: patterns_res = (0, 0, "None")
        p_score, p_dir, p_name = patterns_res
        details['Patterns'] = {'score': p_score, 'dir': p_dir, 'name': p_name}
        
        # Cycle
        cycle_res = results.get('Cycle')
        if cycle_res is None: cycle_res = ("NEUTRAL", 0)
        c_phase, c_score = cycle_res
        c_dir = 0
        if c_phase == "MANIPULATION_BUY": c_dir = 1
        elif c_phase == "MANIPULATION_SELL": c_dir = -1
        details['Cycle'] = {'phase': c_phase, 'score': c_score, 'dir': c_dir}
        
        # SupplyDemand
        sd_res = results.get('SupplyDemand')
        if sd_res is None: sd_res = (0, 0, {})
        sd_score, sd_dir, sd_info = sd_res
        details['SupplyDemand'] = {'score': sd_score, 'dir': sd_dir, 'info': sd_info}
        
        # Divergence
        div_res = results.get('Divergence')
        if div_res is None: div_res = (0, 0, "None")
        d_score, d_dir, d_type = div_res
        details['Divergence'] = {'score': d_score, 'dir': d_dir, 'type': d_type}
        
        # Volatility
        vol_res = results.get('Volatility')
        if vol_res is None: vol_res = (0, 0)
        v_score, v_dir = vol_res
        details['Volatility'] = {'score': v_score, 'dir': v_dir}
        
        # Kinematics
        kin_res = results.get('Kinematics')
        if kin_res is None: kin_res = (0, 0, 0, 0, 0)
        k_vel, k_acc, k_score, k_angle, k_energy = kin_res
        k_dir = 1 if k_score > 0 else -1 if k_score < 0 else 0
        details['Kinematics'] = {'vel': k_vel, 'acc': k_acc, 'score': abs(k_score), 'angle': k_angle, 'energy': k_energy}
        
        # Fractal Vision
        fractal_res = results.get('Fractal')
        if fractal_res is None: fractal_res = {'score': 0, 'h4_structure': 'NEUTRAL'}
        f_score = fractal_res['score']
        f_dir = 1 if f_score > 0 else -1 if f_score < 0 else 0
        details['Fractal'] = fractal_res
        
        # Math Core
        math_res = results.get('Math')
        if math_res is None: math_res = {'regime_prob': 0.5, 'hurst': 0.5, 'entropy': 0, 'kalman_diff': 0}
        regime_prob = math_res['regime_prob']
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
        q_tunnel = quantum_res['tunneling_prob']
        is_excited = quantum_res.get('is_excited', False)
        details['Quantum'] = quantum_res
        
        # Cortex Memory
        cortex_prob = results.get('Cortex')
        if cortex_prob is None: cortex_prob = 0.5
        details['Cortex'] = {'bullish_prob': cortex_prob}
        
        # Prediction Engine (Monte Carlo)
        pred_res = results.get('Prediction')
        if pred_res is None: pred_res = {'prob_bullish': 0.5, 'skew': 0}
        mc_bullish = pred_res['prob_bullish']
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
        # 1. Sniper Level + Reversal Pattern + Divergence
        if s_score > 50 and p_name != "None" and d_type != "None":
            if s_dir == p_dir == d_dir:
                logger.info(f"GOLDEN SETUP DETECTED: Sniper + {p_name} + {d_type}. FULL SEND!")
                decision = "BUY" if s_dir == 1 else "SELL"
                # Dynamic Score: Base 95 + excess Sniper Score
                base_dynamic = 95.0 + (s_score - 50) * 0.1
                score = base_dynamic if decision == "BUY" else -base_dynamic
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

        # Calculate Base Vector (Foundational Modules)
        v_trend = t_score * t_dir * w_trend
        v_sniper = s_score * s_dir * w_sniper
        v_quant = q_score * q_dir * w_quant
        v_pattern = (p_score + confluence_boost) * p_dir * w_pattern
        v_cycle = c_score * c_dir * w_cycle
        v_sd = sd_score * sd_dir * w_sd
        v_div = d_score * d_dir * w_div
        v_kin = abs(k_score) * k_dir * w_kin
        v_fractal = abs(f_score) * f_dir * w_fractal
        
        prelim_vector = v_trend + v_sniper + v_quant + v_pattern + v_cycle + v_sd + v_div + v_kin + v_fractal + (sc_impact * 20)
        
        # --- PHASE 8: MULTI-EYE ALIGNMENT BOOSTS (Contextual Authority) ---
        # 15. Oracle Swing Alignment
        if oracle_dir != 0:
            if (oracle_dir == 1 and prelim_vector > 0) or (oracle_dir == -1 and prelim_vector < 0):
                prelim_vector += 30 * oracle_dir
                logger.info(f"CONFLUENCE: Oracle Swing Alignment! Global Bias {oracle_res['decision']} confirms trade.")
            else:
                prelim_vector -= 20 * oracle_dir
                logger.warning(f"DIVERGENCE: Oracle Swing Bias {oracle_res['decision']} opposes prelim vector. Reducing confidence.")

        # 16. Council Position Alignment (The Macro Anchor)
        if council_dir != 0:
            if (council_dir == 1 and prelim_vector > 0) or (council_dir == -1 and prelim_vector < 0):
                boost = 50 if "STRONG" in council_anchor else 30
                prelim_vector += boost * council_dir
                logger.info(f"CONFLUENCE: Council Position Alignment! Secular Anchor {council_anchor} confirms direction.")
            else:
                penalty = 40 if "STRONG" in council_anchor else 20
                prelim_vector -= penalty * council_dir
                logger.warning(f"DANGER: Major Divergence! Secular Anchor {council_anchor} opposes trade direction.")
                
        # 17. Overlord Meta-Directive
        if overlord_dir != 0:
            if (overlord_dir == 1 and prelim_vector > 0) or (overlord_dir == -1 and prelim_vector < 0):
                boost = 60 if abs(overlord_score) > 80 else 30
                prelim_vector += boost * overlord_dir
                logger.info(f"CONFLUENCE: Overlord Master Directive {overlord_res['decision']} confirms trade. Authority Boost: {boost}")
            else:
                prelim_vector -= 40 * overlord_dir
                logger.warning(f"CRITICAL DIVERGENCE: Overlord Directive {overlord_res['decision']} opposes consensus! Reducing power.")
                
        # 18. Sovereign Alignment (The Universal Law)
        if sov_dir != 0:
            if (sov_dir == 1 and prelim_vector > 0) or (sov_dir == -1 and prelim_vector < 0):
                boost = 80 if "STRONG" in sovereign_res['decision'] else 40
                prelim_vector += boost * sov_dir
                logger.info(f"SOVEREIGN ALIGNMENT: {sovereign_res['decision']}. Universal coherence triggers major confidence boost (+{boost}).")
            else:
                 prelim_vector -= 60 * sov_dir
                 logger.warning(f"SOVEREIGN VETO: Multi-scale fractals oppose entry direction! DANGER.")
                 
        # 19. Singularity Convergence
        if is_singularity:
            prelim_vector += 100 * (1 if prelim_vector > 0 else -1 if prelim_vector < 0 else 0)
            logger.info("SINGULARITY DETECTED: Geometric Manifold has folded. Absolute Certainty Surge (+100).")

        total_vector = prelim_vector

        # Volatility Gatekeeper
        if v_score == 0:
            logger.info("Volatility Guard: Market too quiet. Veto.")
            return "WAIT", 0, details
        
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
        logger.debug(f"Consensus Breakdown: Trend={v_trend:.2f} Sniper={v_sniper:.2f} Quant={v_quant:.2f} Pat={v_pattern:.2f} Cycle={v_cycle:.2f} SD={v_sd:.2f} Div={v_div:.2f} Kin={v_kin:.2f} Frac={v_fractal:.2f} | Health={health_score}")
        
        final_score = abs(total_vector)
        
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
        if final_score > threshold: 
            if total_vector > 0:
                decision = "BUY"
            else:
                decision = "SELL"
                
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
            
        # Heartbeat Log (Show activity even if WAIT)
        if decision == "WAIT":
            logger.info(f"Consensus Heartbeat: Trend={t_score:.0f} Sniper={s_score:.0f} Quant={q_score:.0f} Vector={total_vector:.2f}")
            
        return decision, total_vector, details
