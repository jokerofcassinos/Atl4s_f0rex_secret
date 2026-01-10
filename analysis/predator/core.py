
import logging
import pandas as pd
from datetime import datetime

# Core PREDATOR Modules
from .liquidity import LiquidityEngineer
from .order_blocks import OrderBlockEngine
from .fvg import FVGTracker
from .time_fractal import TimeFractalEngine
from .displacement import DisplacementEngine

# Existing Systems (Legacy/Neural)
from analysis.deep_cognition import DeepCognition
from analysis.scalper_swarm import ScalpSwarm

logger = logging.getLogger("PredatorCore")

class PredatorCore:
    """
    PREDATOR Protocol v1.0 Core Engine
    
    Orchestrates the 9-Dimensional Confluence Matrix:
    1. Time Fractal (Kill Zones + M8)
    2. Liquidity (Accumulation/Sweep)
    3. Order Blocks (Institutional Footprint)
    4. FVG (Imbalance Magnets)
    5. Displacement (Commitment)
    6. MTF Confluence (H4/H1 Bias)
    7. Physics (Entropy/Energy) [Via DeepCognition]
    8. Neural Ensemble (DeepBrain + Swarm)
    9. Risk Geometry (Structure-based R:R)
    """
    
    def __init__(self):
        # Initialize Sub-Engines
        self.liquidity = LiquidityEngineer()
        self.order_blocks = OrderBlockEngine()
        self.fvg = FVGTracker()
        self.time_engine = TimeFractalEngine()
        self.displacement = DisplacementEngine()
        
        # Legacy/Neural Integration
        self.deep_brain = DeepCognition()
        self.swarm = ScalpSwarm()
        
        # Confluence Weights (Tunable)
        self.weights = {
            'time': 1.0,         # Fractal timing is prerequisite
            'liquidity': 1.5,    # Sweeps are high probability
            'order_block': 1.3,  # Structure backing
            'fvg': 1.0,          # Magnets
            'displacement': 1.2, # Momentum confirmation
            'mtf': 1.5,          # Trend alignment (H4/H1)
            'neural': 1.0        # AI Consensus
        }
        
    def detect_trend(self, df: pd.DataFrame) -> str:
        """Simple EMA Trend Detection"""
        if df is None or len(df) < 50: return "NEUTRAL"
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        price = df['close'].iloc[-1]
        
        if price > ema20 > ema50: return "BULLISH"
        if price < ema20 < ema50: return "BEARISH"
        return "NEUTRAL"

    def evaluate(self, data_map: dict, tick: dict, current_time: datetime) -> dict:
        """
        Main decision loop. 
        Returns {execute: bool, signal: str, score: float, breakdown: dict}
        """
        df_m1 = data_map.get('M1')
        df_m5 = data_map.get('M5')
        df_h1 = data_map.get('H1')
        df_h4 = data_map.get('H4')
        
        # Breakdown record
        dims = {}
        score = 0.0
        
        # === DIMENSION 1: TIME FRACTAL ===
        # First Gate: Is it a valid time to trade?
        time_res = self.time_engine.get_time_score(current_time)
        dims['time'] = time_res
        
        # Critical VETO: If Q1 Dead Zone or non-tradeable
        if not time_res['tradeable']:
            return {'execute': False, 'signal': 'WAIT', 'reason': f"Time Block: {time_res['reason']}"}
            
        score += time_res['score'] * self.weights['time']
        
        # === DIMENSION 6: MTF CONFLUENCE (Bias) ===
        # We check this early to establish directional bias
        h4_bias = self.detect_trend(df_h4)
        h1_bias = self.detect_trend(df_h1)
        
        mtf_bias = "NEUTRAL"
        if h4_bias == h1_bias and h4_bias != "NEUTRAL":
            mtf_bias = "BUY" if h4_bias == "BULLISH" else "SELL"
            score += 2.0 * self.weights['mtf']
        elif h4_bias != "NEUTRAL":
             # H4 is king, but weaker score if H1 disagrees
             mtf_bias = "BUY" if h4_bias == "BULLISH" else "SELL"
             score += 1.0 * self.weights['mtf']
             
        dims['mtf'] = {'h4': h4_bias, 'h1': h1_bias, 'bias': mtf_bias}
        
        # If we have no macro bias, we are very careful (scalp only)
        # But for now, let's allow it if local structure is strong
        
        # === DIMENSION 2: LIQUIDITY SWEEP ===
        # Check for sweeps in direction of bias (or reversal if bias neutral)
        # Actually liquidity sweep often REVERSES the micro trend to join Macro trend.
        
        # Let's check both directions
        sweep_buy = self.liquidity.detect_liquidity_sweep(df_m5, "BUY")
        sweep_sell = self.liquidity.detect_liquidity_sweep(df_m5, "SELL")
        
        target_dir = None
        
        if sweep_buy['detected']:
            dims['liquidity'] = sweep_buy
            score += 3.0 * self.weights['liquidity']
            target_dir = "BUY"
        elif sweep_sell['detected']:
            dims['liquidity'] = sweep_sell
            score += 3.0 * self.weights['liquidity']
            target_dir = "SELL"
        else:
            dims['liquidity'] = {'detected': False}
            
        # Conflict check: If Sweep says BUY but H4 says SELL, that's tricky.
        # Usually a sweep into H4 resistance is a great sell. 
        # A sweep of lows in an uptrend is a great buy.
        # So alignment is key.
        
        if target_dir and mtf_bias != "NEUTRAL" and target_dir != mtf_bias:
             # Counter-trend sweep? Risky.
             score -= 2.0 # Penalty
        
        # === DIMENSION 3: ORDER BLOCKS ===
        # Check for OB in the target direction
        ob_res = {'detected': False}
        if target_dir:
             ob_res = self.order_blocks.detect_valid_ob(df_m5, target_dir)
             if ob_res['detected']:
                 score += 2.0 * self.weights['order_block']
        
        dims['order_block'] = ob_res
        
        # === DIMENSION 4: FVG ENTRY ===
        # Check if we are entering a discount FVG (for buy) or premium FVG (for sell)
        fvgs = self.fvg.detect_fvg(df_m5)
        in_fvg = False
        if target_dir:
             in_fvg = self.fvg.price_entering_fvg(fvgs, tick['last'], target_dir)
             
        dims['fvg'] = {'in_fvg': in_fvg, 'count': len(fvgs)}
        if in_fvg:
            score += 1.5 * self.weights['fvg']
            
        # === DIMENSION 5: DISPLACEMENT ===
        # Has the move started?
        disp = self.displacement.detect_displacement(df_m5)
        dims['displacement'] = disp
        
        if disp['detected']:
            if target_dir and disp['direction'] == target_dir:
                score += 2.0 * self.weights['displacement']
            elif target_dir and disp['direction'] != target_dir:
                # Strong move against us? abort
                score -= 3.0
                
        # === DIMENSION 8: NEURAL ENSEMBLE ===
        # Consult the Deep Brain
        # We need to map our score to the 'trend_score' input it expects
        # It expects roughly 0-10, we have... well, let's just pass raw.
        
        neural_decision, state, future_prob, orbit, micro = self.deep_brain.consult_subconscious(
            trend_score=score / 2, # Normalize slightly
            volatility_score=0, # TODO: Add volatility engine
            pattern_score=score,
            smc_score=score,
            df_m5=df_m5,
            live_tick=tick
        )
        
        dims['neural'] = {'decision': neural_decision, 'state': state}
        
        # Neural Score Integration
        # Neural returns -1 to 1
        if target_dir == "BUY":
            if neural_decision > 0.3: score += 2.0 * self.weights['neural']
            elif neural_decision < -0.3: score -= 3.0
        elif target_dir == "SELL":
            if neural_decision < -0.3: score += 2.0 * self.weights['neural']
            elif neural_decision > 0.3: score -= 3.0
            
        # === FINAL DECISION ===
        
        # Threshold calculation
        # Baseline = 7.0
        threshold = 7.0
        
        # Dynamic adjustments
        if mtf_bias == "NEUTRAL": threshold += 2.0 # Need more evidence if no trend
        if score < 0: score = 0 # Floor
        
        # If we have no strong technical setup (No Sweep, No OB), it's hard to justify
        if not dims['liquidity']['detected'] and not dims['order_block']['detected']:
             # Unless Neural is screaming conviction?
             if abs(neural_decision) < 0.8:
                 # No Setup + No Strong AI = No Trade
                 return {'execute': False, 'signal': 'WAIT', 'score': score, 'reason': 'No Setup (Sweep/OB)'}

        # Direction determination (if not set by sweep)
        final_signal = target_dir if target_dir else ("BUY" if neural_decision > 0.5 else ("SELL" if neural_decision < -0.5 else "WAIT"))
        
        execute = score >= threshold and final_signal != "WAIT"
        
        return {
            'execute': execute,
            'signal': final_signal,
            'score': score,
            'threshold': threshold,
            'breakdown': dims,
            'reason': f"Score {score:.1f}/{threshold} | {dims['time']['reason']}"
        }
