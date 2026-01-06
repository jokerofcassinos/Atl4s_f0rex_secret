
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.black_swan_adversary import BlackSwanAdversary  # Reuse logic
from core.agi.thought_tree import ThoughtTree
from core.agi.decision_memory import ModuleDecisionMemory
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

logger = logging.getLogger("VetoSwarm")

class VetoSwarm(SubconsciousUnit):
    """
    The Censor. 
    A Swarm of 'No-sayers' that must be silenced for a trade to pass.
    
    Phase 6: Enhanced with recursive thinking and decision memory.
    """
    def __init__(self):
        super().__init__("Veto_Swarm")
        self.adversary = BlackSwanAdversary()
        
        # Phase 6: Recursive Thinking
        self.thought_tree = ThoughtTree("Veto_Swarm", max_depth=5)
        self.decision_memory = ModuleDecisionMemory("Veto_Swarm", max_memory=500)

        # Phase 9: Swarm-level AGI integration
        self.agi_adapter = AGISwarmAdapter("Veto_Swarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        # VetoSwarm runs slightly differently. It inspects the 'Proposed Decision' if available.
        # But 'process' is usually run in parallel before decision. 
        # So we VETO based on Market State, effectively flagging "Toxic Conditions".
        
        df_m5 = context.get('df_m5')
        market_state = context.get('market_state', {})
        
        if df_m5 is None: return None
        
        votes = []
        
        # 1. Stress Test Agent (Black Swan Monitor)
        # If market volatility is crazy high, pre-emptively VETO.
        # 1. Stress Test Agent (Black Swan Monitor)
        # If market volatility is crazy high, pre-emptively VETO.
        
        # Calculate ATR if missing
        if 'ATR' not in df_m5.columns:
            high_low = df_m5['high'] - df_m5['low']
            high_close = np.abs(df_m5['high'] - df_m5['close'].shift())
            low_close = np.abs(df_m5['low'] - df_m5['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            current_atr = atr.iloc[-1]
        else:
            current_atr = df_m5.iloc[-1]['ATR']
            
        atr_pct = (current_atr / df_m5.iloc[-1]['close']) * 100
        if atr_pct > 0.5: # Extreme Volatility (>0.5% M5 range)
             votes.append("VETO: Extreme Volatility (Black Swan Risk)")
             
        # 2. Reality Agent (Physics Check)
        close = df_m5['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if loss > 0:
            rsi = 100 - (100 / (1 + (gain/loss)))
            if rsi > 85: votes.append("VETO: Reality Check (RSI > 85 Overbought)")
            if rsi < 15: votes.append("VETO: Reality Check (RSI < 15 Oversold)")
            
        # 3. Trap Agent (Volume Disconnect)
        # Price Up, Volume Down = Trap
        price_trend = df_m5['close'].iloc[-1] - df_m5['close'].iloc[-5]
        vol_trend = df_m5['volume'].iloc[-1] - df_m5['volume'].iloc[-5]
        
        # Simple divergence check
        if abs(price_trend) > 0 and (np.sign(price_trend) != np.sign(vol_trend)):
             # Rising Price, Falling Volume?
             if price_trend > 0 and vol_trend < 0:
                 pass # Warning, but maybe not hard veto unless extreme
                 
        # 4. Weekend Guard (Dynamic)
        # Check if today is Saturday(5) or Sunday(6)
        current_time = pd.Timestamp.now()
        day_of_week = current_time.dayofweek # 0=Mon, 6=Sun
        
        # Check Config for "Weekend Mode" implied by Profile
        # If Virtual SL is wide (>15) it's likely Crypto/Weekend profile.
        # Or check spread_limit (0.05 vs 0.02).
        config = context.get('config', {})
        is_crypto_profile = config.get('spread_limit', 0.0) >= 0.04
        
        if day_of_week >= 5: # Saturday or Sunday
            if not is_crypto_profile:
                 symbol = context.get('symbol', 'UNKNOWN')
                 votes.append(f"VETO: Market Closed (Weekend) for {symbol} (Profile: Forex)")
            else:
                 # It is Crypto Profile, allow it.
                 pass
                 
        # Phase 6: Recursive Thinking about each veto
        # Perguntas recursivas: "Por que veto?", "Foi correto?", "Qual contexto?"
        veto_reasoning = {}
        
        if votes:
            reason = " | ".join(votes)
            
            # Cria árvore de pensamento para este veto
            root_node_id = self.thought_tree.create_node(
                question="Why am I vetoing this trade?",
                context={'votes': votes, 'market_state': market_state},
                confidence=1.0
            )
            
            # Responde com os motivos
            self.thought_tree.answer_node(root_node_id, reason, confidence=1.0)
            
            # Pergunta recursiva 1: "Foi correto?"
            child1_id = self.thought_tree.create_node(
                question="Was this veto correct?",
                parent_id=root_node_id,
                context={'veto_reason': reason}
            )
            
            # Busca vetos similares no passado
            similar_vetos = self.decision_memory.find_similar_decisions(
                {'votes': votes, 'market_state': market_state}, limit=5
            )
            
            if similar_vetos:
                # Analisa se os vetos passados foram corretos
                correct_vetos = [v for v in similar_vetos if v.result == "WIN" or v.result == "BREAKEVEN"]
                correctness_rate = len(correct_vetos) / len(similar_vetos) if similar_vetos else 0.0
                answer1 = f"Similar past vetos had {correctness_rate:.1%} correctness rate"
                self.thought_tree.answer_node(child1_id, answer1, confidence=correctness_rate)
            else:
                self.thought_tree.answer_node(child1_id, "No similar past vetos found", confidence=0.5)
            
            # Pergunta recursiva 2: "Qual contexto?"
            child2_id = self.thought_tree.create_node(
                question="What is the context of this veto?",
                parent_id=root_node_id,
                context={'votes': votes}
            )
            
            context_analysis = f"Market state: {market_state}. Veto reasons: {len(votes)} conditions detected."
            self.thought_tree.answer_node(child2_id, context_analysis, confidence=0.8)
            
            # Registra veto na memória
            veto_id = self.decision_memory.record_decision(
                decision="VETO",
                score=-100.0,  # Veto sempre tem score negativo
                context={'votes': votes, 'market_state': market_state, 'df_m5_length': len(df_m5) if df_m5 is not None else 0},
                reasoning=reason,
                confidence=1.0
            )
            
            # Adiciona perguntas recursivas à memória
            self.decision_memory.add_recursive_question(veto_id, "Why did I veto?", reason)
            self.decision_memory.add_recursive_question(veto_id, "Was this correct?", 
                                                       answer1 if similar_vetos else "Unknown")
            self.decision_memory.add_recursive_question(veto_id, "What is the context?", context_analysis)

            # Phase 9: Swarm-level thought integration in global AGI
            symbol = context.get("symbol", "UNKNOWN")
            timeframe = context.get("timeframe", "M5")
            market_snapshot = {
                "price": float(df_m5["close"].iloc[-1]),
                "atr_pct": float(atr_pct),
            }
            swarm_output = {
                "decision": "VETO",
                "score": -100.0,
                "votes": votes,
                "reason": reason,
            }
            swarm_thought: SwarmThoughtResult = self.agi_adapter.think_on_swarm_output(
                symbol=symbol,
                timeframe=timeframe,
                market_state={**market_state, **market_snapshot},
                swarm_output=swarm_output,
            )

            logger.info(f"VETO SWARM BLOCK: {reason}")
            logger.debug(
                "Phase 9: VetoSwarm thought root=%s scenario_count=%s",
                swarm_thought.thought_root_id,
                swarm_thought.meta.get("scenario_count"),
            )

            return SwarmSignal(
                source=self.name,
                signal_type="VETO",
                confidence=100.0,
                timestamp=0,
                meta_data={
                    'reason': reason,
                    'thought_tree_nodes': len(self.thought_tree.nodes),
                    'recursive_analysis': {
                        'similar_past_vetos': len(similar_vetos),
                        'correctness_rate': len([v for v in similar_vetos if v.result == "WIN"]) / len(similar_vetos) if similar_vetos else 0.0
                    },
                    'agi_thought_root_id': swarm_thought.thought_root_id,
                    'agi_scenarios': swarm_thought.meta.get("scenario_count", 0),
                }
            )
        
        # Se não há veto, também registra (para aprendizado)
        no_veto_id = self.decision_memory.record_decision(
            decision="ALLOW",
            score=0.0,
            context={'market_state': market_state, 'df_m5_length': len(df_m5) if df_m5 is not None else 0},
            reasoning="No veto conditions detected",
            confidence=0.5
        )
            
        return None
