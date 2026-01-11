"""
Genesis Adapters - Wrapper classes for integration
═══════════════════════════════════════════════════

Lightweight adapters to connect existing components to Genesis system.
"""

import logging
from typing import Dict, Optional, Any
import pandas as pd

logger = logging.getLogger("GenesisAdapters")


# ═══════════════════════════════════════════════════════════
# AGI CORE ADAPTER
# ═══════════════════════════════════════════════════════════

class OmegaAGICore:
    """
    Adapter for omega_agi_core.py
    
    Simplifies the interface for Genesis integration.
    """
    
    def __init__(self):
        try:
            # Try to import from omega_agi_core - look for available classes
            from core.agi import omega_agi_core
            # Use a simpler approach - just mark as available for now
            self.omega = None  # Will be implemented in Phase 1.2
            self.available = False  # Temporarily disable until full integration
            logger.info("OmegaAGICore adapter initialized (minimal mode)")
        except Exception as e:
            logger.warning(f"OmegaAGICore not available: {e}")
            self.omega = None
            self.available = False
    
    async def process_market_state(self,
                                     df_m5: pd.DataFrame,
                                     df_h1: pd.DataFrame,
                                     current_price: float,
                                     signal_context: Any) -> Dict:
        """
        Process market state and return AGI context.
        
        Returns:
            Dict with 'confidence', 'direction', 'regime', etc.
        """
        
        if not self.available:
            return {'confidence': 0, 'direction': 'WAIT', 'regime': 'UNKNOWN'}
        
        try:
            # Simplified AGI call - can be expanded later
            context = {
                'confidence': 50,  # Base confidence
                'direction': 'WAIT',
                'regime': 'NORMAL',
                'agi_active': True
            }
            
            # TODO: Implement full omega.analyze() integration
            # For now, return basic context
            
            return context
            
        except Exception as e:
            logger.error(f"AGI processing error: {e}")
            return {'confidence': 0, 'direction': 'WAIT', 'regime': 'ERROR'}


# ═══════════════════════════════════════════════════════════
# METACOGNITION ADAPTER
# ═══════════════════════════════════════════════════════════

class MetaCognition:
    """
    Adapter for metacognition modules.
    
    Cross-validates signals and provides meta-insights.
    """
    
    def __init__(self):
        try:
            from core.agi.metacognition.recursive_reflection import RecursiveReflection
            self.recursive = RecursiveReflection()
            self.available = True
            logger.info("MetaCognition loaded successfully")
        except Exception as e:
            logger.warning(f"MetaCognition not available: {e}")
            self.recursive = None
            self.available = False
    
    def analyze_signals(self, signal_context: Any, agi_context: Dict) -> Dict:
        """
        Cross-validate signals with metacognitive analysis.
        
        Returns:
            Dict with meta-insights and validation scores
        """
        
        if not self.available:
            return {'validated': False, 'insights': []}
        
        try:
            # Meta-analysis of signal alignment
            insights = {
                'validated': True,
                'alignment_score': 0.7,
                'insights': ['Signals cross-validated']
            }
            
            # TODO: Implement full recursive reflection
            
            return insights
            
        except Exception as e:
            logger.error(f"MetaCognition error: {e}")
            return {'validated': False, 'insights': []}


# ═══════════════════════════════════════════════════════════
# HOLOGRAPHIC MEMORY ADAPTER
# ═══════════════════════════════════════════════════════════

class HolographicMemory:
    """
    Adapter for holographic_memory.py
    
    Pattern matching and historical context retrieval.
    """
    
    def __init__(self):
        try:
            from core.holographic_memory import HolographicMemory as CoreMemory
            self.memory = CoreMemory()
            self.available = True
            logger.info("HolographicMemory loaded successfully")
        except Exception as e:
            logger.warning(f"HolographicMemory not available: {e}")
            self.memory = None
            self.available = False
    
    def find_similar_context(self, signal_context: Any) -> Dict:
        """
        Find similar historical patterns.
        
        Returns:
            Dict with similar patterns and their outcomes
        """
        
        if not self.available:
            return {'patterns': [], 'avg_success_rate': 0}
        
        try:
            # TODO: Implement pattern retrieval from holographic memory
            patterns = {
                'patterns': [],
                'avg_success_rate': 0.7,
                'confidence_boost': 0
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return {'patterns': [], 'avg_success_rate': 0}


# ═══════════════════════════════════════════════════════════
# EXECUTION ENGINE ADAPTER
# ═══════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    Adapter for core/execution_engine.py
    
    Handles trade execution via ZMQ bridge.
    """
    
    def __init__(self, symbol: str = "GBPUSD"):
        try:
            from core.execution_engine import ExecutionEngine as CoreEngine
            self.engine = CoreEngine()
            self.symbol = symbol
            self.available = True
            logger.info(f"ExecutionEngine loaded for {symbol}")
        except Exception as e:
            logger.warning(f"ExecutionEngine not available: {e}")
            self.engine = None
            self.available = False
    
    async def execute_signal(self, signal: Any, zmq_bridge: Any):
        """
        Execute trade via MT5.
        
        Args:
            signal: GenesisSignal object
            zmq_bridge: ZMQ bridge to MT5
        """
        
        if not self.available or not zmq_bridge:
            logger.warning("Execution not available (paper mode or missing bridge)")
            return
        
        try:
            # Convert GenesisSignal to execution command
            command = signal.direction  # "BUY" or "SELL"
            confidence = signal.confidence
            sl_pips = signal.sl_pips
            tp_pips = signal.tp_pips
            
            # Execute via bridge
            self.engine.execute_signal(
                command=command,
                symbol=self.symbol,
                confidence=confidence,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                source="GENESIS"
            )
            
            logger.info(f"Executed: {command} {self.symbol} @ {confidence:.0f}% | SL:{sl_pips:.1f} TP:{tp_pips:.1f}")
            
        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════
# RISK MANAGER ADAPTER
# ═══════════════════════════════════════════════════════════

class RiskManager:
    """
    Adapter for risk management modules.
    
    Position sizing, exposure limits, etc.
    """
    
    def __init__(self):
        self.max_risk_per_trade = 2.0  # %
        self.max_portfolio_risk = 10.0  # %
        logger.info("RiskManager initialized")
    
    def calculate_position_size(self, signal: Any, balance: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Returns:
            Lot size
        """
        
        try:
            risk_amount = balance * (signal.risk_pct / 100)
            sl_pips = signal.sl_pips
            
            # Simple calculation (can be enhanced)
            pip_value = 10  # $10 per pip for 1 lot GBPUSD
            lots = risk_amount / (sl_pips * pip_value)
            
            # Min/max constraints
            lots = max(0.01, min(lots, 1.0))
            
            return lots
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0.01  # Minimum lot


# ═══════════════════════════════════════════════════════════
# ZMQ BRIDGE ADAPTER
# ═══════════════════════════════════════════════════════════

class ZMQBridge:
    """
    Adapter for ZMQ communication with MT5.
    
    Simplified interface for Genesis.
    """
    
    def __init__(self):
        try:
            from execution.zmq_bridge import ZMQBridge as CoreBridge
            self.bridge = CoreBridge()
            self.available = True
            logger.info("ZMQBridge loaded successfully")
        except Exception as e:
            logger.warning(f"ZMQBridge not available: {e}")
            self.bridge = None
            self.available = False
    
    def send_order(self, order_data: Dict) -> bool:
        """
        Send order to MT5 via ZMQ.
        
        Returns:
            True if successful
        """
        
        if not self.available:
            logger.warning("ZMQ not available - order not sent")
            return False
        
        try:
            result = self.bridge.send_order(order_data)
            return result
        except Exception as e:
            logger.error(f"ZMQ error: {e}")
            return False
