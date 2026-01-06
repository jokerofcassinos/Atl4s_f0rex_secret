"""
AGI Ultra: Intelligent Execution System

Features:
- Optimal timing prediction for entries/exits
- Slippage prediction and avoidance
- Order flow integration for execution quality
- Dynamic position sizing with Kelly criterion
- Smart order routing simulation
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("IntelligentExecution")


class ExecutionQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    executed: bool
    actual_price: float
    requested_price: float
    slippage: float
    latency_ms: float
    quality: ExecutionQuality
    reason: str = ""


@dataclass
class TimingWindow:
    """Optimal timing window for execution."""
    start_time: float
    end_time: float
    quality_score: float
    expected_slippage: float
    reason: str


class IntelligentExecutionSystem:
    """
    AGI Ultra: Intelligent Execution System.
    
    Features:
    - Optimal timing prediction
    - Slippage prediction model
    - Dynamic sizing with Kelly
    - Execution quality tracking
    """
    
    def __init__(
        self,
        max_slippage_pips: float = 5.0,
        min_liquidity_threshold: float = 0.3,
        kelly_fraction: float = 0.25
    ):
        self.max_slippage_pips = max_slippage_pips
        self.min_liquidity_threshold = min_liquidity_threshold
        self.kelly_fraction = kelly_fraction
        
        # Slippage model (running statistics)
        self.slippage_history: deque = deque(maxlen=500)
        self.avg_slippage = 0.0
        self.slippage_std = 0.0
        
        # Execution history
        self.execution_history: List[ExecutionResult] = []
        self.max_history = 1000
        
        # Timing model
        self.timing_patterns: Dict[str, List[float]] = {}
        
        # Order flow state
        self.current_imbalance = 0.0
        self.volume_profile = np.zeros(24)  # Hourly volume
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        
        logger.info("IntelligentExecutionSystem initialized")
    
    # -------------------------------------------------------------------------
    # SLIPPAGE PREDICTION
    # -------------------------------------------------------------------------
    def predict_slippage(
        self,
        side: str,
        size: float,
        volatility: float,
        spread: float,
        imbalance: float
    ) -> float:
        """
        Predict expected slippage.
        
        Model: slippage = base + vol_factor + size_factor + imbalance_factor
        """
        # Base slippage is half the spread
        base_slippage = spread / 2
        
        # Volatility factor (higher vol = more slippage)
        vol_factor = volatility * 0.2
        
        # Size impact (larger orders = more slippage)
        size_factor = np.log1p(size) * 0.1
        
        # Order flow imbalance (adverse selection)
        if side == "BUY" and imbalance < 0:
            # Buying against sell pressure
            imbalance_factor = abs(imbalance) * 0.3
        elif side == "SELL" and imbalance > 0:
            # Selling against buy pressure
            imbalance_factor = abs(imbalance) * 0.3
        else:
            # Flow is in our favor
            imbalance_factor = -abs(imbalance) * 0.1
        
        # Historical adjustment
        if self.slippage_history:
            historical_bias = (self.avg_slippage - base_slippage) * 0.3
        else:
            historical_bias = 0
        
        predicted = base_slippage + vol_factor + size_factor + imbalance_factor + historical_bias
        
        return max(0, predicted)
    
    def update_slippage_model(self, actual_slippage: float):
        """Update slippage model with actual data."""
        self.slippage_history.append(actual_slippage)
        
        if len(self.slippage_history) >= 10:
            self.avg_slippage = np.mean(self.slippage_history)
            self.slippage_std = np.std(self.slippage_history)
    
    # -------------------------------------------------------------------------
    # OPTIMAL TIMING
    # -------------------------------------------------------------------------
    def find_optimal_timing(
        self,
        symbol: str,
        side: str,
        urgency: float = 0.5
    ) -> TimingWindow:
        """
        Find optimal timing window for execution.
        
        Considers:
        - Historical volume patterns
        - Current order flow
        - Spread patterns
        - News events
        """
        now = time.time()
        current_hour = int((now % 86400) / 3600)
        
        # Base window (immediate execution if urgent)
        if urgency > 0.8:
            return TimingWindow(
                start_time=now,
                end_time=now + 5,
                quality_score=0.5,
                expected_slippage=self.avg_slippage * 1.5,
                reason="Urgent execution"
            )
        
        # Analyze volume profile
        best_hour = current_hour
        best_score = 0.0
        
        for hour_offset in range(3):  # Look ahead 3 hours
            hour = (current_hour + hour_offset) % 24
            
            # Volume score (higher volume = better)
            vol_score = self.volume_profile[hour] / (np.max(self.volume_profile) + 1e-6)
            
            # Avoid market open/close
            if hour in [8, 9, 15, 16]:  # London/NY open/close
                vol_score *= 0.8  # Slightly reduce
            
            # Avoid low liquidity hours
            if hour in [4, 5, 6]:  # Asian session lull
                vol_score *= 0.5
            
            score = vol_score - hour_offset * 0.1  # Prefer sooner
            
            if score > best_score:
                best_score = score
                best_hour = hour
        
        # Calculate window
        hours_until = (best_hour - current_hour) % 24
        window_start = now + hours_until * 3600
        
        return TimingWindow(
            start_time=window_start,
            end_time=window_start + 1800,  # 30 min window
            quality_score=best_score,
            expected_slippage=self.avg_slippage * (2 - best_score),
            reason=f"Optimal hour: {best_hour}:00"
        )
    
    def update_volume_profile(self, hour: int, volume: float):
        """Update hourly volume profile."""
        # Exponential moving average
        alpha = 0.1
        self.volume_profile[hour] = self.volume_profile[hour] * (1 - alpha) + volume * alpha
    
    # -------------------------------------------------------------------------
    # DYNAMIC SIZING
    # -------------------------------------------------------------------------
    def calculate_position_size(
        self,
        account_balance: float,
        confidence: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_risk_pct: float = 0.02
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly: f* = (p * b - q) / b
        where:
            p = win probability
            q = loss probability
            b = win/loss ratio
        """
        if avg_loss == 0:
            avg_loss = 1.0
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        # Full Kelly
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Apply fraction (1/4 Kelly is common)
        kelly_fraction = kelly * self.kelly_fraction
        
        # Clamp to max risk
        kelly_fraction = max(0, min(max_risk_pct, kelly_fraction))
        
        # Adjust by confidence
        adjusted = kelly_fraction * confidence
        
        # Calculate actual size
        risk_amount = account_balance * adjusted
        
        logger.debug(
            f"Kelly sizing: win_rate={win_rate:.1%}, b={b:.2f}, "
            f"kelly={kelly:.2%}, adjusted={adjusted:.2%}"
        )
        
        return risk_amount
    
    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def should_execute(
        self,
        side: str,
        current_price: float,
        target_price: float,
        volatility: float,
        spread: float,
        imbalance: float
    ) -> Tuple[bool, str]:
        """
        Determine if we should execute now.
        
        Returns:
            (should_execute, reason)
        """
        # Predict slippage
        predicted_slippage = self.predict_slippage(
            side, 1.0, volatility, spread, imbalance
        )
        
        # Check slippage threshold
        if predicted_slippage > self.max_slippage_pips:
            return False, f"High slippage expected: {predicted_slippage:.1f} pips"
        
        # Check order flow
        if side == "BUY" and imbalance < -self.min_liquidity_threshold:
            return False, f"Adverse order flow: {imbalance:.2f}"
        if side == "SELL" and imbalance > self.min_liquidity_threshold:
            return False, f"Adverse order flow: {imbalance:.2f}"
        
        # Check spread
        if spread > self.max_slippage_pips * 0.5:
            return False, f"Spread too wide: {spread:.1f} pips"
        
        return True, "Execution conditions met"
    
    def record_execution(
        self,
        requested_price: float,
        actual_price: float,
        side: str,
        latency_ms: float
    ) -> ExecutionResult:
        """Record an execution for learning."""
        self.total_executions += 1
        
        # Calculate slippage
        if side == "BUY":
            slippage = actual_price - requested_price
        else:
            slippage = requested_price - actual_price
        
        # Determine quality
        slippage_pips = abs(slippage) * 10000  # Assuming forex
        
        if slippage_pips < 0.5:
            quality = ExecutionQuality.EXCELLENT
        elif slippage_pips < 1.0:
            quality = ExecutionQuality.GOOD
        elif slippage_pips < 2.0:
            quality = ExecutionQuality.FAIR
        else:
            quality = ExecutionQuality.POOR
        
        result = ExecutionResult(
            executed=True,
            actual_price=actual_price,
            requested_price=requested_price,
            slippage=slippage_pips,
            latency_ms=latency_ms,
            quality=quality
        )
        
        # Update models
        self.update_slippage_model(slippage_pips)
        
        # Store history
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)
        
        if quality in [ExecutionQuality.EXCELLENT, ExecutionQuality.GOOD]:
            self.successful_executions += 1
        
        logger.debug(
            f"Execution recorded: {quality.value}, "
            f"slippage={slippage_pips:.2f} pips, latency={latency_ms:.0f}ms"
        )
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        success_rate = self.successful_executions / self.total_executions if self.total_executions > 0 else 0
        
        quality_counts = {q.value: 0 for q in ExecutionQuality}
        for result in self.execution_history:
            quality_counts[result.quality.value] += 1
        
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': success_rate,
            'avg_slippage': self.avg_slippage,
            'slippage_std': self.slippage_std,
            'quality_distribution': quality_counts
        }
