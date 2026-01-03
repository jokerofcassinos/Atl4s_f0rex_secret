
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("OpportunityFlow")

@dataclass
class AssetFocus:
    symbol: str
    volatility_score: float # 0.0 to 100.0 (ATR/Volume based)
    trend_clarity: float    # 0.0 to 100.0 (ADX/Slope based)
    liquidity_rank: int     # 1, 2, 3
    is_tradable: bool

class OpportunityFlowManager:
    """
    The All-Seeing Eye's Lens.
    Allocates 'Attention' (CPU & Capital) to the most profitable flow.
    """
    def __init__(self):
        self.focus_scores: Dict[str, AssetFocus] = {}
        self.active_symbols = ["XAUUSD", "BTCXAU", "XAUAUD"]
        self.current_focus = "XAUUSD" # Default

    def calculate_focus(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """
        Determines which asset deserves the 'Main Brain' attention.
        Args:
            market_data: Dictionary mapping symbol -> DF_M5
        Returns:
            The symbol to focus on.
        """
        best_score = -1.0
        best_symbol = self.current_focus

        for symbol in self.active_symbols:
            if symbol not in market_data or market_data[symbol].empty:
                continue
            
            df = market_data[symbol]
            if len(df) < 50: continue

            # 1. Volatility (Opportunity Size)
            # Normalized ATR (ATR / Close Price) * 10000 to get basis points equivalent
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 1.0
            close = df['close'].iloc[-1]
            norm_vol = (atr / close) * 10000 
            
            # 2. Trend Clarity (Probability of Clean Move)
            # Simple Slope Check or ADX if available
            # Using simple close-to-close deviation sum for now as proxy
            ma_20 = df['close'].rolling(20).mean().iloc[-1]
            deviation = abs(close - ma_20)
            trend_score = (deviation / close) * 10000

            # Composite Score
            # We want High Volatility AND High Trend (Breakout)
            # Or High Volatility + Low Trend (Range Scalping), but Breakout is better for 1:3000
            
            total_score = (norm_vol * 1.5) + (trend_score * 1.0)
            
            self.focus_scores[symbol] = AssetFocus(
                symbol=symbol,
                volatility_score=norm_vol,
                trend_clarity=trend_score,
                liquidity_rank=1, # Todo: hook up to Spread monitor
                is_tradable=True
            )
            
            if total_score > best_score:
                best_score = total_score
                best_symbol = symbol

        # Hysteresis: Don't switch unless new score is 10% better to avoid thrashing
        old_score = self._get_score(self.current_focus)
        
        if best_score > (old_score * 1.1):
            if self.current_focus != best_symbol:
                logger.info(f"FOCUS SHIFT: {self.current_focus} -> {best_symbol} (Score: {best_score:.1f})")
            self.current_focus = best_symbol
            
        return self.current_focus

    def _get_score(self, symbol: str) -> float:
        if symbol in self.focus_scores:
            f = self.focus_scores[symbol]
            return (f.volatility_score * 1.5) + (f.trend_clarity * 1.0)
        return 0.0

    def get_current_focus(self) -> str:
        return self.current_focus

