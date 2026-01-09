import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

# Configure logging
logger = logging.getLogger("CrisisManager")

@dataclass
class SymbolCrisisState:
    symbol: str
    daily_pnl: float = 0.0
    consecutive_losses_long: int = 0
    consecutive_losses_short: int = 0
    is_inverted_long: bool = False  # If True, Long signals become Short
    is_inverted_short: bool = False # If True, Short signals become Long
    last_drawdown_high: float = 0.0 # Peak equity for this symbol

class CrisisManager:
    def __init__(self):
        self.states: Dict[str, SymbolCrisisState] = {}
        # Vampire Mode Settings (Tiered)
        self.tier1_trigger = 0.60  # Profit to trigger Half-Stop
        self.tier2_trigger = 1.20  # Profit to trigger BreakEven
        
        # Inversion Settings
        self.inversion_threshold = 3 # Consecutive losses to trigger inversion
        
        logger.info("Crisis Manager Initialized: Smart Vampire & Inverse Polarity Active.")

    def get_state(self, symbol: str) -> SymbolCrisisState:
        if symbol not in self.states:
            self.states[symbol] = SymbolCrisisState(symbol=symbol)
        return self.states[symbol]

    def register_trade_result(self, symbol: str, profit: float, direction: str, structure_intact: bool = False):
        """
        Updates logic based on closed trade results.
        PHASE 24: Relativity Engine (Context-Aware Inversion).
        """
        state = self.get_state(symbol)
        state.daily_pnl += profit
        
        # Update Consecutive Losses / Inversion Logic
        if profit < 0:
            # IT WAS A LOSS
            if direction == 'BUY':
                state.consecutive_losses_long += 1
                state.consecutive_losses_short = 0 # Reset opposite
                if state.consecutive_losses_long >= self.inversion_threshold:
                    # PHASE 24: RELATIVITY CHECK
                    if structure_intact:
                        logger.info(f"🧱 ACCUMULATION DETECTED: {symbol} Longs failing ({state.consecutive_losses_long}x) buy Structure Intact. VETOING Inversion to allow Accumulation.")
                    else:
                        state.is_inverted_long = True
                        logger.warning(f"🚨 INVERSION TRIGGERED: {symbol} Longs failed {state.consecutive_losses_long}x AND Structure Broken. Flipping Long Bias to SHORT.")
                        
            elif direction == 'SELL':
                state.consecutive_losses_short += 1
                state.consecutive_losses_long = 0 # Reset opposite
                if state.consecutive_losses_short >= self.inversion_threshold:
                    # PHASE 24: RELATIVITY CHECK
                    if structure_intact:
                        logger.info(f"🧱 DISTRIBUTION DETECTED: {symbol} Shorts failing ({state.consecutive_losses_short}x) but Structure Intact. VETOING Inversion to allow Accumulation.")
                    else:
                        state.is_inverted_short = True
                        logger.warning(f"🚨 INVERSION TRIGGERED: {symbol} Shorts failed {state.consecutive_losses_short}x AND Structure Broken. Flipping Short Bias to LONG.")
        else:
            # IT WAS A WIN - Reset counters and inversion
            if direction == 'BUY':
                state.consecutive_losses_long = 0
                if state.is_inverted_long:
                    state.is_inverted_long = False
                    logger.info(f"✅ INVERSION RESET: {symbol} Longs are working again.")
            elif direction == 'SELL':
                state.consecutive_losses_short = 0
                if state.is_inverted_short:
                    state.is_inverted_short = False
                    logger.info(f"✅ INVERSION RESET: {symbol} Shorts are working again.")

    def check_vampire_mode(self, ticket: int, current_profit: float, open_price: float, current_sl: float, direction: int) -> Optional[float]:
        """
        Applies Smart Tiered Vampire Logic.
        Returns NEW SL price if adjustment is needed, else None.
        direction: 0 for Buy, 1 for Sell (MT5 standard)
        """
        # We need point size to calculate price moves roughly, but here we deal with raw profit $
        # Ideally we act on Profit $ primarily as requested.
        
        new_sl = None
        
        # TIER 2: BreakEven (Risk Free)
        if current_profit >= self.tier2_trigger:
            # Calculate BE Price (+ small buffer)
            # Assuming we accept slight slippage, we want effectively Open Price
            # MT5 BE is usually OpenPrice +/- small delta
            
            # Simple heuristic: If we haven't moved SL to at least BE yet
            # Check if current SL is worse than Open Price
            
            is_buy = (direction == 0)
            
            # Target BE Price (Open +/- tiny buffer)
            # Since we don't have point size easily, we just use open price
            target_sl = open_price 
            
            if is_buy:
                # If SL is below Open, move to Open
                if current_sl < target_sl:
                    new_sl = target_sl
                    logger.info(f"🧛 VAMPIRE TIER 2 (BE): Ticket {ticket} Profit ${current_profit:.2f}. Moving SL to {new_sl}")
            else: # Sell
                # If SL is above Open, move to Open
                if current_sl > target_sl:
                    new_sl = target_sl
                    logger.info(f"🧛 VAMPIRE TIER 2 (BE): Ticket {ticket} Profit ${current_profit:.2f}. Moving SL to {new_sl}")

        # TIER 1: Half-Stop (Risk Reduction)
        elif current_profit >= self.tier1_trigger:
            # Reduce Risk by 50%
            # New SL = (Open + Current_SL) / 2
            
            is_buy = (direction == 0)
            mid_point = (open_price + current_sl) / 2
            
            if is_buy:
                # Only move UP
                if mid_point > current_sl:
                     # Check if we are already better than mid_point (e.g. from Trailing)
                     # For now, just simplistic check
                     new_sl = mid_point
                     logger.info(f"🧛 VAMPIRE TIER 1 (Half-Risk): Ticket {ticket} Profit ${current_profit:.2f}. Moving SL to {new_sl}")
            else: # Sell
                # Only move DOWN
                if mid_point < current_sl:
                    new_sl = mid_point
                    logger.info(f"🧛 VAMPIRE TIER 1 (Half-Risk): Ticket {ticket} Profit ${current_profit:.2f}. Moving SL to {new_sl}")
                    
        return new_sl

    def check_equity_shield(self, symbol: str, trades: List[Dict], structure_intact: bool = False, max_dd_limit: float = -300.0, oversold_severity: float = 0.0) -> Optional[int]:
        """
        PHASE 23.5: Elastic Equity Shield.
        PHASE 23.6: Rubber Band Protocol (Estilingue).
        
        Args:
            structure_intact: If True, allow 20% Grace Zone.
            max_dd_limit: Base limit (e.g. -300).
            oversold_severity: 0.0 to 1.0. High value means Extreme Oversold (Rubber Band potential).
                               If > 0.0, we expand the limit significantly to catch the V-Shape.
        """
        # 1. Calculate Total Floating PNL for this Symbol
        symbol_trades = [t for t in trades if t.get('symbol') == symbol]
        if not symbol_trades: return None
        
        total_pnl = sum([float(t.get('profit', 0.0)) for t in symbol_trades])
        
        # 2. Determine Dynamic Limit
        effective_limit = max_dd_limit
        
        # A. Rubber Band Logic (Priority over Structure)
        # If market is crashing hard (Oversold), we expand the shield to let it breathe.
        # Max Expansion: 3x the limit (e.g. -300 -> -900)
        # Expansion Factor = 1 + (severity * 2.0)
        # e.g. Severity 0.0 -> Factor 1.0 -> Limit -300
        # e.g. Severity 0.5 -> Factor 2.0 -> Limit -600
        # e.g. Severity 1.0 -> Factor 3.0 -> Limit -900
        if oversold_severity > 0.3:
            expansion_factor = 1.0 + (oversold_severity * 2.0)
            effective_limit = max_dd_limit * expansion_factor
            # logger.debug(f"🏹 RUBBER BAND: {symbol} Severity {oversold_severity:.2f}. Expanded Limit: ${effective_limit:.2f}")
            
        elif structure_intact:
             # B. Structural Grace (Standard)
             effective_limit = max_dd_limit * 1.20 # e.g. -300 * 1.2 = -360
        
        # C. HARD FLOOR (The Concrete Wall)
        # Never allow more than -1000 regardless of logic.
        HARD_FLOOR = -1000.0
        if effective_limit < HARD_FLOOR:
            effective_limit = HARD_FLOOR

        # 3. Check Breach
        if total_pnl < effective_limit:
            logger.warning(f"🛡️ EQUITY SHIELD BREACHED: {symbol} Total PnL ${total_pnl:.2f} < Limit ${effective_limit:.2f} (Severity: {oversold_severity:.2f})")
            
            # 4. Surgical Pruning
            sorted_trades = sorted(symbol_trades, key=lambda x: float(x.get('profit', 0.0)))
            worst_trade = sorted_trades[0]
            
            worst_ticket = worst_trade.get('ticket')
            worst_pnl = float(worst_trade.get('profit', 0.0))
            
            logger.warning(f"✂️ SURGICAL PRUNING: Closing Worst Trade Ticket {worst_ticket} (${worst_pnl:.2f}) to save the rest.")
            return worst_ticket
            
        return None

    def get_bias_modifier(self, symbol: str) -> Dict[str, str]:
        """
        Returns bias override if inversion is active.
        """
        state = self.get_state(symbol)
        modifiers = {}
        if state.is_inverted_long:
            modifiers['BUY'] = 'SELL' # Logic: If AGI says BUY, we SELL
        if state.is_inverted_short:
            modifiers['SELL'] = 'BUY' # Logic: If AGI says SELL, we BUY
            
        return modifiers
