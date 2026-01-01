import logging

logger = logging.getLogger("Atl4s-TradeManager")

class TradeManager:
    def __init__(self):
        self.partial_tp_done = {} # Track if partial TP has been taken for a ticket

    def check_trailing_stop(self, position, current_price, structure_low, structure_high):
        """
        Calculates new Stop Loss based on Market Structure (Smart Trailing).
        Returns:
            new_sl (float) or None if no update needed.
        """
        ticket = position['ticket']
        entry_price = position['open_price']
        current_sl = position['sl']
        direction = 1 if position['type'] == 0 else -1 # 0=Buy, 1=Sell
        
        new_sl = None
        
        # R-Multiple Calculation
        risk = abs(entry_price - current_sl)
        if risk == 0: return None
        
        profit_pips = (current_price - entry_price) * direction if direction == 1 else (entry_price - current_price)
        r_multiple = profit_pips / risk
        
        # Logic:
        # 1. If > 1R, Move to Breakeven (Entry +/- small buffer)
        # 2. If > 2R, Trail behind Structure
        
        if direction == 1: # BUY
            # Breakeven
            if r_multiple > 1.0 and current_sl < entry_price:
                new_sl = entry_price + 0.10 # Small buffer
                logger.info(f"Trade {ticket}: Moving SL to Breakeven.")
                return new_sl
            
            # Smart Trail
            if r_multiple > 2.0 and structure_low > current_sl:
                # Only move SL up
                new_sl = structure_low - 0.10 # Buffer below swing low
                logger.info(f"Trade {ticket}: Smart Trailing SL to {new_sl} (Structure Low).")
                return new_sl

        else: # SELL
            # Breakeven
            if r_multiple > 1.0 and current_sl > entry_price:
                new_sl = entry_price - 0.10
                logger.info(f"Trade {ticket}: Moving SL to Breakeven.")
                return new_sl
                
            # Smart Trail
            if r_multiple > 2.0 and structure_high < current_sl:
                # Only move SL down
                new_sl = structure_high + 0.10 # Buffer above swing high
                logger.info(f"Trade {ticket}: Smart Trailing SL to {new_sl} (Structure High).")
                return new_sl
                
        return None

    def check_partial_tp(self, position, current_price):
        """
        Checks if Partial TP (50%) should be taken at 1.5R.
        Returns:
            action (dict) or None
        """
        ticket = position['ticket']
        
        # Check if already done
        if self.partial_tp_done.get(ticket, False):
            return None
            
        entry_price = position['open_price']
        current_sl = position['sl']
        volume = position['volume']
        direction = 1 if position['type'] == 0 else -1
        
        if volume < 0.02: # Cannot split 0.01
            return None
            
        risk = abs(entry_price - current_sl)
        if risk == 0: return None
        
        current_profit = (current_price - entry_price) * direction if direction == 1 else (entry_price - current_price)
        r_multiple = current_profit / risk
        
        if r_multiple >= 1.5:
            # Trigger Partial Close
            close_vol = round(volume / 2, 2)
            if close_vol < 0.01: close_vol = 0.01
            
            self.partial_tp_done[ticket] = True
            logger.info(f"Trade {ticket}: Triggering Partial TP ({close_vol} lots) at 1.5R.")
            
            return {
                "action": "CLOSE_PARTIAL",
                "ticket": ticket,
                "volume": close_vol
            }
            
        return None

    def adjust_tp_for_volatility(self, base_tp, entry_price, direction, volatility_state):
        """
        Expands TP if volatility is expanding.
        """
        if volatility_state == "EXPANDING":
            # Increase TP distance by 20%
            dist = abs(base_tp - entry_price)
            new_dist = dist * 1.20
            
            if direction == 1:
                return entry_price + new_dist
            else:
                return entry_price - new_dist
        
        return base_tp

    def check_hard_exit(self, position, hard_tp_amount, hard_sl_amount):
        """
        Instant Exit Logic (Virtual TP/SL).
        Checks if floating profit/loss exceeds hard dollar thresholds.
        Returns: Action Dict or None
        """
        profit = position['profit'] # Floating PnL in USD
        
        # 1. Virtual Take Profit (Active Grab)
        # If profit >= 0.70 (or configured amount), Close Instantly.
        if profit >= hard_tp_amount:
             return {
                "action": "CLOSE_FULL",
                "ticket": position['ticket'],
                "reason": f"Virtual TP Hit (${profit:.2f})"
             }
             
        # 2. Virtual Stop Loss (Safety Net)
        # If loss > 1.00 (negative profit < -1.00), Close Instantly.
        # Note: profit is negative for loss.
        # if profit <= -hard_sl_amount:
        #      return {
        #         "action": "CLOSE_FULL",
        #         "ticket": position['ticket'],
        #         "reason": f"Virtual SL Hit (${profit:.2f})"
        #      }
             
        return None
