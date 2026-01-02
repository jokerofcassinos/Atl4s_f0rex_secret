import logging

logger = logging.getLogger("Atl4s-TradeManager")

class TradeManager:
    def __init__(self):
        self.partial_tp_done = {} # Track if partial TP has been taken for a ticket
        self.peak_profits = {} # Track Max Profit for Trailing Protection

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

    def check_hard_exit(self, position, hard_tp_amount, hard_sl_amount, micro_metrics=None):
        """
        Instant Exit Logic (Virtual TP/SL) with Quantum Railgun Upgrade.
        Checks if floating profit/loss exceeds hard dollar thresholds.
        Returns: Action Dict or None
        """
        profit = position['profit'] # Floating PnL in USD
        
        # 1. Virtual Take Profit (Active Grab) with Quantum Railgun Logic
        # 1. Virtual Take Profit (Active Grab) with Quantum Railgun (TUNED)
        if profit >= hard_tp_amount:
             # Check for Momentum Extension (Greed Protocol) - STRICT MODE
             should_extend = False
             reason_ext = ""
             
             # GREED CAP: If we already have 1.5x the Target, don't wait. Take it.
             if profit > (hard_tp_amount * 1.5):
                 should_extend = False
             elif micro_metrics:
                 velocity = micro_metrics.get('velocity', 0)
                 entropy = micro_metrics.get('entropy', 1.0)
                 trade_type = position['type']
                 
                 # QUANTUM RAILGUN: STRICTER REQUIREMENTS (User: "Wait less")
                 # Needs Stronger Velocity (0.7) and Lower Entropy (0.4) to JUSTIFY waiting.
                 is_favorable = False
                 if trade_type == 0 and velocity > 0.7: is_favorable = True
                 elif trade_type == 1 and velocity < -0.7: is_favorable = True
                 
                 if is_favorable and entropy < 0.4:
                     should_extend = True
                     reason_ext = f"Velocity {velocity:.2f} (Explosive) >> Entropy {entropy:.2f}"
                     
             if should_extend:
                 return {
                     "action": "EXTEND",
                     "ticket": position['ticket'],
                     "reason": f"⚡ QUANTUM RAILGUN: Smashing TP (${profit:.2f}) | {reason_ext}" 
                 }
            
             return {
                "action": "CLOSE_FULL",
                "ticket": position['ticket'],
                "reason": f"Virtual TP Hit (${profit:.2f}) - INSTANT"
             }
             
        # 2. Virtual Stop Loss (Safety Net)
        # if profit <= -hard_sl_amount:
        #      return { ... }
              
        return None

    def __init__(self):
        self.partial_tp_done = {} # Track if partial TP has been taken for a ticket
        self.peak_profits = {} # Track Max Profit per ticket for Trailing Exit

    # ... (other methods unchanged) ...

    def check_exhaustion_exit(self, position, micro_metrics):
        """
        Lethal TP logic (Exhaustion & Profit Erosion Protection).
        """
        profit = position['profit']
        ticket = position['ticket']
        
        # Track Peak Profit
        current_peak = self.peak_profits.get(ticket, 0.0)
        if profit > current_peak:
            self.peak_profits[ticket] = profit
            current_peak = profit
            
        if profit <= 5.0: return None # Ignore small/negative profits
        
        # Metrics
        velocity = micro_metrics.get('velocity', 0) # Raw Velocity (Signed)
        abs_velocity = abs(velocity)
        entropy = micro_metrics.get('entropy', 0)
        trade_type = position.get('type') # 0=Buy, 1=Sell
        
        reason = ""
        should_close = False
        
        # --- 0. RAPID REVERSAL KILL (Panic Button) ---
        # User Feedback: "Candle went back up fast".
        # If we have significant profit ($100+) and Velocity flips aggressively against us.
        if profit > 100.0:
            if trade_type == 0 and velocity < -0.2: # Long, but price crashing down
                 should_close = True
                 reason = f"Lethal TP: MOMENTUM FLIPPED (Vel {velocity:.2f}) | Securing ${profit:.2f}"
            elif trade_type == 1 and velocity > 0.2: # Short, but price rocketing up
                 should_close = True
                 reason = f"Lethal TP: MOMENTUM FLIPPED (Vel {velocity:.2f}) | Securing ${profit:.2f}"
                 
            if should_close:
                 logger.info(f"🚨 {reason}")
                 return {
                    "action": "CLOSE_FULL",
                    "ticket": ticket,
                    "reason": reason
                 }
        
        # --- 1. PROFIT EROSION PROTECTION (The Ratchet) ---
        
        drop_percent = 0.0
        if current_peak > 0:
            drop_percent = (current_peak - profit) / current_peak
        
        # Tier 0: Peak > $30 -> Allow 50% Drop (Secure $15)
        if current_peak > 30.0 and drop_percent > 0.50:
            should_close = True
            reason = f"Lethal TP: Profit Erosion 50% (Peak ${current_peak:.2f} -> ${profit:.2f})"
            
        # Tier 1: Peak > $50 -> Allow 40% Drop (Secure $30)
        elif current_peak > 50.0 and drop_percent > 0.40:
            should_close = True
            reason = f"Lethal TP: Profit Erosion 40% (Peak ${current_peak:.2f} -> ${profit:.2f})"
            
        # Tier 2: Peak > $100 -> Allow 30% Drop (Secure $70)
        elif current_peak > 100.0 and drop_percent > 0.30:
            should_close = True
            reason = f"Lethal TP: Profit Erosion 30% (Peak ${current_peak:.2f} -> ${profit:.2f})"
            
        # Tier 3: Peak > $200 -> Allow 10% Drop (TITANIUM LOCK - Secure $180)
        # User: "Had 300, lost 100". With this, exit at 270.
        elif current_peak > 200.0 and drop_percent > 0.10:
             should_close = True
             reason = f"Lethal TP: Titanium Lock 10% (Peak ${current_peak:.2f} -> ${profit:.2f})"
             
        # Tier 4: Peak > $300 -> Allow 5% Drop (Secure $285)
        elif current_peak > 300.0 and drop_percent > 0.05:
             should_close = True
             reason = f"Lethal TP: Diamond Lock 5% (Peak ${current_peak:.2f} -> ${profit:.2f})"

        if should_close:
             logger.info(f"🛡️ {reason}")
             return {
                "action": "CLOSE_FULL",
                "ticket": ticket,
                "reason": reason
             }
        
        # --- 2. PHYSICS EXHAUSTION ---
        # 1. Stagnation Exit
        if profit > 6.0 and abs_velocity < 0.15:
            should_close = True
            reason = f"Lethal TP: Stagnation (Vel {abs_velocity:.2f}) at ${profit:.2f}"
            
        # 2. Chaos Protection
        elif profit > 10.0 and entropy > 0.6:
            should_close = True
            reason = f"Lethal TP: Chaos (Ent {entropy:.2f}) at ${profit:.2f}"
            
        # 3. Momentum Decay
        elif profit > 20.0 and abs_velocity < 0.3:
             should_close = True
             reason = f"Lethal TP: Momentum Fading at ${profit:.2f}"
             
        if should_close:
             logger.info(f"🔫 {reason}")
             return {
                "action": "CLOSE_FULL",
                "ticket": ticket,
                "reason": reason
             }
             
        return None
