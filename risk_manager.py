import logging
import config

logger = logging.getLogger("Atl4s-Risk")

class RiskManager:
    def __init__(self):
        self.initial_capital = config.INITIAL_CAPITAL
        self.current_capital = config.INITIAL_CAPITAL
        self.risk_per_trade = config.RISK_PER_TRADE # Base Risk
        self.max_drawdown = 0.50 # 50% hard stop
        self.peak_capital = config.INITIAL_CAPITAL # High Water Mark

    def update_capital(self, new_balance):
        """Updates capital and High Water Mark."""
        self.current_capital = new_balance
        if new_balance > self.peak_capital:
            self.peak_capital = new_balance

    def calculate_position_size(self, entry_price, stop_loss_price, current_balance, win_rate=0.40, profit_factor=1.5):
        """
        Calculates position size using Kelly Criterion and Drawdown Control.
        """
        self.update_capital(current_balance)
        
        # 1. Hard Stop Check
        if current_balance < self.initial_capital * (1 - self.max_drawdown):
            logger.critical("Max Drawdown Hit! Trading Halted.")
            return 0

        # 2. Drawdown Control (Survival Mode)
        # If DD > 5%, cut risk by 50%
        drawdown = (self.peak_capital - current_balance) / self.peak_capital
        risk_multiplier = 1.0
        
        if drawdown > 0.05:
            risk_multiplier = 0.5
            logger.warning(f"Drawdown Control Active (DD: {drawdown*100:.1f}%). Risk reduced by 50%.")

        # 3. Kelly Criterion Estimation
        # Kelly % = W - (1-W)/R
        # W = Win Rate, R = Reward/Risk Ratio
        # We use a fractional Kelly (e.g., Half Kelly) for safety.
        
        # Default conservative values if no history
        W = max(win_rate, 0.30) 
        R = max(profit_factor, 1.0) # Usually 1.5 to 2.0
        
        kelly_pct = W - ((1 - W) / R)
        
        # Safety Cap: Never risk more than 5% even if Kelly says so
        # And never risk negative
        kelly_pct = max(0, min(kelly_pct, 0.05))
        
        # Apply Half Kelly for safety
        base_risk_pct = kelly_pct * 0.5
        
        # Fallback to config if Kelly is too low (e.g. during learning)
        # But respect the cap.
        final_risk_pct = max(base_risk_pct, 0.01) # Min 1%
        final_risk_pct = min(final_risk_pct, config.RISK_PER_TRADE) # Cap at Config limit (5%)
        
        # Apply Drawdown Multiplier
        final_risk_pct *= risk_multiplier
        
        risk_amount = current_balance * final_risk_pct
        
        # Distance in points
        sl_dist = abs(entry_price - stop_loss_price)
        
        if sl_dist == 0:
            return 0
            
        tick_value = 100 # Standard for XAUUSD 1.00 move
        
        lots = risk_amount / (sl_dist * tick_value)
        lots = round(lots, 2)
        
        if lots < 0.01:
            logger.warning(f"Calculated lot size {lots} too small. Risk: ${risk_amount:.2f}, SL: {sl_dist:.2f}")
            return 0.01 # Minimum size
            
        logger.info(f"Risk Calc: Bal ${current_balance:.2f}, Risk ${risk_amount:.2f} ({final_risk_pct*100:.1f}%), SL {sl_dist:.2f}, Lots {lots}")
        return lots

    def get_stop_loss(self, entry_price, direction, atr_value):
        """
        Calculates Stop Loss price based on ATR or Structure.
        """
        # ATR Multiplier
        multiplier = 1.5
        dist = atr_value * multiplier
        
        if direction == 1: # Buy
            sl = entry_price - dist
        else: # Sell
            sl = entry_price + dist
            
        return sl

    def get_take_profit(self, entry_price, direction, stop_loss_price):
        """
        Calculates Take Profit based on R:R ratio.
        """
        rr_ratio = 2.0 # Target 1:2
        risk = abs(entry_price - stop_loss_price)
        reward = risk * rr_ratio
        
        if direction == 1:
            tp = entry_price + reward
        else:
            tp = entry_price - reward
            
        return tp

    def calculate_dynamic_lot(self, current_equity):
        """
        Calculates lot size based on linear scaling.
        OLD: (Equity / 30) * 0.02 -> Too Aggressive ($5k = 3.3 lots).
        NEW: (Equity / 500) * 0.05 -> Safer ($5k = 0.50 lots).
        """
        # Conservative Scaling
        base_runs = current_equity / 500.0 # How many '500 blocks' do we have?
        base_lots = 0.05 # Per $500
        
        raw_lots = base_runs * base_lots
        
        # Minimum Floor
        if raw_lots < 0.01: raw_lots = 0.01
        
        # Round to 2 decimals
        final_lots = round(raw_lots, 2)
        
        # Hard Safety Cap (User "Scared" of 3.3)
        # We cap Base Lots at 0.40. 
        # With Singularity (3x), this becomes 1.2 lots (Max Aggression).
        if final_lots > 0.40: final_lots = 0.40
        
        return final_lots

    def check_margin_survival(self, account_info):
        """
        Checks if we have enough free margin to open a trade.
        Returns True if safe, False if critical.
        """
        if not account_info:
            return True # Assume safe if no data
            
        margin_free = account_info.get('margin_free', 0)
        margin_level = account_info.get('margin_level', 9999)
        
        # Critical Stop if Margin Level < 150%
        if margin_level < 150:
            logger.warning(f"CRITICAL MARGIN: Level {margin_level:.2f}% too low. No new trades.")
            return False
            
        # Absolute buffer (e.g. need at least $10 free)
        if margin_free < 10:
            logger.warning(f"CRITICAL MARGIN: Free Margin ${margin_free:.2f} too low.")
            return False
            
        return True
