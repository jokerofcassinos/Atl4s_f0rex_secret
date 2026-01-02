import logging
import config
import numpy as np

logger = logging.getLogger("Atl4s-NeuralRisk")

class NeuralRiskManager:
    def __init__(self):
        self.initial_capital = config.INITIAL_CAPITAL
        self.current_capital = config.INITIAL_CAPITAL
        self.risk_per_trade = config.RISK_PER_TRADE # Base Risk (e.g., 0.02)
        self.max_drawdown = 0.50 # 50% hard stop
        self.peak_capital = config.INITIAL_CAPITAL # High Water Mark
        
        # Neural Parameters
        self.base_win_rate = 0.45
        self.base_rr = 1.5

    def update_capital(self, new_balance):
        """Updates capital and High Water Mark."""
        self.current_capital = new_balance
        if new_balance > self.peak_capital:
            self.peak_capital = new_balance

    def check_margin_survival(self, acc_stats):
        """
        Checks if the account has sufficient margin to survive new trades.
        Returns True if safe to trade, False otherwise.
        """
        if not acc_stats:
            logger.warning("Margin Check: No account stats available. Assuming unsafe.")
            return False
            
        margin_level = acc_stats.get('margin_level', 0)
        margin_free = acc_stats.get('margin_free', 0)
        
        # Safety Thresholds
        CRITICAL_MARGIN_LEVEL = 150.0 # Percentage
        
        if margin_level > 0 and margin_level < CRITICAL_MARGIN_LEVEL:
            logger.warning(f"Margin Safety: CRITICAL LEVEL ({margin_level:.2f}%). Halting Trading.")
            return False
            
        if margin_free < 0:
            logger.critical(f"Margin Safety: NEGATIVE FREE MARGIN (${margin_free:.2f}). Emergency Halt.")
            return False
            
        return True

    def calculate_safe_margin_lots(self, equity, margin_free, current_price, leverage=500):
        """
        Calculates the absolute maximum NEW lots allowed based on FREE MARGIN.
        """
        if current_price <= 0: return 0.01
        
        # Contract Size for XAUUSD is 100
        contract_size = 100 
        
        # Margin Required for 1.0 Lot = (Price * Contract) / Leverage
        margin_per_lot = (current_price * contract_size) / leverage
        
        if margin_per_lot <= 0: return 0.01
        
        # Max New Lots = Free Margin / MarginPerLot
        # Use margin_free if available (and positive), else fallback to equity (risky)
        available_funds = margin_free if margin_free > 0 else 0
        
        max_theoretical_lots = available_funds / margin_per_lot
        
        # Safety Buffer: Use only 90% of available free margin
        safe_lots = max_theoretical_lots * 0.90
        
        return round(safe_lots, 2)

    def calculate_dynamic_lot(self, current_equity):
        """
        Calculates the base lot size based on linear equity scaling.
        Logic: $30 Equity -> 0.02 Lots.
        """
        # Base Ratio: 0.02 lots per $30
        # Ratio = 0.02 / 30 = 0.000666...
        base_ratio = 0.02 / 30.0
        
        raw_lots = current_equity * base_ratio
        
        # Round to 2 decimal places (standard lot step)
        lots = round(raw_lots, 2)
        
        # Hard limits
        lots = max(0.01, lots) # Minimum 0.01
        # lots = min(lots, 5.0) # Optional Max Cap for safety
        
        # Log occasionally? Too verbose for every tick maybe, but useful for debugging
        # logger.debug(f"Dynamic Lot Calc: Equity ${current_equity:.2f} -> {lots} Lots")
        
        return lots

    def calculate_dynamic_kelly(self, win_rate, profit_factor):
        """
        Calculates Kelly Percentage.
        K = W - (1-W)/R
        """
        if profit_factor == 0: return 0
        k = win_rate - ((1 - win_rate) / profit_factor)
        return max(0, k)

    def calculate_position_size(self, entry_price, stop_loss_price, current_balance, 
                              consensus_score=50, regime_prob=0.5, cortex_conf=0.5):
        """
        Calculates position size using Neural Logic (Dynamic Kelly + Circuit Breaker).
        """
        self.update_capital(current_balance)
        
        # 1. Hard Stop Check
        if current_balance < self.initial_capital * (1 - self.max_drawdown):
            logger.critical("Max Drawdown Hit! Trading Halted.")
            return 0

        # 2. Circuit Breaker (Exponential Drawdown Reduction)
        drawdown = (self.peak_capital - current_balance) / self.peak_capital
        risk_multiplier = 1.0
        
        if drawdown > 0.05:
            # Exponential decay: Risk decreases rapidly as DD increases
            # At 10% DD, multiplier ~ 0.6
            # At 20% DD, multiplier ~ 0.36
            risk_multiplier = np.exp(-5 * (drawdown - 0.05))
            logger.warning(f"Circuit Breaker Active (DD: {drawdown*100:.1f}%). Risk Multiplier: {risk_multiplier:.2f}")

        # 3. Dynamic Confidence Adjustment
        # Base Win Rate is adjusted by Consensus Score and Regime Probability
        # If Score is 90 and Regime is 0.9 -> High Confidence -> Higher Win Rate Estimate
        
        # Normalize inputs
        norm_score = consensus_score / 100.0
        
        # Estimated Win Rate
        # We assume our base win rate improves with higher confidence
        est_win_rate = self.base_win_rate + (norm_score - 0.5) * 0.2 + (regime_prob - 0.5) * 0.1 + (cortex_conf - 0.5) * 0.1
        est_win_rate = min(0.80, max(0.30, est_win_rate)) # Cap between 30% and 80%
        
        # Estimated R:R (Reward to Risk)
        # We assume we target at least 1.5, but maybe higher in strong trends
        est_rr = self.base_rr
        if regime_prob > 0.7:
            est_rr = 2.0 # Trend following targets higher R:R
            
        # 4. Dynamic Kelly Calculation
        kelly_pct = self.calculate_dynamic_kelly(est_win_rate, est_rr)
        
        # Safety: Use Half Kelly or Quarter Kelly
        # We scale Kelly fraction based on confidence too!
        kelly_fraction = 0.3 + (norm_score * 0.2) # 0.3 to 0.5 Kelly
        
        target_risk_pct = kelly_pct * kelly_fraction
        
        # Absolute Safety Caps
        target_risk_pct = min(target_risk_pct, 0.05) # Never more than 5%
        target_risk_pct = max(target_risk_pct, 0.005) # Minimum 0.5% risk to make it worth it
        
        # Apply Circuit Breaker
        final_risk_pct = target_risk_pct * risk_multiplier
        
        # Calculate Amount
        risk_amount = current_balance * final_risk_pct
        
        # Calculate Lots
        sl_dist = abs(entry_price - stop_loss_price)
        if sl_dist == 0: return 0
        
        tick_value = 100 # XAUUSD standard
        lots = risk_amount / (sl_dist * tick_value)
        lots = round(lots, 2)
        
        if lots < 0.01:
            logger.warning(f"Calculated lots {lots} too small. Risk: ${risk_amount:.2f}")
            return 0.01
            
        logger.info(f"Neural Risk: Bal ${current_balance:.0f} | Conf: {norm_score:.2f} | Kelly: {target_risk_pct*100:.1f}% | Final Risk: {final_risk_pct*100:.2f}% | Lots: {lots}")
        return lots

    def get_geometric_stop(self, entry_price, direction, atr_value, kinematics_energy=0, wavelet_power=0, uncertainty=0):
        """
        Calculates a 'Living Stop Loss' based on Market Physics.
        Expands in high energy (to avoid noise) and contracts in high coherence (precision).
        """
        # Base Stop
        base_dist = atr_value * 1.5
        
        # 1. Phase Space Expansion (Kinematics)
        # If Orbit Energy is high (>0.8), market is moving fast. 
        # We need wider stops to accommodate volatility/noise orbit.
        expansion_factor = 1.0
        if kinematics_energy > 0.8:
            expansion = (kinematics_energy - 0.8) * 0.5 # e.g. Energy 1.8 -> +0.5 expansion
            expansion_factor += min(expansion, 1.0) # Cap at +100%
            logger.info(f"Geometric Stop: High Orbit Energy ({kinematics_energy:.2f}). Expanding Stop by {(expansion_factor-1)*100:.0f}%.")
            
        # 2. Wavelet Contraction (Coherence)
        # If Coherence is high (>0.8), the trend is pure. We can trust it more.
        # Tighter stop allowed.
        contraction_factor = 1.0
        if wavelet_power > 0.8: # Using Coherence as 'Power' proxy here
            contraction_factor = 0.8 # Shrink by 20%
            logger.info(f"Geometric Stop: High Coherence ({wavelet_power:.2f}). Contracting Stop to 80%.")
            
        final_dist = base_dist * expansion_factor * contraction_factor
        
        # 3. Heisenberg Limit Check
        # Uncertainty is the absolute floor for stop distance based on momentum
        if final_dist < uncertainty:
            logger.info(f"Geometric Stop: Hitting Heisenberg Limit. Stop widened to {uncertainty:.2f}")
            final_dist = uncertainty
            
        # Calculate Price
        if direction == 1:
            return entry_price - final_dist
        else:
            return entry_price + final_dist

    def get_quantum_stop(self, entry_price, direction, quantum_data, atr_value):
        """
        Calculates Stop Loss using Quantum Barriers if available, else ATR.
        DEPRECATED/FALLBACK: Prefer get_geometric_stop for full physics.
        """
        # Default ATR Stop
        atr_sl_dist = atr_value * 1.5
        
        if direction == 1: # Buy
            default_sl = entry_price - atr_sl_dist
        else: # Sell
            default_sl = entry_price + atr_sl_dist
            
        # Check Quantum Data
        uncertainty = 0
        if quantum_data:
             uncertainty = quantum_data.get('uncertainty', 0)
        
        # Heisenberg Principle: Stop must be at least 'uncertainty' away
        # If uncertainty is high (high momentum), we need breathing room.
        min_stop_dist = max(atr_sl_dist, uncertainty)
        
        # Check barriers
        if not quantum_data or 'nearest_level' not in quantum_data:
            barrier_sl_dist = min_stop_dist
        else:
            q_level = quantum_data['nearest_level']
            q_prob = quantum_data.get('tunneling_prob', 1.0)
            
            # If tunneling probability is LOW (< 0.3), strict barrier.
            if q_prob < 0.4:
                if direction == 1 and q_level < entry_price:
                     barrier_sl = q_level - (atr_value * 0.2)
                     # Ensure barrier SL is not TOO close (violate Heisenberg)
                     if (entry_price - barrier_sl) < uncertainty:
                         logger.info(f"Quantum Stop: Barrier too close ({entry_price - barrier_sl:.2f} < {uncertainty:.2f}). Expanding to Heisenberg Uncertainty.")
                         barrier_sl = entry_price - uncertainty
                     
                     if abs(entry_price - barrier_sl) < (atr_value * 3.0):
                         logger.info(f"Quantum Stop: Using Barrier/Heisenberg at {barrier_sl:.2f}")
                         return barrier_sl
                         
                elif direction == -1 and q_level > entry_price:
                     barrier_sl = q_level + (atr_value * 0.2)
                     if (barrier_sl - entry_price) < uncertainty:
                         barrier_sl = entry_price + uncertainty
                         
                     if abs(entry_price - barrier_sl) < (atr_value * 3.0):
                         logger.info(f"Quantum Stop: Using Barrier/Heisenberg at {barrier_sl:.2f}")
                         return barrier_sl
        
        # Fallback to Min Stop Dist (ATR or Heisenberg)
        if direction == 1:
            return entry_price - min_stop_dist
        else:
            return entry_price + min_stop_dist

    def calculate_quantum_size(self, entry_price, stop_loss_price, current_balance, 
                             consensus_score=50, regime_prob=0.5, cortex_conf=0.5,
                             wavelet_coherence=0, topology_loops=0, kinematics_score=0):
        """
        Calculates 'Quantum Sized' position based on Signal Quality, Physics, and History.
        """
        self.update_capital(current_balance)
        
        # 1. Hard Limits
        if current_balance < self.initial_capital * (1 - self.max_drawdown):
            logger.critical("Max Drawdown Hit! Trading Halted.")
            return 0
            
        sl_dist = abs(entry_price - stop_loss_price)
        if sl_dist == 0: return 0
        tick_value = 100 # XAUUSD

        # 2. Base Neural Calculation (Kelly)
        # We reuse the logic but in a cleaner flow
        norm_score = consensus_score / 100.0
        
        # Base Win Rate & RR
        est_win_rate = self.base_win_rate + (norm_score - 0.5) * 0.2 + (regime_prob - 0.5) * 0.1
        est_rr = self.base_rr
        if regime_prob > 0.7: est_rr = 2.5
        
        kelly_pct = self.calculate_dynamic_kelly(est_win_rate, est_rr)
        base_risk_pct = kelly_pct * 0.4 # Default 40% Kelly
        
        # 3. Quantum Multipliers
        size_multiplier = 1.0
        
        # A. Physics Boost (Clean & Fast)
        if wavelet_coherence > 0.8 and kinematics_score > 70:
            size_multiplier *= 1.5
            logger.info("Quantum Sizing: Physics Boost (High Coherence + Speed). Size x1.5")
        elif wavelet_coherence < 0.3:
            size_multiplier *= 0.6
            logger.info("Quantum Sizing: Noise Penalty (Low Coherence). Size x0.6")
            
        # B. Topology Boost (Arbitrage)
        if topology_loops > 0:
            size_multiplier *= 1.3
            logger.info("Quantum Sizing: Topology Boost (Loop Detected). Size x1.3")
            
        # C. Cortex Boost (Memory)
        if cortex_conf > 0.75:
            size_multiplier *= 1.2
            logger.info("Quantum Sizing: Cortex Boost (High Confidence Memory). Size x1.2")
        elif cortex_conf < 0.4:
            size_multiplier *= 0.8
            logger.info("Quantum Sizing: Cortex Penalty (Low Confidence). Size x0.8")
            
        # Apply Multiplier
        target_risk_pct = base_risk_pct * size_multiplier
        
        # 4. Circuit Breaker (Drawdown)
        drawdown = (self.peak_capital - current_balance) / self.peak_capital
        if drawdown > 0.05:
            dd_dampener = np.exp(-5 * (drawdown - 0.05))
            target_risk_pct *= dd_dampener
            logger.warning(f"Quantum Sizing: Circuit Breaker active. Risk reduced by {1-dd_dampener:.2f}%")
            
        # 5. Review & Cap
        target_risk_pct = min(target_risk_pct, 0.05) # Max 5%
        target_risk_pct = max(target_risk_pct, 0.002) # Min 0.2%
        
        risk_amount = current_balance * target_risk_pct
        lots = risk_amount / (sl_dist * tick_value)
        lots = round(lots, 2)
        
        if lots < 0.01: lots = 0.01
        
        logger.info(f"Quantum Size: Bal ${current_balance:.0f} | BaseRisk: {base_risk_pct*100:.2f}% | Mult: {size_multiplier:.2f} | FinalRisk: {target_risk_pct*100:.2f}% | Lots: {lots}")
        return lots

    def get_take_profit(self, entry_price, direction, stop_loss_price, regime_prob=0.5):
        """
        Calculates Take Profit.
        """
        risk = abs(entry_price - stop_loss_price)
        
        # Dynamic R:R
        # If Trending (High Regime Prob), aim higher.
        rr_ratio = 1.5 + (regime_prob * 1.5) # 1.5 to 3.0
        
        reward = risk * rr_ratio
        
        if direction == 1:
            tp = entry_price + reward
        else:
            tp = entry_price - reward
            
        return tp
