import logging
import pandas as pd
import numpy as np
import yfinance as yf
import config
from src.macro_math import MacroMath
from datetime import datetime, timedelta

logger = logging.getLogger("Atl4s-SixthEye")

class SixthEye:
    """
    The Oracle Council (Position System).
    Responsible for Secular Trends and Macro Fundamentals.
    - Monitors Monthly (MN) and Weekly (W1) timeframes.
    - Calculates Implicit Real Rates (Yields - Inflation expectations).
    - Proxies COT (Commitment of Traders) via volume/spread persistence.
    - Provides a "Macro Anchor" for the entire system.
    """
    def __init__(self):
        self.macro_context = {}
        self.last_macro_sync = datetime.min
        self.regime_prob = 0.5 # Bayesian Prior
        
    def sync_macro_fundamentals(self):
        """Fetches macro data for Real Rate calculations (e.g., TIPS proxies)."""
        if datetime.now() - self.last_macro_sync < timedelta(days=1):
            return
            
        logger.info("Syncing Macro Fundamentals (TIPS, CPI Proxies)...")
        # Gold is highly correlated to REAL RATES (Yield - Inflation)
        # We use TIPS (TIP ETF) or generic Yield Spread as a proxy.
        try:
            tip = yf.download("TIP", period="1y", interval="1d", progress=False)
            if isinstance(tip.columns, pd.MultiIndex): tip.columns = tip.columns.get_level_values(0)
            self.macro_context['TIP'] = tip.rename(columns={"Close": "close"})
            
            # US10Y is already in config.INTERMARKET_SYMBOLS but we might need it specifically here
            tnx = yf.download("^TNX", period="1y", interval="1d", progress=False)
            if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = tnx.columns.get_level_values(0)
            self.macro_context['Yield'] = tnx.rename(columns={"Close": "close"})
        except Exception as e:
            logger.error(f"Macro Sync Error: {e}")
            
        self.last_macro_sync = datetime.now()

    def analyze_secular_trend(self, data_map):
        """Analyzes MN and W1 for decade-scale trend persistence."""
        df_mn = data_map.get('MN')
        df_w1 = data_map.get('W1')
        
        score = 0
        details = {}
        
        if df_mn is not None and len(df_mn) > 12:
            # Check 12-month Rolling Mean (1-year cycle)
            ma12 = df_mn['close'].rolling(12).mean()
            if df_mn['close'].iloc[-1] > ma12.iloc[-1]:
                score += 25 # Secular Bullish
                details['MN_SECULAR'] = "BULLISH"
            else:
                score -= 25 # Secular Bearish
                details['MN_SECULAR'] = "BEARISH"
                
        if df_w1 is not None and len(df_w1) > 52:
            # Check 52-week Rolling Mean (1-year trend)
            ma52 = df_w1['close'].rolling(52).mean()
            current_w = df_w1['close'].iloc[-1]
            if current_w > ma52.iloc[-1]:
                score += 15
                details['W1_SECULAR'] = "BULLISH"
            else:
                score -= 15
                details['W1_SECULAR'] = "BEARISH"
                
        return score, details

    def calculate_implicit_real_rates(self, df_gold):
        """Calculates direction of Real Rates and Cointegration divergence."""
        self.sync_macro_fundamentals()
        
        tip = self.macro_context.get('TIP')
        yields = self.macro_context.get('Yield')
        
        if tip is None or yields is None or len(tip) < 20 or df_gold is None:
            return 0, {}
            
        # 1. Cointegration Analysis (Gold vs TIP)
        # Gold is a long-term inflation hedge, should be cointegrated with TIP ETF prices.
        n = min(len(tip), len(df_gold))
        y = df_gold['close'].values.flatten()[-n:]
        x = tip['close'].values.flatten()[-n:]
        
        coint_res = MacroMath.calculate_cointegration(y, x)
        
        # 2. Real Rate Direction Bias
        tip_slope = (tip['close'].iloc[-1] - tip['close'].iloc[-10]) / 10
        yield_slope = (yields['close'].iloc[-1] - yields['close'].iloc[-10]) / 10
        
        real_rate_bias = 0
        if tip_slope > 0 and yield_slope < 0: real_rate_bias = 30
        elif tip_slope < 0 and yield_slope > 0: real_rate_bias = -30
        else: real_rate_bias = 15 if tip_slope > yield_slope else -15
            
        return real_rate_bias, coint_res

    def analyze_regime_bayesian(self, df_gold):
        """Updates Bayesian probability of Expansion vs Equilibrium."""
        if df_gold is None or len(df_gold) < 20: return 0.5
        
        close = df_gold['close'].values
        self.regime_prob = MacroMath.bayesian_regime_detect(close, prev_prob=self.regime_prob)
        return self.regime_prob

    def cot_proxy_analysis(self, df_w1):
        """Proxies Commitment of Traders (COT) via volume persistence."""
        if df_w1 is None or len(df_w1) < 13: return 0
        
        # Institutional 'Managed Money' accumulation usually shows as:
        # 1. Price rising on rising volume over multiple weeks.
        # 2. Narrowing spreads on pullbacks (absorption).
        
        recent_vol = df_w1['volume'].iloc[-4:].mean()
        prev_vol = df_w1['volume'].iloc[-13:-4].mean()
        
        price_change = (df_w1['close'].iloc[-1] - df_w1['close'].iloc[-4]) / df_w1['close'].iloc[-4]
        
        bias = 0
        if price_change > 0 and recent_vol > prev_vol:
            bias = 20 # Institutional Accumulation
        elif price_change < 0 and recent_vol > prev_vol:
            bias = -20 # Institutional Distribution
            
        return bias

    def deliberate(self, data_map):
        """Main entry point for Sexto Olho."""
        df_mn = data_map.get('MN')
        df_w1 = data_map.get('W1')
        
        secular_score, secular_details = self.analyze_secular_trend(data_map)
        real_rate_bias, coint_res = self.calculate_implicit_real_rates(df_w1)
        cot_bias = self.cot_proxy_analysis(df_w1)
        regime_prob = self.analyze_regime_bayesian(df_w1)
        
        # Cointegration Alpha: Fade extreme divergence if cointegrated
        coint_alpha = 0
        if coint_res.get('stat_score', 0) > 0.7:
            z = coint_res['z_score']
            if z > 2.0: coint_alpha = -20 # Overvalued relative to TIP
            elif z < -2.0: coint_alpha = 20 # Undervalued relative to TIP

        total_score = secular_score + real_rate_bias + cot_bias + coint_alpha
        
        # Bayesian Filtering on Anchor
        if regime_prob > 0.8: total_score *= 1.2 # High confidence in Expansion
        elif regime_prob < 0.2: total_score *= 0.8 # High uncertainty
        
        anchor = "WAIT"
        if total_score > 50: anchor = "STRONG_BUY"
        elif total_score > 20: anchor = "BUY"
        elif total_score < -50: anchor = "STRONG_SELL"
        elif total_score < -20: anchor = "SELL"
        
        return {
            'anchor': anchor,
            'score': total_score,
            'details': secular_details,
            'macro_bias': real_rate_bias,
            'cot_sentiment': cot_bias,
            'cointegration': coint_res,
            'bayesian_regime': regime_prob
        }
