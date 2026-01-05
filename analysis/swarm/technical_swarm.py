import logging
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.technical_library import TechnicalLibrary

logger = logging.getLogger("TechnicalSwarm")

class TechnicalSwarm(SubconsciousUnit):
    """
    The Technical Consensus.
    Aggregates signals from all major technical indicators.
    """
    def __init__(self):
        super().__init__("Technical_Swarm")
        self.lib = TechnicalLibrary()

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        close = df_m5['close'].iloc[-1]
        
        # 1. Collect Signals
        votes = [] # 1 for Buy, -1 for Sell
        reasons = []
        
        # RSI
        rsi = self.lib.calculate_rsi(df_m5['close'])
        if rsi < 30: 
            votes.append(1)
            reasons.append(f"RSI Oversold ({rsi:.1f})")
        elif rsi > 70: 
            votes.append(-1)
            reasons.append(f"RSI Overbought ({rsi:.1f})")
            
        # Bollinger
        upper, mid, lower = self.lib.calculate_bollinger_bands(df_m5['close'])
        if close > upper: 
            votes.append(-1)
            reasons.append("Price > Bollinger Upper")
        elif close < lower: 
            votes.append(1)
            reasons.append("Price < Bollinger Lower")
            
        # MACD
        macd, signal, hist = self.lib.calculate_macd(df_m5['close'])
        if macd > signal: votes.append(0.5) # Weak Buy
        else: votes.append(-0.5) # Weak Sell
        
        # Stochastic
        k, d = self.lib.calculate_stochastic(df_m5)
        if k < 20 and d < 20: 
            votes.append(1)
            reasons.append("Stoch Oversold")
        elif k > 80 and d > 80: 
            votes.append(-1)
            reasons.append("Stoch Overbought")
            
        # VWAP
        vwap = self.lib.calculate_vwap(df_m5)
        if vwap > 0:
            if close > vwap: votes.append(0.5) # Bullish context
            else: votes.append(-0.5) # Bearish context

        # CCI
        cci = self.lib.calculate_cci(df_m5)
        if cci > 100: votes.append(-1)
        elif cci < -100: votes.append(1)
        
        # Williams %R
        wr = self.lib.calculate_williams_r(df_m5)
        if wr < -80: votes.append(1) # Oversold
        elif wr > -20: votes.append(-1) # Overbought

        # Ichimoku (Cloud check)
        tenkan, kijun, senkou_a, senkou_b = self.lib.calculate_ichimoku(df_m5)
        if close > senkou_a and close > senkou_b:
            votes.append(1) # Above Cloud
        elif close < senkou_a and close < senkou_b:
            votes.append(-1) # Below Cloud
            
        # 2. Consensus Logic
        total_score = sum(votes)
        
        # Normalization (Approx max possible score is ~7-8)
        # If absolute score > 4, it's strong.
        
        signal_type = "WAIT"
        confidence = 0.0
        
        if total_score >= 3.0:
            signal_type = "BUY"
            confidence = 75.0 + min(total_score * 5, 20)
        elif total_score <= -3.0:
            signal_type = "SELL"
            confidence = 75.0 + min(abs(total_score) * 5, 20)
            
        if signal_type != "WAIT":
            summary = ", ".join(reasons) if reasons else "Confluence of Minor Indicators"
            return SwarmSignal(
                source=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=0,
                meta_data={'reason': summary, 'score': total_score}
            )
            
        return None
