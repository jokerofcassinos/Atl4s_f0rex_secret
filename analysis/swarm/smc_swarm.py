import logging
import time
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.smc_engine import SmartMoneyEngine

logger = logging.getLogger("SmartMoneySwarm")

class SmartMoneySwarm(SubconsciousUnit):
    """
    The Visualizer.
    Finds Institutional Setups and DRAWS them on the chart.
    """
    def __init__(self, bridge=None):
        super().__init__("SmartMoney_Swarm")
        self.engine = SmartMoneyEngine()
        self.bridge = bridge # Reference to ZmqBridge to send draw commands
        self.last_draw_time = 0
        
    def set_bridge(self, bridge):
        self.bridge = bridge

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 100: return None
        
        # Multi-Timeframe Context (H1)
        data_map = context.get('data_map', {})
        df_h1 = data_map.get('H1')
        h1_trend = "NEUTRAL"
        
        if df_h1 is not None and len(df_h1) > 20:
             # Calculate simple EMA 20 on H1
             closes = df_h1['close']
             ema_20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
             current_h1_price = closes.iloc[-1]
             
             if current_h1_price > ema_20:
                 h1_trend = "BULLISH"
             else:
                 h1_trend = "BEARISH"
        
        # 1. Detect structures
        fvgs = self.engine.detect_fvgs(df_m5)
        sweeps = self.engine.detect_liquidity_grabs(df_m5)
        
        # 2. DRAW ON CHART (Throttled 5s)
        if self.bridge and time.time() - self.last_draw_time > 5.0:
            # Get symbol from context or fallback
            tick = context.get('tick', {})
            symbol = tick.get('symbol', 'XAUUSD')
            
            # Draw FVGs - extend to CURRENT time so they appear near current price
            current_time = int(time.time())  # Use current time for right edge
            price_close = df_m5['close'].iloc[-1]
            # atr = df_m5['high'].iloc[-1] - df_m5['low'].iloc[-1] # Rough ATR
            
            for i, fvg in enumerate(fvgs):
                # Filter: Only draw relevant FVGs (near price)
                # If FVG is miles away, skip it to reduce clutter
                dist_percent = abs(fvg['bottom'] - price_close) / price_close
                if dist_percent > 0.005: # >0.5% away (approx 100 pips on Gold), skip
                     continue
                     
                name_top = f"FVG_Top_{i}"
                name_bot = f"FVG_Bot_{i}"
                name_text = f"FVG_Label_{i}"
                
                # Colors: Neon Green / Neon Red
                # MQL Integer Colors: 
                # Green = 65280 (Lime)
                # Red = 255 (Red)
                # Gold = 55295 (Yellow) for text
                
                if fvg['type'] == 'BULL_FVG':
                     color = 65280 # Lime
                     label = "DEMAND GAP"
                     c_text = 65280
                else:
                     color = 255 # Red
                     label = "SUPPLY GAP"
                     c_text = 255

                # Extend projection 20 candles
                future_time = current_time + (300 * 20) 
                
                # Use LINES instead of RECT to be minimalist
                # Top Line
                self.bridge.send_draw_line(symbol, name_top, fvg['top'], fvg['top'], fvg['time'], future_time, color)
                # Bottom Line
                self.bridge.send_draw_line(symbol, name_bot, fvg['bottom'], fvg['bottom'], fvg['time'], future_time, color)
                
                # Text in the middle
                mid_price = (fvg['top'] + fvg['bottom']) / 2
                self.bridge.send_draw_text(symbol, name_text, mid_price, current_time, label, c_text)
                
            # Draw Sweeps with context
            for i, sweep in enumerate(sweeps):
                name = f"SWEEP_{i}"
                c = 42495 # Orange
                label = "v LIQ GRAB" if sweep['type'] == 'SWEEP_LOW' else "^ LIQ GRAB"
                self.bridge.send_draw_text(symbol, name, sweep['level'], sweep['time'], label, c)
                
            self.last_draw_time = time.time()

        # 3. Generate Signal with HTF Filter
        signal_type = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Logic: Liquidity Sweeps (Stop Hunts)
        if sweeps:
            last_sweep = sweeps[-1]
            if last_sweep['type'] == 'SWEEP_HIGH':
                # Swept High -> Potential SELL
                # FILTER: Only Sell if H1 is BEARISH or we have extreme extension
                if h1_trend == "BEARISH":
                    signal_type = "SELL"
                    confidence = 85.0
                    reason = "SMC: Sweep of Highs (Pro-Trend)"
                elif h1_trend == "BULLISH":
                    # Counter-trend sweep. Much riskier.
                    signal_type = "SELL"
                    confidence = 55.0 # Weak signal, likely vetoed
                    reason = "SMC: Sweep of Highs (Counter-Trend - risky)"
                    
            elif last_sweep['type'] == 'SWEEP_LOW':
                # Swept Low -> Potential BUY
                # FILTER: Only Buy if H1 is BULLISH
                if h1_trend == "BULLISH":
                    signal_type = "BUY"
                    confidence = 85.0
                    reason = "SMC: Sweep of Lows (Pro-Trend)"
                elif h1_trend == "BEARISH":
                    signal_type = "BUY"
                    confidence = 55.0 # Weak
                    reason = "SMC: Sweep of Lows (Counter-Trend - risky)"
        
        # Logic: FVG Retests
        if fvgs and signal_type == "WAIT":
            last_fvg = fvgs[-1]
            price = df_m5['close'].iloc[-1]
            
            if last_fvg['type'] == 'BULL_FVG':
                if last_fvg['bottom'] <= price <= last_fvg['top']:
                    if h1_trend == "BULLISH":
                        signal_type = "BUY"
                        confidence = 80.0
                        reason = "SMC: Bullish FVG Retest (Pro-Trend)"
                    else:
                        # Retesting a demand zone in a downtrend -> likely to fail
                        confidence = 0.0 
                        
            elif last_fvg['type'] == 'BEAR_FVG':
                if last_fvg['bottom'] <= price <= last_fvg['top']:
                    if h1_trend == "BEARISH":
                        signal_type = "SELL"
                        confidence = 80.0
                        reason = "SMC: Bearish FVG Retest (Pro-Trend)"
                    else:
                        confidence = 0.0

        if signal_type != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'reason': reason}
            )
            
        return None
