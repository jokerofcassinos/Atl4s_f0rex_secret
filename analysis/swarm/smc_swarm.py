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
        
        # 1. Detect structures
        fvgs = self.engine.detect_fvgs(df_m5)
        sweeps = self.engine.detect_liquidity_grabs(df_m5)
        
        # 2. DRAW ON CHART (Throttled 5s)
        if self.bridge and time.time() - self.last_draw_time > 5.0:
            # Get symbol from context or fallback
            tick = context.get('tick', {})
            symbol = tick.get('symbol', 'XAUUSD')
            
            # Clear old? No clear command yet. We overwrite by name.
            
            # Draw FVGs - extend to CURRENT time so they appear near current price
            current_time = int(time.time())  # Use current time for right edge
            for i, fvg in enumerate(fvgs):
                name = f"FVG_{i}"
                color = 0x00FF00 if fvg['type'] == 'BULL_FVG' else 0x0000FF # Green/Red(BGR)
                # MT5 Color: Red=255(0x0000FF), Green=32768(0x008000), Blue=16711680(0xFF0000) ? 
                # MQL5 uses integer representation. clrRed=255. clrLime=65280 (0x00FF00).
                
                c = 65280 if fvg['type'] == 'BULL_FVG' else 255 # Green / Red
                
                # Extend rectangle from FVG start time to CURRENT time (visible on chart)
                self.bridge.send_draw_rect(symbol, name, fvg['bottom'], fvg['top'], fvg['time'], current_time, c)
                
            # Draw Sweeps
            for i, sweep in enumerate(sweeps):
                name = f"SWEEP_{i}"
                c = 16776960 # Aqua 
                self.bridge.send_draw_text(symbol, name, sweep['level'], sweep['time'], "LIQ GRAB", c)
                
            self.last_draw_time = time.time()

        # 3. Generate Signal
        signal_type = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Logic: If we just swept a high, SELL.
        if sweeps:
            last_sweep = sweeps[-1]
            if last_sweep['type'] == 'SWEEP_HIGH':
                signal_type = "SELL"
                confidence = 85.0
                reason = "SMC: Liquidity Sweep of Highs (Stop Hunt)"
            elif last_sweep['type'] == 'SWEEP_LOW':
                signal_type = "BUY"
                confidence = 85.0
                reason = "SMC: Liquidity Sweep of Lows (Stop Hunt)"
        
        # Logic: If inside a BULL FVG, BUY.
        if fvgs:
            # Check if current price is inside the most recent FVG
            last_fvg = fvgs[-1]
            price = df_m5['close'].iloc[-1]
            
            if last_fvg['type'] == 'BULL_FVG':
                if last_fvg['bottom'] <= price <= last_fvg['top']:
                    signal_type = "BUY"
                    confidence = 80.0
                    reason = "SMC: Retest of Bullish FVG"
            elif last_fvg['type'] == 'BEAR_FVG':
                if last_fvg['bottom'] <= price <= last_fvg['top']:
                    signal_type = "SELL"
                    confidence = 80.0
                    reason = "SMC: Retest of Bearish FVG"

        if signal_type != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'reason': reason}
            )
            
        return None
