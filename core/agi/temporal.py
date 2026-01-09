
import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger("Temporal")

class FractalTimeScaleIntegrator:
    """
    System 24: Fractal Time Scale Integrator.
    "Time is not linear, it is fractal."
    Ensures coherence across M1, M5, H1, D1 execution frames.
    """
    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.weights = {'1m': 0.1, '5m': 0.2, '15m': 0.2, '1h': 0.3, '4h': 0.2}

    def calculate_fractal_coherence(self, market_data_map: Dict[str, Any]) -> float:
        """
        Returns a score from -1.0 (Full Bearish Coherence) to +1.0 (Full Bullish Coherence).
        0.0 means Chaos/Noise (Timeframes disagree).
        """
        score = 0.0
        total_weight = 0.0
        
        for tf, weight in self.weights.items():
            if tf in market_data_map:
                df = market_data_map[tf]
                if df is not None and not df.empty:
                    # Simple Trend Proxy: SMA20 vs SMA50 or just Price > SMA50
                    # For prototype, we use last candle direction + sma slope if available
                    # Assuming df has 'close', 'open'
                    
                    last = df.iloc[-1]
                    trend = 0.0
                    
                    # Candle Direction
                    if last['close'] > last['open']: trend += 0.5
                    elif last['close'] < last['open']: trend -= 0.5
                    
                    # Moving Average Check (if exists)
                    if 'sma_50' in df.columns:
                         if last['close'] > last['sma_50']: trend += 0.5
                         else: trend -= 0.5
                    
                    score += trend * weight
                    total_weight += weight
                    
        if total_weight == 0: return 0.0
        return score / total_weight

    def detect_temporal_dilation(self, market_data_map: Dict[str, Any]) -> str:
        """
        Detects if "Time is speeding up" (Lower TFs exploding relative to Higher TFs).
        High Volatility on M1 while H1 is flat = Pre-Breakout or Noise.
        """
        if '1m' not in market_data_map or '1h' not in market_data_map:
             return "NORMAL"
             
        df_m1 = market_data_map.get('1m')
        df_h1 = market_data_map.get('1h')
        
        if df_m1 is None or df_h1 is None: return "NORMAL"
        
        # Calculate ATR/Range relative to price
        def get_rel_range(df):
            if df.empty: return 0
            dev = (df['high'] - df['low']).mean()
            return dev / df['close'].mean()
            
        vol_m1 = get_rel_range(df_m1.tail(10)) * 60 # Annualize/Normalize to Hour
        vol_h1 = get_rel_range(df_h1.tail(10))
        
        ratio = vol_m1 / (vol_h1 + 1e-9)
        
        if ratio > 5.0:
            return "DILATION_FAST" # M1 is screaming, H1 sleeping
        elif ratio < 0.2:
            return "DILATION_SLOW" # H1 moving, M1 dead
            
        return "SYNCHRONIZED"

class ChronosPattern:
    """
    Analyzes Time-Based Fractals (ICT/Smart Money Concepts).
    Universal Session Support: NY, London, Tokyo, Sydney.
    """
    def __init__(self, utc_offset_hours=2):
        self.offset = utc_offset_hours 
        self.manipulation_ranges = {} # Store ranges per session
        
        # DEFINITIONS (NY Local Time 0-24h)
        self.SESSION_DEF = {
            'NY':     {'open': 8.0,  'manip_start': 8.5, 'manip_end': 9.5, 'expand_start': 9.5, 'reversal': 10.0},
            'LONDON': {'open': 2.0,  'manip_start': 2.0, 'manip_end': 3.0, 'expand_start': 3.0, 'reversal': 4.0}, # 2 AM NY = 7 AM London
            'TOKYO':  {'open': 19.0, 'manip_start': 19.0, 'manip_end': 20.0, 'expand_start': 20.0, 'reversal': 21.0}
        }
        
    def analyze_session_fractal(self, tick: Dict[str, Any], market_data_map: Dict[str, Any], symbol: str = "XAUUSD") -> Dict[str, Any]:
        """
        Detects Manipulation, Expansion, and Distribution phases for the relevant session.
        """
        # 1. Get Current NY Time
        import datetime
        now = datetime.datetime.now()
        ny_hour = now.hour - self.offset
        # Wrap around 24h
        if ny_hour < 0: ny_hour += 24
        if ny_hour >= 24: ny_hour -= 24
        
        ny_minute = now.minute
        ny_time = ny_hour + (ny_minute / 60.0)
        
        # 2. Determine Relevant Sessions based on Asset
        sessions = self._get_relevant_sessions(symbol)
        
        best_narrative = "QUIET"
        best_phase = "OUTSIDE"
        best_bias = "NEUTRAL"
        
        # 3. Check each relevant session
        for sess_name in sessions:
            defs = self.SESSION_DEF[sess_name]
            
            # Check if we are in this session's window (approx 4h block)
            # Simple window check: Open to Open+4h
            start = defs['manip_start']
            end = start + 4.0
            
            # Handle wrap around midnight (Tokyo)
            in_window = False
            if start < end:
                if start <= ny_time < end: in_window = True
            else: # Crosses midnight
                if ny_time >= start or ny_time < (end % 24): in_window = True
                
            if not in_window: continue
            
            # We are in window, analyze fractal
            phase, nar, bias = self._analyze_specific_session(sess_name, defs, ny_time, tick)
            
            if phase != "OUTSIDE":
                best_phase = phase
                best_narrative = f"{sess_name}: {nar}"
                if bias != "NEUTRAL": best_bias = bias
                break # Prioritize first match? Or merge? Usually only 1 active.
                
        return {
            "chronos_phase": best_phase,
            "chronos_narrative": best_narrative,
            "chronos_bias": best_bias,
            "ny_time": f"{int(ny_hour):02d}:{int(ny_minute):02d}"
        }

    def _get_relevant_sessions(self, symbol: str) -> list:
        s = symbol.upper()
        if "JPY" in s: return ['TOKYO', 'LONDON', 'NY'] # JPY moves in all, but Tokyo key
        if "EUR" in s or "GBP" in s or "CHF" in s: return ['LONDON', 'NY']
        if "AUD" in s or "NZD" in s: return ['TOKYO', 'LONDON'] # Proxy Sydney as Tokyo start
        if "USD" in s or "CAD" in s: return ['NY', 'LONDON']
        return ['NY', 'LONDON'] # Default to major liq

    def _analyze_specific_session(self, name, defs, ny_time, tick):
        phase = "OUTSIDE"
        narrative = "WAITING"
        bias = "NEUTRAL"
        price = tick.get('last', (tick['bid']+tick.get('ask', tick['bid']))/2)
        
        # Initialize storage if needed
        if name not in self.manipulation_ranges:
            self.manipulation_ranges[name] = {'high': -1.0, 'low': 999999.0}
            
        # Time logic handling wrap-around is tricky, assume standardized input for now
        # Simplifying: Just check offsets from 'start'
        
        # Normalize time dist from start
        # If ny_time < start (e.g. 1 vs 19), add 24 to ny_time
        calc_time = ny_time
        if calc_time < defs['manip_start']: calc_time += 24
        
        dt = calc_time - defs['manip_start']
        
        # Phase A: Manipulation Window
        manip_dur = defs['manip_end'] - defs['manip_start']
        if 0 <= dt < manip_dur:
            phase = "MANIPULATION_ZONE"
            r = self.manipulation_ranges[name]
            
            if r['high'] == -1.0: # Reset if stale? 
                 # For now assuming persistent object per day. 
                 # Ideally reset if dt is small
                 pass
            
            # Update Range
            if dt < 0.1: # First 6 mins, reset
                 r['high'] = price
                 r['low'] = price
            else:
                 r['high'] = max(r['high'], price)
                 r['low'] = min(r['low'], price)
                 
            self.manipulation_ranges[name] = r
            narrative = f"Building Range"

        # Phase B: Expansion
        elif manip_dur <= dt < (defs['reversal'] - defs['manip_start']):
            phase = "EXPANSION_ZONE"
            r = self.manipulation_ranges[name]
            if r['high'] > 0:
                if price > r['high']:
                    bias = "BULLISH"
                    narrative = "Expansion UP"
                elif price < r['low']:
                    bias = "BEARISH"
                    narrative = "Expansion DOWN"
                else:
                    narrative = "Inside Range"
                    
        # Phase C: Reversal
        elif dt >= (defs['reversal'] - defs['manip_start']):
             phase = "DISTRIBUTION_ZONE"
             narrative = "Macro Reversal"
             
        return phase, narrative, bias

class QuarterlyCycle:
    """
    Implements IPDA (Interbank Price Delivery Algorithm) Quarterly Theory.
    Based on 90-minute Macro Cycles starting from Midnight NY.
    Each 90m Cycle has 4 Quarters (22.5m each).
    """
    def __init__(self, utc_offset_hours=2):
        self.offset = utc_offset_hours

    def analyze_cycle(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the current True Day Cycle (90m) and Phase (A-M-D-X).
        """
        import datetime
        now = datetime.datetime.now()
        ny_hour = now.hour - self.offset
        
        # Normalize to 0-24h
        if ny_hour < 0: ny_hour += 24
        if ny_hour >= 24: ny_hour -= 24
        
        ny_min = now.minute
        
        # Total minutes from Midnight NY
        total_minutes = (ny_hour * 60) + ny_min
        
        # 90-minute Cycle Count
        cycle_idx = int(total_minutes // 90)
        minutes_into_cycle = total_minutes % 90
        
        # Determine Quarter (22.5m blocks)
        quarter_idx = int(minutes_into_cycle // 22.5) # 0, 1, 2, 3
        
        # Interpret Phase
        phase = "WAITING"
        bias = "NEUTRAL"
        
        if quarter_idx == 0:
            phase = "ACCUMULATION"    # Trap Volume / Build Positions (Q1)
        elif quarter_idx == 1:
            phase = "MANIPULATION"    # Judas Swing / Fakeout (Q2)
        elif quarter_idx == 2:
            phase = "DISTRIBUTION"    # True Move / Expansion (Q3)
        elif quarter_idx == 3:
            phase = "CONTINUATION/X"  # Reversal or Continuation (Q4)
            
        return {
            "ipda_cycle_idx": cycle_idx, # Which 90m block of the day
            "quarter_idx": quarter_idx + 1, # 1-4
            "quarter_phase": phase,
            "minutes_into_cycle": minutes_into_cycle,
            "description": f"IPDA Cycle {cycle_idx+1} | Q{quarter_idx+1}: {phase}"
        }
