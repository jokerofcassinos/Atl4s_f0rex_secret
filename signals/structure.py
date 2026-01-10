"""
Laplace Demon - Structure Analysis Module v2.0

Implements institutional SMC (Smart Money Concepts) and BlackRock patterns:

1. SMC Core:
   - Order Blocks
   - Fair Value Gaps (FVG)
   - Break of Structure (BOS)
   - Change of Character (CHoCH)
   - Premium/Discount Zones

2. BlackRock/Aladdin Patterns:
   - Seek and Destroy (Liquidity Sweep)
   - Iceberg Detection (Absorption Blocks)
   - Month-End Rebalancing Filter

3. Institutional Levels:
   - .00/.20/.50/.80 Grid
   - IPDA Data Ranges (20/40/60 Day)
   - True Day Open vs Midnight Open

Named after Laplace's Demon - the deterministic intelligence that knows
all positions of atoms and can predict the future perfectly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum
import logging

logger = logging.getLogger("Laplace-Structure")


class StructureType(Enum):
    """Market structure types."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


@dataclass
class OrderBlock:
    """Represents an institutional order block."""
    type: str  # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    midpoint: float
    timestamp: datetime
    strength: float  # 0-100
    tested: bool = False
    invalidated: bool = False


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (imbalance)."""
    type: str  # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    midpoint: float
    timestamp: datetime
    filled_pct: float = 0.0


@dataclass
class LiquidityPool:
    """Represents external liquidity (stop hunt targets)."""
    level: float
    type: str  # "HIGH" or "LOW"
    strength: int  # Number of touches/rejections
    timestamp: datetime
    swept: bool = False


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer
    
    Implements ICT concepts:
    - Order Blocks
    - Fair Value Gaps
    - Break of Structure
    - Optimal Trade Entry (OTE)
    """
    
    def __init__(self):
        self.order_blocks: List[OrderBlock] = []
        self.fvgs: List[FairValueGap] = []
        self.liquidity_pools: List[LiquidityPool] = []
        self.structure_trend: StructureType = StructureType.RANGING
        
    def analyze(self, df: pd.DataFrame, current_price: float = None) -> Dict:
        """
        Full SMC analysis on price data.
        
        Returns:
            Dict with order_blocks, fvgs, trend, and entry signals
        """
        if df is None or len(df) < 20:
            return {'error': 'Insufficient data'}
        
        # Normalize columns in case of MultiIndex (yfinance)
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            df.columns = [str(c).lower() for c in df.columns]
        except:
            pass
        
        if current_price is None:
            current_price = float(df['close'].iloc[-1])
        
        try:
            # Detect components
            self._detect_order_blocks(df)
            self._detect_fvgs(df)
            self._detect_liquidity_pools(df)
            self._determine_structure(df)
            
            # Validate existing components
            self._validate_order_blocks(current_price)
            self._update_fvg_fill(current_price)
            
            # Find entry opportunities
            entry = self._find_entry_signal(df, current_price)
            
            return {
                'trend': self.structure_trend.value,
                'active_order_blocks': [ob for ob in self.order_blocks if not ob.invalidated][-5:],
                'active_fvgs': [f for f in self.fvgs if f.filled_pct < 100][-5:],
                'liquidity_pools': self.liquidity_pools[-5:],
                'entry_signal': entry
            }
        except Exception as e:
            logger.error(f"STRUCTURE CRASH: {e}")
            return {
                'trend': 'RANGING',
                'active_order_blocks': [],
                'active_fvgs': [],
                'entry_signal': {'direction': None, 'confidence': 0, 'reason': 'Error'}
            }
    
    def _detect_order_blocks(self, df: pd.DataFrame):
        """
        [AGI PERCEPTION] Validated Order Block Detection.
        
        A raw candle pattern is NOT an Order Block. 
        It only becomes one if it causes a Break of Structure (BOS).
        """
        if len(df) < 20: return
        self.order_blocks = []
        
        # We scan for Swing Points first to define structure
        highs = df['high']
        lows = df['low']
        closes = df['close']
        opens = df['open']
        
        # 1. Identify Candidate Blocks (Pattern Recognition)
        # Bullish: Down-candle before Up-move
        # Bearish: Up-candle before Down-move
        
        for i in range(len(df) - 10, 5, -1): # Scan backwards from recent
            # Check Bullish OB Candidate
            if closes.iloc[i] < opens.iloc[i]: # Bearish candle
                # Check for Displacement relative to this candle
                # We need to see price break the HIGH of this candle + structure
                
                # Search forward for BOS (Break of local swing high)
                local_high = highs.iloc[i-3:i].max() # Immediate pre-structure
                
                caused_bos = False
                displacement_strength = 0.0
                
                for j in range(i+1, min(i+10, len(df))):
                    # Displacement Check
                    body_ratio = abs(closes.iloc[j] - opens.iloc[j]) / (abs(closes.iloc[i] - opens.iloc[i]) + 0.00001)
                    if body_ratio > 3.0: # Huge expansion
                         displacement_strength = 100.0
                    
                    # BOS Check
                    if closes.iloc[j] > local_high:
                        caused_bos = True
                        break
                    
                    # Invalidated if we go lower before breaking up
                    if lows.iloc[j] < lows.iloc[i]:
                        break
                
                if caused_bos:
                    ob = OrderBlock(
                        type="BULLISH",
                        top=highs.iloc[i],
                        bottom=lows.iloc[i], 
                        midpoint=(highs.iloc[i] + lows.iloc[i])/2,
                        timestamp=df.index[i],
                        strength=80.0 if displacement_strength > 0 else 50.0 # Validated
                    )
                    self.order_blocks.append(ob)

            # Check Bearish OB Candidate
            elif closes.iloc[i] > opens.iloc[i]: # Bullish candle
                # Check for Displacement DOWN
                local_low = lows.iloc[i-3:i].min()
                
                caused_bos = False
                displacement_strength = 0.0
                
                for j in range(i+1, min(i+10, len(df))):
                    # Displacement Check
                    body_ratio = abs(closes.iloc[j] - opens.iloc[j]) / (abs(closes.iloc[i] - opens.iloc[i]) + 0.00001)
                    if body_ratio > 3.0: 
                         displacement_strength = 100.0
                    
                    # BOS Check
                    if closes.iloc[j] < local_low:
                        caused_bos = True
                        break
                    
                    if highs.iloc[j] > highs.iloc[i]:
                        break
                
                if caused_bos:
                    ob = OrderBlock(
                        type="BEARISH",
                        top=highs.iloc[i],
                        bottom=lows.iloc[i],
                        midpoint=(highs.iloc[i] + lows.iloc[i])/2,
                        timestamp=df.index[i],
                        strength=80.0 if displacement_strength > 0 else 50.0
                    )
                    self.order_blocks.append(ob)
                    
        # Sort and prune
        self.order_blocks.sort(key=lambda x: x.timestamp)
        self.order_blocks = self.order_blocks[-5:] # Only Keep Fresh Context
    
    def _detect_fvgs(self, df: pd.DataFrame):
        """
        Detect Fair Value Gaps (imbalances) - Vectorized.
        
        Bullish FVG: Gap between candle[i] high and candle[i+2] low
        Bearish FVG: Gap between candle[i] low and candle[i+2] high
        """
        if len(df) < 3:
            return
        
        # Reset list to avoid duplicates if re-running
        self.fvgs = []
        
        high = df['high']
        low = df['low']
        
        # Calculate shifts
        # c1 = i, c2 = i+1, c3 = i+2
        c1_high = high
        c1_low = low
        c3_high = high.shift(-2)
        c3_low = low.shift(-2)
        
        # Bullish identification
        # c3.low > c1.high
        bullish_gaps = c3_low - c1_high
        bullish_mask = bullish_gaps > 0
        
        # Bearish identification
        # c3.high < c1.low
        bearish_gaps = c1_low - c3_high
        bearish_mask = bearish_gaps > 0
        
        # Only process the last 30 detected gaps to save memory/time
        # Get indices where gaps exist
        bull_indices = np.where(bullish_mask)[0]
        bear_indices = np.where(bearish_mask)[0]
        
        # Limit to recent ones if list is huge, but usually we iterate small checks in real-time
        # For backtest optimization, we only really care about the most recent ones for interaction
        
        # Add Bullish FVGs (Take last 15)
        for i in bull_indices[-15:]:
            if i + 2 >= len(df): continue
            gap_size = bullish_gaps.iloc[i]
            c2_time = df.index[i+1]
            
            fvg = FairValueGap(
                type="BULLISH",
                top=float(c3_low.iloc[i]),
                bottom=float(c1_high.iloc[i]),
                midpoint=float((c3_low.iloc[i] + c1_high.iloc[i]) / 2),
                timestamp=c2_time
            )
            self.fvgs.append(fvg)
            
        # Add Bearish FVGs (Take last 15)
        for i in bear_indices[-15:]:
            if i + 2 >= len(df): continue
            gap_size = bearish_gaps.iloc[i]
            c2_time = df.index[i+1]
            
            fvg = FairValueGap(
                type="BEARISH",
                top=float(c1_low.iloc[i]),
                bottom=float(c3_high.iloc[i]),
                midpoint=float((c1_low.iloc[i] + c3_high.iloc[i]) / 2),
                timestamp=c2_time
            )
            self.fvgs.append(fvg)
        
        # Sort by timestamp
        self.fvgs.sort(key=lambda x: x.timestamp)
    
    def _detect_liquidity_pools(self, df: pd.DataFrame, lookback: int = 50):
        """
        [AGI PERCEPTION] Fractal Liquidity Scanner.
        Detects specific swing points that the algorithm targets.
        """
        if len(df) < lookback: return
        self.liquidity_pools = []
        
        # Define Swings: High surrounded by 2 lower highs
        # Fractal Period 5
        
        highs = df['high']
        lows = df['low']
        
        for i in range(5, len(df)-5):
            # Fractal High
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                
                # Check if it was swept later?
                # For now, just mark it as a pool
                pool = LiquidityPool(
                    level=highs.iloc[i],
                    type="HIGH",
                    strength=50, # Baseline
                    timestamp=df.index[i]
                )
                self.liquidity_pools.append(pool)
                
            # Fractal Low
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                
                pool = LiquidityPool(
                    level=lows.iloc[i],
                    type="LOW",
                    strength=50,
                    timestamp=df.index[i]
                )
                self.liquidity_pools.append(pool)
                
        # Filter: Only keep pools that haven't been massively violated (> 20 pips)
        # But we WANT them to be swept by < 5 pips (Turtle Soup).
        
        current_price = df['close'].iloc[-1]
        
        active_pools = []
        for pool in self.liquidity_pools:
            # Check interaction with recent price (last 10 candles)
            recent_high = highs.iloc[-10:].max()
            recent_low = lows.iloc[-10:].min()
            
            # TURTLE SOUP LOGIC (AGI)
            # If High Pool was breached by 0.1 to 5.0 pips, mark as SWEPT (Reversal Signal)
            if pool.type == "HIGH":
                diff = recent_high - pool.level
                if 0.00001 < diff < 0.0005: # Swept
                     pool.swept = True
                     pool.strength = 100 # Maximum Interest
                elif diff > 0.0010: # Broken (Trend)
                     continue # Ignore broken pools
            
            if pool.type == "LOW":
                diff = pool.level - recent_low
                if 0.00001 < diff < 0.0005: # Swept
                     pool.swept = True
                     pool.strength = 100
                elif diff > 0.0010: # Broken
                     continue
            
            active_pools.append(pool)
            
        self.liquidity_pools = active_pools[-5:] # Keep closest/freshest
    
    def _determine_structure(self, df: pd.DataFrame):
        """Determine market structure (bullish/bearish/ranging)."""
        if len(df) < 20:
            return
        
        # Simple swing structure
        recent = df.iloc[-20:]
        midpoint = len(recent) // 2
        
        first_half_high = recent.iloc[:midpoint]['high'].max()
        first_half_low = recent.iloc[:midpoint]['low'].min()
        second_half_high = recent.iloc[midpoint:]['high'].max()
        second_half_low = recent.iloc[midpoint:]['low'].min()
        
        if second_half_high > first_half_high and second_half_low > first_half_low:
            self.structure_trend = StructureType.BULLISH
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            self.structure_trend = StructureType.BEARISH
        else:
            self.structure_trend = StructureType.RANGING
    
    def _validate_order_blocks(self, current_price: float):
        """Mark OBs as tested or invalidated."""
        for ob in self.order_blocks:
            if ob.invalidated:
                continue
            
            if ob.type == "BULLISH":
                # Tested if price touched zone
                if ob.bottom <= current_price <= ob.top:
                    ob.tested = True
                # Invalidated if price closes below
                if current_price < ob.bottom:
                    ob.invalidated = True
            else:  # BEARISH
                if ob.bottom <= current_price <= ob.top:
                    ob.tested = True
                if current_price > ob.top:
                    ob.invalidated = True
    
    def _update_fvg_fill(self, current_price: float):
        """Update FVG fill percentage."""
        for fvg in self.fvgs:
            gap_size = fvg.top - fvg.bottom
            if gap_size <= 0:
                fvg.filled_pct = 100
                continue
            
            if fvg.type == "BULLISH":
                if current_price <= fvg.bottom:
                    fvg.filled_pct = 100
                elif current_price < fvg.top:
                    filled = fvg.top - current_price
                    fvg.filled_pct = (filled / gap_size) * 100
            else:
                if current_price >= fvg.top:
                    fvg.filled_pct = 100
                elif current_price > fvg.bottom:
                    filled = current_price - fvg.bottom
                    fvg.filled_pct = (filled / gap_size) * 100
    
    def _find_entry_signal(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Find optimal entry based on SMC confluence."""
        signal = {
            'direction': None,
            'confidence': 0,
            'sl_price': None,
            'tp_price': None,
            'reason': 'No confluence'
        }
        
        # Check for OB + FVG confluence
        for ob in self.order_blocks:
            if ob.invalidated:
                continue
            
            # Price in OB zone
            if ob.bottom <= current_price <= ob.top:
                # Check for FVG confluence
                for fvg in self.fvgs:
                    if fvg.filled_pct >= 100:
                        continue
                    
                    # Same direction FVG overlapping
                    if fvg.type == ob.type:
                        overlap = max(0, min(ob.top, fvg.top) - max(ob.bottom, fvg.bottom))
                        if overlap > 0:
                            signal = {
                                'direction': 'BUY' if ob.type == "BULLISH" else 'SELL',
                                'confidence': min(90, 50 + ob.strength * 0.3 + (100 - fvg.filled_pct) * 0.2),
                                'sl_price': ob.bottom - 0.0010 if ob.type == "BULLISH" else ob.top + 0.0010,
                                'tp_price': None,  # Calculate based on structure
                                'reason': f'{ob.type} OB + FVG confluence'
                            }
                            break
        
        return signal


class InstitutionalLevels:
    """
    Institutional Price Levels
    
    Implements:
    - .00/.20/.50/.80 Grid (Psychological levels)
    - IPDA Data Ranges (20/40/60 day high/low)
    - True Day Open vs Midnight Open
    """
    
    def __init__(self):
        self.midnight_open: float = 0.0
        self.ny_open: float = 0.0
        self.ipda_levels: Dict = {}
    
    def get_psychological_grid(self, current_price: float) -> Dict:
        """
        Get nearest .00/.20/.50/.80 levels.
        
        These are institutional order placement zones.
        """
        base = int(current_price)
        fractional = current_price - base
        
        # Find nearest level
        levels = [0.00, 0.20, 0.50, 0.80, 1.00]
        
        above = []
        below = []
        
        for offset in range(-1, 3):
            for lvl in levels:
                full_level = base + offset + lvl
                if full_level > current_price:
                    above.append(full_level)
                elif full_level < current_price:
                    below.append(full_level)
        
        above.sort()
        below.sort(reverse=True)
        
        return {
            'nearest_above': above[:3],
            'nearest_below': below[:3],
            'level_type': self._classify_level(fractional)
        }
    
    def _classify_level(self, frac: float) -> str:
        """Classify how close we are to a key level."""
        key_levels = [0.00, 0.20, 0.50, 0.80]
        for lvl in key_levels:
            if abs(frac - lvl) < 0.01:  # Within 1 pip
                return f"AT_{int(lvl*100):02d}"
        return "BETWEEN"
    
    def calculate_ipda_ranges(self, df_daily: pd.DataFrame) -> Dict:
        """
        Calculate IPDA (Interbank Price Delivery Algorithm) ranges.
        
        The algorithm looks back 20/40/60 days for liquidity targets.
        """
        if df_daily is None or len(df_daily) < 60:
            return {}
        
        ranges = {}
        
        for days in [20, 40, 60]:
            if len(df_daily) >= days:
                recent = df_daily.iloc[-days:]
                ranges[f'{days}_day'] = {
                    'high': recent['high'].max(),
                    'low': recent['low'].min(),
                    'range': recent['high'].max() - recent['low'].min()
                }
        
        self.ipda_levels = ranges
        return ranges
    
    def set_session_opens(self, df: pd.DataFrame, current_date: datetime = None):
        """
        Set key session open prices.
        
        - Midnight Open (00:00 NY)
        - NY Open (08:30 NY)
        """
        if df is None or len(df) < 50:
            return
        
        if current_date is None:
            current_date = datetime.now()
        
        # Find midnight candle (00:00 hour)
        midnight_mask = df.index.hour == 0
        midnight_candles = df[midnight_mask]
        if len(midnight_candles) > 0:
            self.midnight_open = midnight_candles.iloc[-1]['open']
        
        # Find NY open candle (13:30 UTC = 08:30 NY)
        ny_mask = (df.index.hour == 13) & (df.index.minute >= 30)
        ny_candles = df[ny_mask]
        if len(ny_candles) > 0:
            self.ny_open = ny_candles.iloc[-1]['open']
    
    def analyze_session_bias(self, current_price: float) -> Dict:
        """
        Analyze bias based on session opens.
        
        - Above both: Strong Bullish
        - Below both: Strong Bearish
        - Between: Consolidation (danger zone)
        """
        if self.midnight_open == 0 or self.ny_open == 0:
            return {'bias': 'UNKNOWN', 'reason': 'Session opens not set'}
        
        above_midnight = current_price > self.midnight_open
        above_ny = current_price > self.ny_open
        
        if above_midnight and above_ny:
            return {
                'bias': 'STRONG_BULLISH',
                'reason': 'Above both session opens',
                'action': 'Only look for LONGS'
            }
        elif not above_midnight and not above_ny:
            return {
                'bias': 'STRONG_BEARISH',
                'reason': 'Below both session opens',
                'action': 'Only look for SHORTS'
            }
        else:
            return {
                'bias': 'NEUTRAL',
                'reason': 'Between session opens - consolidation',
                'action': 'AVOID or tight range trades only'
            }


class BlackRockPatterns:
    """
    BlackRock/Aladdin Pattern Detection
    
    Implements institutional execution patterns:
    1. Seek and Destroy (Liquidity sweep)
    2. Iceberg Detection (Absorption)
    3. Month-End Rebalancing
    """
    
    def __init__(self):
        self.last_seek_destroy: Optional[Dict] = None
        self.iceberg_zones: List[Dict] = []
        
    def detect_seek_and_destroy(self, df: pd.DataFrame) -> Dict:
        """
        Detect "Seek and Destroy" pattern.
        
        Pattern: Price takes out both highs AND lows in rapid succession.
        This is the Aladdin creating liquidity by stopping both sides.
        
        After the pattern, price typically returns to the mid-point.
        """
        if df is None or len(df) < 10:
            return {'detected': False}
        
        recent = df.iloc[-10:]
        
        # Look for Outside Bar or rapid high/low sweep
        for i in range(1, len(recent)):
            candle = recent.iloc[i]
            prev_high = recent.iloc[:i]['high'].max()
            prev_low = recent.iloc[:i]['low'].min()
            
            # Outside bar pattern
            if candle['high'] > prev_high and candle['low'] < prev_low:
                midpoint = (candle['high'] + candle['low']) / 2
                
                self.last_seek_destroy = {
                    'detected': True,
                    'high': candle['high'],
                    'low': candle['low'],
                    'midpoint': midpoint,
                    'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                    'action': 'Wait for return to midpoint, then trade mean reversion'
                }
                return self.last_seek_destroy
        
        return {'detected': False}
    
    def detect_iceberg_absorption(self, df: pd.DataFrame, tolerance_pips: float = 2.0) -> Dict:
        """
        Detect Iceberg Order / Absorption Block.
        
        Pattern: Multiple candles with wicks touching same level but bodies closing away.
        This indicates an algorithm absorbing all orders at a fixed price.
        
        When absorption ends, price will explode in the direction of the bodies.
        """
        if df is None or len(df) < 5:
            return {'detected': False}
        
        recent = df.iloc[-10:]
        pip_size = 0.0001  # GBPUSD
        tolerance = tolerance_pips * pip_size
        
        # Check for absorption at lows (bullish absorption)
        low_levels = recent['low'].values
        
        for i in range(len(low_levels) - 2):
            base_low = low_levels[i]
            touches = 0
            bullish_closes = 0
            
            for j in range(i, min(i + 5, len(low_levels))):
                if abs(low_levels[j] - base_low) < tolerance:
                    touches += 1
                    if recent.iloc[j]['close'] > recent.iloc[j]['open']:
                        bullish_closes += 1
            
            if touches >= 3 and bullish_closes >= 2:
                zone = {
                    'type': 'BULLISH_ABSORPTION',
                    'level': base_low,
                    'touches': touches,
                    'strength': bullish_closes / touches * 100,
                    'entry_trigger': recent['high'].iloc[-3:].max(),
                    'sl': base_low - (tolerance * 2),
                    'reason': f'Iceberg BUY wall detected with {touches} touches'
                }
                self.iceberg_zones.append(zone)
                return {'detected': True, 'zone': zone}
        
        # Check for absorption at highs (bearish absorption)
        high_levels = recent['high'].values
        
        for i in range(len(high_levels) - 2):
            base_high = high_levels[i]
            touches = 0
            bearish_closes = 0
            
            for j in range(i, min(i + 5, len(high_levels))):
                if abs(high_levels[j] - base_high) < tolerance:
                    touches += 1
                    if recent.iloc[j]['close'] < recent.iloc[j]['open']:
                        bearish_closes += 1
            
            if touches >= 3 and bearish_closes >= 2:
                zone = {
                    'type': 'BEARISH_ABSORPTION',
                    'level': base_high,
                    'touches': touches,
                    'strength': bearish_closes / touches * 100,
                    'entry_trigger': recent['low'].iloc[-3:].min(),
                    'sl': base_high + (tolerance * 2),
                    'reason': f'Iceberg SELL wall detected with {touches} touches'
                }
                self.iceberg_zones.append(zone)
                return {'detected': True, 'zone': zone}
        
        return {'detected': False}
    
    def is_month_end_rebalancing(self, current_date: datetime = None) -> Dict:
        """
        Check if we're in month-end rebalancing window.
        
        Last 3 business days of month: Funds must rebalance portfolios.
        Expect strong, momentum-ignoring moves especially at 16:00 London.
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Get last day of month
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)
        
        last_day = next_month - timedelta(days=1)
        
        # Count business days until end of month
        days_to_end = 0
        check_date = current_date
        while check_date <= last_day:
            if check_date.weekday() < 5:  # Monday-Friday
                days_to_end += 1
            check_date += timedelta(days=1)
        
        in_window = days_to_end <= 3
        
        # Check if we're near London Fix time (16:00 UTC)
        at_london_fix = current_date.hour == 16
        
        return {
            'in_rebalancing_window': in_window,
            'days_to_month_end': days_to_end,
            'at_london_fix': at_london_fix,
            'action': 'Trade AGAINST monthly trend if in window' if in_window else 'Normal trading',
            'warning': 'MONTH-END REBALANCING: Expect abnormal flows!' if in_window and at_london_fix else None
        }


class VectorCandleTheory:
    """
    Vector Candle Theory (Market Maker Pain)
    
    A "Vector Candle" is 200%+ larger than average.
    It's not trend - it's the Market Maker trapping traders.
    The algorithm MUST return to relieve these positions.
    """
    
    def __init__(self, atr_multiplier: float = 3.0):
        self.atr_multiplier = atr_multiplier
        self.last_vector: Optional[Dict] = None
    
    def detect_vector_candle(self, df: pd.DataFrame) -> Dict:
        """
        Detect Vector Candle (oversized displacement).
        
        After detection, wait for opposite color candle, then fade.
        """
        if df is None or len(df) < 20:
            return {'detected': False}
        
        # Calculate ATR
        atr = self._calculate_atr(df, 14)
        
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        
        # Vector candle = body > 3x ATR
        if body > atr * self.atr_multiplier:
            direction = "UP" if last_candle['close'] > last_candle['open'] else "DOWN"
            
            self.last_vector = {
                'detected': True,
                'direction': direction,
                'open': last_candle['open'],
                'close': last_candle['close'],
                'high': last_candle['high'],
                'low': last_candle['low'],
                'body_size': body,
                'atr': atr,
                'ratio': body / atr,
                'target': last_candle['open'],  # Price of origin
                'action': f'Wait for reversal candle, then FADE {direction}'
            }
            return self.last_vector
        
        return {'detected': False}
    
    def check_reversal_entry(self, df: pd.DataFrame) -> Dict:
        """
        Check if reversal candle formed after vector.
        
        If vector was UP and next candle is DOWN, enter SHORT targeting vector origin.
        """
        if self.last_vector is None or not self.last_vector.get('detected'):
            return {'entry': False}
        
        if len(df) < 2:
            return {'entry': False}
        
        last_candle = df.iloc[-1]
        candle_dir = "UP" if last_candle['close'] > last_candle['open'] else "DOWN"
        
        # Reversal = opposite direction candle
        if candle_dir != self.last_vector['direction']:
            entry_dir = "SELL" if self.last_vector['direction'] == "UP" else "BUY"
            
            return {
                'entry': True,
                'direction': entry_dir,
                'entry_price': last_candle['close'],
                'target': self.last_vector['target'],
                'sl': self.last_vector['high'] if entry_dir == "SELL" else self.last_vector['low'],
                'reason': f'Vector Recovery: Fading {self.last_vector["direction"]} vector'
            }
        
        return {'entry': False}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return 0.0020  # Default 20 pips
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        return np.mean(tr[-period:])


class GannGeometry:
    """
    Gann's Law of Vibration (Geometric Grid)
    
    The number 144 (12x12) is the "signature" of the market.
    Price finds support/resistance at 36/72/144 pip intervals.
    """
    
    SACRED_NUMBERS = [36, 72, 144, 288]  # Gann's geometric intervals
    
    def __init__(self, pip_size: float = 0.0001):
        self.pip_size = pip_size
    
    def calculate_gann_grid(self, swing_low: float) -> Dict:
        """
        Calculate Gann geometric levels from a swing point.
        
        Price tends to find resistance/support at these exact intervals.
        """
        levels = {}
        
        for interval in self.SACRED_NUMBERS:
            price_interval = interval * self.pip_size
            levels[f'L{interval}'] = swing_low + price_interval
        
        return {
            'anchor': swing_low,
            'levels': levels,
            'action': 'Use for partial TP or reversal zones'
        }
    
    def check_geometric_exhaustion(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check if price has moved exactly 144 points (or multiple).
        
        If yes, expect reversal or consolidation.
        """
        if df is None or len(df) < 50:
            return {'exhausted': False}
        
        if direction == "UP":
            move_start = df['low'].iloc[-50:].min()
            current = df['close'].iloc[-1]
            move_pips = (current - move_start) / self.pip_size
        else:
            move_start = df['high'].iloc[-50:].max()
            current = df['close'].iloc[-1]
            move_pips = (move_start - current) / self.pip_size
        
        # Check if near sacred number
        for num in self.SACRED_NUMBERS:
            if abs(move_pips - num) < 5:  # Within 5 pips
                return {
                    'exhausted': True,
                    'sacred_number': num,
                    'actual_move': move_pips,
                    'recommendation': f'CAUTION: {direction} move hit {num} pip Gann level. Expect reversal.'
                }
        
        return {'exhausted': False, 'current_move': move_pips}


class TeslaVortex:
    """
    Tesla 3-6-9 Pattern (Cycle Exhaustion)
    
    The market rarely makes more than 3 impulses without correction.
    At 9 consecutive candles, reversal probability is ~70%.
    """
    
    def count_consecutive_candles(self, df: pd.DataFrame) -> Dict:
        """
        Count consecutive same-color candles.
        
        At 9 consecutive, prepare for reversal.
        """
        if df is None or len(df) < 10:
            return {'count': 0}
        
        recent = df.iloc[-15:]
        
        bullish_count = 0
        bearish_count = 0
        
        # Count from end
        for i in range(len(recent) - 1, -1, -1):
            candle = recent.iloc[i]
            if candle['close'] > candle['open']:
                if bearish_count > 0:
                    break
                bullish_count += 1
            else:
                if bullish_count > 0:
                    break
                bearish_count += 1
        
        count = max(bullish_count, bearish_count)
        direction = "BULLISH" if bullish_count > bearish_count else "BEARISH"
        
        result = {
            'count': count,
            'direction': direction,
            'warning': None,
            'action': 'Normal'
        }
        
        if count >= 9:
            result['warning'] = f'VORTEX 9: {count} consecutive {direction} candles. HIGH reversal probability!'
            result['action'] = f'Prepare to FADE {direction}'
        elif count >= 6:
            result['warning'] = f'VORTEX 6: {count} consecutive candles. Momentum slowing.'
            result['action'] = 'Reduce position size'
        elif count >= 3:
            result['warning'] = f'VORTEX 3: {count} consecutive. Normal impulse wave.'
        
        return result
    
    def count_impulse_waves(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Count major impulse waves (swings).
        
        After 3 impulses in same direction, expect deep correction.
        """
        if df is None or len(df) < lookback:
            return {'waves': 0}
        
        # Simplified swing detection
        recent = df.iloc[-lookback:]
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
        
        # Count higher highs / lower lows
        hh_count = 0
        ll_count = 0
        
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i-1][1]:
                hh_count += 1
            else:
                break
        
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] < swing_lows[i-1][1]:
                ll_count += 1
            else:
                break
        
        waves = max(hh_count, ll_count)
        direction = "UP" if hh_count > ll_count else "DOWN"
        
        result = {
            'waves': waves,
            'direction': direction,
            'exhausted': waves >= 3,
            'action': f'Wait for correction' if waves >= 3 else 'Normal trading'
        }
        
        if waves >= 3:
            result['warning'] = f'3 DRIVES COMPLETE: {waves} {direction} waves. Expect deep correction!'
        
        return result
