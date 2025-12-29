import pandas as pd
import logging

logger = logging.getLogger("Atl4s-SupplyDemand")

class SupplyDemand:
    def __init__(self):
        pass

        self.zones = [] # List of active zones: {'type': 'DEMAND', 'top': 0, 'bottom': 0, 'touches': 0, 'strength': 100, 'created_at': index}

    def _get_psychological_score(self, price):
        """Checks proximity to 00 or 50 levels."""
        decimal_part = price % 10
        # Check XX00 or XX50
        dist_00 = abs(decimal_part - 0)
        dist_50 = abs(decimal_part - 5)
        
        if dist_00 < 0.5 or dist_00 > 9.5: return 20 # Strong Psych Level
        if dist_50 < 0.5: return 10 # Mid Psych Level
        return 0

    def analyze(self, df):
        """
        Identifies Supply (RBD) and Demand (DBR) zones with Advanced Logic.
        Returns:
            score (int): 0-100
            direction (int): 1 (Buy/Demand), -1 (Sell/Supply), 0 (Neutral)
            zone_info (dict): Details about the zone
        """
        if df is None or len(df) < 50:
            return 0, 0, {}
            
        df = df.copy()
        curr_price = df.iloc[-1]['close']
        
        # 1. Detect New Zones (Simplified Scan)
        # We scan the last 50 candles for fresh zones
        new_zones = []
        df['body'] = abs(df['close'] - df['open'])
        avg_body = df['body'].rolling(20).mean()
        avg_vol = df['volume'].rolling(20).mean()
        
        for i in range(len(df)-2, len(df)-50, -1):
            c_out = df.iloc[i]
            c_base = df.iloc[i-1]
            c_in = df.iloc[i-2]
            
            # Logic for DBR/RBD (Same as before but stricter)
            is_strong_green = c_out['close'] > c_out['open'] and c_out['body'] > avg_body.iloc[i] * 1.5
            is_strong_red = c_out['close'] < c_out['open'] and c_out['body'] > avg_body.iloc[i] * 1.5
            is_base = c_base['body'] < (avg_body.iloc[i] * 0.6)
            
            # Quality Calculation: Breakout Intensity + Volume Surge + Age Bias
            intensity = c_out['body'] / (avg_body.iloc[i] + 1e-9)
            vol_surge = c_out['volume'] / (avg_vol.iloc[i] + 1e-9)
            
            # Age Bias: Closer to current candle = higher score
            age_factor = (50 - (len(df) - 1 - i)) / 50.0  # 1.0 (new) to 0.0 (old)
            
            quality = min(98, 45 + (intensity * 12) + (vol_surge * 10) + (age_factor * 5))
            
            if is_strong_green and is_base and c_in['close'] < c_in['open']:
                # Demand
                top = max(c_base['open'], c_base['close'])
                bottom = c_base['low']
                new_zones.append({'type': 'DEMAND', 'top': top, 'bottom': bottom, 'touches': 0, 'strength': 100, 'quality': quality})
                
            elif is_strong_red and is_base and c_in['close'] > c_in['open']:
                # Supply
                top = c_base['high']
                bottom = min(c_base['open'], c_base['close'])
                new_zones.append({'type': 'SUPPLY', 'top': top, 'bottom': bottom, 'touches': 0, 'strength': 100, 'quality': quality})

        # 2. Evaluate Current Price vs Zones (The Fortress Logic)
        best_score = 0
        best_dir = 0
        best_info = {}
        
        for zone in new_zones:
            # Check if price is interacting with this zone
            # Demand: Price inside or near top
            if zone['type'] == 'DEMAND':
                if zone['bottom'] <= curr_price <= zone['top'] * 1.01:
                    # Interaction!
                    base_score = zone.get('quality', 69.5)
                    
                    # Permeability Penalty (Simulated)
                    # If price has been here before in the last 20 candles, reduce score
                    # (This is a stateless approximation of touches)
                    recent_touches = df[(df['low'] <= zone['top']) & (df.index > df.index[-20])].shape[0]
                    if recent_touches > 3:
                        zone['strength'] = 20 # Weak
                        logger.info("Fortress: Demand Zone Weakened (Too many touches)")
                    elif recent_touches > 1:
                        zone['strength'] = 80 # Used
                    
                    # Psychological Boost
                    psych_boost = self._get_psychological_score(zone['top'])
                    
                    final_score = (base_score * (zone['strength'] / 100.0)) + psych_boost
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_dir = 1
                        best_info = zone
                        
            # Supply: Price inside or near bottom
            elif zone['type'] == 'SUPPLY':
                if zone['bottom'] * 0.99 <= curr_price <= zone['top']:
                    base_score = zone.get('quality', 69.5)
                    
                    recent_touches = df[(df['high'] >= zone['bottom']) & (df.index > df.index[-20])].shape[0]
                    if recent_touches > 3:
                        zone['strength'] = 20
                        logger.info(f"Fortress: Supply Zone Weakened ({recent_touches} touches)")
                    elif recent_touches > 1:
                        zone['strength'] = 80
                        
                    psych_boost = self._get_psychological_score(zone['bottom'])
                    
                    final_score = (base_score * (zone['strength'] / 100.0)) + psych_boost
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_dir = -1
                        best_info = zone

        if best_score > 0:
            logger.info(f"Fortress S&R: {best_info['type']} at {best_info['bottom']}-{best_info['top']} (Score: {best_score:.1f})")
            
        return best_score, best_dir, best_info
