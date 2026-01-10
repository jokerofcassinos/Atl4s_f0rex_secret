
from datetime import datetime

class TimeFractalEngine:
    """
    Combines ICT Kill Zones with M8 Fibonacci Time Quarters.
    
    Philosophy:
    - Macros operate at specific times (Kill Zones) to inject volatility.
    - Micro-algos operate on fractal cycles (8-minute blocks).
    """
    
    # ICT Kill Zones (UTC hours - approximation)
    # Ideally should be configurable per broker offset.
    # Standard: London Open (7-10), NY Open (12-15), London Close (15-17)
    KILL_ZONES = {
        'LONDON_OPEN': (7, 10),
        'NY_OPEN': (12, 15),
        'LONDON_CLOSE': (15, 17),
        'ASIAN_RANGE': (0, 4) # Accumulation
    }
    
    def get_time_score(self, current_time: datetime) -> dict:
        """
        Evaluates temporal probability.
        
        Returns:
            score: -999 (VETO) to +5 (High Prob)
            tradeable: bool
            reason: str
        """
        score = 0
        reasons = []
        is_tradeable = True
        
        # 1. Kill Zone Check (Macro)
        hour = current_time.hour
        in_zone = False
        
        for zone_name, (start, end) in self.KILL_ZONES.items():
            if start <= hour < end:
                in_zone = True
                score += 3
                reasons.append(f"In {zone_name}")
                break
                
        if not in_zone:
            # Outside kill zones, volatility is lower (lunch, evening)
            pass 
        
        # 2. M8 Fractal Cycle (Micro)
        # 8-minute blocks: 0-2 (Q1), 2-4 (Q2), 4-6 (Q3), 6-8 (Q4)
        minutes = current_time.minute
        seconds = current_time.second
        
        # Calculate position in 8-min cycle
        cycle_minute = minutes % 8
        seconds_into_m8 = (cycle_minute * 60) + seconds
        
        if seconds_into_m8 < 120:
            # Q1: 0-2 mins - DEAD ZONE / ACCUMULATION
            # Algorithms reset. Usually choppy or counter-trend.
            score = -999
            is_tradeable = False
            reasons.append("Q1 Dead Zone")
            
        elif seconds_into_m8 < 240:
            # Q2: 2-4 mins - MANIPULATION / JUDAS SWING
            # Often sets the trap. Risky but high reward if fading.
            score -= 1
            reasons.append("Q2 Trap Zone")
            
        elif seconds_into_m8 < 360:
            # Q3: 4-6 mins - DISTRIBUTION / EXPANSION
            # The "Golden Minute". Valid moves happen here.
            score += 2
            reasons.append("Q3 Golden Zone")
            
        else:
            # Q4: 6-8 mins - CONTINUATION / DECAY
            # Move is done.
            score += 0
            reasons.append("Q4 Decay")
            
        return {
            'score': score,
            'tradeable': is_tradeable,
            'reason': ', '.join(reasons)
        }
