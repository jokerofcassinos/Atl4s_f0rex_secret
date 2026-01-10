
class SpreadSimulator:
    """
    Simulates realistic spreads for different currency pairs and market conditions.
    """
    
    # Base spreads in standard pips (1 pip = 0.0001 for forex)
    # XAUUSD is in dollars (e.g. 0.25 = 25 cents)
    BASE_SPREADS = {
        'EURUSD': 0.00008,  # 0.8 pips
        'GBPUSD': 0.00020,  # 2.0 pips (20 points for Standard Account)
        'USDJPY': 0.009,    # 0.9 pips (JPY quoting)
        'USDCAD': 0.00013,  # 1.3 pips
        'USDCHF': 0.00015,  # 1.5 pips
        'XAUUSD': 0.25,     # 25 cents
    }
    
    # Artificial Padding (to force conservative backtesting)
    SPREAD_PADDING_POINTS = {
        'GBPUSD': 0.00005,  # Extra 5 points safety margin
    }
    
    def __init__(self):
        pass
        
    def get_spread(self, symbol: str, volatility_factor: float = 1.0, current_hour: int = 14) -> float:
        """
        Returns the dynamic spread for the given symbol and conditions.
        """
        base = self.BASE_SPREADS.get(symbol, 0.00010) # Default 1 pip
        padding = self.SPREAD_PADDING_POINTS.get(symbol, 0.0)
        
        # Adjust for JPY pairs if symbol not in map but contains JPY
        if symbol not in self.BASE_SPREADS and "JPY" in symbol:
             base = 0.012 # 1.2 pips
             
        # Session Factor: Spreads are higher during rollover (21-23 GMT often, assume 17-18 EST = 22-23)
        session_factor = 1.0
        if current_hour >= 21 or current_hour <= 1:
            session_factor = 2.5 # Huge spread during rollover
            
        # Volatility Factor (e.g. News)
        final_spread = (base * volatility_factor * session_factor) + padding
        
        return final_spread
