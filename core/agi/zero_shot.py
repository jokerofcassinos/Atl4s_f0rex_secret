
import logging
import re
from typing import Dict, Any

logger = logging.getLogger("ZeroShot")

class ZeroShotAnalyst:
    """
    Phase 147: Zero-Shot Analyst.
    
    System #2: Cognitive Architecture.
    Enables the AGI to reason about assets it has never encountered before
    by mapping them to known ontological categories.
    """
    def __init__(self):
        self.known_classes = {
            "CRYPTO": {"volatility": "EXTREME", "session": "24/7", "safe_haven": False},
            "FOREX_MAJOR": {"volatility": "LOW", "session": "24/5", "safe_haven": True},
            "FOREX_EXOTIC": {"volatility": "HIGH", "session": "24/5", "safe_haven": False},
            "COMMODITY": {"volatility": "MODERATE", "session": "FUTURES", "safe_haven": "VARIES"},
            "INDEX": {"volatility": "MODERATE", "session": "MARKET_HOURS", "safe_haven": False},
        }
        
    def classify_asset(self, symbol: str) -> str:
        """
        Infers asset class from symbol string using Zero-Shot heuristics.
        """
        symbol = symbol.upper()
        
        # Heuristic 1: Crypto (BTC, ETH, etc)
        if any(x in symbol for x in ["BTC", "ETH", "SOL", "DOGE", "SHIB", "XRP"]):
            return "CRYPTO"
            
        # Heuristic 2: Indices (US30, NAS100, SPX)
        if any(x in symbol for x in ["US30", "NAS", "SPX", "GER30", "HK50"]):
            return "INDEX"
            
        # Heuristic 3: Commodities (XAU, XAG, OIL, WTI)
        if any(x in symbol for x in ["XAU", "XAG", "OIL", "WTI", "NG"]):
            return "COMMODITY"
            
        # Heuristic 4: Forex Majors (USD, EUR, JPY, GBP combinations)
        majors = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]
        # If symbol contains two majors (e.g. EURUSD)
        matches = sum(1 for m in majors if m in symbol)
        if matches >= 2:
            return "FOREX_MAJOR"
            
        # Fallback: Exotic Forex or Unknown
        if "USD" in symbol:
            return "FOREX_EXOTIC"
            
        return "UNKNOWN"
        
    def get_trading_params(self, symbol: str) -> Dict[str, Any]:
        """
        Generates trading parameters for a completely null/unknown asset
        by assuming its class properties.
        """
        asset_class = self.classify_asset(symbol)
        profile = self.known_classes.get(asset_class, self.known_classes["FOREX_EXOTIC"])
        
        params = {}
        
        # Risk Multiplier
        if profile['volatility'] == "EXTREME":
            params['risk_mult'] = 0.5 # Half size for crypto
        elif profile['volatility'] == "LOW":
            params['risk_mult'] = 1.0 # Full size for majors
        else:
            params['risk_mult'] = 0.75
            
        # Stop Loss Padding
        if profile['volatility'] == "EXTREME":
            params['sl_padding'] = "WIDE"
        else:
            params['sl_padding'] = "TIGHT"
            
        logger.info(f"ZERO-SHOT: Analyzed novel asset {symbol}. Class: {asset_class}. Params: {params}")
        return params
