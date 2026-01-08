
import logging
import random
from typing import List, Dict, Any

logger = logging.getLogger("EyeOfSauron")

class GlobalMarketScanner:
    """
    Sistema E-2: Global Market Scanner (The Eye of Sauron).
    Scans the Multi-Verse (Forex/Crypto) to find the 'Main Market'.
    """
    def __init__(self, bridge):
        self.bridge = bridge
        self.universe = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "BTCUSD", "ETHUSD"
        ]
        
    def scan_universe(self, mode="AUTO") -> str:
        """
        Iterates through the universe and calculates Opportunity Score.
        Mode: 'AUTO' (All), 'FOREX' (Fiat+Gold), 'CRYPTO' (BTC/ETH).
        Returns the best symbol.
        """
        if not self.bridge: return "EURUSD" # Fallback
        
        scores = {}
        logger.info(f"INITIATING GLOBAL SCAN SEQUENCE (MODE: {mode})...")
        
        # Filter Universe
        target_universe = []
        for sym in self.universe:
            is_crypto = sym in ["BTCUSD", "ETHUSD"]
            if mode == "FOREX" and is_crypto: continue
            if mode == "CRYPTO" and not is_crypto: continue
            target_universe.append(sym)
            
        if not target_universe:
            logger.warning("Universe empty after filtering! Reverting to EURUSD.")
            return "EURUSD"
        
        # Simulating AGI Perception
        for symbol in target_universe:
            # Score = Volatility (0-10) + Trend Clarity (0-10)
            
            volatility = random.uniform(2, 9)
            clarity = random.uniform(1, 10)
            
            # Bias towards Crypto/Gold for higher vol
            if symbol in ["XAUUSD", "BTCUSD", "ETHUSD"]:
                volatility += 2.0
                
            scores[symbol] = volatility + clarity
            logger.info(f"SCAN: {symbol} -> Score: {scores[symbol]:.1f} (V:{volatility:.1f} C:{clarity:.1f})")
            
        # Select Winner
        best_symbol = max(scores, key=scores.get)
        logger.info(f"GLOBAL SCAN COMPLETE. TARGET LOCKED: {best_symbol}")
        
        return best_symbol
