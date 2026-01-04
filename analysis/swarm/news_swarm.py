
import logging
import time
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal
from core.news_scraper import NewsScraper

logger = logging.getLogger("NewsSwarm")

class NewsSwarm(SubconsciousUnit):
    """
    Phase 107: The Oracle of Delphi (News Sentiment Swarm).
    
    Reads the 'Global Consciousness' (Internet) to determine Macro Bias.
    - Risk-On (Bullish): Stocks/Crypto/Yields Up.
    - Risk-Off (Bearish): Gold/Bonds/Dollar Up.
    
    Acts as a 'Supreme Filter' for high-level market conditions.
    """
    
    def __init__(self):
        super().__init__("News_Swarm")
        self.scraper = NewsScraper()
        self.last_scrape_time = 0
        self.scrape_interval = 3600 * 4 # Every 4 hours
        self.current_bias = 0.0
        self.last_news = []
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        # 1. Check if we need to refresh the Oracle
        now = time.time()
        if now - self.last_scrape_time > self.scrape_interval:
            logger.info("ORACLE: Consulting the Internet...")
            try:
                self.current_bias = self.scraper.get_market_sentiment()
                self.last_scrape_time = now
                logger.info(f"ORACLE: Global Sentiment Score: {self.current_bias:.3f}")
            except Exception as e:
                logger.error(f"ORACLE: Vision Clouded. Error: {e}")
                
        # 2. Interpret Bias
        # Bias range: -1.0 (Apocalypse) to +1.0 (Utopia)
        # Threshold: +/- 0.25 to matter
        
        signal = "WAIT"
        conf = 0.0
        meta = {'bias': self.current_bias}
        
        if abs(self.current_bias) > 0.15:
            # Significant Sentiment
            if self.current_bias > 0:
                # Positive Sentiment (Risk-On) -> Good for BTC, Stocks. Bad for Gold/USD?
                # Simplification: Positive = BUY Asset (Bullishness)
                # Beware Inverse Correlations!
                signal = "BUY"
                conf = 50.0 + (abs(self.current_bias) * 50) # Max 100
                meta['reason'] = f"News Oracle: Positive Sentiment ({self.current_bias:.2f})"
            else:
                signal = "SELL"
                conf = 50.0 + (abs(self.current_bias) * 50)
                meta['reason'] = f"News Oracle: Negative Sentiment ({self.current_bias:.2f})"
        else:
            return None # Neutral noise
            
        return SwarmSignal(
            source=self.name,
            signal_type=signal,
            confidence=min(conf, 99.0),
            timestamp=now,
            meta_data=meta
        )
