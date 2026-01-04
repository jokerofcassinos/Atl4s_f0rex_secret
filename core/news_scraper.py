
import logging
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from typing import List, Dict, Tuple
import re

logger = logging.getLogger("NewsScraper")

class NewsScraper:
    """
    Phase 107: The Oracle of Delphi (News Scraper).
    Fetches RSS feeds from major financial news sources and performs basic Sentiment Analysis.
    """
    
    FEEDS = [
        # Investing.com Forex News
        "https://www.investing.com/rss/news_1.rss", 
        # DailyFX Real Time News
        "https://www.dailyfx.com/feeds/market-news",
        # CNBS Top News (General Macro)
        "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    ]
    
    # Basic Bag-of-Words Sentiment Dictionary
    # We focus on keywords that move Gold/Dollar/Crypto
    KEYWORDS = {
        'hawk': -0.8, 'hike': -0.7, 'inflation': -0.6, 'soar': 0.5, 'surge': 0.6,
        'plunge': -0.8, 'crash': -0.9, 'crisis': -0.7, 'war': -0.5, 'tension': -0.3,
        'record': 0.4, 'rally': 0.7, 'bull': 0.6, 'bear': -0.6,
        'stimulus': 0.7, 'cut': 0.5, 'dovish': 0.6, 'support': 0.3,
        'resistance': -0.2, 'breakout': 0.6, 'collapse': -0.8,
        'bitcoin': 0.0, 'gold': 0.0, # Just nouns
        'sec': -0.4, 'ban': -0.8, 'regulation': -0.3, 'approval': 0.7
    }
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_all(self) -> List[Dict]:
        """
        Fetches news from all feeds.
        Returns list of {'title': str, 'link': str, 'sentiment': float}
        """
        news_items = []
        
        for url in self.FEEDS:
            try:
                # logger.info(f"Fetching news from {url}...")
                response = requests.get(url, headers=self.headers, timeout=5)
                if response.status_code == 200:
                    items = self._parse_rss(response.content)
                    news_items.extend(items)
                else:
                    logger.warning(f"Failed to fetch {url}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                
        return news_items

    def _parse_rss(self, xml_content) -> List[Dict]:
        items = []
        try:
            root = ET.fromstring(xml_content)
            # RSS 2.0 usually has channel/item
            for item in root.findall('./channel/item'):
                title = item.find('title').text
                link = item.find('link').text
                
                if title:
                    sentiment = self._analyze_sentiment(title)
                    items.append({
                        'title': title,
                        'link': link,
                        'sentiment': sentiment
                    })
        except Exception as e:
            pass # XML parsing error usually
            
        return items

    def _analyze_sentiment(self, text: str) -> float:
        """
        Calculates a score between -1.0 (Bearish/Negative) and 1.0 (Bullish/Positive).
        This is a crude heuristic but faster than loading a jagged LLM locally.
        """
        text = text.lower()
        score = 0.0
        word_count = 0
        
        # Tokenize (simple split)
        tokens = re.findall(r'\w+', text)
        
        for token in tokens:
            if token in self.KEYWORDS:
                score += self.KEYWORDS[token]
                word_count += 1
                
        # Keyword Context? "Gold plunges" -> Gold Negative. "Dollar plunges" -> Gold Positive?
        # This requires Subject-Verb association.
        # For now, we assume General Market Sentiment (Risk-On vs Risk-Off).
        # "Rally" is Risk-On (+). "Crash" is Risk-Off (-).
        
        # Normalized score
        if word_count > 0:
            final_score = score / max(1, word_count)
            # Clamp
            return max(-1.0, min(final_score, 1.0))
        
        return 0.0

    def get_market_sentiment(self) -> float:
        """
        Returns the aggregate sentiment of the market.
        """
        news = self.fetch_all()
        if not news: return 0.0
        
        total = 0.0
        count = 0
        for n in news:
            if n['sentiment'] != 0.0:
                total += n['sentiment']
                count += 1
                
        if count == 0: return 0.0
        
        return total / count
