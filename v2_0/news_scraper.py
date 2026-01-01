import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime
import config

logger = logging.getLogger("Atl4s-News")

class NewsScraper:
    def __init__(self):
        self.url = "https://www.investing.com/economic-calendar/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_high_impact_events(self):
        """
        Scrapes Investing.com for today's high impact events.
        Returns a list of events.
        """
        try:
            logger.info("Fetching Economic Calendar...")
            response = requests.get(self.url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch news. Status: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"id": "economicCalendarData"})
            
            if not table:
                logger.error("Could not find calendar table.")
                return []

            events = []
            rows = table.find_all("tr", {"class": "js-event-item"})
            
            for row in rows:
                try:
                    # Extract Impact
                    impact_cell = row.find("td", {"class": "sentiment"})
                    if not impact_cell: continue
                    
                    # Count filled bull icons for impact
                    bulls = impact_cell.find_all("i", {"class": "grayFullBullishIcon"})
                    impact = len(bulls)
                    
                    # Filter: Only High Impact (3 bulls)
                    if impact < 3:
                        continue

                    # Extract Time
                    time_cell = row.find("td", {"class": "time"})
                    event_time_str = time_cell.text.strip()
                    
                    # Extract Currency
                    currency_cell = row.find("td", {"class": "flagCur"})
                    currency = currency_cell.text.strip()
                    
                    # Filter: Only USD (affects XAUUSD directly)
                    if "USD" not in currency:
                        continue

                    # Extract Event Name
                    event_cell = row.find("td", {"class": "event"})
                    event_name = event_cell.text.strip()

                    events.append({
                        "time": event_time_str,
                        "currency": currency,
                        "impact": impact,
                        "event": event_name
                    })
                    
                except Exception as e:
                    continue

            logger.info(f"Found {len(events)} high-impact USD events today.")
            return events

        except Exception as e:
            logger.error(f"Error scraping news: {e}")
            return []

if __name__ == "__main__":
    scraper = NewsScraper()
    events = scraper.get_high_impact_events()
    for e in events:
        print(e)
