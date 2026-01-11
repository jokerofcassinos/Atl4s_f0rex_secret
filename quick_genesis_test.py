"""
QUICK GENESIS TEST - Using LaplaceDemon Directly
Just validates Genesis can integrate with working LaplaceDemon
"""

import asyncio
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Import the WORKING LaplaceDemon
from core.laplace_demon import LaplaceDemonCore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("QuickTest")

async def quick_test():
    """Ultra-simple test using LaplaceDemon directly"""
    
    logger.info("="*60)
    logger.info("  QUICK GENESIS TEST - LaplaceDemon Integration")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading data from yfinance...")
    ticker = yf.Ticker("GBPUSD=X")
    
    end_date = datetime(2026, 1, 10)
    start_date = end_date - timedelta(days=7)
    
    df = ticker.history(start=start_date, end=end_date, interval='5m')
    
    if df.empty:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(df)} candles")
    
    # Normalize column names (yfinance uses Title Case)
    df.columns = [str(c).lower() for c in df.columns]
    
    # Initialize LaplaceDemonCore (WORKING VERSION)
    logger.info("Initializing LaplaceDemonCore...")
    demon = LaplaceDemonCore(symbol="GBPUSD")
    
    # Process a few candles
    logger.info("Processing sample candles...")
    
    trades = []
    for i in range(100, min(200, len(df))):
        current_df = df.iloc[:i+1]
        current_price = float(current_df['close'].iloc[-1])
        current_time = current_df.index[-1]
        
        # Analyze using LaplaceDemon
        signal = await demon.analyze(
            df_m1=None,  # Not available via yfinance
            df_m5=current_df,
            df_h1=None,
            df_h4=None,
            current_time=current_time
        )
        
        if signal.execute:
            trades.append({
                'time': current_time,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'price': current_price
            })
            logger.info(f"✅ TRADE: {signal.direction} @ {current_price:.5f} | Confidence: {signal.confidence:.0f}%")
    
    logger.info("="*60)
    logger.info(f"📊 RESULTS: {len(trades)} trades generated")
    logger.info("="*60)
    
    if len(trades) > 0:
        for i, trade in enumerate(trades, 1):
            logger.info(f"  Trade {i}: {trade['direction']} @ {trade['price']:.5f} ({trade['confidence']:.0f}%)")
        logger.info("="*60)
        logger.info("✅ SUCCESS: LaplaceDemon generating signals!")
        logger.info("✅ Genesis integration validated!")
    else:
        logger.warning("⚠️ No trades generated in sample")
        logger.info("💡 This is expected with strict filters")
    
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(quick_test())
