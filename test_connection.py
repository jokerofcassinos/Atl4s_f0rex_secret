import logging
import time
from bridge import ZmqBridge

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Atl4s-Test")

def main():
    logger.info("--- Atl4s-Forex Connection Test ---")
    logger.info("1. Initializing Python Server...")
    
    try:
        bridge = ZmqBridge()
        logger.info("2. Server Listening on Port 5555.")
        logger.info("3. Please attach 'Atl4sBridge' EA to a chart in MT5 now.")
        
        logger.info("Waiting for connection...")
        
        while True:
            # The bridge accepts connection in a background thread.
            # We just check if we are receiving data.
            
            tick = bridge.get_tick()
            if tick:
                logger.info(f"SUCCESS! Received Tick from MT5: {tick}")
                logger.info("Connection is WORKING.")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Test Cancelled.")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'bridge' in locals():
            bridge.close()
        logger.info("Test Finished.")

if __name__ == "__main__":
    main()
