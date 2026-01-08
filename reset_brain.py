import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BrainWiper")

LINES = [
    r"d:\Atl4s-Forex\brain\holographic_plate.pkl",
    r"d:\Atl4s-Forex\brain\holographic_plate.npy",
    r"d:\Atl4s-Forex\trade_history_db.json"
]

def wipe_memory():
    logger.info("Initializing Brain Wipe Protocol...")
    time.sleep(1)
    
    deleted_count = 0
    for file_path in LINES:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Successfully DELETED: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        else:
            logger.info(f"File not found (already clean): {file_path}")
            
    if deleted_count > 0:
        logger.info(f"Process Complete. {deleted_count} memory files vaporized.")
        logger.info("The Bot is now a Blank Slate.")
    else:
        logger.info("Process Complete. No memory files were found.")

if __name__ == "__main__":
    wipe_memory()
    input("\nPress Enter to exit...")
