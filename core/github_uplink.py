import os
import subprocess
import logging
import datetime

logger = logging.getLogger("InfinityUplink")

class GithubUplink:
    """
    Infinity Uplink: Automated Repository Synchronization.
    Ensures the AGI's code evolution is persisted to the cloud.
    """
    
    @staticmethod
    def status():
        """Check git status."""
        try:
            result = subprocess.run('git status', shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logger.error(f"Uplink Status Error: {e}")
            return "Error"

    @staticmethod
    def sync_codebase(message: str = None):
        """
        Performs a full sync cycle: Add -> Commit -> Push.
        """
        try:
            logger.info("Uplink: Initiating Synchronization Sequence...")
            
            # 1. Add All
            subprocess.run('git add .', shell=True, check=True)
            
            # 2. Commit
            if not message:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"AGI Auto-Evolution: {timestamp}"
            
            # Use quotes for safety
            commit_cmd = f'git commit -m "{message}"'
            subprocess.run(commit_cmd, shell=True) # Don't check=True, might be nothing to commit
            
            # 3. Push
            logger.info("Uplink: Transmitting Data to Remote...")
            push_res = subprocess.run('git push origin main', shell=True, capture_output=True, text=True)
            
            if push_res.returncode == 0:
                logger.info("Uplink: Transmission Successful. Codebase Persisted.")
                return True
            else:
                logger.error(f"Uplink Transmission Failed: {push_res.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Uplink Critical Failure: {e}")
            return False

    @staticmethod
    def safe_push():
        """
        Future: Run tests before pushing.
        """
        GithubUplink.sync_codebase("AGI Safety Push")
