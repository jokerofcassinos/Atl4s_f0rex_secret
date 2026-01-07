import os
import sys
import subprocess
import logging
import time

logger = logging.getLogger("SystemGuard")

class SystemGuard:
    """
    Guardian of the Process Table.
    Detects and eliminates 'Zombie' processes that hog resources or ports.
    """
    
    @staticmethod
    def kill_zombies_on_port(port: int = 5557):
        """
        Finds any process listening on the specified port and terminates it.
        Windows ONLY (uses netstat/taskkill).
        """
        try:
            # 1. Find PID using netstat
            # netstat -ano | findstr :5557
            # Output line example: "  TCP    0.0.0.0:5557           0.0.0.0:0              LISTENING       1234"
            
            logger.info(f"SystemGuard: Scanning for Zombies on Port {port}...")
            
            # Execute netstat
            cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if not result.stdout:
                logger.info("SystemGuard: Sector Clear. No Zombies detected.")
                return

            lines = result.stdout.strip().split('\n')
            pids_to_kill = set()
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    # Last element is usually PID
                    pid = parts[-1]
                    # Check if it's really the port we want
                    local_addr = parts[1]
                    if f":{port}" in local_addr:
                        pids_to_kill.add(pid)
            
            current_pid = str(os.getpid())
            
            for pid in pids_to_kill:
                if pid == current_pid:
                    continue # Don't commit suicide
                
                logger.warning(f"SystemGuard: ZOMBIE DETECTED (PID: {pid}). Initiating Purge...")
                # Kill it
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"SystemGuard: Zombie (PID: {pid}) Eliminated.")
                
            # Wait a moment for OS to release socket
            time.sleep(1.0)
            
        except Exception as e:
            logger.error(f"SystemGuard Error: {e}")

    @staticmethod
    def clean_slate():
        """
        Aggressive cleanup if needed.
        """
        pass
