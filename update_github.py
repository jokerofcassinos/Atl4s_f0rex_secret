import subprocess
import logging
from datetime import datetime
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GitAutoUpdate")

def run_command(command):
    try:
        # Using shell=True for windows command compatibility
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def get_current_branch():
    success, output = run_command("git branch --show-current")
    if success:
        return output
    return None

def update_github():
    logger.info("--- Atl4s-Forex Intelligent Updater ---")
    
    # 0. Detect Branch
    branch = get_current_branch()
    if not branch:
        logger.error("Could not detect git branch.")
        return
        
    logger.info(f"Active Branch: {branch}")
    
    # Determine Context
    version_label = "v1.0"
    if "v2" in branch:
        version_label = "v2.0"
        
    logger.info(f"Target Version: {version_label}")

    # 1. Check Status
    success, output = run_command("git status --porcelain")
    if not success:
        logger.error(f"Failed to check git status: {output}")
        return

    if not output:
        logger.info("No changes to commit.")
        
        # Pull latest even if no commits
        logger.info(f"Pulling latest for {branch}...")
        run_command(f"git pull origin {branch}")
        return

    # 2. Add All Changes
    logger.info("Staging changes...")
    success, output = run_command("git add .")
    if not success:
        logger.error(f"Failed to add files: {output}")
        return

    # 3. Commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Auto-Update ({version_label}): {timestamp} [{branch}]"
    
    logger.info(f"Committing: '{commit_msg}'")
    success, output = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        logger.error(f"Failed to commit: {output}")
        return

    # 4. Push to Specific Branch
    logger.info(f"Pushing to origin {branch}...")
    success, output = run_command(f"git push origin {branch}")
    if not success:
        logger.error(f"Push failed: {output}")
    else:
        logger.info(f">>> GitHub Update Successful ({version_label})! <<<")

if __name__ == "__main__":
    update_github()
