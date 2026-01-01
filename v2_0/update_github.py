import subprocess
import logging
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GitAutoUpdate")

def run_command(command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def update_github():
    logger.info("--- Starting GitHub Auto-Update ---")
    
    # 1. Check Status
    success, output = run_command("git status --porcelain")
    if not success:
        logger.error(f"Failed to check git status: {output}")
        return

    if not output:
        logger.info("No changes to commit.")
        return

    # 2. Add All Changes
    logger.info("Staging changes...")
    success, output = run_command("git add .")
    if not success:
        logger.error(f"Failed to add files: {output}")
        return

    # 3. Commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Auto-Update: System Enhancement {timestamp}"
    logger.info(f"Committing with message: '{commit_msg}'")
    success, output = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        logger.error(f"Failed to commit: {output}")
        return

    # 4. Push
    logger.info("Pushing to remote...")
    success, output = run_command("git push")
    if not success:
        logger.error(f"Push failed: {output}")
    else:
        logger.info(">>> GitHub Update Successful! <<<")

if __name__ == "__main__":
    update_github()
