import subprocess
import logging
import os
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Atl4s-v2-Uploader")

def run_command(command, cwd=None):
    try:
        # If no cwd provided, use the repo root (parent of this script)
        if not cwd:
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True, cwd=cwd)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def upload_v2():
    logger.info("--- Atl4s-Forex v2.0 GitHub Uploader ---")
    
    # Repo Root Calculation
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Targeting Repository: {repo_root}")

    # 1. Check Status
    success, output = run_command("git status --porcelain")
    if not success:
        logger.error(f"Git Error: {output}")
        return

    if not output:
        logger.info("No changes found to commit.")
        return

    # 2. Add v2_0 specific folder
    logger.info("Staging v2.0 files...")
    # adding v2_0/ specifically (the folder name in repo root)
    success, output = run_command("git add v2_0/") 
    if not success:
        logger.error(f"Failed to add v2_0 files: {output}")
        # Fallback to adding all if v2_0 add fails (unlikely)
        logger.warning("Fallback: Adding all changes...")
        run_command("git add .")

    # 3. Commit
    commit_msg = "feat(v2.0): RELEASE v2.0 - Dynamic Risk Manager & Interactive Startup"
    logger.info(f"Committing: '{commit_msg}'")
    
    success, output = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        logger.error(f"Commit Failed: {output}")
        return

    # 4. Push
    logger.info("Pushing to GitHub...")
    success, output = run_command("git push")
    if not success:
        logger.error(f"Push Failed: {output}")
    else:
        logger.info(">>> SUCCESS: version 2.0 uploaded to GitHub! <<<")

if __name__ == "__main__":
    upload_v2()
