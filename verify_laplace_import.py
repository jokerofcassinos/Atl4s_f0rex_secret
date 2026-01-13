import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import LaplaceDemonCore...")
    from core.laplace_demon import LaplaceDemonCore
    print("SUCCESS: LaplaceDemonCore imported.")
    
    # Try to instantiate to check __init__ logic
    demon = LaplaceDemonCore()
    print("SUCCESS: LaplaceDemonCore instantiated.")
    
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
