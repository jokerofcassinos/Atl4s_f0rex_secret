
import sys
import os

# Add project root to path
sys.path.append('d:/Atl4s-Forex')

try:
    print("Importing OmegaAGICore...")
    from core.agi.omega_agi_core import OmegaAGICore
    print("Import Successful.")
    
    print("Testing Instantiation...")
    # Mock dependencies
    class MockWhy: pass
    class MockSim: pass
    
    core = OmegaAGICore(MockWhy(), MockSim())
    if hasattr(core, 'correlation_synapse'):
        print("CorrelationSynapse found in core.")
        print("Integration Verified.")
    else:
        print("ERROR: correlation_synapse missing!")
        
    # Test Method Signature
    from inspect import signature
    sig = signature(core.synthesize_singularity_decision)
    if 'open_positions' in sig.parameters:
        print("Method signature updated correctly.")
    else:
        print("ERROR: Method signature missing 'open_positions'")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
