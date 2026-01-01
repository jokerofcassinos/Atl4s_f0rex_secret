import logging
from analysis.tenth_eye import TenthEye

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify-Architect")

def test_architect():
    print("\n" + "="*50)
    print("TESTING TENTH EYE (THE ARCHITECT)")
    print("="*50 + "\n")
    
    arch = TenthEye()
    
    # 1. High Coherence
    print("Scenario 1: High System Coherence (Consensus Alignment)")
    results_aligned = {
        'Eye1': {'decision': 'BUY'},
        'Eye2': {'decision': 'BUY'},
        'Eye3': {'decision': 'WAIT'}
    }
    res = arch.deliberate(results_aligned)
    print(f"Status: {res['status']} | Coherence: {res['coherence']:.2f} | Veto: {res['veto']}")
    
    # 2. High Conflict
    print("\nScenario 2: System Conflict (Split Decisions)")
    results_conflict = {
        'Eye1': {'decision': 'BUY'},
        'Eye2': {'decision': 'SELL'},
        'Eye3': {'decision': 'BUY'},
        'Eye4': {'decision': 'SELL'}
    }
    res_conf = arch.deliberate(results_conflict)
    print(f"Status: {res_conf['status']} | Coherence: {res_conf['coherence']:.2f} | Veto: {res_conf['veto']}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_architect()
