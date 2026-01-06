import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from cpp_core.agi_bridge import get_agi_bridge

def diagnose():
    bridge = get_agi_bridge().hdc
    print(f"HDC Available: {bridge.available}")
    
    # 1. Test Scalar Encoding Determinism
    print("\n--- Scalar Encoding ---")
    s1 = bridge.encode_scalar(50.0, 0, 100)
    s2 = bridge.encode_scalar(50.0, 0, 100)
    sim_s = bridge.cosine_similarity(s1, s2)
    print(f"Scalar Self-Sim: {sim_s} (Expect 1.0)")
    
    # 2. Test Bind Determinism
    print("\n--- Bind ---")
    v1 = bridge.random_hv()
    v2 = bridge.random_hv()
    b1 = bridge.bind(v1, v2)
    b2 = bridge.bind(v1, v2)
    sim_b = bridge.cosine_similarity(b1, b2)
    print(f"Bind Self-Sim: {sim_b} (Expect 1.0)")
    
    # 3. Test Bundle Determinism
    print("\n--- Bundle ---")
    t1 = bridge.bundle([v1, v2])
    t2 = bridge.bundle([v1, v2])
    sim_t = bridge.cosine_similarity(t1, t2)
    print(f"Bundle Self-Sim: {sim_t} (Expect 1.0)")

if __name__ == "__main__":
    diagnose()
