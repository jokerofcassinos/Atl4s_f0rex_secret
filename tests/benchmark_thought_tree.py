
import sys
import os
import time
import uuid
# Add current dir to path to import core
sys.path.append(os.getcwd())

from core.agi.thought_tree import GlobalThoughtOrchestrator

def benchmark():
    orch = GlobalThoughtOrchestrator()
    mod_names = ["Trend", "Sniper", "Quant", "Oracle", "Council"]
    trees = [orch.get_or_create_tree(name) for name in mod_names]
    
    print("Populating 5000 nodes...")
    start_pop = time.time()
    for i in range(1000):
        for tree in trees:
            tree.create_node(f"Why did I decide {'BUY' if i % 2 == 0 else 'SELL'}? Context: {uuid.uuid4()}")
    
    # Force initial indexing
    orch._index_all_new_nodes()
    print(f"Population took: {time.time() - start_pop:.2f}s")
    
    print("\nRunning Similarity Search Benchmark...")
    query = "Why did I decide BUY?"
    
    start_search = time.time()
    for _ in range(100):
         orch.find_similar_thoughts("Trend", query, threshold=0.5)
    
    end_search = time.time()
    avg_time = (end_search - start_search) / 100
    print(f"100 Searches took: {end_search - start_search:.4f}s")
    print(f"Average Search Time: {avg_time:.6f}s")
    
    if avg_time < 0.01:
        print("\n✅ SUCCESS: Search performance is high!")
    else:
        print("\n❌ WARNING: Search performance still slow.")

if __name__ == "__main__":
    benchmark()
