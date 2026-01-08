import logging
import asyncio
from core.swarm_orchestrator import SwarmOrchestrator
from core.grandmaster import GrandMaster
from core.zmq_bridge import ZmqBridge

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase4")

class MockBus:
    def get_recent_thoughts(self): return []
    def register_thought(self, t): pass

class MockAgent:
    def __init__(self, name): self.name = name
    def process(self, ctx): return None

class MockNeuro:
    def get_dynamic_weights(self): return {} 

class MockAttention:
    def forward(self, x): return ([0.5]*64, [0.1]*10)

# Mock GrandMaster (New Class)
class MockApexGM:
    def perceive_and_decide(self, ctx):
        print(f"[GM] Perceiving Context: {ctx}")
        return "BUY" # Intervention!

async def test_civil_war():
    print("\n--- TEST: CIVIL WAR TIE-BREAKER ---")
    
    # Init Orchestrator with Mock GM
    gm = MockApexGM()
    orch = SwarmOrchestrator(MockBus(), None, MockNeuro(), MockAttention(), grandmaster=gm)
    
    # Force Swarm State (Mocking internal synthesis)
    # We can't easily force synthesis without mocking thoughts, 
    # but let's test the Logic Block by calling a helper or mocking the method?
    # Actually, synthesize_thoughts takes explicit args!
    
    from core.interfaces import SwarmSignal
    
    # Create conflicting thoughts
    thoughts = []
    # 2500 Buy Score
    thoughts.append(SwarmSignal(source="AgentA", signal_type="BUY", confidence=100.0, meta_data={}, timestamp=12345))
    # 2500 Sell Score
    thoughts.append(SwarmSignal(source="AgentB", signal_type="SELL", confidence=100.0, meta_data={}, timestamp=12345))
    
    # Need to boost them to > 2000 total.
    # We can just mock synthesize_thoughts to simulate the scores if needed, 
    # but better to pass thoughts that sum up.
    # Orchestrator uses weights. default 1.0.
    # To get > 2000, we need lots of agents or high weights.
    # Let's fake the weights or use many signals.
    
    # Actually, let's subclass Orchestrator to inject the scores directly for testing the 'if' block.
    # Or just rely on the fact that I reviewed the code.
    
    # Let's try to pass many signals.
    for i in range(25):
        thoughts.append(SwarmSignal(source=f"B_{i}", signal_type="BUY", confidence=100.0, meta_data={}, timestamp=12345))
        thoughts.append(SwarmSignal(source=f"S_{i}", signal_type="SELL", confidence=100.0, meta_data={}, timestamp=12345))
        
    decision, score, meta = orch.synthesize_thoughts(thoughts, {})
    
    print(f"Decision: {decision}")
    print(f"Reason: {meta.get('reason')}")
    
    if decision == "BUY" and meta.get('reason') == "GrandMaster Tie-Break":
        print("PASS: GrandMaster successfully intervened in Civil War.")
    else:
        print(f"FAIL: Decision was {decision}. Expected BUY via Intervention.")

def test_visual_routing():
    print("\n--- TEST: ZMQ VISUAL ROUTING ---")
    bridge = ZmqBridge(port=5558)
    # Mock Clients
    bridge.clients['USDJPYm'] = "SocketObj"
    
    # Test Routing Logic (by calling send_command and checking logs/logic)
    # Since we can't check what was sent to socket easily without mocking socket,
    # We rely on no errors.
    
    try:
        bridge.send_draw_text("USDJPY", "TestObj", 1.0, 1000, "Hello", "Red")
        print("PASS: send_draw_text executed without error (Log should show routing).")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    asyncio.run(test_civil_war())
    test_visual_routing()
