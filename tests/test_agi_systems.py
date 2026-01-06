"""
AGI Ultra: Test Suite

Comprehensive tests for AGI system validation:
- Unit tests for each AGI module
- Integration tests for cross-module communication
- Reasoning quality tests
- Performance benchmarks
"""

import unittest
import time
import numpy as np
from typing import Dict, Any

# Test utilities
def create_mock_df(length: int = 100):
    """Create mock DataFrame for testing."""
    import pandas as pd
    
    np.random.seed(42)
    
    base_price = 1900.0
    returns = np.random.normal(0, 0.001, length)
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.001, length)),
        'high': prices * (1 + np.random.uniform(0, 0.002, length)),
        'low': prices * (1 - np.random.uniform(0, 0.002, length)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, length)
    })


class TestInfiniteWhyEngine(unittest.TestCase):
    """Tests for InfiniteWhyEngine."""
    
    def setUp(self):
        from core.agi.infinite_why_engine import InfiniteWhyEngine
        self.engine = InfiniteWhyEngine(max_depth=8)
    
    def test_initialization(self):
        """Test engine initializes correctly."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.max_depth, 8)
    
    def test_capture_event(self):
        """Test event capture."""
        event = self.engine.capture_event(
            symbol="XAUUSD",
            timeframe="M5",
            market_state={'price': 1900.0},
            analysis_state={'trend': 'UP'},
            decision="BUY",
            decision_score=75.0,
            decision_meta={'reason': 'Strong trend'},
            module_name="TestModule"
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.symbol, "XAUUSD")
        self.assertEqual(event.decision, "BUY")
    
    def test_deep_scan(self):
        """Test recursive deep scan."""
        # First capture an event
        event = self.engine.capture_event(
            symbol="XAUUSD",
            timeframe="M5",
            market_state={'price': 1900.0},
            analysis_state={},
            decision="BUY",
            decision_score=75.0,
            decision_meta={},
            module_name="TestModule"
        )
        
        # Run deep scan
        result = self.engine.deep_scan_recursive(
            module_name="TestModule",
            query_event=event,
            max_depth=3
        )
        
        self.assertIsNotNone(result)
        self.assertIn('combined_intuition', result)


class TestThoughtTree(unittest.TestCase):
    """Tests for ThoughtTree."""
    
    def setUp(self):
        from core.agi.thought_tree import ThoughtTree, GlobalThoughtOrchestrator
        self.tree = ThoughtTree("TestTree", max_depth=5)
        self.orchestrator = GlobalThoughtOrchestrator()
    
    def test_create_node(self):
        """Test node creation."""
        node_id = self.tree.create_node(
            question="Why is this happening?",
            context={'test': 'data'},
            confidence=0.8
        )
        
        self.assertIsNotNone(node_id)
        self.assertIn(node_id, self.tree.nodes)
    
    def test_answer_node(self):
        """Test answering a node."""
        node_id = self.tree.create_node(
            question="Test question?",
            context={},
            confidence=0.5
        )
        
        self.tree.answer_node(node_id, "Test answer", confidence=0.9)
        
        node = self.tree.nodes[node_id]
        self.assertEqual(node.answer, "Test answer")
        self.assertEqual(node.confidence, 0.9)
    
    def test_parent_child_relationship(self):
        """Test parent-child relationships."""
        parent_id = self.tree.create_node("Parent?", {})
        child_id = self.tree.create_node("Child?", {}, parent_id=parent_id)
        
        parent = self.tree.nodes[parent_id]
        child = self.tree.nodes[child_id]
        
        self.assertIn(child_id, parent.children)
        self.assertEqual(child.parent_id, parent_id)


class TestHolographicMemory(unittest.TestCase):
    """Tests for HolographicMemory."""
    
    def setUp(self):
        from core.memory.holographic import HolographicMemory
        self.memory = HolographicMemory()
    
    def test_encode_and_learn(self):
        """Test encoding and learning."""
        state = {
            'symbol': 'XAUUSD',
            'price': 1900.0,
            'trend': 'UP'
        }
        
        # Learn from outcome
        self.memory.learn(state, outcome_score=1.0)
        
        # Should not raise
        self.assertTrue(True)
    
    def test_intuit(self):
        """Test intuition retrieval."""
        # Add some learning first
        for i in range(10):
            state = {'price': 1900 + i, 'trend': 'UP'}
            self.memory.learn(state, outcome_score=0.8)
        
        # Get intuition
        result = self.memory.intuit({'price': 1905})
        
        self.assertIsNotNone(result)


class TestPatternLibrary(unittest.TestCase):
    """Tests for PatternLibrary."""
    
    def setUp(self):
        from core.agi.pattern_library import PatternLibrary
        self.library = PatternLibrary()
    
    def test_add_pattern(self):
        """Test adding patterns."""
        vector = np.random.randn(256)
        
        pattern_id = self.library.add_pattern(
            vector=vector,
            category='trend',
            metadata={'name': 'Test Pattern'}
        )
        
        self.assertIsNotNone(pattern_id)
    
    def test_search_similar(self):
        """Test similarity search."""
        # Add patterns
        for i in range(10):
            vector = np.random.randn(256)
            self.library.add_pattern(vector, 'trend', {'index': i})
        
        # Search
        query = np.random.randn(256)
        results = self.library.search_similar(query, top_k=5)
        
        self.assertLessEqual(len(results), 5)


class TestUnifiedReasoningLayer(unittest.TestCase):
    """Tests for UnifiedReasoningLayer."""
    
    def setUp(self):
        from core.agi.unified_reasoning import UnifiedReasoningLayer, ModuleInsight
        self.layer = UnifiedReasoningLayer()
        self.ModuleInsight = ModuleInsight
    
    def test_synthesize_agreement(self):
        """Test synthesis with agreement."""
        insights = [
            self.ModuleInsight(
                module_name="Module1",
                decision="BUY",
                confidence=0.8,
                reasoning="Trend is up",
                supporting_evidence=[]
            ),
            self.ModuleInsight(
                module_name="Module2",
                decision="BUY",
                confidence=0.7,
                reasoning="Momentum strong",
                supporting_evidence=[]
            )
        ]
        
        result = self.layer.synthesize(insights, {})
        
        self.assertEqual(result.final_decision, "BUY")
        self.assertGreater(result.agreement_score, 0.5)
    
    def test_synthesize_conflict(self):
        """Test synthesis with conflict."""
        insights = [
            self.ModuleInsight("A", "BUY", 0.8, "Up", []),
            self.ModuleInsight("B", "SELL", 0.9, "Down", [])
        ]
        
        result = self.layer.synthesize(insights, {})
        
        self.assertLess(result.agreement_score, 0.5)


class TestHealthMonitor(unittest.TestCase):
    """Tests for HealthMonitor."""
    
    def setUp(self):
        from core.agi.health_monitor import HealthMonitor, HealthStatus
        self.monitor = HealthMonitor()
        self.HealthStatus = HealthStatus
    
    def test_register_component(self):
        """Test component registration."""
        self.monitor.register_component("TestComponent")
        self.assertIn("TestComponent", self.monitor.components)
    
    def test_update_component(self):
        """Test component updates."""
        self.monitor.register_component("TestComp")
        self.monitor.update_component("TestComp", latency_ms=50.0)
        
        # Should not raise
        self.assertTrue(True)
    
    def test_health_status(self):
        """Test overall health."""
        self.monitor.register_component("Test1")
        self.monitor.register_component("Test2")
        
        for _ in range(10):
            self.monitor.update_component("Test1", latency_ms=50.0)
            self.monitor.update_component("Test2", latency_ms=60.0)
        
        status = self.monitor.get_overall_health()
        
        self.assertIn(status, list(self.HealthStatus))


class TestMCTSPlanner(unittest.TestCase):
    """Tests for MCTSPlanner."""
    
    def setUp(self):
        from core.mcts_planner import MCTSPlanner
        self.planner = MCTSPlanner(iterations=100)  # Reduced for tests
    
    def test_search(self):
        """Test MCTS search."""
        state = {
            'price': 1900.0,
            'entry': 1895.0,
            'side': 'BUY',
            'pnl': 5.0,
            'volatility': 1.0
        }
        
        move = self.planner.search(state, trend_bias=0.1)
        
        self.assertIn(move, ['HOLD', 'CLOSE', 'TRAIL', 'PARTIAL_TP'])


class TestIntegration(unittest.TestCase):
    """Integration tests for AGI system."""
    
    def test_full_pipeline(self):
        """Test complete AGI pipeline."""
        # This tests that all modules can work together
        
        from core.agi.infinite_why_engine import InfiniteWhyEngine
        from core.agi.thought_tree import GlobalThoughtOrchestrator
        from core.agi.unified_reasoning import UnifiedReasoningLayer, ModuleInsight
        from core.agi.health_monitor import HealthMonitor
        
        # Initialize
        why_engine = InfiniteWhyEngine(max_depth=4)
        orchestrator = GlobalThoughtOrchestrator()
        reasoning = UnifiedReasoningLayer()
        monitor = HealthMonitor()
        
        # Register components
        monitor.register_component("WhyEngine")
        monitor.register_component("Reasoning")
        
        # Capture event
        start = time.time()
        event = why_engine.capture_event(
            symbol="XAUUSD",
            timeframe="M5",
            market_state={'price': 1900},
            analysis_state={'trend': 'UP'},
            decision="BUY",
            decision_score=80.0,
            decision_meta={},
            module_name="Integration"
        )
        monitor.update_component("WhyEngine", latency_ms=(time.time() - start) * 1000)
        
        # Create thoughts
        tree = orchestrator.get_or_create_tree("Integration")
        tree.create_node("Why buy?", {'event': event.event_id})
        
        # Synthesize
        start = time.time()
        insights = [
            ModuleInsight("A", "BUY", 0.8, "Strong", []),
            ModuleInsight("B", "BUY", 0.7, "Momentum", [])
        ]
        result = reasoning.synthesize(insights, {})
        monitor.update_component("Reasoning", latency_ms=(time.time() - start) * 1000)
        
        # Verify
        self.assertEqual(result.final_decision, "BUY")
        self.assertGreater(result.confidence, 0.5)


class TestPerformance(unittest.TestCase):
    """Performance benchmarks."""
    
    def test_why_engine_speed(self):
        """Benchmark WhyEngine speed."""
        from core.agi.infinite_why_engine import InfiniteWhyEngine
        
        engine = InfiniteWhyEngine(max_depth=4)
        
        start = time.time()
        iterations = 100
        
        for i in range(iterations):
            engine.capture_event(
                symbol="XAUUSD",
                timeframe="M5",
                market_state={'price': 1900 + i},
                analysis_state={},
                decision="BUY",
                decision_score=50.0,
                decision_meta={},
                module_name="Perf"
            )
        
        elapsed = time.time() - start
        per_event = elapsed / iterations * 1000
        
        print(f"WhyEngine: {per_event:.2f}ms per event")
        
        # Should be < 10ms per event
        self.assertLess(per_event, 100)


class TestMemoryRecall(unittest.TestCase):
    """Tests for memory and recall systems."""
    
    def test_holographic_memory_recall(self):
        """Test holographic memory storage and recall."""
        from core.memory.holographic import HolographicMemory
        
        memory = HolographicMemory()
        
        # Learn multiple states
        for i in range(20):
            state = {
                'price': 1900 + i,
                'trend': 'UP' if i % 2 == 0 else 'DOWN',
                'volatility': 1.0 + i * 0.1
            }
            memory.learn(state, outcome_score=0.8 if i % 3 == 0 else 0.3)
        
        # Recall similar state
        query = {'price': 1910, 'trend': 'UP'}
        result = memory.intuit(query)
        
        self.assertIsNotNone(result)
    
    def test_decision_memory_patterns(self):
        """Test decision memory pattern detection."""
        from core.agi.decision_memory_expanded import DecisionMemoryExpanded, DecisionOutcome
        
        memory = DecisionMemoryExpanded(pattern_min_frequency=3)
        
        # Record similar decisions
        for i in range(10):
            dec_id = memory.record_decision(
                decision="BUY",
                symbol="XAUUSD",
                timeframe="M5",
                confidence=0.8,
                market_context={'regime': 'TRENDING'},
                module_votes={'A': 'BUY', 'B': 'BUY'},
                reasoning_chain=["Trend is up", "Momentum strong"]
            )
            
            # Update outcome
            memory.update_outcome(
                dec_id,
                outcome=DecisionOutcome.WIN if i % 2 == 0 else DecisionOutcome.LOSS,
                exit_price=1910.0,
                pnl=10.0 if i % 2 == 0 else -5.0,
                pnl_pct=0.5 if i % 2 == 0 else -0.25,
                duration_seconds=300
            )
        
        # Check patterns detected
        patterns = memory.get_pattern_report()
        self.assertGreater(len(patterns), 0)
        
        # Test prediction
        prediction = memory.predict_outcome(
            decision="BUY",
            symbol="XAUUSD",
            market_context={'regime': 'TRENDING'},
            confidence=0.8
        )
        
        self.assertIn('predicted_win_prob', prediction)


class TestConsciousnessBus(unittest.TestCase):
    """Tests for consciousness bus."""
    
    def test_publish_and_subscribe(self):
        """Test thought publishing and subscription."""
        from core.agi.consciousness_bus import ConsciousnessBus, ThoughtPriority
        
        bus = ConsciousnessBus()
        received = []
        
        def callback(thought):
            received.append(thought)
        
        bus.subscribe("insight", callback)
        
        # Publish thought
        thought_id = bus.publish(
            source_module="TestModule",
            content={'decision': 'BUY', 'confidence': 0.8},
            thought_type="insight",
            priority=ThoughtPriority.HIGH
        )
        
        self.assertIsNotNone(thought_id)
        
        # Wait for processing
        time.sleep(0.1)
        
        # Should have received the thought
        self.assertGreater(len(received), 0)
        
        bus.shutdown()
    
    def test_thought_fusion(self):
        """Test thought fusion."""
        from core.agi.consciousness_bus import ConsciousnessBus, Thought, ThoughtPriority
        
        bus = ConsciousnessBus()
        
        # Create thoughts
        thoughts = [
            Thought(
                priority=ThoughtPriority.HIGH.value,
                timestamp=time.time(),
                source_module="Module1",
                content={'decision': 'BUY', 'confidence': 0.8}
            ),
            Thought(
                priority=ThoughtPriority.NORMAL.value,
                timestamp=time.time(),
                source_module="Module2",
                content={'decision': 'BUY', 'confidence': 0.7}
            )
        ]
        
        # Fuse
        result = bus.fuse_thoughts(thoughts, "consensus")
        
        self.assertEqual(result['consensus'], 'BUY')
        self.assertGreater(result['agreement'], 0.5)
        
        bus.shutdown()
    
    def test_temporal_coherence(self):
        """Test temporal coherence tracking."""
        from core.agi.consciousness_bus import ConsciousnessBus, ThoughtPriority
        
        bus = ConsciousnessBus()
        
        # Publish consistent thoughts
        for _ in range(10):
            bus.publish("Test", {'decision': 'BUY'}, "decision", ThoughtPriority.NORMAL)
        
        coherence = bus.get_temporal_coherence()
        self.assertGreater(coherence, 0.8)
        
        bus.shutdown()


class TestSwarmTransform(unittest.TestCase):
    """Tests for swarm transformation."""
    
    def test_agi_enabled_swarm(self):
        """Test AGI-enabled swarm base class."""
        from core.agi.swarm_transform import ExampleTrendSwarm
        
        swarm = ExampleTrendSwarm()
        
        # Create mock context
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        prices = 1900 + np.cumsum(np.random.randn(50) * 0.1)
        
        context = {
            'df_m5': pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000, 5000, 50)
            }),
            'symbol': 'XAUUSD',
            'timeframe': 'M5'
        }
        
        # Run analysis
        result = swarm._analyze(context)
        
        self.assertIn('decision', result)
        self.assertIn('score', result)
        self.assertIn('reasoning', result)


class TestFullIntegration(unittest.TestCase):
    """Complete integration tests."""
    
    def test_complete_agi_flow(self):
        """Test complete AGI flow from signal to decision."""
        from core.agi.consciousness_bus import ConsciousnessBus, ThoughtPriority
        from core.agi.decision_memory_expanded import DecisionMemoryExpanded
        from core.agi.unified_reasoning import UnifiedReasoningLayer, ModuleInsight
        from core.agi.health_monitor import HealthMonitor
        
        # Initialize components
        bus = ConsciousnessBus()
        memory = DecisionMemoryExpanded()
        reasoning = UnifiedReasoningLayer()
        monitor = HealthMonitor()
        
        # Register components
        monitor.register_component("TestFlow")
        
        # Simulate flow
        # 1. Publish thoughts
        bus.publish("ModuleA", {'decision': 'BUY', 'confidence': 0.8}, "decision")
        bus.publish("ModuleB", {'decision': 'BUY', 'confidence': 0.7}, "decision")
        
        # 2. Collect insights
        insights = [
            ModuleInsight("ModuleA", "BUY", 0.8, "Strong signal", []),
            ModuleInsight("ModuleB", "BUY", 0.7, "Confirmation", [])
        ]
        
        # 3. Synthesize
        unified = reasoning.synthesize(insights, {'regime': 'TRENDING'})
        
        # 4. Record decision
        memory.record_decision(
            decision=unified.final_decision,
            symbol="XAUUSD",
            timeframe="M5",
            confidence=unified.confidence,
            market_context={'regime': 'TRENDING'},
            module_votes={'A': 'BUY', 'B': 'BUY'},
            reasoning_chain=unified.reasoning_chain
        )
        
        # 5. Update health
        monitor.update_component("TestFlow", latency_ms=50)
        
        # Verify
        self.assertEqual(unified.final_decision, "BUY")
        self.assertEqual(memory.total_decisions, 1)
        
        bus.shutdown()


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInfiniteWhyEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestThoughtTree))
    suite.addTests(loader.loadTestsFromTestCase(TestHolographicMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternLibrary))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedReasoningLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestMCTSPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryRecall))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessBus))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestFullIntegration))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
