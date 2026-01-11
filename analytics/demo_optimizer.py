"""
Quick ML Optimizer Demo

Simple demonstration of optimization capabilities
"""

import sys
sys.path.insert(0, 'D:/Atl4s-Forex')

from analytics.ml_optimizer import MLOptimizer
from analytics.trade_analyzer import TradeAnalyzer

print("="*60)
print("GENESIS ML OPTIMIZER - DEMO")
print("="*60)
print()

# Load analyzer
analyzer = TradeAnalyzer(db_path="data/genesis_trades.json")
print(f"✅ Loaded {len(analyzer.trades)} trades from database")
print()

# Create optimizer
optimizer = MLOptimizer(analyzer)
print("✅ ML Optimizer initialized")
print()

# Get suggestions
print("🧠 Analyzing trade patterns...")
suggestions = optimizer.analyze_optimal_parameters(days=30)
print(f"✅ Found {len(suggestions)} optimization opportunities")
print()

if suggestions:
    print("="*60)
    print("TOP 5 RECOMMENDATIONS")
    print("="*60)
    print()
    
    for i, s in enumerate(suggestions[:5], 1):
        print(f"{i}. {s.parameter_name.upper()}")
        print(f"   Current:     {s.current_value}")
        print(f"   Suggested:   {s.suggested_value}")
        print(f"   Improvement: +{s.expected_improvement:.1f}% WR")
        print(f"   Confidence:  {s.confidence:.0f}%")
        print(f"   💡 {s.evidence}")
        print()
    
    # Quick wins
    quick_wins = [s for s in suggestions if s.confidence >= 75 and s.expected_improvement >= 3]
    
    if quick_wins:
        print("="*60)
        print("⚡ QUICK WINS (High confidence, easy to implement)")
        print("="*60)
        print()
        for win in quick_wins:
            print(f"• {win.parameter_name}: {win.evidence}")
        print()
    
    # Total potential
    total = sum(s.expected_improvement for s in suggestions[:3])
    print("="*60)
    print(f"💰 TOTAL POTENTIAL: +{total:.1f}% Win Rate Improvement")
    print("="*60)
else:
    print("Need more trade data for optimization.")

print()
print("✅ Analysis complete!")
