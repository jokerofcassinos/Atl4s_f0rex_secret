from risk_manager import RiskManager
import time

def test_risk_systems():
    rm = RiskManager()
    print("🛡️ [TEST] Testing Advanced Risk Systems...")
    
    # 1. Test Drawdown Oracle
    print("\n🔮 [TEST] 1. Drawdown Oracle (Volatility Adjustment)")
    
    balance = 10000
    entry = 2000.0
    stop = 1990.0 # 10 points
    
    # Case A: High Volatility ("Samba")
    lots_high_vol = rm.calculate_oracle_lots(balance, entry, stop, volatility_score=85, structure_score=50)
    print(f"   Case A (High Vol): Lots = {lots_high_vol} (Expected: Reduced)")
    
    # Case B: Sniper ("Concrete")
    lots_sniper = rm.calculate_oracle_lots(balance, entry, stop, volatility_score=20, structure_score=90)
    print(f"   Case B (Sniper):   Lots = {lots_sniper} (Expected: Boosted)")
    
    # Case C: Normal
    lots_normal = rm.calculate_oracle_lots(balance, entry, stop, volatility_score=50, structure_score=50)
    print(f"   Case C (Normal):   Lots = {lots_normal} (Expected: Standard)")
    
    if lots_sniper > lots_normal > lots_high_vol:
        print("   ✅ Oracle Logic Validated.")
    else:
        print("   ❌ Oracle Logic Failed.")

    # 2. Test Time Decay TP ("Descending Staircase")
    print("\n📉 [TEST] 2. Time Decay TP")
    
    start_time = time.time() - (5 * 3600) # 5 hours ago
    current_time = time.time()
    
    entry_price = 1.0000
    initial_tp = 1.0100 # 100 pips
    
    # 5 Hours Decay
    decayed_tp = rm.calculate_decayed_tp(entry_price, initial_tp, start_time, current_time)
    dist_initial = abs(initial_tp - entry_price)
    dist_decayed = abs(decayed_tp - entry_price)
    
    print(f"   Initial TP Dist: {dist_initial:.4f}")
    print(f"   Decayed TP Dist (5h): {dist_decayed:.4f}")
    
    percent_drop = (dist_initial - dist_decayed) / dist_initial
    print(f"   Decay %: {percent_drop*100:.1f}%")
    
    if dist_decayed < dist_initial:
        print("   ✅ Decay Logic Validated.")
    else:
        print("   ❌ Decay Logic Failed.")

if __name__ == "__main__":
    test_risk_systems()
