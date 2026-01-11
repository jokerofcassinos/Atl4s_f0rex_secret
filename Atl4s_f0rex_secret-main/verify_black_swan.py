import logging
from analysis.black_swan_adversary import BlackSwanAdversary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify-Adversary")

def test_black_swan():
    print("\n" + "="*50)
    print("TESTING BLACK-SWAN ADVERSARY")
    print("="*50 + "\n")
    
    adv = BlackSwanAdversary()
    
    entry = 2000.0
    atr = 2.0 # 2 dollar ATR
    
    # 1. High-Safety Trade (SL is far)
    print("Scenario 1: Defensive Sl (3.0 * ATR)")
    sl_safe = entry - (3.0 * atr)
    is_safe, prob = adv.audit_trade("BUY", entry, sl_safe, atr)
    print(f"Decision: BUY | SL: {sl_safe} | Is Safe: {is_safe} | Survival Prob: {prob:.2f}")
    
    # 2. Fragile Trade (SL is too close)
    print("\nScenario 2: Aggressive SL (0.5 * ATR) - High risk of being stopped by noise")
    sl_fragile = entry - (0.5 * atr)
    is_safe, prob = adv.audit_trade("BUY", entry, sl_fragile, atr)
    print(f"Decision: BUY | SL: {sl_fragile} | Is Safe: {is_safe} | Survival Prob: {prob:.2f}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_black_swan()
