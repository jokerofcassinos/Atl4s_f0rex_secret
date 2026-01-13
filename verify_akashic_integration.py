from analysis.agi.pattern_hunter import PatternHunter
from datetime import datetime

def test_hunter():
    print("🧪 [TEST] waking up Pattern Hunter...")
    try:
        hunter = PatternHunter()
        if not hunter.memory.is_trained:
            print("❌ [TEST] Memory is NOT trained (Brain Dead).")
            return
    except Exception as e:
        print(f"❌ [TEST] Failed to init Hunter: {e}")
        return

    print("✅ [TEST] Hunter Awake & Memory Loaded.")
    
    # Test Query
    print("❓ [TEST] Querying Oracle with dummy market state...")
    try:
        prediction = hunter.analyze_live_market(
            candle_open=1.3000,
            candle_high=1.3010,
            candle_low=1.2990,
            candle_close=1.3005,
            volume=500,
            rsi=55.0,
            atr=0.0010,
            timestamp=datetime.now()
        )
        print(f"🔮 [TEST] Prediction: {prediction}")
        
        if prediction['status'] == 'SUCCESS':
             print("✅ [TEST] ORACLE SPOKE. System Operational.")
        else:
             print("⚠️ [TEST] Oracle Silent (Low Confidence or Error).")
             
    except Exception as e:
        print(f"❌ [TEST] Query Failed: {e}")

if __name__ == "__main__":
    test_hunter()
