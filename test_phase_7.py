
import logging
import pandas as pd
import numpy as np
from core.agi.temporal import FractalTimeScaleIntegrator
from core.agi.abstraction import AbstractPatternSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase7")

def test_phase_7():
    logger.info("--- Testing Phase 7: Temporal & Abstraction ---")
    
    # 1. Temporal
    temporal = FractalTimeScaleIntegrator()
    
    # Mock Dataframes
    df_bullish = pd.DataFrame({
        'open': [100, 101, 102],
        'close': [101, 102, 103],
        'high': [102, 103, 104],
        'low': [99, 100, 101]
    })
    
    df_bearish = pd.DataFrame({
        'open': [100, 99, 98],
        'close': [99, 98, 97],
        'high': [100, 99, 98],
        'low': [98, 97, 96]
    })
    
    market_map = {'1m': df_bullish, '5m': df_bullish, '1h': df_bearish} # Mixed
    
    coherence = temporal.calculate_fractal_coherence(market_map)
    logger.info(f"Fractal Coherence (Mixed): {coherence:.2f} (Expected near 0)")
    
    dilation = temporal.detect_temporal_dilation(market_map)
    logger.info(f"Time Dilation: {dilation}")
    
    if abs(coherence) < 0.5:
        logger.info("SUCCESS: Temporal Coherence logic functional.")
    else:
        logger.warning(f"FAILURE: Coherence {coherence} too high for mixed input.")
        
    # 2. Abstraction
    abstractor = AbstractPatternSynthesizer()
    
    # Compression Data (Flat)
    flat_prices = [100, 100.1, 99.9, 100.0, 100.1, 100.0]
    signature = abstractor.synthesize_signature(flat_prices, window=5)
    logger.info(f"Signature (Flat): {signature}")
    
    match, score = abstractor.find_structural_similarity(signature)
    logger.info(f"Pattern Match: {match} ({score:.2f})")
    
    if match == "COMPRESSION":
        logger.info("SUCCESS: Abstraction detected COMPRESSION.")
    else:
        logger.warning(f"FAILURE: Expected COMPRESSION, got {match}")

if __name__ == "__main__":
    test_phase_7()
