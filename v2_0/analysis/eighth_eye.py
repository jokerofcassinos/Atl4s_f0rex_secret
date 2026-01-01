import logging
import numpy as np
import pandas as pd
from src.macro_math import MacroMath

logger = logging.getLogger("Atl4s-EighthEye")

class EighthEye:
    """
    The Sovereign (Universal System).
    Checks for Cross-Timeframe Fractal Coherence.
    - Multi-Scale Wavelet Alignment
    - Harmonic Resonance (Timeframe Synchronization)
    - Sovereign Override for High-Probability Strikes
    """
    def __init__(self):
        # Sovereign uses persistent multi-scale memory if needed, 
        # but macro_math does most of the heavy lifting.
        logger.info("Initializing Sovereign Eye (Fractal Geometry)...")

    def calculate_coherence_matrix(self, data_map):
        """
        Calculates Trend Coherence between M5 and higher timeframes (H1, H4, D1).
        If coherence is high (> 0.7) across ALL scales, we have Sovereign Alignment.
        """
        df_m5 = data_map.get('M5')
        if df_m5 is None or len(df_m5) < 64:
            return 0, {}

        m5_close = df_m5['close'].values.flatten()[-64:] # Need powers of 2 for wavelet simplicity
        
        scales = ['H1', 'H4', 'D1']
        coherences = {}
        total_alignment = 0
        
        for tf in scales:
            df_tf = data_map.get(tf)
            if df_tf is not None and len(df_tf) >= 32:
                # We need to align the series. 
                # Simplest way: take last N bars of the higher TF.
                tf_close = df_tf['close'].values.flatten()[-32:]
                
                # Use MacroMath Wavelet Tool
                # Returns trend_coherence and noise_coherence
                res = MacroMath.wavelet_haar_mra(m5_close, tf_close)
                coherences[tf] = res['trend_coherence']
                
                if res['trend_coherence'] > 0.6: total_alignment += 1
                elif res['trend_coherence'] < -0.6: total_alignment -= 1
            else:
                coherences[tf] = 0
                
        return total_alignment, coherences

    def deliberate(self, data_map):
        """
        Highest Authority Check.
        """
        alignment_count, coherences = self.calculate_coherence_matrix(data_map)
        
        # Determine Direction
        direction = "WAIT"
        score = 0
        
        # If alignment is 3, all 3 higher TFs confirm M5 Trend direction
        if alignment_count == 3:
            direction = "STRONG_BUY"
            score = 100
        elif alignment_count == -3:
            direction = "STRONG_SELL"
            score = -100
        elif alignment_count >= 2:
            direction = "BUY"
            score = 60
        elif alignment_count <= -2:
            direction = "SELL"
            score = -60
            
        return {
            'decision': direction,
            'score': score,
            'coherences': coherences,
            'alignment': alignment_count
        }
