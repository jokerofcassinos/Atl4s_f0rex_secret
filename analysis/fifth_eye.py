import logging
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.fft import fft, fftfreq
import config
from src.macro_math import MacroMath
from datetime import datetime, timedelta

logger = logging.getLogger("Atl4s-FifthEye")

class FifthEye:
    """
    The Oracle (Swing System).
    Responsible for high-timeframe persistence and intermarket context.
    - Operates on H4, D1, W1.
    - Monitors Intermarket Correlations (DXY, Yields).
    - Detects Spectral Cycles (DFT).
    - Provides Swing Levels (ADR).
    """
    def __init__(self):
        self.intermarket_data = {}
        self.last_sync = datetime.min
        self.sync_enabled = True
        
    def sync_intermarket(self):
        """Fetches intermarket data once every 4 hours."""
        if not self.sync_enabled or datetime.now() - self.last_sync < timedelta(hours=4):
            return
            
        logger.info("Syncing Intermarket Data (DXY, US10Y, SPX)...")
        for name, ticker in config.INTERMARKET_SYMBOLS.items():
            try:
                df = yf.download(ticker, period="1mo", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self.intermarket_data[name] = df.rename(columns={"Close": "close"})
            except Exception as e:
                logger.error(f"Failed to sync {name}: {e}")
                
        self.last_sync = datetime.now()

    def analyze_structure(self, data_map):
        """Analyzes H4, D1, W1 for structural alignment."""
        score = 0
        details = {}
        
        # 1. MTF Trend Alignment
        for tf in ['H4', 'D1', 'W1']:
            df = data_map.get(tf)
            if df is not None and len(df) > 20:
                ma20 = df['close'].rolling(20).mean()
                if df['close'].iloc[-1] > ma20.iloc[-1]:
                    score += 10 # Bullish bias
                    details[f'{tf}_bias'] = "BULLISH"
                else:
                    score -= 10 # Bearish bias
                    details[f'{tf}_bias'] = "BEARISH"
                    
        return score, details

    def analyze_intermarket(self, df_gold):
        """Detects intermarket divergences using Wavelet Coherence."""
        self.sync_intermarket()
        bias = 0
        
        if df_gold is None or len(df_gold) < 32: return 0
        gold_close = df_gold['close'].values.flatten()[-32:]
        
        # Gold vs DXY (Wavelet)
        dxy = self.intermarket_data.get('DXY')
        if dxy is not None and len(dxy) >= 32:
            dxy_close = dxy['close'].values.flatten()[-32:]
            w_res = MacroMath.wavelet_haar_mra(gold_close, dxy_close)
            # DXY is generally INVERSE to Gold.
            # If high coherence on Trend (negative), it confirms the bias.
            if w_res['trend_coherence'] < -0.6: bias += 20 * (-w_res['trend_coherence'])
            elif w_res['trend_coherence'] > 0.4: bias -= 15 # Abnormal positive correlation
            
        # Gold vs Yields (Wavelet)
        yields = self.intermarket_data.get('US10Y')
        if yields is not None and len(yields) >= 32:
            yield_close = yields['close'].values.flatten()[-32:]
            w_res = MacroMath.wavelet_haar_mra(gold_close, yield_close)
            if w_res['trend_coherence'] < -0.6: bias += 20 * (-w_res['trend_coherence'])
            elif w_res['trend_coherence'] > 0.4: bias -= 15
            
        return bias

    def detect_cycles(self, df):
        """Uses Discrete Fourier Transform to detect dominant cycle period."""
        if df is None or len(df) < 100: return 0
        
        close = df['close'].values
        # Detrend for FFT
        detrended = close - np.linspace(close[0], close[-1], len(close))
        
        # FFT
        N = len(detrended)
        yf_fft = fft(detrended)
        xf = fftfreq(N, 1)[:N//2]
        
        # Find dominant frequency (spectral peak)
        magnitudes = 2.0/N * np.abs(yf_fft[0:N//2])
        # Skip DC component at index 0
        peak_idx = np.argmax(magnitudes[1:]) + 1
        
        dominant_period = 1 / xf[peak_idx] if xf[peak_idx] != 0 else 0
        return dominant_period

    def calculate_adr_levels(self, df_d1):
        """Calculates TP/SL targets based on GARCH-adjusted Average Daily Range."""
        if df_d1 is None or len(df_d1) < 20:
            return None
            
        returns = df_d1['close'].pct_change().dropna().values
        garch_vol = MacroMath.garch_11_forecast(returns)
        
        # Scalar to ADR-like units
        curr_price = df_d1['close'].iloc[-1]
        adr_garch = curr_price * garch_vol
        
        return {
            'adr': adr_garch,
            'pivot_up': curr_price + (adr_garch * 0.5),
            'pivot_down': curr_price - (adr_garch * 0.5),
            'model': 'GARCH(1,1)'
        }

    def deliberate(self, data_map):
        """Main entry point for Quinto Olho."""
        struct_score, struct_details = self.analyze_structure(data_map)
        
        df_d1 = data_map.get('D1')
        macro_score = self.analyze_intermarket(df_d1)
        
        dominant_cycle = self.detect_cycles(df_d1)
        adr_levels = self.calculate_adr_levels(df_d1)
        
        total_score = struct_score + macro_score
        
        direction = "WAIT"
        if total_score > 30: direction = "BUY"
        elif total_score < -30: direction = "SELL"
        
        return {
            'decision': direction,
            'score': total_score,
            'details': struct_details,
            'cycle_period': dominant_cycle,
            'adr_levels': adr_levels
        }
