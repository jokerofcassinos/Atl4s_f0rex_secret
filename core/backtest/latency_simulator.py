
import random
from datetime import timedelta
import pandas as pd

class LatencySimulator:
    """
    Simulates network latency and execution delays (e.g. 100ms).
    Also handles slippage.
    """
    
    def __init__(self, latency_ms: int = 100):
        self.latency_ms = latency_ms
        
    def simulate_fill(self, signal_price: float, signal_time, df_m1: pd.DataFrame, direction: str) -> tuple[float, float]:
        """
        Simulates execution with latency.
        
        Args:
            signal_price: The price at the moment of signal.
            signal_time: The timestamp of the signal.
            df_m1: The M1 dataframe to look ahead for price changes.
            direction: "BUY" or "SELL"
            
        Returns:
            (fill_price, slippage_amount)
        """
        # 1. Calculate the 'Real' time of execution
        fill_time = signal_time + timedelta(milliseconds=self.latency_ms)
        
        # 2. Find the candle that contains the fill time
        # This is an approximation. In a real tick backtest we would jump forward N ticks.
        # In M1 OHLC, 100ms is usually within the same candle or the very next one.
        
        # If we are strictly OHLC, we can't see "inside" the candle at 100ms.
        # We will approximate: 
        # If latency is small (< 1s), we assume price might drift slightly from 'close' of signal tick 
        # (which isn't realistic if signal is on close).
        # Actually, "Signal on Close" means we execute at the OPEN of the NEXT candle usually.
        # If we signal intra-candle, we execute 100ms later.
        
        # Let's assume the signal came at the exact provided 'signal_time'.
        # We look for the price at signal_time + latency.
        
        # PROBABILISTIC SLIPPAGE MODEL
        # Instead of strict data lookup (which is hard with only M1 OHLC), we use a probabilistic drift 
        # based on recent volatility.
        
        # Estimate 1-second volatility from the current M1 candle range
        try:
            current_candle = df_m1[df_m1.index <= signal_time].iloc[-1]
            candle_range = current_candle['high'] - current_candle['low']
            # Approx volatility per 100ms (assuming linear-ish movement or random walk)
            # Volatility scales with square root of time? 
            # Let's keep it simple: 10% of the M1 range is the max drift for 100ms? No that's huge.
            # M1 range is 60000ms. 100ms is 1/600th.
            # So drift is likely very small, but non-zero.
            
            # Let's add explicit slippage noise.
            # 0.5 pips max slippage typically.
            
            volatility_pip_scale = candle_range * 0.05 # 5% of candle range as variance base
            
            # Random slippage: 
            # Skewed against the trader? Usually yes.
            # Slippage is (+) for bad fill, (-) for good fill (rare).
            
            # Random drift (-0.2 to +0.8) * volatility
            # Mostly positive slippage (bad for us)
            noise_factor = random.uniform(-0.2, 1.0) 
            slippage = volatility_pip_scale * noise_factor * 0.1 # Scaling down to 100ms realm
            
            # Ensure slippage is not zero but small
            if abs(slippage) < 0.00001: slippage = 0.00001 * random.choice([1, -1])
            
            # Apply slippage in direction of "worse price"
            # BUY: Add slippage
            # SELL: Subtract slippage (price goes down, we sell lower) -> Wait, sell lower is bad.
            # Sell Logic: Price = 1.2000. Bad fill = 1.1998. So we Subtract slippage.
            
            fill_price = signal_price
            
            if direction == "BUY":
                fill_price += max(0, slippage) # Mostly paying more
            else:
                fill_price -= max(0, slippage) # Mostly selling for less
                
            actual_slippage = abs(fill_price - signal_price)
            return fill_price, actual_slippage

        except Exception:
            # Fallback
            return signal_price, 0.0
