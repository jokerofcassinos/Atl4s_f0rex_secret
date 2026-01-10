
import pandas as pd
from .spread_simulator import SpreadSimulator
from .latency_simulator import LatencySimulator

class RealisticExecutor:
    """
    Combines Spread and Latency simulation to provide realistic trade execution.
    """
    
    def __init__(self, latency_ms: int = 100, symbol: str = "GBPUSD"):
        self.spread_sim = SpreadSimulator()
        self.latency_sim = LatencySimulator(latency_ms)
        self.symbol = symbol
        
    def get_current_tick(self, timestamp, price_close: float) -> dict:
        """
        Generates a realistic tick with dynamic spread.
        """
        spread = self.spread_sim.get_spread(self.symbol, current_hour=timestamp.hour)
        
        # We assume 'price_close' is the MID price or BID?
        # Standard OHLC is usually BID.
        # So BID = close
        # ASK = close + spread
        
        bid = price_close
        ask = bid + spread
        
        return {
            'time': timestamp,
            'bid': bid,
            'ask': ask,
            'spread': spread
        }
        
    def execute_order(self, direction: str, signal_price: float, signal_time, df_m1: pd.DataFrame) -> dict:
        """
        Executes an order with latency and slippage.
        """
        fill_price, slippage = self.latency_sim.simulate_fill(
            signal_price=signal_price,
            signal_time=signal_time,
            df_m1=df_m1,
            direction=direction
        )
        
        return {
            'fill_price': fill_price,
            'slippage': slippage,
            'requested_price': signal_price,
            'execution_time': signal_time # Logic time is same, but price reflects delay
        }
