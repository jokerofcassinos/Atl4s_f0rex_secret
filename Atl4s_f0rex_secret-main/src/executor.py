import MetaTrader5 as mt5
import logging

logger = logging.getLogger("Executor")

class Executor:
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.magic_number = 47145 # ATL4S

    def execute(self, decision):
        """
        Places order on MT5.
        decision: {'action': 'BUY', 'lot': 0.01, 'sl': 1234, 'tp': 1240}
        """
        if decision['action'] == "HOLD":
            return

        # Prepare request
        action_type = mt5.TRADE_ACTION_DEAL
        order_type = mt5.ORDER_TYPE_BUY if decision['action'] == 'BUY' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if decision['action'] == 'BUY' else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": action_type,
            "symbol": self.symbol,
            "volume": decision['lot'],
            "type": order_type,
            "price": price,
            "sl": decision['sl'],
            "tp": decision['tp'],
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "Atl4s-Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Failed: {result.comment} ({result.retcode})")
        else:
            logger.info(f"Order Executed: {decision['action']} {decision['lot']} lots @ {price}")
            
    def get_open_positions(self):
        return mt5.positions_get(symbol=self.symbol)
