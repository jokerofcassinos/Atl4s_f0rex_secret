import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("Atl4s-MT5Monitor")

class MT5Monitor:
    def __init__(self):
        self.connected = False
        self._initialize()

    def _initialize(self):
        try:
            if not mt5.initialize():
                logger.error(f"MT5 Init Failed: {mt5.last_error()}")
                self.connected = False
            else:
                logger.info("MT5 Python API Connected.")
                self.connected = True
        except Exception as e:
            logger.error(f"MT5 Init Exception: {e}")
            self.connected = False

    def get_account_summary(self):
        """
        Returns basic account info: Balance, Equity, Profit (floating).
        """
        if not self.connected: 
            if not self._initialize(): return None
            
        info = mt5.account_info()
        if info is None:
            return None
        
        return {
            'balance': info.balance,
            'equity': info.equity,
            'profit': info.profit,
            'margin_free': info.margin_free
        }

    def get_open_positions(self, symbol=None):
        """
        Retrieves all open positions.
        Returns: List of dicts (Ticket, Type, Volume, Profit, OpenPrice, SL, TP)
        """
        if not self.connected: 
             if not self._initialize(): return []
             
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                return []
                
            results = []
            for pos in positions:
                results.append({
                    'ticket': pos.ticket,
                    'type': pos.type, # 0=Buy, 1=Sell
                    'volume': pos.volume,
                    'profit': pos.profit,
                    'open_price': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'symbol': pos.symbol
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def analyze_manual_performance(self, magic_number=0, days=30):
        """
        Analyzes trade history.
        magic_number=0 usually implies manual trades.
        """
        if not self.connected: self._initialize()

        from_date = datetime.now() - timedelta(days=days)
        deals = mt5.history_deals_get(from_date, datetime.now())

        if deals is None:
            logger.warning("No history deals found.")
            return {
                'total_trades': 0, 'win_rate': 0.0, 'profit': 0.0, 'loss': 0.0, 'net': 0.0
            }

        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        
        # Filter: Entry Out (Closed trades) and Magic Number
        # deal_type: 1 (EXIT) ? No, MT5 deals are In/Out. entry=1 (Out) is closing.
        # But profit is recorded on the deal that closes relevant position? 
        # Actually simplest is to sum profit of all deals with magic=0.
        
        # Filter by Magic Number (0 for manual)
        manual_deals = df[df['magic'] == magic_number]
        
        # Filter for deals that have PnL (usually ENTRY_OUT or ENTRY_INOUT)
        # Entry types: 0=IN, 1=OUT, 2=INOUT, 3=OUT_BY
        # We generally look at profit != 0
        closed_deals = manual_deals[manual_deals['profit'] != 0]

        if closed_deals.empty:
            return {'total_trades': 0, 'win_rate': 0.0, 'profit': 0.0, 'loss': 0.0, 'net': 0.0}

        wins = closed_deals[closed_deals['profit'] > 0]
        losses = closed_deals[closed_deals['profit'] < 0]

        total_profit = wins['profit'].sum()
        total_loss = losses['profit'].sum()
        net_profit = total_profit + total_loss

        win_count = len(wins)
        loss_count = len(losses)
        total_count = win_count + loss_count

        win_rate = (win_count / total_count * 100) if total_count > 0 else 0.0

        return {
            'total_trades': total_count,
            'win_rate': win_rate,
            'profit': total_profit,
            'loss': abs(total_loss),
            'net': net_profit,
            'accuracy_label': f"{win_rate:.1f}%"
        }
