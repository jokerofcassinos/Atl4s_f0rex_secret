import logging
import asyncio
from datetime import datetime
from typing import Optional, Any, List, Dict
from pathlib import Path
import json

logger = logging.getLogger("TelegramNotifier")

class TelegramNotifier:
    """Telegram notification system for Genesis"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self.message_queue = []
        
        # Config file
        self.config_file = Path("config/telegram.json")
        self._load_config()
        
        if self.enabled:
            logger.info("Telegram Notifier enabled")
        else:
            logger.info("Telegram Notifier disabled (no credentials)")
    
    def _load_config(self):
        """Load Telegram config from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.bot_token = self.bot_token or config.get('bot_token')
                    self.chat_id = self.chat_id or config.get('chat_id')
                    self.enabled = bool(self.bot_token and self.chat_id)
            except Exception as e:
                logger.warning(f"Could not load Telegram config: {e}")
    
    def save_config(self, bot_token: str, chat_id: str):
        """Save Telegram credentials"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump({
                'bot_token': bot_token,
                'chat_id': chat_id
            }, f, indent=2)
        
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = True
        logger.info("Telegram config saved")
    
    async def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send message via Telegram"""
        if not self.enabled:
            logger.debug(f"Telegram disabled, would send: {text[:50]}...")
            return False
        
        try:
            import aiohttp
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent")
                        return True
                    else:
                        logger.error(f"Telegram error: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
            
    def send_sync(self, text: str):
        """Synchronous send wrapper"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Avoid nested loops
                asyncio.create_task(self.send_message(text))
            else:
                loop.run_until_complete(self.send_message(text))
        except:
            try:
                asyncio.run(self.send_message(text))
            except:
                pass

    async def notify_trade_entry(self, direction: str, symbol: str, entry: float,
                           sl: float, tp: float, confidence: float, setup: str):
        """Notify trade entry"""
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        message = f"""
{emoji} *NEW TRADE - {direction}*

📊 *{symbol}*
Entry: `{entry:.5f}`
SL: `{sl:.5f}`
TP: `{tp:.5f}`
🎯 Setup: `{setup}`
🔥 Conf: `{confidence:.1f}%`
"""
        await self.send_message(message)

    async def notify_trade_exit(self, symbol: str, entry: float, exit: float,
                          pnl_dollars: float, pnl_pips: float, reason: str, source: str = "UNKNOWN"):
        """Notify trade exit"""
        emoji = "💰" if pnl_dollars > 0 else "❌"
        
        message = f"""
{emoji} *TRADE CLOSED - {symbol}*

Entry: `{entry:.5f}`
Exit: `{exit:.5f}`
Result: `${pnl_dollars:+.2f}` ({pnl_pips:+.1f} pips)
🎯 Setup: `{source}`
🚪 Reason: `{reason}`
"""
        await self.send_message(message)

    async def notify_risk_alert(self, level: str, category: str, message: str):
        """Notify risk alert"""
        emoji = "🚨" if level == "CRITICAL" else "⚠️" if level == "WARNING" else "ℹ️"
        
        text = f"""
{emoji} *RISK ALERT - {level}*

Category: {category}
{message}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(text)

    async def notify_backtest_report(self, symbol: str, start_date: str, end_date: str, results: Any):
        """Send comprehensive backtest report"""
        wr = getattr(results, 'win_rate', 0)
        profit = getattr(results, 'net_profit', 0)
        trades = getattr(results, 'total_trades', 0)
        pf = getattr(results, 'profit_factor', 0)
        mdd = getattr(results, 'max_drawdown_pct', 0)
        
        status = "🔥" if wr >= 60 else "✅" if wr >= 50 else "⚠️"
        
        message = f"""
{status} *BACKTEST COMPLETE - {symbol}*
{start_date} ➔ {end_date}

📊 *Win Rate: {wr:.1f}%*
💰 *Net Profit: ${profit:.2f}*
📉 *Profit Factor: {pf:.1f}*
📉 *Max Drawdown: {mdd:.1f}%*

📈 Total Trades: {trades}
✅ Wins: {getattr(results, 'winning_trades', 0)}
❌ Losses: {getattr(results, 'losing_trades', 0)}

🚀 *Genesis System Online*
"""
        await self.send_message(message)

    def notify_system_status(self, status: str, details: str = ""):
        """Notify system (sync fallback)"""
        emoji = "🟢" if status == "ONLINE" else "🔴" if status == "OFFLINE" else "🟡"
        message = f"{emoji} *GENESIS {status}*\n\n{details}"
        self.send_sync(message)

_notifier = None

def get_notifier():
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
