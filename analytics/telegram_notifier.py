"""
Genesis Telegram Notifier

Real-time notifications via Telegram:
- Trade alerts
- Performance updates
- Risk warnings
- Daily summaries
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path
import json

logger = logging.getLogger("TelegramNotifier")


class TelegramNotifier:
    """
    Telegram notification system for Genesis
    
    Sends real-time alerts for:
    - Trade entries/exits
    - Performance milestones  
    - Risk alerts
    - Daily/weekly summaries
    """
    
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
        except ImportError:
            logger.warning("aiohttp not installed, using queue mode")
            self.message_queue.append(text)
            return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    def send_sync(self, text: str):
        """Synchronous send wrapper"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_message(text))
            else:
                loop.run_until_complete(self.send_message(text))
        except:
            # Fallback for new event loop
            asyncio.run(self.send_message(text))
    
    # Pre-formatted message templates
    
    def notify_trade_entry(self, direction: str, symbol: str, entry: float,
                           sl: float, tp: float, confidence: float, setup: str):
        """Notify trade entry"""
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        message = f"""
{emoji} *NEW TRADE - {direction}*

📊 *{symbol}*
Entry: `{entry:.5f}`
SL: `{sl:.5f}`
TP: `{tp:.5f}`

🎯 Setup: {setup}
💪 Confidence: {confidence:.0f}%
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_sync(message)
    
    def notify_trade_exit(self, direction: str, symbol: str, entry: float,
                          exit_price: float, profit: float, pips: float, win: bool):
        """Notify trade exit"""
        emoji = "✅" if win else "❌"
        profit_emoji = "💰" if profit > 0 else "📉"
        
        message = f"""
{emoji} *TRADE CLOSED - {'WIN' if win else 'LOSS'}*

📊 *{symbol}* {direction}
Entry: `{entry:.5f}`
Exit: `{exit_price:.5f}`

{profit_emoji} P/L: *${profit:.2f}* ({pips:.1f} pips)
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_sync(message)
    
    def notify_risk_alert(self, level: str, category: str, message: str):
        """Notify risk alert"""
        if level == "CRITICAL":
            emoji = "🚨"
        elif level == "WARNING":
            emoji = "⚠️"
        else:
            emoji = "ℹ️"
        
        text = f"""
{emoji} *RISK ALERT - {level}*

Category: {category}
{message}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_sync(text)
    
    def notify_daily_summary(self, trades: int, wins: int, profit: float,
                              win_rate: float, best_trade: float, worst_trade: float):
        """Send daily summary"""
        status = "✅" if win_rate >= 70 else "⚠️"
        
        message = f"""
📊 *DAILY SUMMARY*

{status} Win Rate: *{win_rate:.1f}%*

📈 Trades: {trades}
✅ Wins: {wins}
❌ Losses: {trades - wins}

💰 Total P/L: *${profit:.2f}*
🏆 Best: ${best_trade:.2f}
📉 Worst: ${worst_trade:.2f}

📅 {datetime.now().strftime('%Y-%m-%d')}
"""
        self.send_sync(message)
    
    def notify_milestone(self, milestone: str, details: str):
        """Notify milestone achievement"""
        message = f"""
🎉 *MILESTONE ACHIEVED!*

🏆 {milestone}

{details}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_sync(message)
    
    def notify_system_status(self, status: str, details: str = ""):
        """Notify system status"""
        if status == "ONLINE":
            emoji = "🟢"
        elif status == "OFFLINE":
            emoji = "🔴"
        else:
            emoji = "🟡"
        
        message = f"""
{emoji} *GENESIS {status}*

{details}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_sync(message)


# Global instance
_notifier = None

def get_notifier() -> TelegramNotifier:
    """Get or create notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


if __name__ == "__main__":
    # Demo mode
    print("="*60)
    print("GENESIS TELEGRAM NOTIFIER - DEMO")
    print("="*60)
    print()
    
    notifier = TelegramNotifier()
    
    print("Telegram Status:", "Enabled" if notifier.enabled else "Disabled (no credentials)")
    print()
    
    # Show sample messages
    print("Sample Trade Entry Message:")
    print("-" * 40)
    print("""
🟢 *NEW TRADE - BUY*

📊 *GBPUSD*
Entry: `1.26500`
SL: `1.26300`
TP: `1.27000`

🎯 Setup: GENESIS_PULLBACK
💪 Confidence: 75%
⏰ 14:30:00
""")
    
    print("Sample Trade Exit Message:")
    print("-" * 40)
    print("""
✅ *TRADE CLOSED - WIN*

📊 *GBPUSD* BUY
Entry: `1.26500`
Exit: `1.26800`

💰 P/L: *$30.00* (30.0 pips)
⏰ 15:45:00
""")
    
    print("Sample Risk Alert:")
    print("-" * 40)
    print("""
🚨 *RISK ALERT - CRITICAL*

Category: DAILY_LOSS
Daily loss $500 exceeds limit - STOP TRADING TODAY!

⏰ 16:00:00
""")
    
    print()
    print("="*60)
    print("To enable Telegram notifications:")
    print("1. Create a bot via @BotFather on Telegram")
    print("2. Get your chat ID via @userinfobot")
    print("3. Run: notifier.save_config('BOT_TOKEN', 'CHAT_ID')")
    print("="*60)
