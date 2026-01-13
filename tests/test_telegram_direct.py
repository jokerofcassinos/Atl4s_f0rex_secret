import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from analytics.telegram_notifier import get_notifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTelegram")

async def test_telegram():
    notifier = get_notifier()
    print(f"Testing Telegram. Enabled: {notifier.enabled}")
    print(f"Token: {notifier.bot_token[:10]}...")
    print(f"Chat ID: {notifier.chat_id}")
    
    success = await notifier.send_message("🚀 Teste de conexão direta do Atl4s-Forex!")
    if success:
        print("✅ Mensagem enviada com sucesso!")
    else:
        print("❌ Falha no envio. Verifique os logs.")

if __name__ == "__main__":
    asyncio.run(test_telegram())
