"""
Test Telegram Connection
"""
import asyncio
import aiohttp

async def test_telegram():
    bot_token = "8348530602:AAGoFcr1zWI2cEML_YdpaVl2Cj1eTES3QZY"
    chat_id = "8198692328"
    
    message = """
🎉 *GENESIS TRADING BOT CONECTADO!*

✅ Telegram configurado com sucesso!
🚀 Sistema pronto para enviar alertas

📊 Você receberá:
- Alertas de trades
- Avisos de risco
- Resumos diários

⏰ Testado em: """ + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if response.status == 200:
                    print("✅ SUCESSO! Mensagem enviada para o Telegram!")
                    print(f"   Message ID: {result.get('result', {}).get('message_id')}")
                else:
                    print(f"❌ Erro: {response.status}")
                    print(f"   {result}")
    except Exception as e:
        print(f"❌ Erro de conexão: {e}")

if __name__ == "__main__":
    asyncio.run(test_telegram())
