import asyncio
import logging
import sys
import os
import time

# Add project root to path
sys.path.append(os.getcwd())

from core.zmq_bridge import ZmqBridge
from core.execution_engine import ExecutionEngine
from analytics.telegram_notifier import get_notifier

# Silence noisy loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logging.getLogger("peewee").setLevel(logging.WARNING)

logger = logging.getLogger("ForceTradeTest")

async def test_execution_flow():
    """
    Forces a live test trade to verify:
    1. ZmqBridge connection.
    2. ExecutionEngine order transmission.
    3. Trade context persistence.
    4. Closure detection and Telegram notification.
    """
    print("\n" + "="*60)
    print(" 🕹️  ATL4S-FOREX: LIVE EXECUTION VERIFICATION")
    print("="*60)
    print("Este script vai forçar uma ordem de 0.01 lotes com TP curto (2 pips).")
    print("Certifique-se de que o main_laplace.py está DESLIGADO.")
    print("="*60 + "\n")

    try:
        # CLEANUP ZOMBIES (Prevent Ghost Notifications)
        context_file = "d:/Atl4s-Forex/data/trade_context.json"
        if os.path.exists(context_file):
            os.remove(context_file)
            print("🧹 Contexto limpo: trade_context.json removido para evitar fantasmas.")

        bridge = ZmqBridge(port=5558)
        notifier = get_notifier()
        executor = ExecutionEngine(bridge, notifier=notifier)
        
        print("⏳ Aguardando conexão do MetaTrader 5...")
        tick = None
        while tick is None:
            tick = bridge.get_tick()
            if tick is None:
                await asyncio.sleep(1)
        
        symbol = tick.get('symbol', 'GBPUSD')
        print(f"✅ Conectado! Recebendo ticks de {symbol} (@ {tick['bid']})")
        
        # --- NOVO: TESTE DE CONECTIVIDADE TELEGRAM ---
        print("\n📡 Testando conexão com Telegram...")
        ping_ok = await notifier.send_message("🔍 *TESTE DE CONEXÃO:* Iniciando fluxo de verificação de execução...")
        if ping_ok:
            print("✅ Telegram OK! Mensagem de teste recebida.")
        else:
            print("❌ ERRO NO TELEGRAM: Verifique seu Token e ChatID no config/telegram.json")
            print("Abortando teste para evitar ordens sem notificação.")
            return

        print(f"\n🚀 DISPARANDO ORDEM DE TESTE: {symbol} | BUY | 0.01 lotes")
        print("   ∟ Alvo: TP de apenas 2 pips para fechamento rápido.")
        
        result = await executor.execute_trade(
            symbol=symbol,
            direction="BUY",
            lots=0.01,
            sl_pips=20,
            tp_pips=2,
            comment="VERIFICATION_TEST"
        )
        
        if result == "SENT":
            print("\n✅ ORDEM ENVIADA AO MT5!")
            print("📱 Notificação de Entrada enviada ao Telegram.")
            print("Monitorando para fechamento (Take Profit hit)...")
            print("Quando a ordem fechar, você deve receber um alerta no Telegram.")
            
            # Monitor loop: 1. Wait for Open, 2. Wait for Close
            timeout = 600 
            start_time = time.time()
            trade_opened = False
            tick_count = 0
            last_print_time = 0
            
            while time.time() - start_time < timeout:
                current_tick = bridge.get_tick()
                if current_tick:
                    tick_count += 1
                    await executor.monitor_positions(current_tick)
                    
                    trades = current_tick.get('trades_json', [])
                    now = time.time()
                    
                    # Periodic Status Update (every 5 seconds)
                    if now - last_print_time > 5:
                        active_tickets = list(executor.trade_sources.keys())
                        pos_count = current_tick.get('positions', 0)
                        print(f"📡 Ticks: {tick_count} | Posições no MT5: {pos_count} | JSON Parseados: {len(trades)} | Rastreio Python: {active_tickets if active_tickets else 'Aguardando...'}")
                        last_print_time = now
                    
                    # Step 1: Confirm Open
                    if not trade_opened and executor.trade_sources:
                        print(f"\n📈 ORDEM REGISTRADA: {list(executor.trade_sources.keys())}")
                        print("Ficaremos em loop até que ela feche no MT5.")
                        trade_opened = True
                    
                    # Step 2: Confirm Close (only after it was confirmed open)
                    if trade_opened and not executor.trade_sources:
                        print("\n🎯 SUCESSO: Ordem fechada detectada!")
                        print("Aguardando 5 segundos para garantir que o Telegram termine de enviar...")
                        await asyncio.sleep(5)
                        print("Verifique seu Telegram para a notificação de PnL.")
                        break
                else:
                    if time.time() - last_print_time > 10:
                        print("⚠️ Sem sinal do MT5 (Ponte pode estar desconectada)")
                        last_print_time = time.time()
                
                await asyncio.sleep(0.5)
            else:
                if not trade_opened:
                    print("\n❌ FALHA: A ordem nunca apareceu no rastreador do Python.")
                    print("Verifique se o MagicNumber e o Símbolo estão corretos no MT5.")
                else:
                    print("\n⌛ TIMEOUT: A ordem não fechou em 10 minutos.")
                
        else:
            print("\n❌ FALHA: A ponte não enviou a ordem. Verifique o log do MT5.")

    except Exception as e:
        logger.error(f"Erro no teste: {e}")
    finally:
        print("\n🏁 Teste finalizado.")

if __name__ == "__main__":
    asyncio.run(test_execution_flow())
