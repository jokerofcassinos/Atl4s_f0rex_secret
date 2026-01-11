import asyncio
from core.neuro_link import ChromeBridge

async def main():
    print("=== INICIANDO NEURO-BRIDGE ===")
    bridge = ChromeBridge()
    
    connected = await bridge.connect()
    if not connected:
        print("FALHA CRITICA: Não foi possível conectar ao Chrome.")
        print("Certifique-se de ter rodado o 'start_neuro_chrome.bat' e que o Chrome está aberto.")
        return

    print("\n>>> MENTOR REMOTO CONECTADO.")
    print(">>> O sistema agora é um híbrido: Local + Web.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("VOCÊ (Local) > ")
        if user_input.lower() in ["exit", "sair"]:
            break
            
        # Enviar
        print("Transcrevendo para a IA remota...")
        await bridge.send_thought(user_input)
        
        # Receber
        print("Aguardando resposta...")
        response = await bridge.listen_response()
        print(f"\n[MENTOR REMOTO] >\n{response}\n")
        print("-" * 50)

    await bridge.close()

if __name__ == "__main__":
    asyncio.run(main())
