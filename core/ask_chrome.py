import asyncio
import sys
from core.neuro_link import ChromeBridge

async def ask(prompt):
    bridge = ChromeBridge()
    if not await bridge.connect():
        print("ERROR: Connection failed")
        return
    
    await bridge.send_thought(prompt)
    response = await bridge.listen_response(timeout=45) # Wait up to 45s
    print(response)
    await bridge.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ask_chrome.py \"Prompt text\"")
        sys.exit(1)
        
    prompt = sys.argv[1]
    asyncio.run(ask(prompt))
