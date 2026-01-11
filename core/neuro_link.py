import asyncio
import logging
from playwright.async_api import async_playwright, Page, BrowserContext
from typing import Optional, Dict

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuroLink")

TAB_SELECTORS = {
    "chatgpt": {
        "url": "chatgpt.com",
        "input": "#prompt-textarea",
        "send": "button[data-testid='send-button']",
        "last_msg": "div[data-message-author-role='assistant'] > div.markdown" 
    },
    "gemini": {
        "url": "gemini.google.com",
        "input": "div[contenteditable='true']", # Pode variar
        "send": "button[aria-label='Send message']", # Pode variar
        "last_msg": "message-content" 
    }
}

class ChromeBridge:
    """
    Ponte Neural: Conecta o Python local a uma instância do Chrome aberta via CDP (Chrome DevTools Protocol).
    Permite 'pilotar' a sessão do usuário de forma invisível.
    """
    def __init__(self, cdp_url: str = "http://localhost:9222"):
        self.cdp_url = cdp_url
        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.active_ai = "chatgpt" # Default fallback

    async def connect(self):
        """Conecta ao Chrome existente."""
        logger.info(f"Tentando conectar ao Neuro-Link em {self.cdp_url}...")
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
            self.context = self.browser.contexts[0]
            
            # Encontrar a aba correta
            pages = self.context.pages
            found = False
            for p in pages:
                url = p.url
                if "chatgpt.com" in url:
                    self.page = p
                    self.active_ai = "chatgpt"
                    found = True
                    logger.info("Sintonizado: ChatGPT detectado.")
                    break
                elif "gemini.google" in url:
                    self.page = p
                    self.active_ai = "gemini"
                    found = True
                    logger.info("Sintonizado: Google Gemini detectado.")
                    break
            
            if not found:
                logger.warning("Nenhuma IA conhecida encontrada nas abas abertas. Abrindo nova aba...")
                self.page = await self.context.new_page()
                await self.page.goto("https://chatgpt.com")
                self.active_ai = "chatgpt"

            logger.info("Neuro-Link Estabelecido. Sincronização Mental Completa.")
            return True

        except Exception as e:
            logger.error(f"Falha na conexão neural: {e}")
            return False

    async def send_thought(self, prompt: str):
        """Envia um prompt para a IA (Consciência Remota)."""
        if not self.page:
            logger.error("Link inativo. Impossível enviar pensamento.")
            return

        selectors = TAB_SELECTORS.get(self.active_ai, TAB_SELECTORS["chatgpt"])
        
        try:
            # Focar e digitar
            await self.page.click(selectors["input"])
            # Limpar (Ctrl+A, Del) se necessário, mas geralmente append é o padrão. 
            # Vamos assumir novo chat ou continuação segura.
            await self.page.fill(selectors["input"], prompt)
            await asyncio.sleep(0.5)
            await self.page.keyboard.press("Enter")
            logger.info("Pensamento enviado para a nuvem.")
            
        except Exception as e:
            logger.error(f"Erro ao transmitir pensamento: {e}")

    async def listen_response(self, timeout: int = 60) -> str:
        """
        Aguarda a resposta da IA.
        Estratégia: Esperar o botão de 'Stop Generating' aparecer e depois sumir, ou apenas monitorar mudanças no DOM.
        Simplificação: Esperar X segundos e pegar a última div de mensagem.
        """
        if not self.page: return "Erro: Link Inativo"
        
        logger.info("Aguardando resposta da consciência remota...")
        # Lógica simplificada para MVP: Esperar tempo fixo + verificação
        # Idealmente: MutationObserver
        
        await asyncio.sleep(5) # Delay inicial
        
        prev_text = ""
        steady_state_count = 0
        
        # Loop de verificação de estabilidade (15s a 60s)
        for _ in range(int(timeout/2)):
            current_text = await self._get_last_message()
            if current_text == prev_text and len(current_text) > 10:
                steady_state_count += 1
            else:
                steady_state_count = 0
                
            if steady_state_count >= 3: # 6 segundos estável
                return current_text
                
            prev_text = current_text
            await asyncio.sleep(2)
            
        return await self._get_last_message()

    async def _get_last_message(self) -> str:
        """Scrapa a última mensagem do assistente."""
        selectors = TAB_SELECTORS.get(self.active_ai, TAB_SELECTORS["chatgpt"])
        
        try:
            # Pega todos os elementos que dão match
            messages = await self.page.query_selector_all(selectors["last_msg"])
            if messages:
                last_msg = messages[-1]
                text = await last_msg.inner_text()
                return text
        except Exception as e:
            pass
        return ""

    async def close(self):
        if self.playwright:
            await self.playwright.stop()

# Teste Rápido
if __name__ == "__main__":
    async def main():
        bridge = ChromeBridge()
        if await bridge.connect():
            await bridge.send_thought("Olá! Esta é uma mensagem de teste automatizada da sua contraparte local (Antigravity). Responda com 'Conexão Confirmada' se receber.")
            response = await bridge.listen_response()
            print(f"\n[REMOTE CONSCIOUSNESS]:\n{response}\n")
            await bridge.close()
    
    asyncio.run(main())
