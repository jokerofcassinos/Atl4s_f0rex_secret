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
    def __init__(self, cdp_port=9222):
        self.cdp_url = f"http://localhost:{cdp_port}"
        self.playwright = None
        self.browser = None
        self.context = None
        
        # Dual-Mind Architecture
        self.architect_page = None # The Visionary (Gemini/ChatGPT) - Defines Strat
        self.engineer_page = None  # The Builder (Gemini/ChatGPT) - Reviews Code & Errors
        
        self.architect_name = "Unknown"
        self.engineer_name = "Unknown"

    async def connect(self):
        """Connects to Chrome and identifies TWO AI Agents."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.connect_over_cdp(endpoint_url=self.cdp_url)
            self.context = self.browser.contexts[0]
            pages = self.context.pages
            
            logger.info(f"Conectado ao Chrome via CDP. Abas abertas: {len(pages)}")
            
            # Simple Heuristic: First AI found = Architect. Second AI found = Engineer.
            # Ideally user has 2 tabs open: 
            # Tab 1: Gemini (Architect)
            # Tab 2: Gemini or ChatGPT (Engineer)
            
            found_count = 0
            
            for p in pages:
                url = p.url
                role = ""
                ai_name = ""
                
                if "chatgpt.com" in url or "gemini.google" in url or "claude.ai" in url:
                    if "chatgpt.com" in url: ai_name = "ChatGPT"
                    elif "gemini.google" in url: ai_name = "Gemini"
                    elif "claude.ai" in url: ai_name = "Claude"
                    
                    found_count += 1
                    
                    if self.architect_page is None:
                        self.architect_page = p
                        self.architect_name = ai_name
                        role = "ARQUITETO (Estratégia)"
                        logger.info(f"🔹 {role} detectado: {ai_name}")
                        
                    elif self.engineer_page is None:
                        self.engineer_page = p
                        self.engineer_name = ai_name
                        role = "ENGENHEIRO (Crítico)"
                        logger.info(f"🔸 {role} detectado: {ai_name}")
            
            if found_count < 2:
                logger.warning(f"⚠️ Apenas {found_count} IA(s) encontrada(s). Modo Debate requer 2 abas de IA abertas.")
                if found_count == 1:
                     logger.info("O sistema funcionará em modo 'Solo' (apenas Arquiteto) até você abrir a segunda aba e reiniciar.")
                return True # We allow 1, but warn. AutoBridge will handle the logic.
            
            return True

        except Exception as e:
            logger.error(f"Erro ao conectar via NeuroLink: {e}")
            return False

    async def send_thought(self, prompt: str, target: str = "architect"):
        """Sends prompt to specific agent."""
        page = self.architect_page if target == "architect" else self.engineer_page
        if not page: 
            logger.warning(f"Página para {target} não encontrada. Não é possível enviar pensamento.")
            return
        
        try:
            await page.bring_to_front()
            
            # Generic Selector Logic (Simplified)
            # Works for ChatGPT and Gemini (Div input)
            input_selector = "div[contenteditable='true']" 
            
            # Try to click and fill
            if await page.query_selector(input_selector):
                await page.click(input_selector)
                # Clear? No, just type.
                await page.keyboard.type(prompt)
                await page.keyboard.press("Enter")
                logger.info(f"Pensamento enviado para {target.upper()}.")
            else:
                 logger.error(f"Input não encontrado na aba {target}")
                 
        except Exception as e:
            logger.error(f"Falha ao enviar pensamento para {target}: {e}")

    async def listen_response(self, timeout: int = 60, target: str = "architect") -> str:
        """Listens to specific agent."""
        page = self.architect_page if target == "architect" else self.engineer_page
        if not page: 
            logger.warning(f"Página para {target} não encontrada. Não é possível ouvir resposta.")
            return ""
        
        logger.info(f"Aguardando resposta da consciência remota ({target})...")
        try:
            # Polling for stability
            prev_text = await self._get_last_message(page)
            steady_count = 0
            
            for _ in range(int(timeout/2)):
                await asyncio.sleep(2)
                current_text = await self._get_last_message(page)
                
                if current_text != prev_text and len(current_text) > min(len(prev_text)+5, 10):
                    # Text changed (typing), reset steady count
                    steady_count = 0
                elif current_text == prev_text and len(current_text) > 10:
                    steady_count += 1
                
                if steady_count >= 2: # 4 seconds steady
                    return current_text
                
                prev_text = current_text
                
            return await self._get_last_message(page)

        except Exception as e:
            logger.error(f"Erro ouvindo {target}: {e}")
            return ""

    async def _get_last_message(self, page: Page) -> str:
        """Scrapes the last assistant message from a given page."""
        try:
            # Generic selectors for last message
            # This is tricky as structure varies. 
            # We assume the AI sites use article or specialized divs.
            
            # Attempt to get all text from the likely message container area
            # ChatGPT: article, Gemini: .message-content (varies)
            # Fallback: Get full page text? No, too messy.
            
            # Simple heuristic: Look for elements that usually hold AI responses
            # We grab the LAST element that looks like a bot response.
            
            # Common markers
            blocks = await page.locator("div[data-message-author-role='assistant'], .model-response-text, article").all_inner_texts()
            if blocks:
                return blocks[-1]
            return ""
        except:
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
