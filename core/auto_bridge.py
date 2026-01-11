import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from core.neuro_link import ChromeBridge

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("neuro_loop.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoBridge")

class AutoBridge:
    def __init__(self):
        self.bridge = ChromeBridge()
        self.running = True
        self.history = []

    async def start_loop(self):
        """Loop de Debate Autônomo: Arquiteto <-> Executor <-> Engenheiro."""
        logger.info("⚡ INICIANDO PROTOCOLO DUAL-MIND (DEBATE REAL) ⚡")
        logger.info("Certifique-se de ter DUAS abas de IA abertas (ex: 2 Geminis, ou 1 Gemini + 1 ChatGPT).")
        
        if not await self.bridge.connect():
            logger.error("❌ FALHA CRÍTICA: Não foi possível conectar.")
            return

        if not self.bridge.engineer_page:
            logger.warning("⚠️ MODO SOLO: Apenas uma IA detectada. O debate será simulado (não ideal).")

        # 1. Kickoff
        boot_msg = (
            "⚠️ PROTOCOLO DUAL-MIND ATIVADO.\n"
            "Eu sou o AutoBridge. Estou conectado a DUAS IAs.\n"
            "- VOCÊ é o ARQUITETO (Líder Estratégico).\n"
            "- A outra aba é o ENGENHEIRO (Crítico/Auditor).\n"
            "\n"
            "Seu objetivo: Definir a estratégia e comandar.\n"
            "O Engenheiro vai revisar seus comandos e analisar os erros antes de você decidir de novo.\n"
            "Comande."
        )
        await self.bridge.send_thought(boot_msg, target="architect")
        
        while self.running:
            try:
                # --- FASE 1: ARQUITETO FALA ---
                logger.info("👂 Ouvindo ARQUITETO...")
                arch_thought = await self.bridge.listen_response(timeout=120, target="architect")
                
                if not arch_thought or arch_thought in self.history:
                    await asyncio.sleep(5)
                    continue
                
                self.history.append(arch_thought)
                logger.info(f"🧠 ARQUITETO DIZ:\n{arch_thought[:200]}...")
                
                # --- FASE 2: AÇÃO (CORPO) ---
                exec_result = await self.execute_thought(arch_thought)
                
                # --- FASE 3: ENGENHEIRO ANALISA (Se existir) ---
                if self.bridge.engineer_page:
                    logger.info("📨 Encaminhando para o ENGENHEIRO (Review)...")
                    
                    engineer_prompt = (
                        f"🔧 **REVIEW REQUEST (De: AutoBridge)**\n"
                        f"O Arquiteto ordenou:\n---\n{arch_thought[:1000]}\n---\n"
                        f"Resultado da Execução:\n---\n{exec_result[-3000:]}\n---\n"
                        f"Analise isso criticamente. Houve erro? A estratégia faz sentido? Qual sua sugestão para o Arquiteto?"
                    )
                    await self.bridge.send_thought(engineer_prompt, target="engineer")
                    
                    logger.info("👂 Ouvindo ENGENHEIRO...")
                    eng_feedback = await self.bridge.listen_response(timeout=120, target="engineer")
                    logger.info(f"👷 ENGENHEIRO DIZ:\n{eng_feedback[:200]}...")
                    
                    # --- FASE 4: RETORNO AO ARQUITETO ---
                    final_report = (
                        f"� **FEEDBACK DO ENGENHEIRO:**\n{eng_feedback}\n\n"
                        f"📜 **LOG TÉCNICO CRU:**\n{exec_result[-1000:]}"
                    )
                else:
                    # Modo Solo
                    final_report = f"✅ EXECUTADO. SAÍDA:\n{exec_result[-2000:]}"

                logger.info("📤 Devolvendo relatório ao ARQUITETO...")
                await self.bridge.send_thought(final_report, target="architect")
                
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Erro no Loop: {e}")
                await asyncio.sleep(5)

    def _analyze_result(self, stdout_content: str) -> str:
        """Gera 'opiniões' para debater com a IA Remota."""
        opinion = ""
        
        # 1. Análise de Erros (Crítico)
        if "Traceback" in stdout_content or "Error:" in stdout_content:
            opinion += "⚠️ **DISCORDO DA ABORDAGEM:** O código quebrou. Veja o Traceback. Não podemos avançar sem corrigir isso. Minha sugestão é revisar as importações ou a sintaxe.\n"
        
        # 2. Análise de Backtest (O Analista Quant)
        elif "sim_report" in stdout_content or os.path.exists("simulation_report.txt"):
            try:
                if os.path.exists("simulation_report.txt"):
                     with open("simulation_report.txt", "r", encoding="utf-8") as f:
                         content = f.read()
                         opinion += f"📈 **MINHA ANÁLISE QUANTITATIVA:**\nLi o resultado. Resumo:\n{content[:400]}\n"
                         
                         if "Win Rate: 0%" in content:
                             opinion += "\n🔥 **PONTO DE DEBATE:** O resultado foi desastroso (0% Win Rate). Sua estratégia atual falhou completamente. Precisamos mudar o oscilador ou os filtros. O que você propõe?\n"
                         elif "Win Rate: 100%" in content:
                             opinion += "\n🧐 **CETICISMO:** 100% de acerto? Isso parece Overfitting. Sugiro testarmos em um período de 'Crise' para validar.\n"
                         else:
                             opinion += "\n✅ **APROVAÇÃO PARCIAL:** Os resultados são promissores, mas podemos otimizar o Drawdown. O que acha de ajustar o Stop Loss?\n"
            except:
                pass

        # 3. Confirmação de Código (O Code Reviewer)
        elif "FILE UPDATED" in stdout_content:
            opinion += "💾 **CODE REVIEW:** Apliquei suas mudanças no arquivo."
            if "laplace_demon.py" in stdout_content:
                opinion += " Você alterou o cérebro do bot. Espero que a lógica 'Sniper' esteja correta. Vamos rodar um teste para provar sua tese?\n"
            
        if not opinion:
            opinion = "Execução limpa, mas estou aguardando sua direção estratégica. Para onde vamos agora?"
            
        return opinion

    async def execute_thought(self, thought: str) -> str:
        """Decodifica e executa intenções da IA (CMD, READ, WRITE)."""
        output = ""
        
        # 1. Tentativa de Aplicar Código (Procura por blocos ```python ... ``` associados a arquivos)
        code_applied = await self._apply_code_blocks(thought)
        if code_applied:
            output += code_applied + "\n"
        
        # 2. Comandos Explícitos
        lines = thought.split('\n')
        for line in lines:
            if "CMD:" in line:
                cmd = line.split("CMD:")[1].strip()
                logger.info(f"🛠️ EXECUTANDO COMANDO: {cmd}")
                try:
                    # Executar comando real
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    
                    if stdout: output += f"[STDOUT]\n{stdout.decode('cp1252', errors='replace').strip()}\n"
                    if stderr: output += f"[STDERR]\n{stderr.decode('cp1252', errors='replace').strip()}\n"
                    
                except Exception as e:
                    output += f"Erro na execução: {e}\n"
            
            elif "READ:" in line:
                path = line.split("READ:")[1].strip()
                try:
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            output += f"[FILE: {path}]\n{content[:3000]}\n(Truncated if too long)"
                    else:
                        output += f"Arquivo não encontrado: {path}\n"
                except Exception as e:
                    output += f"Erro ao ler arquivo: {e}\n"

        return output

    async def _apply_code_blocks(self, text: str) -> str:
        """
        Analisa o texto procurando por padrões de escrita de arquivo.
        Padrão esperado:
        #### 📄 ARQUIVO: `caminho/do/arquivo.py`
        ```python
        codigo...
        ```
        """
        import re
        
        report = ""
        # Regex para capturar caminho e conteúdo
        # Procura por: "ARQUIVO: `path`" ... code block
        # Esta regex é simplificada e assume que o bloco de código vem logo após o cabeçalho
        pattern = r"ARQUIVO:\s*[`'\"]?([^`'\n\r]+)[`'\"]?.*?```(?:\w+)?\s(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            return ""

        for path, content in matches:
            path = path.strip()
            # Limpeza básica do conteúdo
            content = content.strip()
            
            try:
                # Segurança: Prevenir escrita fora do diretório (básico)
                if ".." in path or not (path.endswith(".py") or path.endswith(".txt") or path.endswith(".md") or path.endswith(".json")):
                     report += f"⚠️ SKIPPED: Caminho inseguro ou extensão inválida: {path}\n"
                     continue

                # Garantir diretório
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"💾 ARQUIVO GRAVADO: {path}")
                report += f"✅ FILE UPDATED: {path} ({len(content.splitlines())} lines)\n"
                
            except Exception as e:
                logger.error(f"Erro ao gravar arquivo {path}: {e}")
                report += f"❌ WRITE ERROR {path}: {e}\n"
                
        return report

if __name__ == "__main__":
    loop = AutoBridge()
    asyncio.run(loop.start_loop())
