import re
import sys
import os

def parse_log(log_path):
    if not os.path.exists(log_path):
        print(f"Erro: Arquivo {log_path} não encontrado.")
        return

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 1. Encontrar todos os IDs de trades com LOSS (vsl_hit)
    # Assumindo que o log tem algo como "Trade #123 ... vsl_hit" ou linhas próximas
    # Ajuste o regex abaixo conforme o formato exato do seu log
    loss_pattern = re.compile(r'Trade #(\d+).*?vsl_hit', re.IGNORECASE | re.DOTALL)
    
    # Se o 'vsl_hit' estiver em uma linha diferente do ID, precisamos de uma lógica de bloco.
    # Esta é uma abordagem genérica baseada na sua descrição:
    
    # Passo A: Identificar quais trades falharam
    failed_trade_ids = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "vsl_hit" in line:
            # Tenta achar o número do trade na mesma linha ou nas 5 anteriores
            for j in range(i, max(0, i-5), -1):
                match = re.search(r'Trade #(\d+)', lines[j])
                if match:
                    failed_trade_ids.append(match.group(1))
                    break
    
    failed_trade_ids = sorted(list(set(failed_trade_ids)))
    print(f"DEBUG: Encontrados {len(failed_trade_ids)} trades com loss.")

    # Passo B: Extrair o contexto de decisão para cada trade falho
    for trade_id in failed_trade_ids:
        print(f"\n{'='*40}")
        print(f" ANALISANDO LOSS: Trade #{trade_id}")
        print(f"{'='*40}")
        
        # Aqui buscamos o bloco onde a decisão foi tomada
        # Regex busca "Trade #ID" até o próximo "Trade #" ou fim do bloco de decisão
        decision_pattern = re.compile(rf"(Trade #{trade_id}.*?)(?=Trade #\d|$)", re.DOTALL)
        match = decision_pattern.search(content)
        
        if match:
            # Mostra apenas os primeiros 2000 caracteres do log do trade para não estourar contexto
            log_segment = match.group(1)[:2000] 
            print(log_segment)
        else:
            print("Log de decisão não encontrado para este ID.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extract_failures.py <caminho_do_log>")
    else:
        parse_log(sys.argv[1])