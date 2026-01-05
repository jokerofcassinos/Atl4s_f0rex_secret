@echo off
echo [ATL4S] Iniciando Auto-Update para Branch v5.0...
echo ---------------------------------------------------

cd /d "%~dp0"

echo [1/3] Adicionando arquivos...
git add .

echo [2/3] Commitando mudancas...
set timestamp=%date:~-4%-%date:~3,2%-%date:~0,2%_%time:~0,2%-%time:~3,2%
git commit -m "Auto-Update: %timestamp%"

echo [3/3] Enviando para GitHub (v5.0)...
git push origin v5.0

echo ---------------------------------------------------
echo [SUCESSO] Update concluido!
pause
