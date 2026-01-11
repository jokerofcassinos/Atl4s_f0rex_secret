@echo off
echo ===================================================
echo   INICIANDO MODO DEPURACAO NEURO-LINK (CHROME)
echoFrom: Antigravity
echo ===================================================
echo.
echo !IMPORTANTE!: Feche TODAS as janelas do Chrome antes de continuar.
echo Se o Chrome ja estiver aberto sem a porta de debug, nao funcionara.
echo.
pause

echo Iniciando Chrome na porta 9222...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\selenium\ChromeProfile"

echo.
echo Chrome iniciado. Por favor:
echo 1. Navegue para o ChatGPT ou Gemini.
echo 2. Faca login se necessario.
echo 3. Volte aqui e execute o script ponte.
echo.
pause
