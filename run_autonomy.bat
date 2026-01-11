@echo off
echo ===================================================
echo   ATIVANDO MODO AUTONOMO (ANTIGRAVITY v2.0)
echo ===================================================
echo.
echo Certifique-se que o Chrome ja esta aberto na porta 9222.
echo (Se nao, rode o start_neuro_chrome.bat antes)
echo.
pause

.\venv\Scripts\python.exe -m core.auto_bridge

pause
