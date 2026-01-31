@echo off
echo ========================================
echo Stopping All Services...
echo ========================================
echo.

REM Kill all uvicorn processes
echo Terminating all uvicorn processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data API*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Analytics API*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Formula API*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Agent API*" 2>nul

REM Alternative: Kill all Python processes running uvicorn
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID:"') do (
    netstat -ano | find "8000" | find "%%a" >nul && taskkill /F /PID %%a 2>nul
    netstat -ano | find "8001" | find "%%a" >nul && taskkill /F /PID %%a 2>nul
    netstat -ano | find "8002" | find "%%a" >nul && taskkill /F /PID %%a 2>nul
    netstat -ano | find "8003" | find "%%a" >nul && taskkill /F /PID %%a 2>nul
)

echo.
echo ✓ All services stopped
echo.
pause
