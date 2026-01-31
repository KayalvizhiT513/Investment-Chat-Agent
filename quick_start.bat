@echo off
echo ========================================
echo Investment Analytics AI Agent
echo Quick Start (No Dependency Check)
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please run start_all_services.bat first
    pause
    exit /b 1
)

REM Start services in separate windows
echo Starting all services...
echo.

start "Data API - Port 8001" cmd /k "venv\Scripts\activate.bat && cd data_api && uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"
timeout /t 2 /nobreak >nul

start "Analytics API - Port 8002" cmd /k "venv\Scripts\activate.bat && cd analytics_api && uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"
timeout /t 2 /nobreak >nul

start "Formula API - Port 8003" cmd /k "venv\Scripts\activate.bat && cd formula_api && uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload"
timeout /t 2 /nobreak >nul

start "Agent API - Port 8000 (MAIN)" cmd /k "venv\Scripts\activate.bat && cd agent_api && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 5 /nobreak >nul

echo ✓ All services started!
echo.
echo Opening application...
start http://localhost:8000

echo.
echo Application running at: http://localhost:8000
echo.
pause
