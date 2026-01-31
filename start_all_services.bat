@echo off
echo ========================================
echo Investment Analytics AI Agent
echo Starting All Services...
echo ========================================
echo.

REM Activate virtual environment
echo [1/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please ensure venv exists. Run: python -m venv venv
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Install/Update dependencies for all services
echo [2/5] Installing dependencies...
echo Installing data_api dependencies...
pip install -q -r data_api\requirements.txt
echo Installing analytics_api dependencies...
pip install -q -r analytics_api\requirements.txt
echo Installing formula_api dependencies...
pip install -q -r formula_api\requirements.txt
echo Installing agent_api dependencies...
pip install -q -r agent_api\requirements.txt
echo ✓ All dependencies installed
echo.

REM Start services in separate windows
echo [3/5] Starting microservices...
echo.

echo Starting data_api on port 8001...
start "Data API - Port 8001" cmd /k "venv\Scripts\activate.bat && cd data_api && uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"
timeout /t 2 /nobreak >nul

echo Starting analytics_api on port 8002...
start "Analytics API - Port 8002" cmd /k "venv\Scripts\activate.bat && cd analytics_api && uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"
timeout /t 2 /nobreak >nul

echo Starting formula_api on port 8003...
start "Formula API - Port 8003" cmd /k "venv\Scripts\activate.bat && cd formula_api && uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload"
timeout /t 2 /nobreak >nul

echo Starting agent_api on port 8000 (Main Service)...
start "Agent API - Port 8000 (MAIN)" cmd /k "venv\Scripts\activate.bat && cd agent_api && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo [4/5] All services started successfully!
echo ========================================
echo.
echo Service Status:
echo ✓ Data API:      http://localhost:8001
echo ✓ Analytics API: http://localhost:8002
echo ✓ Formula API:   http://localhost:8003
echo ✓ Agent API:     http://localhost:8000 (MAIN - Frontend)
echo.
echo ========================================
echo [5/5] Opening main application...
echo ========================================
echo.

REM Wait a bit for services to fully start
timeout /t 5 /nobreak >nul

REM Open the main application in browser
start http://localhost:8000

echo.
echo ========================================
echo Application is now running!
echo ========================================
echo.
echo Main Interface: http://localhost:8000
echo.
echo To stop all services:
echo - Close all the terminal windows that opened
echo - Or run: stop_all_services.bat
echo.
echo Press any key to keep this window open...
pause >nul
