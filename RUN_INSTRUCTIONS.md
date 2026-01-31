# 🚀 How to Run the Investment Analytics AI Agent

## Prerequisites

- Python 3.8+ installed
- Virtual environment created at `venv/`
- Environment variables configured in `.env` file:
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
  - `OPENAI_API_KEY`
  - `DATA_API_URL=http://localhost:8001`
  - `ANALYTICS_API_URL=http://localhost:8002/analytics`

---

## Quick Start (Recommended)

### Option 1: Full Start with Dependency Installation

```bash
start_all_services.bat
```

This script will:
1. ✅ Activate virtual environment
2. ✅ Install/update all dependencies
3. ✅ Start all 4 microservices
4. ✅ Open the application in your browser

### Option 2: Quick Start (Skip Dependency Check)

```bash
quick_start.bat
```

Use this if dependencies are already installed.

---

## Manual Start (Individual Services)

If you prefer to start services individually:

### 1. Activate Virtual Environment
```bash
venv\Scripts\activate
```

### 2. Start Each Service

**Terminal 1 - Data API (Port 8001):**
```bash
cd data_api
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Analytics API (Port 8002):**
```bash
cd analytics_api
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

**Terminal 3 - Formula API (Port 8003):**
```bash
cd formula_api
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

**Terminal 4 - Agent API (Port 8000 - MAIN):**
```bash
cd agent_api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the Application
Open your browser and navigate to: **http://localhost:8000**

---

## Stopping the Services

### Option 1: Using Stop Script
```bash
stop_all_services.bat
```

### Option 2: Manual Stop
Close all the terminal windows running the services.

---

## Service Architecture

| Service | Port | Description | Dependencies |
|---------|------|-------------|--------------|
| **agent_api** | 8000 | Main conversational AI agent + Frontend | OpenAI, Supabase, Analytics API |
| **data_api** | 8001 | Data access layer for portfolios/benchmarks | Supabase |
| **analytics_api** | 8002 | Performance analytics computation | Supabase, NumPy |
| **formula_api** | 8003 | Formula computation engine | NumPy, Pandas |

---

## API Endpoints

### Agent API (Main Interface)
- **Frontend:** http://localhost:8000
- **Chat Endpoint:** http://localhost:8000/chat

### Data API
- **Health Check:** http://localhost:8001/
- **Portfolios:** http://localhost:8001/portfolios
- **Benchmarks:** http://localhost:8001/benchmarks

### Analytics API
- **Health Check:** http://localhost:8002/
- **Compute Analytics:** http://localhost:8002/analytics

### Formula API
- **Health Check:** http://localhost:8003/

---

## Troubleshooting

### Issue: Virtual environment not found
**Solution:** Create a new virtual environment:
```bash
python -m venv venv
```

### Issue: Port already in use
**Solution:** 
1. Run `stop_all_services.bat` to kill existing processes
2. Or manually kill processes using ports 8000-8003:
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Missing dependencies
**Solution:** Install dependencies manually:
```bash
venv\Scripts\activate
pip install -r agent_api\requirements.txt
pip install -r analytics_api\requirements.txt
pip install -r data_api\requirements.txt
pip install -r formula_api\requirements.txt
```

### Issue: Environment variables not loaded
**Solution:** Verify `.env` file exists in the root directory with all required variables.

---

## Development Mode

All services run with `--reload` flag, which means:
- ✅ Auto-reload on code changes
- ✅ Hot reloading for faster development
- ✅ No need to restart services after code updates

---

## Production Deployment

For production deployment, refer to:
- `data_api/render.yaml` - Render.com deployment config
- Individual service README files for cloud deployment instructions

---

## Support

For issues or questions:
1. Check the main README.md
2. Review individual service documentation
3. Check the demo video: https://www.loom.com/share/ebc2bededa4548baa15e479609c1202d
