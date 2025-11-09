from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# -----------------------------
#  Configuration
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Performance Data API",
    description="Unified API to access portfolio and benchmark data with date range filters.",
    version="1.0.0"
)


# -----------------------------
#  Health Check
# -----------------------------
@app.get("/", tags=["Health"])
def root():
    return {"message": "✅ Data API is live and connected to Supabase."}


# -----------------------------
#  Portfolios
# -----------------------------
@app.get("/portfolios", tags=["Portfolios"])
def get_all_portfolios():
    data = supabase.table("portfolios").select("*").execute()
    return data.data


@app.get("/portfolios/{portfolio_id}", tags=["Portfolios"])
def get_portfolio_by_id(portfolio_id: int):
    data = supabase.table("portfolios").select("*").eq("id", portfolio_id).execute()
    return data.data[0] if data.data else {"error": "Portfolio not found."}


# -----------------------------
#  Portfolio Returns
# -----------------------------
@app.get("/portfolio-returns", tags=["Portfolio Returns"])
def get_portfolio_returns(
    portfolio_id: int = Query(..., description="Portfolio ID"),
    start_date: str = Query(None, description="Filter by month_end_date >= start_date"),
    end_date: str = Query(None, description="Filter by month_end_date <= end_date"),
):
    """Return monthly portfolio returns, optionally filtered by date range."""
    query = supabase.table("portfolio_returns").select("*").eq("portfolio_id", portfolio_id)
    if start_date:
        query = query.gte("month_end_date", start_date)
    if end_date:
        query = query.lte("month_end_date", end_date)
    result = query.execute()
    return result.data


# -----------------------------
#  Benchmarks
# -----------------------------
@app.get("/benchmarks", tags=["Benchmarks"])
def get_all_benchmarks():
    data = supabase.table("benchmarks").select("*").execute()
    return data.data


@app.get("/benchmarks/{benchmark_id}", tags=["Benchmarks"])
def get_benchmark_by_id(benchmark_id: int):
    data = supabase.table("benchmarks").select("*").eq("id", benchmark_id).execute()
    return data.data[0] if data.data else {"error": "Benchmark not found."}


# -----------------------------
#  Benchmark Returns
# -----------------------------
@app.get("/benchmark-returns", tags=["Benchmark Returns"])
def get_benchmark_returns(
    benchmark_id: int = Query(..., description="Benchmark ID"),
    start_date: str = Query(None, description="Filter by month_end_date >= start_date"),
    end_date: str = Query(None, description="Filter by month_end_date <= end_date"),
):
    """Return monthly benchmark returns, optionally filtered by date range."""
    query = supabase.table("benchmark_returns").select("*").eq("benchmark_id", benchmark_id)
    if start_date:
        query = query.gte("month_end_date", start_date)
    if end_date:
        query = query.lte("month_end_date", end_date)
    result = query.execute()
    return result.data
