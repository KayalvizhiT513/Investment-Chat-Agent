from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Investment Performance Analytics API",
    description="API to compute performance analytics by fetching data from Supabase and applying formulas.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------

# Formulas (copied from formula_api for integration)

def volatility(returns):
    return np.sqrt(np.sum((np.array(returns) - np.mean(returns)) ** 2) / (len(returns) - 1))

def beta(portfolio_returns, benchmark_returns):
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    return cov_matrix[0, 1] / cov_matrix[1, 1]

def sharpe_ratio(returns, risk_free_return):
    excess_return = np.array(returns) - risk_free_return
    mean_excess_return = np.mean(excess_return)
    vol = volatility(returns)
    return mean_excess_return / vol

def tracking_error(portfolio_returns, benchmark_returns):
    diff = np.array(portfolio_returns) - np.array(benchmark_returns)
    return np.sqrt(np.sum((diff - np.mean(diff)) ** 2) / (len(diff) - 1))

def information_ratio(portfolio_returns, benchmark_returns):
    active_return = np.array(portfolio_returns) - np.array(benchmark_returns)
    mean_active_return = np.mean(active_return)
    te = tracking_error(portfolio_returns, benchmark_returns)
    return mean_active_return / te

# -----------------------------

# Helper functions to fetch data

def get_portfolio_returns(portfolio_name: str, start_date: str = None, end_date: str = None) -> List[float]:
    # Get portfolio id by name
    portfolio = supabase.table("portfolios").select("id").eq("portfolio_name", portfolio_name).execute()
    if not portfolio.data:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_name}' not found")
    portfolio_id = portfolio.data[0]["id"]

    query = supabase.table("portfolio_returns").select("month_end_date, portfolio_return").eq("portfolio_id", portfolio_id).order("month_end_date")
    if start_date:
        query = query.gte("month_end_date", start_date)
    if end_date:
        query = query.lte("month_end_date", end_date)
    data = query.execute()
    returns = [row["portfolio_return"] for row in data.data]
    return returns

def get_benchmark_returns(benchmark_name: str, start_date: str = None, end_date: str = None) -> List[float]:
    benchmark = supabase.table("benchmarks").select("id").eq("benchmark_name", benchmark_name).execute()
    if not benchmark.data:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_name}' not found")
    benchmark_id = benchmark.data[0]["id"]

    query = supabase.table("benchmark_returns").select("month_end_date, benchmark_return").eq("benchmark_id", benchmark_id).order("month_end_date")
    if start_date:
        query = query.gte("month_end_date", start_date)
    if end_date:
        query = query.lte("month_end_date", end_date)
    data = query.execute()
    returns = [row["benchmark_return"] for row in data.data]
    return returns

def get_risk_free_rate(risk_free_portfolio_name: str, start_date: str = None, end_date: str = None) -> float:
    # Assuming risk_free is a portfolio with returns as rates
    returns = get_portfolio_returns(risk_free_portfolio_name, start_date, end_date)
    # Perhaps average or last, but for simplicity, assume constant or mean
    return np.mean(returns) if returns else 0.0

# -----------------------------

# Routes

@app.get("/")
def root():
    return {"message": "✅ Analytics API is live."}

@app.get("/analytics")
def compute_analytics(
    portfolio_name: str = Query(None, description="Name of the portfolio"),
    benchmark_name: str = Query(None, description="Name of the benchmark"),
    risk_free_portfolio_name: str = Query(None, description="Name of the risk-free portfolio"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    metrics: List[str] = Query(["volatility", "beta", "sharpe_ratio", "tracking_error", "information_ratio"], description="List of metrics to compute")
):
    try:
        # Fetch data
        portfolio_returns = get_portfolio_returns(portfolio_name, start_date, end_date)
        benchmark_returns = get_benchmark_returns(benchmark_name, start_date, end_date) if benchmark_name else []
        risk_free_rate = get_risk_free_rate(risk_free_portfolio_name, start_date, end_date) if risk_free_portfolio_name else 0.0

        results = {}
        for metric in metrics:
            print(f"Computing metric: {metric}")
            if metric == "volatility":
                results[metric] = volatility(portfolio_returns)
                print("Computed volatility:", results[metric])
            elif metric == "beta":
                results[metric] = beta(portfolio_returns, benchmark_returns)
            elif metric == "sharpe_ratio":
                results[metric] = sharpe_ratio(portfolio_returns, risk_free_rate)
            elif metric == "tracking_error":
                results[metric] = tracking_error(portfolio_returns, benchmark_returns)
            elif metric == "information_ratio":
                results[metric] = information_ratio(portfolio_returns, benchmark_returns)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")

        return {"portfolio": portfolio_name, "benchmark": benchmark_name, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
