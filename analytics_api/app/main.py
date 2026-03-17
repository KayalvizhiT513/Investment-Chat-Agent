from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
import os
from typing import List
from urllib.parse import urlparse
import numpy as np

load_dotenv()


def _read_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value.strip().strip('"').strip("'")
    return None


SUPABASE_URL = _read_env("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = _read_env("SUPABASE_KEY", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "Supabase configuration missing. Set SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) "
        "and SUPABASE_KEY (or SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY)."
    )

parsed = urlparse(SUPABASE_URL)
if not parsed.scheme or not parsed.netloc:
    raise ValueError(
        f"Invalid SUPABASE_URL: {SUPABASE_URL!r}. Expected format: https://<project-ref>.supabase.co"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Investment Performance Analytics API",
    description="API to compute performance analytics by fetching data from Supabase and applying formulas.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _assert_min_points(series: List[float], label: str):
    if len(series) < 2:
        raise HTTPException(status_code=400, detail=f"{label} needs at least 2 return points")


def _is_dns_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "name or service not known" in text
        or "nodename nor servname provided" in text
        or "temporary failure in name resolution" in text
    )


def _raise_supabase_error(exc: Exception):
    if _is_dns_error(exc):
        host = urlparse(SUPABASE_URL).netloc
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot resolve Supabase host '{host}'. "
                "Verify SUPABASE_URL in Render environment variables and ensure outbound DNS/network is available."
            ),
        )
    raise HTTPException(status_code=500, detail=f"Supabase query failed: {exc}")


def volatility(returns):
    _assert_min_points(returns, "Portfolio returns")
    return np.sqrt(np.sum((np.array(returns) - np.mean(returns)) ** 2) / (len(returns) - 1))


def beta(portfolio_returns, benchmark_returns):
    _assert_min_points(portfolio_returns, "Portfolio returns")
    _assert_min_points(benchmark_returns, "Benchmark returns")
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    denom = cov_matrix[1, 1]
    if denom == 0:
        raise HTTPException(status_code=400, detail="Benchmark variance is zero; beta is undefined")
    return cov_matrix[0, 1] / denom


def sharpe_ratio(returns, risk_free_return):
    _assert_min_points(returns, "Portfolio returns")
    excess_return = np.array(returns) - risk_free_return
    mean_excess_return = np.mean(excess_return)
    vol = volatility(returns)
    if vol == 0:
        raise HTTPException(status_code=400, detail="Volatility is zero; sharpe_ratio is undefined")
    return mean_excess_return / vol


def tracking_error(portfolio_returns, benchmark_returns):
    _assert_min_points(portfolio_returns, "Portfolio returns")
    _assert_min_points(benchmark_returns, "Benchmark returns")
    diff = np.array(portfolio_returns) - np.array(benchmark_returns)
    return np.sqrt(np.sum((diff - np.mean(diff)) ** 2) / (len(diff) - 1))


def information_ratio(portfolio_returns, benchmark_returns):
    active_return = np.array(portfolio_returns) - np.array(benchmark_returns)
    mean_active_return = np.mean(active_return)
    te = tracking_error(portfolio_returns, benchmark_returns)
    if te == 0:
        raise HTTPException(status_code=400, detail="Tracking error is zero; information_ratio is undefined")
    return mean_active_return / te


def get_portfolio_returns(portfolio_name: str, start_date: str = None, end_date: str = None) -> List[float]:
    portfolio = supabase.table("portfolios").select("id").eq("portfolio_name", portfolio_name).execute()
    if not portfolio.data:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_name}' not found")
    portfolio_id = portfolio.data[0]["id"]

    query = (
        supabase.table("portfolio_returns")
        .select("month_end_date, portfolio_return")
        .eq("portfolio_id", portfolio_id)
        .order("month_end_date")
    )
    if start_date:
        query = query.gte("month_end_date", start_date)
    if end_date:
        query = query.lte("month_end_date", end_date)
    data = query.execute()
    return [row["portfolio_return"] for row in data.data]


def get_benchmark_returns(benchmark_name: str, start_date: str = None, end_date: str = None) -> List[float]:
    benchmark = supabase.table("benchmarks").select("id").eq("benchmark_name", benchmark_name).execute()
    if not benchmark.data:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_name}' not found")
    benchmark_id = benchmark.data[0]["id"]

    query = (
        supabase.table("benchmark_returns")
        .select("month_end_date, benchmark_return")
        .eq("benchmark_id", benchmark_id)
        .order("month_end_date")
    )
    if start_date:
        query = query.gte("month_end_date", start_date)
    if end_date:
        query = query.lte("month_end_date", end_date)
    data = query.execute()
    return [row["benchmark_return"] for row in data.data]


def get_risk_free_rate(risk_free_portfolio_name: str, start_date: str = None, end_date: str = None) -> float:
    returns = get_portfolio_returns(risk_free_portfolio_name, start_date, end_date)
    return float(np.mean(returns)) if returns else 0.0


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
    metrics: List[str] = Query(
        ["volatility", "beta", "sharpe_ratio", "tracking_error", "information_ratio"],
        description="List of metrics to compute",
    ),
):
    if not portfolio_name:
        return {"message": "✅ Analytics API is live."}

    try:
        portfolio_returns = get_portfolio_returns(portfolio_name, start_date, end_date)
        benchmark_returns = get_benchmark_returns(benchmark_name, start_date, end_date) if benchmark_name else []
        risk_free_rate = get_risk_free_rate(risk_free_portfolio_name, start_date, end_date) if risk_free_portfolio_name else 0.0

        results = {}
        for metric in metrics:
            if metric == "volatility":
                results[metric] = volatility(portfolio_returns)
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
    except HTTPException:
        raise
    except Exception as exc:
        _raise_supabase_error(exc)
