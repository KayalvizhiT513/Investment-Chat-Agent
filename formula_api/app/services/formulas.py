import numpy as np
import pandas as pd

# --- Volatility (standard deviation of returns) ---
def volatility(returns):
    return np.sqrt(np.sum((np.array(returns) - np.mean(returns)) ** 2) / (len(returns) - 1))

# --- Beta relative to a benchmark ---
def beta(portfolio_returns, benchmark_returns):
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    return cov_matrix[0, 1] / cov_matrix[1, 1]

# --- Sharpe ratio (ex-post) ---
def sharpe_ratio(returns, risk_free_return):
    excess_return = np.array(returns) - risk_free_return
    mean_excess_return = np.mean(excess_return)      # ✅ use np.mean instead of np.avg
    vol = volatility(returns)
    return mean_excess_return / vol

# --- Tracking error (ex-post) ---
def tracking_error(portfolio_returns, benchmark_returns):
    diff = np.array(portfolio_returns) - np.array(benchmark_returns)
    return np.sqrt(np.sum((diff - np.mean(diff)) ** 2) / (len(diff) - 1))

# --- Information ratio (ex-post) ---
def information_ratio(portfolio_returns, benchmark_returns):
    # active return = portfolio - benchmark
    active_return = np.array(portfolio_returns) - np.array(benchmark_returns)
    mean_active_return = np.mean(active_return)
    te = tracking_error(portfolio_returns, benchmark_returns)
    return mean_active_return / te

    
