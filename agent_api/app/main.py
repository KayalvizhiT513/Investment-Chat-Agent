from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.responses import FileResponse
import os
from pathlib import Path
import json
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import difflib
from datetime import datetime, timedelta
import requests

# Load environment variables
load_dotenv()

# Configurations
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_API_URL = os.getenv("DATA_API_URL")
ANALYTICS_API_URL = os.getenv("ANALYTICS_API_URL")  # Assuming analytics_api runs on 8002

if not OPENAI_API_KEY:
    raise ValueError("Missing environment variables: OPENAI_API_KEY is required for the agent to run.")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        print(f"Warning: could not initialize Supabase client. Falling back to local sample data. {exc}")
else:
    print("Supabase credentials not configured. Using local sample data for portfolios/benchmarks.")

app = FastAPI(title="Conversational Analytics Agent", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []  # List of {"role": "user" or "assistant", "content": str}

class ChatResponse(BaseModel):
    response: str
    parameters: Optional[Dict] = None
    results: Optional[Dict] = None
    reset_history: bool = False

# Available metrics and required params
METRICS_REQUIREMENTS = {
    "volatility": ["portfolio_name", "start_date", "end_date"],
    "beta": ["portfolio_name", "benchmark_name", "start_date", "end_date"],
    "sharpe_ratio": ["portfolio_name", "risk_free_portfolio_name", "start_date", "end_date"],
    "tracking_error": ["portfolio_name", "benchmark_name", "start_date", "end_date"],
    "information_ratio": ["portfolio_name", "benchmark_name", "start_date", "end_date"]
}

FALLBACK_PORTFOLIOS = [
    "Growth Plus",
    "Global Dividend",
    "Secure Income",
    "Global Macro Opportunities"
]

FALLBACK_BENCHMARKS = [
    "Secure Income Benchmark",
    "MSCI World",
    "S&P 500"
]

def get_portfolios():
    if not supabase:
        return FALLBACK_PORTFOLIOS
    try:
        data = supabase.table("portfolios").select("portfolio_name").execute()
        names = [row["portfolio_name"] for row in data.data]
        return names or FALLBACK_PORTFOLIOS
    except Exception as exc:
        print(f"Warning: unable to fetch portfolios from Supabase. Using fallback list. {exc}")
        return FALLBACK_PORTFOLIOS

def get_benchmarks():
    if not supabase:
        return FALLBACK_BENCHMARKS
    try:
        data = supabase.table("benchmarks").select("benchmark_name").execute()
        names = [row["benchmark_name"] for row in data.data]
        return names or FALLBACK_BENCHMARKS
    except Exception as exc:
        print(f"Warning: unable to fetch benchmarks from Supabase. Using fallback list. {exc}")
        return FALLBACK_BENCHMARKS

def fuzzy_match(name: str, options: List[str], cutoff=0.6):
    matches = difflib.get_close_matches(name, options, n=5, cutoff=cutoff)
    return matches

def validate_date(date_str: str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Check if it's end of month
        next_month = dt.replace(day=28) + timedelta(days=4)  # Go to next month
        end_of_month = next_month - timedelta(days=next_month.day)
        if dt != end_of_month:
            return end_of_month.strftime("%Y-%m-%d")
        return date_str
    except ValueError:
        return None

def parse_with_llm(message: str, history: List[Dict]):
    prompt = f"""
You are an assistant that parses user requests for investment analytics calculations.
Extract the following parameters from the conversation:
- portfolio_name: Name of the portfolio
- benchmark_name: Name of the benchmark
- risk_free_portfolio_name: Name of the risk-free portfolio
- start_date: Start date in YYYY-MM-DD format (end of month preferred)
- end_date: End date in YYYY-MM-DD format (end of month preferred)
- metrics: List of metrics to compute from ["volatility", "beta", "sharpe_ratio", "tracking_error", "information_ratio"]

If a parameter is not mentioned, set it to null.
Return only a valid JSON object with these keys. No extra text.

Conversation history:
{chr(10).join([f"{msg['role']}: {msg['content']}" for msg in history])}

Current user message: {message}
"""
    system_prompt = "You are a helpful assistant that extracts structured parameters from user requests."
    OPENAI_MODEL = "gpt-5"
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        content = chat_completion.choices[0].message.content
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        print(f"Error parsing with LLM: {str(e)}")
        return {}

def check_completeness(params: Dict):
    metrics = params.get("metrics", [])
    print("Checking completeness for metrics:", metrics)
    missing = []
    for metric in metrics:
        print(f"Checking requirements for metric: {metric}")
        req = METRICS_REQUIREMENTS.get(metric, [])
        print(f"Metric '{metric}' requires parameters: {req}")
        for r in req:
            if not params.get(r):
                missing.append(r)
    missing = list(set(missing))
    if not metrics:
        missing.append("metrics")
    return missing

def generate_response(params: Dict, missing: List[str], portfolios: List[str], benchmarks: List[str]):
    if missing:
        # missing in pascal case
        missing_pcase = [m.replace("_", " ").title() for m in missing]
        response = "I need the following information to compute: " + ", ".join(missing_pcase) + ". "
        if "portfolio_name" in missing and params.get("portfolio_name"):
            suggestions = fuzzy_match(params["portfolio_name"], portfolios)
            if suggestions:
                response += f"Did you mean one of these portfolios: {', '.join(suggestions)}? "
            else:
                response += f"Available portfolios: {', '.join(portfolios)}. "
        if "benchmark_name" in missing and params.get("benchmark_name"):
            suggestions = fuzzy_match(params["benchmark_name"], benchmarks)
            if suggestions:
                response += f"Did you mean one of these benchmarks: {', '.join(suggestions)}? "
            else:
                response += f"Available benchmarks: {', '.join(benchmarks)}. "
        if "risk_free_portfolio_name" in missing and params.get("risk_free_portfolio_name"):
            suggestions = fuzzy_match(params["risk_free_portfolio_name"], portfolios)
            if suggestions:
                response += f"Did you mean one of these risk-free portfolios: {', '.join(suggestions)}? "
            else:
                response += f"Available portfolios: {', '.join(portfolios)}. "
        # Similar for risk_free
        return response, None, False
    else:
        # Compute
        try:
            # Validate dates
            if params.get("start_date"):
                params["start_date"] = validate_date(params["start_date"])
            if params.get("end_date"):
                params["end_date"] = validate_date(params["end_date"])

            # Call analytics API
            query_params = {
                "portfolio_name": params["portfolio_name"] if params.get("portfolio_name") else None,
                "benchmark_name": params.get("benchmark_name") if params.get("benchmark_name") else None,
                "risk_free_portfolio_name": params.get("risk_free_portfolio_name") if params.get("risk_free_portfolio_name") else None,
                "start_date": params.get("start_date") if params.get("start_date") else None,
                "end_date": params.get("end_date") if params.get("end_date") else None,
                "metrics": params["metrics"] if params.get("metrics") else None,
            }
            resp = requests.get(ANALYTICS_API_URL, params=query_params)
            if resp.status_code == 200:
                results = resp.json()
                response = f"Computed analytics for {params['portfolio_name']}: {results}"
                return response, results, True
            else:
                return f"Error computing analytics: {resp.text}", None, False
        except Exception as e:
            return f"Error: {str(e)}", None, False

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    portfolios = get_portfolios()
    benchmarks = get_benchmarks()

    # Parse parameters
    print("Message:", request.message, "History:", request.conversation_history)
    params = parse_with_llm(request.message, request.conversation_history)
    print("Parsed parameters:", params)

    # Check completeness
    missing = check_completeness(params)
    print("Missing parameters:", missing)

    # Check missing portfolios/benchmarks
    if params.get("portfolio_name"):
        if params["portfolio_name"] not in portfolios:
            suggestions = fuzzy_match(params["portfolio_name"], portfolios)
            missing.append("portfolio_name")
    if params.get("benchmark_name"):
        if params["benchmark_name"] not in benchmarks:
            suggestions = fuzzy_match(params["benchmark_name"], benchmarks)
            missing.append("benchmark_name")
    if params.get("risk_free_portfolio_name"):
        if params["risk_free_portfolio_name"] not in portfolios:
            suggestions = fuzzy_match(params["risk_free_portfolio_name"], portfolios)
            missing.append("risk_free_portfolio_name")

    # Generate response
    response, results, reset_history = generate_response(params, missing, portfolios, benchmarks)
    if reset_history:
        request.conversation_history = []
        request.message = None
        reset_history = False

    print("Final response:", response, "results:", results)
    if results:
        # Convert all key-value pairs into readable phrases
        formatted_metrics = []
        for key, value in results["results"].items():
            if isinstance(value, (int, float)):
                formatted_metrics.append(f"{key.replace('_', ' ').capitalize()} is {value:.6f}")
            else:
                formatted_metrics.append(f"{key.replace('_', ' ').capitalize()} is {value}")
        
        # Join multiple metrics if needed
        response_text = "; ".join(formatted_metrics)
    else:
        response_text = response

    return ChatResponse(
        response=response_text,
        parameters=params if missing else None,
        results=results,
        reset_history=reset_history
    )


@app.get("/")
def root():
    return FileResponse(Path(__file__).parent.parent / "index.html")
