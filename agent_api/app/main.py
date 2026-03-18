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
import re

# Load environment variables
load_dotenv(override=True)

# Configurations
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_API_URL = os.getenv("DATA_API_URL", "http://localhost:8001")
ANALYTICS_API_URL = os.getenv("ANALYTICS_API_URL", "http://localhost:8002/analytics")

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
    options: Optional[Dict] = None  # clickable options e.g. {"metrics": [...], "portfolios": [...]}
    missing: Optional[List[str]] = None  # fields still missing, so frontend can grey them out

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
    OPENAI_MODEL = "gpt-3.5-turbo"
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log the request payload
        print("LLM Request Payload:", messages)

        chat_completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )

        # Log the raw response
        print("LLM Raw Response:", chat_completion)

        content = chat_completion.choices[0].message.content
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        print(f"Error parsing with LLM: {str(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return {}



def _normalize_metric_list(metrics):
    if isinstance(metrics, list):
        return metrics
    if isinstance(metrics, str) and metrics.strip():
        return [metrics.strip()]
    return []


def extract_params_from_history(history: List[Dict[str, str]]) -> Dict:
    """Extract last known structured params from prior assistant/user messages."""
    extracted: Dict = {}
    date_pattern = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})")

    for msg in history or []:
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        # Parse 'key: value' lines from previous 'Extracted Parameters' blocks
        for line in content.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"portfolio_name", "benchmark_name", "risk_free_portfolio_name", "start_date", "end_date"} and value:
                extracted[key] = value
            elif key == "metrics" and value:
                extracted[key] = _normalize_metric_list(value)

        # Parse inline date range like '31-1-2023 to 31-5-2024'
        matches = date_pattern.findall(content)
        if len(matches) >= 2:
            d1, d2 = matches[0], matches[1]
            extracted["start_date"] = f"{int(d1[2]):04d}-{int(d1[1]):02d}-{int(d1[0]):02d}"
            extracted["end_date"] = f"{int(d2[2]):04d}-{int(d2[1]):02d}-{int(d2[0]):02d}"

    if "metrics" in extracted:
        extracted["metrics"] = _normalize_metric_list(extracted["metrics"])
    return extracted


def merge_params(current: Dict, previous: Dict, message: str, portfolios: List[str], benchmarks: List[str]) -> Dict:
    merged = dict(previous or {})

    for key, value in (current or {}).items():
        if value is not None and value != "":
            merged[key] = value

    merged["metrics"] = _normalize_metric_list(merged.get("metrics"))

    msg = (message or "").strip()
    lower = msg.lower()

    # Handle contextual chip messages: "X as benchmark" / "X as risk free portfolio"
    rf_match = re.search(r"^(.+?)\s+as\s+(?:the\s+)?risk.free", lower)
    bench_match = re.search(r"^(.+?)\s+as\s+(?:the\s+)?benchmark", lower)
    if rf_match:
        name = rf_match.group(1).strip()
        for p in portfolios:
            if p.lower() == name:
                merged["risk_free_portfolio_name"] = p
                return merged  # field resolved, no further overrides needed
    elif bench_match:
        name = bench_match.group(1).strip()
        for b in benchmarks:
            if b.lower() == name:
                merged["benchmark_name"] = b
                return merged

    # If user replies with just a portfolio/benchmark name, keep prior context and update only that field.
    for p in portfolios:
        if lower == p.lower():
            merged["portfolio_name"] = p
            break
    for b in benchmarks:
        if lower == b.lower():
            merged["benchmark_name"] = b
            break

    # Parse dates from short follow-up message format dd-mm-yyyy to dd-mm-yyyy
    range_match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})\s*(?:to|-)\s*(\d{1,2})[-/](\d{1,2})[-/](\d{4})", lower)
    if range_match:
        d1, m1, y1, d2, m2, y2 = range_match.groups()
        merged["start_date"] = f"{int(y1):04d}-{int(m1):02d}-{int(d1):02d}"
        merged["end_date"] = f"{int(y2):04d}-{int(m2):02d}-{int(d2):02d}"

    return merged

def check_completeness(params: Dict):
    metrics = params.get("metrics") or []
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

def _analytics_url_candidates(base_url: str) -> List[str]:
    """Return likely analytics URLs, covering both base and /analytics variants."""
    if not base_url:
        return []

    cleaned = base_url.rstrip("/")
    candidates = [base_url]

    if cleaned.endswith("/analytics"):
        base_candidate = cleaned[: -len("/analytics")] or "/"
        candidates.append(base_candidate)
    else:
        candidates.append(f"{cleaned}/analytics")

    deduped = []
    for url in candidates:
        if url and url not in deduped:
            deduped.append(url)
    return deduped


def generate_response(params: Dict, missing: List[str], portfolios: List[str], benchmarks: List[str]):
    if missing:
        metrics = params.get("metrics") or []
        metric_display = " and ".join(m.replace("_", " ") for m in metrics) if metrics else None
        options: Dict = {}
        parts = []

        if "metrics" in missing:
            parts.append("Sure! What would you like to calculate? Pick a metric to get started:")
            options["metrics"] = list(METRICS_REQUIREMENTS.keys())
        elif metric_display:
            parts.append(f"Got it! To calculate {metric_display}, I just need a couple more details.")
        else:
            parts.append("Almost there — just a couple more details needed.")

        for m in missing:
            if m == "portfolio_name":
                invalid = params.get("portfolio_name")
                if invalid:
                    suggestions = fuzzy_match(invalid, portfolios)
                    if suggestions:
                        parts.append(f"I couldn't find \"{invalid}\" — did you mean one of these?")
                        options["portfolios"] = suggestions
                    else:
                        parts.append(f"I couldn't find \"{invalid}\". Which portfolio would you like to analyze?")
                        options["portfolios"] = portfolios
                    params["portfolio_name"] = None  # clear invalid value
                else:
                    parts.append("Which portfolio would you like to analyze?")
                    options["portfolios"] = portfolios
            elif m == "benchmark_name":
                invalid = params.get("benchmark_name")
                if invalid:
                    suggestions = fuzzy_match(invalid, benchmarks)
                    if suggestions:
                        parts.append(f"I couldn't find benchmark \"{invalid}\" — did you mean one of these?")
                        options["benchmarks"] = suggestions
                    else:
                        parts.append(f"I couldn't find benchmark \"{invalid}\". Which benchmark should I compare against?")
                        options["benchmarks"] = benchmarks
                    params["benchmark_name"] = None
                else:
                    parts.append("Which benchmark should I compare against?")
                    options["benchmarks"] = benchmarks
            elif m == "risk_free_portfolio_name":
                # Should not happen — risk-free is auto-set; handle gracefully just in case
                params["risk_free_portfolio_name"] = "US Dollar Risk Free Rate"
            elif m == "start_date":
                parts.append("What start date should I use? (e.g. 2023-01-31)")
            elif m == "end_date":
                parts.append("What end date should I use? (e.g. 2023-12-31)")

        response = " ".join(parts)
        return response, None, False, options, missing
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

            last_error = ""
            for analytics_url in _analytics_url_candidates(ANALYTICS_API_URL):
                resp = requests.get(analytics_url, params=query_params, timeout=20)
                if resp.status_code == 200:
                    results = resp.json()
                    if results and "results" in results:
                        formatted_metrics = []
                        for key, value in results["results"].items():
                            if isinstance(value, (int, float)):
                                formatted_metrics.append(f"{key.replace('_', ' ').capitalize()} is {value:.6f}")
                            else:
                                formatted_metrics.append(f"{key.replace('_', ' ').capitalize()} is {value}")

                        response_text = f"Here are the analytics for {results.get('portfolio', 'the portfolio')}: " + "; ".join(formatted_metrics) + "."
                        return response_text, results, True, None, []

                    if isinstance(results, dict) and "message" in results:
                        last_error = f"Analytics endpoint {analytics_url} returned a health response instead of computed metrics."
                    else:
                        last_error = f"Analytics endpoint {analytics_url} returned no computed metrics."
                    continue

                if resp.status_code == 429:
                    return "The analytics service is currently rate-limited. Please try again in a moment.", None, False, None, []

                last_error = f"Analytics endpoint {analytics_url} returned HTTP {resp.status_code}: {resp.text}"

            return f"Hmm, I ran into a problem fetching the results: {last_error}", None, False, None, []
        except Exception as e:
            return f"Something went wrong while computing analytics: {str(e)}", None, False, None, []

CAPABILITIES_TEXT = (
    "I can help you calculate investment analytics for your portfolios. Here's what I support:\n"
    "- **Volatility** — measures portfolio risk over a date range\n"
    "- **Beta** — compares portfolio sensitivity to a benchmark\n"
    "- **Sharpe Ratio** — risk-adjusted return vs a risk-free portfolio\n"
    "- **Tracking Error** — deviation from a benchmark\n"
    "- **Information Ratio** — active return per unit of active risk\n\n"
    "Just tell me which metric you'd like, the portfolio name, and the date range — and I'll take care of the rest!"
)

GREETING_KEYWORDS = {"hi", "hello", "hey", "hiya", "howdy", "greetings", "sup", "good morning", "good afternoon", "good evening"}

def is_greeting_or_general(message: str, params: Dict) -> bool:
    """Return True if the message is a casual greeting or general question with no analytics intent."""
    msg_lower = message.lower().strip().rstrip("!.,?")
    # Explicit greetings
    if msg_lower in GREETING_KEYWORDS:
        return True
    # Help / capability queries
    if any(kw in msg_lower for kw in ("what can you do", "what do you do", "how can you help", "help me", "capabilities", "features")):
        return True
    # No analytics params extracted at all
    has_any_param = any(v for v in params.values() if v is not None and v != "" and v != [])
    if not has_any_param:
        words = msg_lower.split()
        if len(words) <= 4:  # very short message with no params → likely casual
            return True
    return False

def generate_conversational_response(message: str, history: List[Dict]) -> str:
    """Use LLM to generate a friendly, context-aware reply for non-analytics messages."""
    system_prompt = (
        "You are a friendly investment analytics assistant. You help users calculate portfolio metrics "
        "like volatility, beta, sharpe ratio, tracking error, and information ratio. "
        "When users greet you or ask general questions, respond warmly and briefly. "
        "Always invite them to ask about portfolio analytics. Keep replies concise — 2-3 sentences max."
    )
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or []):
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})
        completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return completion.choices[0].message.content.strip()
    except Exception:
        return "Hey there! I'm your investment analytics assistant. Ask me to calculate metrics like volatility, beta, or Sharpe ratio for any portfolio!"

def interpret_casual_dates(message: str):
    """Interpret casual date references like 'past two months' or 'past two years' into structured dates."""
    today = datetime.today()
    if "past two months" in message.lower():
        start_date = (today.replace(day=1) - timedelta(days=1)).replace(day=1)  # First day of two months ago
        end_date = today.replace(day=1) - timedelta(days=1)  # Last day of last month
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "past two years" in message.lower():
        start_date = today.replace(year=today.year - 2, month=1, day=1)  # First day of two years ago
        end_date = today.replace(year=today.year - 1, month=12, day=31)  # Last day of last year
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    return None, None

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Log the incoming request
        print("Incoming request:", request.dict())

        portfolios = get_portfolios()
        benchmarks = get_benchmarks()

        # Interpret casual date references
        start_date, end_date = interpret_casual_dates(request.message)

        # Parse parameters
        print("Message:", request.message, "History:", request.conversation_history)
        params = parse_with_llm(request.message, request.conversation_history)
        history_params = extract_params_from_history(request.conversation_history)
        params = merge_params(params, history_params, request.message, portfolios, benchmarks)
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        print("Parsed parameters:", params)

        # Auto-set the risk-free portfolio — there is only one option
        RISK_FREE_PORTFOLIO = "US Dollar Risk Free Rate"
        if "sharpe_ratio" in (params.get("metrics") or []) and not params.get("risk_free_portfolio_name"):
            params["risk_free_portfolio_name"] = RISK_FREE_PORTFOLIO

        # Check completeness
        missing = check_completeness(params)
        print("Missing parameters:", missing)

        # Check missing portfolios/benchmarks (case-insensitive lookup)
        portfolios_lower = {p.lower(): p for p in portfolios}
        benchmarks_lower = {b.lower(): b for b in benchmarks}

        if params.get("portfolio_name"):
            canonical = portfolios_lower.get(params["portfolio_name"].lower())
            if canonical:
                params["portfolio_name"] = canonical
            else:
                missing.append("portfolio_name")
        if params.get("benchmark_name"):
            canonical = benchmarks_lower.get(params["benchmark_name"].lower())
            if canonical:
                params["benchmark_name"] = canonical
            else:
                missing.append("benchmark_name")
        if params.get("risk_free_portfolio_name"):
            canonical = portfolios_lower.get(params["risk_free_portfolio_name"].lower())
            if canonical:
                params["risk_free_portfolio_name"] = canonical
            else:
                missing.append("risk_free_portfolio_name")

        # Handle "list" command
        if "list" in request.message.lower():
            response = (
                f"Here's what I have available:\n"
                f"- **Portfolios:** {', '.join(portfolios)}\n"
                f"- **Benchmarks:** {', '.join(benchmarks)}\n"
                f"- **Metrics:** {', '.join(METRICS_REQUIREMENTS.keys())}\n"
                f"- **Example date range:** {datetime.today().replace(day=1).strftime('%Y-%m-%d')} to {(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')}"
            )
            return ChatResponse(response=response, parameters=None, results=None, reset_history=False,
                                options={"metrics": list(METRICS_REQUIREMENTS.keys()),
                                         "portfolios": portfolios, "benchmarks": benchmarks})

        # Handle greetings and general conversation
        if is_greeting_or_general(request.message, params):
            response = generate_conversational_response(request.message, request.conversation_history)
            return ChatResponse(response=response, parameters=None, results=None, reset_history=False)

        # Generate response
        response, results, reset_history, options, missing_out = generate_response(params, missing, portfolios, benchmarks)
        print("Final response:", response, "results:", results)
        if reset_history:
            request.conversation_history = []
            request.message = None
            reset_history = False

        return ChatResponse(
            response=response,
            parameters=params,
            results=results,
            reset_history=reset_history,
            options=options,
            missing=missing_out
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Error in /chat:", tb)
        return ChatResponse(
            response="Oops, something went wrong on my end. Please try again!",
            parameters=None,
            results=None,
            reset_history=False
        )


@app.get("/")
def root():
    return FileResponse(Path(__file__).parent.parent / "index.html")

@app.get("/data")
def data_viewer():
    return FileResponse(Path(__file__).parent.parent / "data_viewer.html")

# ── Data API proxy routes (avoids CORS issues in the browser) ──
@app.get("/api/portfolios")
def proxy_portfolios():
    resp = requests.get(f"{DATA_API_URL}/portfolios", timeout=10)
    return resp.json()

@app.get("/api/benchmarks")
def proxy_benchmarks():
    resp = requests.get(f"{DATA_API_URL}/benchmarks", timeout=10)
    return resp.json()

@app.get("/api/portfolio-returns")
def proxy_portfolio_returns(portfolio_id: int, start_date: str = None, end_date: str = None):
    params = {"portfolio_id": portfolio_id}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    resp = requests.get(f"{DATA_API_URL}/portfolio-returns", params=params, timeout=10)
    return resp.json()

@app.get("/api/benchmark-returns")
def proxy_benchmark_returns(benchmark_id: int, start_date: str = None, end_date: str = None):
    params = {"benchmark_id": benchmark_id}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    resp = requests.get(f"{DATA_API_URL}/benchmark-returns", params=params, timeout=10)
    return resp.json()
