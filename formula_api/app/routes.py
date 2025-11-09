from fastapi import APIRouter, HTTPException
from app.models import ComputeRequest, ComputeResponse
from app.services import formulas

router = APIRouter()

@router.get("/metrics")
def list_metrics():
    """
    List all available ex-post performance metrics.
    """
    return [
        {"name": "volatility", "description": "Standard deviation of realized portfolio returns"},
        {"name": "beta", "description": "Sensitivity of portfolio returns to benchmark returns"},
        {"name": "sharpe_ratio", "description": "Ex-post Sharpe ratio — realized excess return per unit of total risk"},
        {"name": "tracking_error", "description": "Ex-post tracking error — volatility of active returns vs benchmark"},
        {"name": "information_ratio", "description": "Ex-post information ratio — mean active return per unit of tracking error"},
    ]


@router.post("/compute", response_model=ComputeResponse)
def compute_metric(request: ComputeRequest):
    """
    Compute a selected ex-post performance metric using the given inputs.
    """
    try:
        fn = getattr(formulas, request.metric)
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Metric '{request.metric}' not implemented")

    try:
        result = fn(**request.inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"metric": request.metric, "value": result}
