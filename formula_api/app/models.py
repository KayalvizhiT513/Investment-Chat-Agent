from pydantic import BaseModel
from typing import Dict, Any

class ComputeRequest(BaseModel):
    metric: str
    inputs: Dict[str, Any]

class ComputeResponse(BaseModel):
    metric: str
    value: float
