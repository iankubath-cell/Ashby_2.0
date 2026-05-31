# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

# Import your existing modules
# Assuming they are in the same folder or ashby3/
from validator import ViraValidator 
from monitor import HomeostaticMonitor

app = FastAPI(title="Ashby-Vira API")

# Define the data structure Base44 will send
class InterventionRequest(BaseModel):
    metrics: dict  # e.g., {"cpu": 0.95, "memory": 0.4}
    intervention: str  # e.g., "SCALE_UP_REPLICAS"

class ValidationResult(BaseModel):
    status: str  # "SAFE", "FROZEN", "INCONCLUSIVE"
    score: float
    reason: str
    checks_passed: int
    checks_total: int

# Initialize your engines (singleton pattern)
validator = ViraValidator()
monitor = HomeostaticMonitor()

@app.get("/health")
def health_check():
    return {"status": "ok", "engine": "Ashby 2.0"}

@app.post("/validate", response_model=ValidationResult)
def validate_intervention(request: InterventionRequest):
    """
    Base44 sends metrics + proposed action here.
    Ashby 2.0 runs the 6-check deterministic validator.
    """
    try:
        # 1. Feed metrics to the monitor to update state
        monitor.update_metrics(request.metrics)
        
        # 2. Run the validator
        # (Assuming your validator takes metrics and action)
        result = validator.validate(
            intervention=request.intervention,
            current_state=monitor.get_state()
        )
        
        # Map your internal result to the API response
        return ValidationResult(
            status=result.status, # "FROZEN", "INCONCLUSIVE", etc.
            score=monitor.get_stability_score(),
            reason=result.reason,
            checks_passed=result.passed_checks,
            checks_total=6
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_system_status():
    """Returns current stability score and state"""
    return {
        "stability_score": monitor.get_stability_score(),
        "state": monitor.get_state(),
        "recent_decisions": monitor.get_history()
    }

if __name__ == "__main__":
    import uvicorn
    # Run on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
