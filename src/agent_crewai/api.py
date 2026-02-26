from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict
from datetime import datetime
import json
import re
import os

from agent_crewai.crew import CrewAiDev

app = FastAPI(title="Component Analysis API")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class ComponentRequest(BaseModel):
    topic: str = Field(..., description="High-level description of the component being manufactured")
    temperature: float = Field(..., description="Minimum operating temperature the component must withstand (°C)")
    tolerance: float = Field(..., description="Required machining tolerance (mm)")
    geometry_complexity: Optional[str] = Field(None, description="Geometry complexity level, e.g. 'simple', 'complex'")
    surface_finish: Optional[str] = Field(None, description="Required surface finish, e.g. 'mirror', 'standard'")
    special_features: Optional[List[str]] = Field(default_factory=list, description="Special machining features required")
    aerospace_required: bool = Field(True, description="Whether aerospace-grade material is required")

    @field_validator("tolerance", mode="before")
    @classmethod
    def parse_tolerance(cls, v):
        """Accept both float (0.01) and string ('±0.01mm') formats."""
        if isinstance(v, str):
            match = re.search(r"[\d.]+", v)
            if match:
                return float(match.group())
            raise ValueError(f"Cannot parse tolerance value from: {v}")
        return v


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    status: str
    request_summary: Dict[str, Any]
    material_analysis: Dict[str, Any]
    machine_analysis: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_agent_json(raw: str) -> Dict[str, Any]:
    """
    Safely parse JSON from agent output.
    Handles markdown fencing (```json ... ```) and stray whitespace.
    """
    if not raw:
        return {}
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return raw string under a key so the response is still valid JSON
        return {"raw_output": cleaned}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/analyse", response_model=AnalysisResponse)
def analyze_component(request: ComponentRequest):
    """
    Runs the two-agent CrewAI pipeline:
      - Agent 1 (material_expert): queries DB, reasons over candidates, selects best material
      - Agent 2 (machine_planner): queries DB with selected material, plans machining strategy

    Returns a structured JSON report covering material recommendation, machine recommendation,
    machining process plan, and cost/risk analysis.
    """
    crew_instance = CrewAiDev(
        host=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        token=os.getenv("DATABRICKS_TOKEN"),
        client_kwargs={"timeout": 600}
    )

    inputs = {
        "topic": request.topic,
        "temperature": request.temperature,
        "tolerance": request.tolerance,
        "geometry_complexity": request.geometry_complexity or "not specified",
        "surface_finish": request.surface_finish or "not specified",
        "special_features": request.special_features or [],
        "aerospace_required": request.aerospace_required,
        "current_year": str(datetime.now().year),
    }

    try:
        crew_result = crew_instance.crew().kickoff(inputs=inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {str(e)}")

    # crew_result.tasks_output is a list ordered by task execution
    tasks_output = getattr(crew_result, "tasks_output", [])
    print(f"Total tasks: {len(tasks_output)}")
    for i, task in enumerate(tasks_output):
        print(f"\n=== Task {i} ===")
        print(f"Type: {type(task)}")
        print(f"Raw: {repr(task.raw)}")

    material_raw = tasks_output[0].raw if len(tasks_output) > 0 else ""
    machine_raw  = tasks_output[1].raw if len(tasks_output) > 1 else crew_result.raw
    
    

    material_data = parse_agent_json(material_raw)
    machine_data  = parse_agent_json(machine_raw)

    return AnalysisResponse(
        status="success",
        request_summary={
            "topic":               request.topic,
            "temperature_c":       request.temperature,
            "tolerance_mm":        request.tolerance,
            "geometry_complexity": request.geometry_complexity,
            "surface_finish":      request.surface_finish,
            "special_features":    request.special_features,
            "aerospace_required":  request.aerospace_required,
        },
        material_analysis=material_data,
        machine_analysis=machine_data,
    )