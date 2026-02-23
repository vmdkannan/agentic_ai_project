from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
from agent_crewai.crew import CrewAiDev

app = FastAPI()


class ComponentRequest(BaseModel):
    temperature: float
    tolerance: str
    topic: str


@app.post("/analyse")
def analyze_component(request: ComponentRequest):

    crew_instance = CrewAiDev(
        host=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        token=os.getenv("DATABRICKS_TOKEN"),
    )

    # Query materials safely
    materials = crew_instance.query_materials(
        temperature=request.temperature,
        aerospace_required=True
    )
    
    materials = crew_instance.sanitize_for_crewai(materials)

    inputs = {
        "crewai_trigger_payload": request.dict(),
        "topic": request.topic,
        "candidate_materials": materials,
        "current_year": str(datetime.now().year)
    }

    result = crew_instance.crew().kickoff(inputs=inputs)

    return {"result": result.raw}