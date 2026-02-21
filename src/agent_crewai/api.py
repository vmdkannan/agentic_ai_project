from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from agent_crewai.crew import CrewAiDev

app = FastAPI()

class ComponentRequest(BaseModel):
    component_name: str
    dimensions: str
    tolerances: str
    material_requirements: str
    norms: str
    topic: str

@app.post("/analyze")
def analyze_component(request: ComponentRequest):
    inputs = {
        "crewai_trigger_payload": request.dict(),
        "topic": request.topic,
        "current_year": str(datetime.now().year)
    }

    result = CrewAiDev().crew().kickoff(inputs=inputs)
    
    # Use .raw to get the string output from the crew
    return {"result": result.raw}