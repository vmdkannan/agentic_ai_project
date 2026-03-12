from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
from databricks import sql
from typing import Optional, List
from decimal import Decimal
from pydantic import BaseModel, Field
import os


# ---------------------------------------------------------------------------
# Pydantic schemas for tool inputs (required by CrewAI tool validation)
# ---------------------------------------------------------------------------

class MaterialQueryInput(BaseModel):
    temperature: float = Field(
        ...,
        description="Minimum operating temperature in Celsius the material must withstand"
    )
    aerospace_required: bool = Field(
        ...,
        description="Whether the material must be aerospace-grade"
    )


class MachineQueryInput(BaseModel):
    material_type: str = Field(
        ...,
        description=(
            "The material CATEGORY used to match machines in the database. "
            "Use the material_type field from the materials query result "
            "(e.g. 'Superalloy', 'Titanium', 'Steel', 'Aluminium'). "
            "Do NOT pass the specific alloy name (e.g. not 'Waspaloy', not 'Inconel 718')."
        )
    )
    required_tolerance: float = Field(
        ...,
        description="Required machining tolerance in mm"
    )
    required_geometry: Optional[str] = Field(
        None,
        description="Required geometry complexity, e.g. 'complex', 'prismatic', 'freeform'"
    )
    required_surface_finish: Optional[str] = Field(
        None,
        description="Required surface finish capability, e.g. 'mirror', 'high', 'standard'"
    )
    required_features: Optional[List[str]] = Field(
        None,
        description="List of special features the machine must support, e.g. ['5-axis', 'deep_bore']"
    )


# ---------------------------------------------------------------------------
# Tools (standalone, injected with DB credentials at runtime)
# ---------------------------------------------------------------------------

class MaterialQueryTool(BaseTool):
    name: str = "query_materials"
    description: str = (
        "Query the Databricks materials database. Returns up to 5 candidate materials "
        "that meet the minimum operating temperature and aerospace-grade requirements, "
        "ordered by highest temperature capability then lowest cost. "
        "Each result includes material_type (e.g. 'Superalloy') which must be passed "
        "to query_machines — not the material_name. "
        "Use this to retrieve candidate materials before making a recommendation."
    )
    args_schema: type[BaseModel] = MaterialQueryInput

    host: str
    http_path: str
    token: str

    def _sanitize(self, data):
        if isinstance(data, list):
            return [self._sanitize(i) for i in data]
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        if isinstance(data, Decimal):
            return float(data)
        return data

    def _run(self, temperature: float, aerospace_required: bool) -> str:
        query = """
        SELECT
            material_name,
            material_type,
            grade,
            machinability_rating,
            tensile_strength_mpa,
            yield_strength_mpa,
            max_operating_temp_c,
            aerospace_grade,
            cost_per_kg
        FROM dbt_industry_dev.source.materials
        WHERE max_operating_temp_c >= :temperature
          AND aerospace_grade = :aerospace_required
        ORDER BY max_operating_temp_c DESC, cost_per_kg ASC
        LIMIT 5
        """
        with sql.connect(
            server_hostname=self.host,
            http_path=self.http_path,
            access_token=self.token,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    {"temperature": float(temperature), "aerospace_required": bool(aerospace_required)},
                )
                rows = cursor.fetchall()

        if not rows:
            return "No materials found matching the given criteria."

        columns = [
            "material_name", "material_type", "grade", "machinability_rating",
            "tensile_strength_mpa", "yield_strength_mpa", "max_operating_temp_c",
            "aerospace_grade", "cost_per_kg",
        ]
        results = self._sanitize([dict(zip(columns, row)) for row in rows])
        return str(results)


class MachineQueryTool(BaseTool):
    name: str = "query_machines"
    description: str = (
        "Query the Databricks machines table. Returns available machines compatible "
        "with the given material_type category and tolerance. "
        "Pass material_type (e.g. 'Superalloy') — NOT the specific alloy name (e.g. not 'Waspaloy'). "
        "Optionally filters by geometry complexity, surface finish capability, and special features. "
        "Use this after the material has been selected to find suitable machines."
    )
    args_schema: type[BaseModel] = MachineQueryInput

    host: str
    http_path: str
    token: str

    def _sanitize(self, data):
        if isinstance(data, list):
            return [self._sanitize(i) for i in data]
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        if isinstance(data, Decimal):
            return float(data)
        return data

    def _run(
        self,
        material_type: str,
        required_tolerance: float,
        required_geometry: Optional[str] = None,
        required_surface_finish: Optional[str] = None,
        required_features: Optional[List[str]] = None,
    ) -> str:

        GEOMETRY_MAP = {
            "simple":   ["prismatic"],
            "moderate": ["prismatic", "freeform"],
            "complex":  ["freeform", "deep_bore", "complex"],
        }

        FINISH_RANK = {"standard": 1, "high": 2, "very high": 3, "mirror": 4}

        # ── 1. Fetch from DB (material + tolerance only) ──────────────────────
        query = """
            SELECT machine_name, machine_type, supported_material_type,
                max_tolerance_mm, geometry_capability,
                surface_finish_capability, special_features, cost_per_hour
            FROM dbt_industry_dev.source.machines
            WHERE array_contains(
                    transform(split(supported_material_type, ','), x -> trim(x)),
                    :material_type
                )
            AND max_tolerance_mm <= :required_tolerance
            AND status = 'available'
            ORDER BY cost_per_hour ASC
            LIMIT 20
        """
        with sql.connect(
            server_hostname=self.host,
            http_path=self.http_path,
            access_token=self.token,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"material_type": material_type,
                                    "required_tolerance": float(required_tolerance)})
                cols = [c[0] for c in cur.description]
                machines = [dict(zip(cols, row)) for row in cur.fetchall()]

        # ── 2. Helper: normalise DB array columns (list or comma-string → list) ─
        def to_list(val) -> list:
            if isinstance(val, list): return val
            if isinstance(val, str):  return [v.strip() for v in val.split(",")]
            return []

        # ── 3. Optional in-memory filters ─────────────────────────────────────
        if required_geometry:
            caps = GEOMETRY_MAP.get(required_geometry.lower(), [required_geometry])
            machines = [m for m in machines
                        if any(c in to_list(m["geometry_capability"]) for c in caps)]

        if required_surface_finish:
            min_rank = FINISH_RANK.get(required_surface_finish.lower(), 1)
            machines = [m for m in machines
                        if FINISH_RANK.get(m["surface_finish_capability"].lower(), 0) >= min_rank]

        if required_features:
            machines = [m for m in machines
                        if all(f in to_list(m["special_features"]) for f in required_features)]

        # ── 4. Return ──────────────────────────────────────────────────────────
        if not machines:
            return (f"No machines found for material='{material_type}', "
                    f"tolerance={required_tolerance}mm, geometry='{required_geometry}', "
                    f"finish='{required_surface_finish}', features={required_features}.")

        return str(self._sanitize(machines))


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

@CrewBase
class CrewAiDev:
    """
    Two-agent sequential crew:
      1. material_expert  — queries DB, reasons over candidates, picks best material
      2. machine_planner  — queries DB with chosen material TYPE, plans machining strategy
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        host: Optional[str] = None,
        http_path: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
        self.token = token or os.getenv("DATABRICKS_TOKEN")

        self.llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.3)
  
        # self.llm = LLM(
        #     model="deepseek/deepseek-chat",
        #     temperature=0.3,
        #     api_key=os.getenv("DEEPSEEK_API_KEY")
        # )
        
        # self.llm = LLM(
        #     model="llama-3.1-8b-instant",
        #     temperature=0.3,
        #     api_key=os.getenv("GROQ_API_KEY"),
        #     base_url="https://api.groq.com/openai/v1"
        # )
        
        self.llm = LLM(
            model="groq/llama-3.3-70b-versatile",  # or mixtral-8x7b-32768
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )



        self._material_tool = MaterialQueryTool(
            host=self.host, http_path=self.http_path, token=self.token
        )
        self._machine_tool = MachineQueryTool(
            host=self.host, http_path=self.http_path, token=self.token
        )

    # ── Agents ──────────────────────────────────────────────────────────────

    @agent
    def material_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["material_expert"],
            llm=self.llm,
            tools=[self._material_tool],
            verbose=True,
        )

    @agent
    def machine_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["machine_planner"],
            llm=self.llm,
            tools=[
                MachineQueryTool(
                    host=self.host,
                    http_path=self.http_path,
                    token=self.token,
                )
            ],
            verbose=True,
        )

    # ── Tasks ────────────────────────────────────────────────────────────────

    @task
    def material_selection_task(self) -> Task:
        return Task(config=self.tasks_config["material_selection_task"])

    @task
    def machine_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config["machine_planning_task"],
            context=[self.material_selection_task()],
        )

    # ── Crew ─────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )