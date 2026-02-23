from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from databricks import sql
from typing import Optional
import os
from decimal import Decimal



@CrewBase
class CrewAiDev():

    def __init__(
        self,
        host: Optional[str] = None,
        http_path: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
        self.token = token or os.getenv("DATABRICKS_TOKEN")

        self.llm = LLM(
            model="gemini/gemini-2.5-flash",
            temperature=0.3
        )
        
    

    def sanitize_for_crewai(self, data):
        if isinstance(data, list):
            return [self.sanitize_for_crewai(item) for item in data]

        if isinstance(data, dict):
            return {k: self.sanitize_for_crewai(v) for k, v in data.items()}

        if isinstance(data, Decimal):
            return float(data)

        return data
        

    def query_materials(self, temperature: float, aerospace_required: bool):

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
                    {
                        "temperature": float(temperature),
                        "aerospace_required": bool(aerospace_required),
                    },
                )
                result = cursor.fetchall()

        columns = [
            "material_name",
            "material_type",
            "grade",
            "machinability_rating",
            "tensile_strength_mpa",
            "yield_strength_mpa",
            "max_operating_temp_c",
            "aerospace_grade",
            "cost_per_kg",
        ]

        return [dict(zip(columns, row)) for row in result]

    @agent
    def material_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['material_expert'],
            llm=self.llm,
            verbose=True
        )

    @task
    def material_selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['material_selection_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )