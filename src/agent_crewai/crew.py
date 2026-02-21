from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from typing import List
from google import genai

@CrewBase
class CrewAiDev():
    """LatestAiDevelopment crew"""

    # 1. Define the LLM inside the class so it's accessible to your methods
    # Using Gemini 3 Flash (recommended for 2026) for speed/cost
    llm = LLM(
        model="gemini/gemini-2.5-flash", 
        temperature=0.7
    )

    @agent
    def precision_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['precision_agent'],
            llm=self.llm, 
            verbose=True
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['component_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        return Crew(
            agents=self.agents, 

            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )