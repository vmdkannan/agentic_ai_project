# AgentCrewai Crew

Welcome to the AgentCrewai Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:

```bash
crewai install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/agent_crewai/config/agents.yaml` to define your agents
- Modify `src/agent_crewai/config/tasks.yaml` to define your tasks
- Modify `src/agent_crewai/crew.py` to add your own logic, tools and specific args
- Modify `src/agent_crewai/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the agent_crewai Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The agent_crewai Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the AgentCrewai Crew or crewAI.

- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.

Docker Commands :

$ docker build -t agent_crewai .

$ docker run --name agent_crewai_app -p 8078:8078 --env-file .env agent_crewai

$ curl -X POST http://localhost:8078/analyze \
 -H "Content-Type: application/json" \
 -d '{
"component_name": "Turbine Blade",
"dimensions": "500mm x 120mm",
"tolerances": "+/- 0.01mm",
"material_requirements": "Inconel 718",
"norms": "ISO 9001, AS9100",
"topic": "Aerospace Manufacturing"
}'
