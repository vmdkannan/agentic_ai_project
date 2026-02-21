FROM python:3.12-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the whole project
COPY . .

# 3. FIX: Add the src directory to the Python Path
ENV PYTHONPATH=/app/src

# 4. Install the local package (this fixes the 'ModuleNotFoundError')
RUN pip install -e .

EXPOSE 8078

# 5. Start Uvicorn pointing to your api file
# If your file is src/agent_crewai/api.py, use this:
CMD ["uvicorn", "agent_crewai.api:app", "--host", "0.0.0.0", "--port", "8078"]