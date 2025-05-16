FROM python:3.9

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY solution_guidance/ /app/solution_guidance/
COPY tests/ /app/tests/

# Copy data folders
COPY cs-train/ /app/cs-train/
COPY cs-production/ /app/cs-production/

# Ensure the models directory exists (though app.py also creates it)
RUN mkdir -p /app/models

# Expose port and set command
EXPOSE 80
CMD ["uvicorn", "solution_guidance.app:app", "--host", "0.0.0.0", "--port", "80"]