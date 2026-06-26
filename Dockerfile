FROM python:3.11-slim

WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8080

# Start server
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8080"]
