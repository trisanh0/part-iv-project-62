# TEMPO Display Day Live Demo Container
# Single-process booth deployment image for opti homelab server or local testing.

FROM python:3.11-slim

WORKDIR /app

# Ensure non-buffered stdout for container logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal required dependencies for web application
RUN pip install --no-cache-dir \
    fastapi==0.143.0 \
    uvicorn==0.54.0 \
    websockets==17.2

# Copy package metadata and demo code
COPY pyproject.toml /app/
COPY demo /app/demo

# Expose HTTP/WebSocket port
EXPOSE 8000

# Run uvicorn server bound to 0.0.0.0
CMD ["python", "-m", "uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
