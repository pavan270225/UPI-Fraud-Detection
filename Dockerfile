# UPI Fraud Detection - Dockerfile
# Multi-stage build for reproducible environment

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy project files
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY notebooks/ ./notebooks/
COPY reports/ ./reports/
COPY PRESENTATION_AND_PROJECT_PLAN.md .

# Create data directory (datasets not included - must be mounted)
RUN mkdir -p data/ieee-cis data/paysim models

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose Streamlit port
EXPOSE 8501

# Default command runs the dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]