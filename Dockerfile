# Multi-stage Docker build for Dataseter
# Maintainer: mkupermann

# Stage 1: Base dependencies
FROM python:3.10-slim as base

# Labels
LABEL maintainer="mkupermann"
LABEL description="Dataseter - Advanced AI Training Dataset Creator"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PDF processing
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    # Web scraping with JS support
    chromium \
    chromium-driver \
    # Build tools
    gcc \
    g++ \
    make \
    # Additional tools
    curl \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as dependencies

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 3: Application
FROM dependencies as app

WORKDIR /app

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY *.py ./
COPY README.md setup.py requirements.txt ./

# Install the package (with fallback)
RUN pip install -e . || echo "Package installation skipped"

# Ensure src is in Python path
ENV PYTHONPATH=/app

# Create directories for data
RUN mkdir -p /data/input /data/output /data/cache

# Create non-root user
RUN useradd -m -u 1000 dataseter && \
    chown -R dataseter:dataseter /app /data

# Switch to non-root user
USER dataseter

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATASETER_CACHE_DIR=/data/cache
ENV TMPDIR=/tmp
ENV PATH="/home/dataseter/.local/bin:${PATH}"

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["dataseter", "web", "--host", "0.0.0.0", "--port", "8080"]