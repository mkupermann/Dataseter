# Multi-stage Docker build for Dataseter

# Stage 1: Base dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PDF processing
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    # Web scraping with JS support
    chromium \
    chromium-driver \
    # Build tools
    gcc \
    g++ \
    make \
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

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create directories for data
RUN mkdir -p /data/input /data/output /data/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATASETER_CACHE_DIR=/data/cache
ENV TMPDIR=/tmp

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["dataseter", "web", "--host", "0.0.0.0", "--port", "8080"]