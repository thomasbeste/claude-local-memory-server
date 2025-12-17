# Claude Memory Server - Docker Image
#
# Build: docker build -t claude-memory .
# Run:   docker run -d -p 8420:8420 -v claude-memory-data:/data claude-memory

FROM python:3.12-slim

WORKDIR /app

# Install system deps for sentence-transformers (optional but recommended)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install with server + embeddings support
RUN pip install --no-cache-dir -e ".[server,embeddings]"

# Create data directory
RUN mkdir -p /data

# Environment defaults
ENV DATA_DIR=/data
ENV PORT=8420
ENV HOST=0.0.0.0

# Expose the port
EXPOSE 8420

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health').raise_for_status()" || exit 1

# Run the server
CMD python -m claude_memory.cli server \
    --host ${HOST} \
    --port ${PORT} \
    --data-dir ${DATA_DIR}
