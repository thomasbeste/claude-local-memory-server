# Claude Local Memory Server - Docker Image
#
# Build: docker build -t claude-local-memory-server .
# Run:   docker run -d -p 8420:8420 -v claude-local-memory-server-data:/data claude-local-memory-server

FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[server]"

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

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
