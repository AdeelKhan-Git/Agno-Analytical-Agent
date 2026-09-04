FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# Install required system packages
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    proxychains4 \
    procps \
    zstd \
    iproute2 \
    iputils-ping \
    bash \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Ollama
# ============================================================
RUN curl -fsSL https://ollama.com/install.sh | sh

# ============================================================
# Application directory
# ============================================================
WORKDIR /app

# ============================================================
# Python dependencies
# ============================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Application files
# ============================================================
COPY agno_agent.py .
COPY column_glossary.json .
COPY column_descriptions.json .
COPY data/ ./data/
COPY learning/ ./learning/
COPY table_descriptions.json .
COPY schema_knowledge.py .
COPY instructions.py .
COPY rebuild.py .
COPY agent_sessions.db .
COPY knowledge_contents.db .
COPY logging_config.py .
COPY db.py .
COPY app.py .

# ============================================================
# Entrypoint
# ============================================================
COPY entrypoint.sh /app/entrypoint.sh

# IMPORTANT:
# Convert Windows CRLF line endings to Linux LF.
# This fixes:
#   ./entrypoint.sh: not found
#   trap: TERM: bad trap
#   Syntax error: "&&" unexpected
#
# Also make the script executable.
RUN sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# ============================================================
# Environment
# ============================================================
ENV OLLAMA_MODEL=qwen3.8:27b
ENV OLLAMA_MODELS=/root/.ollama/models

# ============================================================
# Ports
# ============================================================
EXPOSE 8501 11434

# ============================================================
# Start application
# ============================================================
# Explicitly invoke Bash instead of relying on executable
# detection/shebang handling.
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]