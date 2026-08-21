FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
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
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agno_agent.py .
COPY glossary_tools.py .
COPY column_glossary.json .
COPY column_descriptions.json .
COPY table_descriptions.json .
COPY schema_knowledge.py .
COPY instructions.py .
COPY rebuild.py .
COPY logging_config.py .
COPY db.py .
COPY app.py .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

ENV OLLAMA_MODEL=granite4.1:30b
ENV OLLAMA_MODELS=/root/.ollama/models

EXPOSE 8501 11434

ENTRYPOINT ["./entrypoint.sh"]