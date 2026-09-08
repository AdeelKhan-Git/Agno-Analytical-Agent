#!/bin/bash

set -e

############################################################
# Start Ollama
############################################################

############################################################
# Ollama runtime tuning — set BEFORE `ollama serve` starts
#
# Keep the global default short so anything WITHOUT an
# explicit keep-alive override can unload on its own. The
# main Granite model AND the embedding model are both kept
# resident deliberately, via their own periodic keep-alive
# pingers below — the search_knowledge_base tool calls the
# embedder on every single knowledge-base search, so it needs
# to stay loaded the same way the main model does, not reload
# from disk on every search.
############################################################

export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2m}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
echo "======================================"
echo "Starting Ollama..."
echo "======================================"

ollama serve &
OLLAMA_PID=$!

# Ensure background processes are cleaned up when the
# container exits or receives a termination signal.
trap 'kill "$OLLAMA_PID" "${KEEPALIVE_PID:-}" "${EMBEDDER_KEEPALIVE_PID:-}" 2>/dev/null || true' EXIT INT TERM

echo "Waiting for Ollama..."

until curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1
do
    sleep 1
done

echo "Ollama is ready."

############################################################
# Pull model if not already present
############################################################

# This MUST match the model configured for the application.
MODEL="${OLLAMA_MODEL:-qwen3.8:27b}"

if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
    echo "Downloading model: $MODEL"

    if ! ollama pull "$MODEL"; then
        echo "======================================"
        echo "FATAL: failed to pull base model '$MODEL'."
        echo "======================================"
        exit 1
    fi
else
    echo "Model already exists."
fi

############################################################
# Build the tuned model variant
#
# Pins:
#   num_ctx
#   num_predict
#   temperature
#
# The tuned model is:
#
#   <base-model>-tuned
#
############################################################

TUNED_MODEL="${MODEL}-tuned"

echo "Building tuned model variant: $TUNED_MODEL (from base: $MODEL)"

cat > /tmp/Modelfile.generated <<EOF
FROM ${MODEL}

PARAMETER num_ctx 49152
PARAMETER temperature 0.2
EOF

if ! ollama create "$TUNED_MODEL" -f /tmp/Modelfile.generated; then
    echo "======================================"
    echo "FATAL: failed to build tuned model variant '$TUNED_MODEL'."
    echo "======================================"
    exit 1
fi

MODEL="$TUNED_MODEL"

echo "Using tuned model variant: $MODEL"

############################################################
# Export tuned model name
############################################################

export OLLAMA_MODEL="$MODEL"

############################################################
# Warm the model
#
# This loads Granite into memory before the first user query.
# Otherwise the first user request would pay the model-loading
# latency.
############################################################

echo "======================================"
echo "Warming up $MODEL..."
echo "======================================"

if curl -s \
    http://127.0.0.1:11434/api/generate \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"hi\", \"stream\": false}" \
    >/dev/null 2>&1
then
    echo "Model warmed and resident in memory."
else
    echo "Warm-up call failed."
    echo "Continuing anyway — first user query will load the model."
fi

############################################################
# Keep the MAIN model resident
#
# Global OLLAMA_KEEP_ALIVE is only 2 minutes.
#
# We periodically send a request with:
#
#   keep_alive = -1
#
# which keeps the main model loaded indefinitely.
############################################################

PING_INTERVAL_SECONDS=90

(
    while true; do

        sleep "$PING_INTERVAL_SECONDS"

        curl -s \
            http://127.0.0.1:11434/api/generate \
            -d "{\"model\": \"$MODEL\", \"prompt\": \" \", \"keep_alive\": -1, \"options\": {\"num_predict\": 1}}" \
            >/dev/null 2>&1 || true

    done
) &

KEEPALIVE_PID=$!

echo "Started main-model ($MODEL) keep-alive pinger (pid $KEEPALIVE_PID, every ${PING_INTERVAL_SECONDS}s)."

############################################################
# Pull embedding model + build semantic search index
############################################################

EMBEDDER_MODEL="${OLLAMA_EMBEDDER_MODEL:-nomic-embed-text}"

LANCEDB_URI="${LANCEDB_URI:-./data/lancedb}"

if ! ollama list | awk '{print $1}' | grep -Fxq "$EMBEDDER_MODEL"; then

    echo "Downloading embedding model: $EMBEDDER_MODEL"

    if ! ollama pull "$EMBEDDER_MODEL"; then
        echo "======================================"
        echo "WARNING: failed to pull embedding model."
        echo "======================================"
        exit 1
    fi

else

    echo "Embedding model already exists."

fi

############################################################
# Create LanceDB directories
############################################################

mkdir -p "$LANCEDB_URI"
mkdir -p "$(dirname "$LANCEDB_URI")"

echo "======================================"
echo "Checking semantic search index..."
echo "======================================"

############################################################
# Build semantic search index if necessary
#
# Your Dockerfile copies:
#
#     rebuild.py
#
# therefore we execute:
#
#     python rebuild.py
#
############################################################

INDEX_MARKER="$LANCEDB_URI/schema_knowledge.lance"

if [ ! -d "$INDEX_MARKER" ] || [ "${FORCE_INDEX_REBUILD:-0}" = "1" ]; then

    echo "Building semantic search index..."

    if python rebuild.py; then
        echo "Semantic search index built successfully."
    else
        echo "======================================"
        echo "WARNING: Index build failed."
        echo "search_schema_knowledge may return empty results."
        echo "======================================"
    fi

else

    echo "Semantic search index already present — skipping rebuild."
    echo "Set FORCE_INDEX_REBUILD=1 to force a rebuild."

fi

############################################################
# Keep the EMBEDDING model resident too
#
# search_knowledge_base calls the embedder on EVERY single
# knowledge-base search the main model makes at query time —
# not just during index building. Unloading it after the
# index build (the previous behavior) meant every real search
# during actual usage had to reload nomic-embed-text from disk
# into VRAM first, adding avoidable latency to every search.
#
# GPU headroom check (see conversation): main tuned model at
# num_ctx=16384 uses ~24GB; nomic-embed-text is only ~0.4GB —
# keeping both resident simultaneously still leaves ~23GB
# free, so there's no capacity reason to unload it.
#
# Same pattern as the main model: one warm-up call, then a
# periodic keep_alive=-1 ping faster than the 2-minute global
# default so it never actually expires.
############################################################

echo "======================================"
echo "Warming up embedding model: $EMBEDDER_MODEL..."
echo "======================================"

if curl -s \
    http://127.0.0.1:11434/api/embed \
    -d "{\"model\": \"$EMBEDDER_MODEL\", \"input\": \"warmup\"}" \
    >/dev/null 2>&1
then
    echo "Embedding model warmed and resident in memory."
else
    echo "Embedding model warm-up call failed."
    echo "Continuing anyway — first search_knowledge_base call will load it."
fi

EMBEDDER_PING_INTERVAL_SECONDS=90

(
    while true; do

        sleep "$EMBEDDER_PING_INTERVAL_SECONDS"

        curl -s \
            http://127.0.0.1:11434/api/embed \
            -d "{\"model\": \"$EMBEDDER_MODEL\", \"input\": \" \", \"keep_alive\": -1}" \
            >/dev/null 2>&1 || true

    done
) &

EMBEDDER_KEEPALIVE_PID=$!

echo "Started embedding-model ($EMBEDDER_MODEL) keep-alive pinger (pid $EMBEDDER_KEEPALIVE_PID, every ${EMBEDDER_PING_INTERVAL_SECONDS}s)."

############################################################
# Start Streamlit
############################################################

echo "======================================"
echo "Starting Streamlit..."
echo "======================================"

exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false