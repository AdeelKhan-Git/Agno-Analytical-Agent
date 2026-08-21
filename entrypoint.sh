############################################################
# Start Ollama
############################################################

echo "======================================"
echo "Starting Ollama..."
echo "======================================"

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."

until curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1
do
    sleep 1
done

echo "Ollama is ready."

############################################################
# Pull model if not already present
############################################################
# This MUST match the default OLLAMA_MODEL agno_agent.py reads — if the two
# ever drift apart, this pulls one model while the app tries to load a
# different one that was never downloaded, and the first real query fails.

MODEL="${OLLAMA_MODEL:-granite4.1:30b}"

if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
    echo "Downloading model: $MODEL"
    ollama pull "$MODEL"
else
    echo "Model already exists."
fi

############################################################
# Warm the model — load it into memory now, not on the first
# user query. Without this, whoever asks the first question
# pays the full model-load latency (can be 30s-2min+ for a
# 27B+ model), which looks like the app hanging or broken.
############################################################

echo "======================================"
echo "Warming up $MODEL..."
echo "======================================"

curl -s http://127.0.0.1:11434/api/generate -d "{\"model\": \"$MODEL\", \"prompt\": \"hi\", \"stream\": false}" >/dev/null 2>&1 \
    && echo "Model warmed and resident in memory." \
    || echo "Warm-up call failed — continuing anyway, first user query will load the model instead."

############################################################
# Pull embedding model + build the semantic search index
#
# The embedder must be pulled the same way as the main model, and the
# LanceDB index directory must exist before anything tries to write to
# it — this is the "data dir not created" failure mode: LANCEDB_URI
# points at a path that has to be created at runtime, not assumed to
# already exist.
############################################################

EMBEDDER_MODEL="${OLLAMA_EMBEDDER_MODEL:-nomic-embed-text}"
LANCEDB_URI="${LANCEDB_URI:-./data/lancedb}"

if ! ollama list | awk '{print $1}' | grep -Fxq "$EMBEDDER_MODEL"; then
    echo "Downloading embedding model: $EMBEDDER_MODEL"
    ollama pull "$EMBEDDER_MODEL"
else
    echo "Embedding model already exists."
fi

mkdir -p "$LANCEDB_URI"
mkdir -p "$(dirname "$LANCEDB_URI")"

echo "======================================"
echo "Checking semantic search index..."
echo "======================================"

# Only build if missing, or if FORCE_INDEX_REBUILD=1 is set — rebuilding
# is cheap but there's no reason to pay it on every restart when
# LANCEDB_URI is a persistent volume. Re-run manually with
# `python rebuild_index.py` any time the JSON docs change.
INDEX_MARKER="$LANCEDB_URI/schema_knowledge.lance"
if [ ! -d "$INDEX_MARKER" ] || [ "${FORCE_INDEX_REBUILD:-0}" = "1" ]; then
    echo "Building semantic search index..."
    python rebuild_index.py || echo "Index build failed — search_schema_knowledge tool will return empty results until this is fixed."
else
    echo "Semantic search index already present — skipping rebuild."
    echo "(Set FORCE_INDEX_REBUILD=1 to force a rebuild after editing the JSON docs.)"
fi

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