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

MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"

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