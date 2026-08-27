############################################################
# Start Ollama
############################################################

############################################################
# Ollama runtime tuning — set BEFORE `ollama serve` starts
#
# No OLLAMA_KEEP_ALIVE was previously set here, meaning every model
# (the main agent model AND nomic-embed-text) fell back to Ollama's
# global default of 5 minutes. That's what actually caused the
# "model predicted to exceed available memory, evicting" event: both
# models compete for the same VRAM under one shared default, with no
# way to say "keep the main model loaded indefinitely, but let the
# embedder unload quickly."
#
# Fix: set a short GLOBAL default here (so the embedder, and anything
# else without an explicit override, unloads quickly and frees VRAM),
# and separately keep the MAIN model resident via an explicit
# background keep-alive pinger (see below, after the model is pulled)
# rather than relying on a per-call keep_alive from Agno's Ollama(...)
# wrapper — that passthrough is not reliably consistent across all
# Ollama client library versions/paths.
############################################################

export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2m}"

echo "======================================"
echo "Starting Ollama..."
echo "======================================"

ollama serve &
OLLAMA_PID=$!

# Ensure background processes (Ollama server, the keep-alive pinger
# started further below) are cleaned up if this script/container is
# stopped, rather than left as orphaned processes.
trap 'kill $OLLAMA_PID ${KEEPALIVE_PID:-} 2>/dev/null' EXIT INT TERM

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
    if ! ollama pull "$MODEL"; then
        echo "FATAL: failed to pull base model '$MODEL' — cannot build the tuned variant without it."
        exit 1
    fi
else
    echo "Model already exists."
fi

############################################################
# Build the tuned model variant — pins num_ctx, num_predict, and
# temperature at the Ollama level, so these apply regardless of what
# any individual client call (Agno's Ollama(...) options=, a curl
# warm-up ping, etc.) does or doesn't set explicitly.
#
# The Modelfile content is generated INLINE here at runtime — no
# separate Modelfile needs to exist in the repo/image. This creates a
# NEW named model ("<base>-tuned") from the base model already pulled
# above; it does not modify or re-download the base model itself.
# MODEL is then repointed at the tuned variant for everything below
# (warm-up call, keep-alive pinger), and this is also what
# agno_agent.py loads via OLLAMA_MODEL.
#
# num_ctx 16384: this Granite architecture disables KV cache shifting,
# so llama.cpp defensively halves the working prompt limit — 16384
# gives ~8192 real usable input tokens instead of truncating at ~4098
# (the exact failure seen in production: "truncating input prompt
# limit=4098 prompt=8490").
# num_predict 1024: hard cap on generation length, so a run-on response
# can't eat VRAM/wall-clock time indefinitely.
# temperature 0.3: low, stable, low-hallucination-risk decoding for SQL
# generation, with a little room left for natural-language explanation
# text (vs. 0.0 fully greedy).
#
# Rebuilt every start (ollama create is fast/near-no-op if the variant
# already exists with the same content), so this always stays in sync
# if these values are changed here later.
############################################################

TUNED_MODEL="${MODEL}-tuned"

echo "Building tuned model variant: $TUNED_MODEL (from base: $MODEL)"

cat > /tmp/Modelfile.generated << EOF
FROM ${MODEL}

PARAMETER num_ctx 16384
PARAMETER num_predict 1024
PARAMETER temperature 0.2
EOF

if ! ollama create "$TUNED_MODEL" -f /tmp/Modelfile.generated; then
    echo "======================================"
    echo "FATAL: failed to build tuned model variant '$TUNED_MODEL'."
    echo "Refusing to fall back to the untuned base model — this app requires"
    echo "the tuned variant (num_ctx=16384, num_predict=1024, temperature=0.3)."
    echo "Check the 'ollama create' output above for the actual error."
    echo "======================================"
    exit 1
fi

MODEL="$TUNED_MODEL"
echo "Using tuned model variant: $MODEL"

# Export so agno_agent.py's build_agent() (via os.environ.get("OLLAMA_MODEL", ...))
# actually loads the SAME model name as whatever was pulled/tuned/warmed
# above — a local shell variable alone is not visible to the Python
# process started later by this same script.
export OLLAMA_MODEL="$MODEL"

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
# Keep the MAIN model resident despite the short global
# OLLAMA_KEEP_ALIVE set above (that short default exists so the
# EMBEDDER unloads quickly and stops competing with the main model
# for VRAM — see the eviction fix further down).
#
# Rather than depend on whichever Ollama client library Agno's
# Ollama(...) wrapper uses to correctly pass a per-call keep_alive
# override (real-world reports show this is inconsistent across
# client paths — some silently fall back to the server default), this
# runs a background pinger that sends a trivial 1-token request with
# an explicit long keep_alive directly against Ollama's REST API on a
# fixed interval, shorter than OLLAMA_KEEP_ALIVE, so the main model
# never actually reaches its idle-unload timeout.
############################################################

PING_INTERVAL_SECONDS=90   # shorter than the 2m global OLLAMA_KEEP_ALIVE

(
    while true; do
        sleep "$PING_INTERVAL_SECONDS"
        curl -s http://127.0.0.1:11434/api/generate \
            -d "{\"model\": \"$MODEL\", \"prompt\": \" \", \"keep_alive\": -1, \"options\": {\"num_predict\": 1}}" \
            >/dev/null 2>&1
    done
) &
KEEPALIVE_PID=$!
echo "Started main-model ($MODEL) keep-alive pinger (pid $KEEPALIVE_PID, every ${PING_INTERVAL_SECONDS}s)."

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
# Explicitly unload the embedder from VRAM right after the index
# build. The global OLLAMA_KEEP_ALIVE (set to a short 2m default
# above) already means the embedder unloads on its own reasonably
# soon — but doing it explicitly here, immediately, means VRAM is
# freed right away instead of waiting up to 2 more minutes, which
# matters right before the main model's next load/reload.
#
# This is what was causing "model predicted to exceed available
# memory, evicting" when granite4.1:30b (or whichever OLLAMA_MODEL is
# configured) tried to load afterward — the embedder (and any other
# stale runner) was still holding VRAM under the previous no-explicit-
# keep_alive setup, where everything defaulted to Ollama's global 5m.
#
# Done via a direct call to Ollama's REST API with keep_alive=0,
# rather than relying on any embedder-library keep_alive passthrough
# (which upstream Ollama issues show can be unreliable depending on
# the client path) — this is the one guaranteed-to-work mechanism.
############################################################

echo "Unloading embedding model from VRAM (keep_alive=0)..."
curl -s http://127.0.0.1:11434/api/generate \
    -d "{\"model\": \"$EMBEDDER_MODEL\", \"keep_alive\": 0}" \
    >/dev/null
echo "Embedding model unloaded."

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