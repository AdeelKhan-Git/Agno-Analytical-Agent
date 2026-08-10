#!/bin/bash
set -e

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

MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"

if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
    echo "Downloading model: $MODEL"
    ollama pull "$MODEL"
else
    echo "Model already exists."
fi

############################################################
# Start Tailscale (optional)
############################################################

if [ -n "$TAILSCALE_AUTHKEY" ]; then

    echo "======================================"
    echo "Starting Tailscale..."
    echo "======================================"

    tailscaled \
        --tun=userspace-networking \
        --state=/tmp/tailscaled.state &

    TAILSCALE_PID=$!

    sleep 5

    tailscale up \
        --authkey="$TAILSCALE_AUTHKEY" \
        --hostname=runpod-agent \
        --accept-routes

    echo "Connected to Tailnet."

    echo "Tailscale IPs:"
    tailscale ip

else

    echo "TAILSCALE_AUTHKEY not provided."
fi

############################################################
# Start Streamlit
############################################################

echo "======================================"
echo "Starting Streamlit..."
echo "======================================"

exec streamlit run app.py