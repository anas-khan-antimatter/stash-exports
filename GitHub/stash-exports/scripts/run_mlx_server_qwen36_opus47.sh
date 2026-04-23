#!/usr/bin/env bash
# Qwen3.6-35B-A3B · Claude Opus 4.7 Reasoning Distilled — OpenAI-compatible MLX server.
#
# Source: lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled (bf16, vision+text).
# Architecture: qwen3_5_moe (256 experts, 8 routed + 1 shared; ~3B active per token).
# Converted locally with `python -m mlx_lm convert --quantize --q-bits 5`; mlx-lm's
# sanitize() strips the vision tower so we serve a text-only model.
#
# A small proxy on MLX_PORT (default 8080) rejects prompts over MLX_MAX_INPUT_TOKENS
# (default 32768) so large prompts do not OOM Metal during prefill. Set
# MLX_MAX_INPUT_TOKENS=0 to disable (risky).
#
# Because A3B activates only ~3B params, prefill memory per step is much smaller
# than dense 31B bf16, so the prefill cap here (MLX_PREFILL_MAX, default 16384)
# is relaxed vs the Gemma launcher. Set MLX_ALLOW_LARGE_PREFILL_STEPS=1 to go
# higher at your own risk.
#
# Remote access (e.g. Cursor on another machine): MLX_NGROK=1 requires `ngrok` on PATH
# and a one-time `ngrok config add-authtoken <token>`. Tunnels public HTTPS → MLX_PORT.
# Warning: the MLX endpoint has no auth; anyone with the URL can use your GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv-gemma4/bin/activate

# Benign noise on exit: mlx/HF sometimes leave multiprocessing semaphores; Python warns at shutdown.
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"

LISTEN_PORT="${MLX_PORT:-8080}"
BACKEND_PORT="${MLX_BACKEND_PORT:-18080}"
PREFILL="${MLX_PREFILL_STEP_SIZE:-4096}"
PREFILL_MAX="${MLX_PREFILL_MAX:-16384}"
if [ "${MLX_ALLOW_LARGE_PREFILL_STEPS:-0}" != "1" ] && [ "$PREFILL" -gt "$PREFILL_MAX" ]; then
  echo "warning: MLX_PREFILL_STEP_SIZE=$PREFILL exceeds cap $PREFILL_MAX for Qwen3.5 MoE A3B 5-bit. Using $PREFILL_MAX. Set MLX_ALLOW_LARGE_PREFILL_STEPS=1 to force." >&2
  PREFILL="$PREFILL_MAX"
fi
MAX_IN="${MLX_MAX_INPUT_TOKENS:-32768}"
# LRU KV prompt cache — 5-bit weights leave plenty of unified memory on 128GB Macs.
CACHE_BYTES="${MLX_PROMPT_CACHE_BYTES:-16GB}"

MODEL_PATH="${MLX_MODEL_PATH:-$HOME/mlx-models/Qwen3.6-35B-A3B-Claude-4.7-Opus-MLX-5bit}"
if [ ! -d "$MODEL_PATH" ]; then
  echo "error: MLX model directory not found: $MODEL_PATH" >&2
  echo "Run the conversion first, e.g.:" >&2
  echo "  python -m mlx_lm convert --hf-path lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled \\" >&2
  echo "    --mlx-path \"$MODEL_PATH\" --quantize --q-bits 5" >&2
  exit 1
fi

for p in "$LISTEN_PORT" "$BACKEND_PORT"; do
  if lsof -ti ":$p" >/dev/null 2>&1; then
    echo "Stopping existing process on port $p..."
    lsof -ti ":$p" | xargs kill -9 2>/dev/null || true
  fi
done
sleep 1

echo "Starting mlx_lm backend on :$BACKEND_PORT (model=$MODEL_PATH, prefill-step-size=$PREFILL, prompt-cache-bytes=$CACHE_BYTES)" >&2
CHAT_TEMPLATE_ARGS="${MLX_CHAT_TEMPLATE_ARGS:-{\"enable_thinking\": false}}"
python -m mlx_lm server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$BACKEND_PORT" \
  --prompt-concurrency 1 \
  --decode-concurrency 1 \
  --prompt-cache-size 2 \
  --prompt-cache-bytes "$CACHE_BYTES" \
  --prefill-step-size "$PREFILL" \
  --chat-template-args "$CHAT_TEMPLATE_ARGS" \
  "$@" &
BACKEND_PID=$!

PROXY_PID=""
NGROK_PID=""
cleanup() {
  [ -n "${NGROK_PID}" ] && kill "$NGROK_PID" 2>/dev/null || true
  [ -n "${PROXY_PID}" ] && kill "$PROXY_PID" 2>/dev/null || true
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export BACKEND_PORT
python -c "
import os, socket, time
port = int(os.environ['BACKEND_PORT'])
for _ in range(120):
    s = socket.socket()
    try:
        s.settimeout(0.25)
        s.connect(('127.0.0.1', port))
        s.close()
        break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit('mlx_lm did not accept connections on port ' + str(port))
"

export MLX_LISTEN_HOST="${MLX_LISTEN_HOST:-127.0.0.1}"
export MLX_LISTEN_PORT="$LISTEN_PORT"
export MLX_BACKEND_PORT="$BACKEND_PORT"
export MLX_MAX_INPUT_TOKENS="$MAX_IN"
# Hugging Face id used by the proxy to load a tokenizer for the token-cap check.
# (mlx_lm itself is serving the *local* MLX_MODEL_PATH.)
export MLX_MODEL_ID="${MLX_MODEL_ID:-lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled}"
# Proxy rewrites Cursor "display name" model fields to this value, so mlx_lm's
# strict model-id check accepts the request. Matches whatever --model is serving.
export MLX_CHAT_MODEL_ID="${MLX_CHAT_MODEL_ID:-$MODEL_PATH}"
# Disable extended thinking — 5-bit quantization causes degenerate loops inside
# <think> blocks.  The model still reasons internally; it just won't emit a
# separate reasoning trace that spirals into word-salad.
export MLX_CHAT_TEMPLATE_ARGS="${MLX_CHAT_TEMPLATE_ARGS:-{\"enable_thinking\": false}}"

echo "Starting input-limit proxy on :$LISTEN_PORT → :$BACKEND_PORT (MLX_MAX_INPUT_TOKENS=$MAX_IN)" >&2

if [ "${MLX_NGROK:-0}" = "1" ]; then
  python "$ROOT/scripts/mlx_openai_input_limit_proxy.py" &
  PROXY_PID=$!
  for _ in $(seq 1 100); do
    if python3 -c "import socket;s=socket.socket();s.settimeout(0.2);s.connect(('127.0.0.1',int('${LISTEN_PORT}')));s.close()" 2>/dev/null; then
      break
    fi
    sleep 0.15
  done
  if ! command -v ngrok >/dev/null 2>&1; then
    echo "error: MLX_NGROK=1 but ngrok not found. Install: https://ngrok.com/download (e.g. brew install ngrok/ngrok/ngrok)" >&2
    exit 1
  fi
  ngrok http "$LISTEN_PORT" --log=stdout >/tmp/ngrok-mlx.log 2>&1 &
  NGROK_PID=$!
  sleep 2
  PUBLIC_URL=""
  PUBLIC_URL=$(curl -fsS "http://127.0.0.1:4040/api/tunnels" 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for t in d.get('tunnels', []):
        u = t.get('public_url', '')
        if u.startswith('https://'):
            print(u)
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
" 2>/dev/null || true)
  echo "" >&2
  if [ -n "$PUBLIC_URL" ]; then
    echo "ngrok → use this as OpenAI-compatible base URL in Cursor (include /v1):" >&2
    echo "  ${PUBLIC_URL}/v1" >&2
  else
    echo "ngrok is running; open http://127.0.0.1:4040 for the public HTTPS URL. Logs: /tmp/ngrok-mlx.log" >&2
  fi
  echo "See docs/cursor-local-mlx-ngrok.txt for Cursor setup and security notes." >&2
  echo "" >&2
  wait "$PROXY_PID"
else
  python "$ROOT/scripts/mlx_openai_input_limit_proxy.py"
fi
