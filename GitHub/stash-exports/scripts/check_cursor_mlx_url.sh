#!/usr/bin/env bash
# Quick checks for Cursor ↔ OpenAI-compatible MLX (local or ngrok).
# Usage:
#   ./scripts/check_cursor_mlx_url.sh http://127.0.0.1:8080/v1
#   ./scripts/check_cursor_mlx_url.sh https://xxxx.ngrok-free.dev/v1
set -euo pipefail
BASE="${1:?Usage: $0 <base-url-ending-in-/v1>}"
KEY="${OPENAI_API_KEY:-mlx}"
echo "== GET /v1/models (adds ngrok header if host looks like ngrok) =="
EXTRA=()
case "$BASE" in
  *ngrok*) EXTRA=(-H "ngrok-skip-browser-warning: true") ;;
esac
if ((${#EXTRA[@]} > 0)); then
  curl -sS "${EXTRA[@]}" -H "Authorization: Bearer ${KEY}" "${BASE%/}/models" | python3 -m json.tool 2>/dev/null || {
    echo "FAILED: not valid JSON (common: ngrok HTML warning page — see docs/cursor-local-mlx-ngrok.txt)" >&2
    curl -sS "${EXTRA[@]}" -H "Authorization: Bearer ${KEY}" "${BASE%/}/models" | head -c 400 >&2
    echo >&2
    exit 1
  }
else
  curl -sS -H "Authorization: Bearer ${KEY}" "${BASE%/}/models" | python3 -m json.tool 2>/dev/null || {
    echo "FAILED: not valid JSON (common: ngrok HTML warning page — see docs/cursor-local-mlx-ngrok.txt)" >&2
    curl -sS -H "Authorization: Bearer ${KEY}" "${BASE%/}/models" | head -c 400 >&2
    echo >&2
    exit 1
  }
fi
echo ""
echo "OK: endpoint returned JSON. Use this exact base URL in Cursor (include /v1)."
