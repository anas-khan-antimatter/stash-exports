#!/usr/bin/env python3
"""
OpenAI-compatible HTTP proxy in front of mlx_lm server.

Rejects /v1/chat/completions and /v1/completions when the tokenized prompt
exceeds MLX_MAX_INPUT_TOKENS, so the client gets a clear JSON error instead of
a Metal OOM abort during prefill.

Strips query strings when forwarding to mlx_lm (Cursor may add ?…); mlx matches
exact paths only and would otherwise return 404 on /v1/chat/completions?foo.

Rewrites JSON "model" to MLX_CHAT_MODEL_ID when the client sends a display name
(e.g. Cursor: "Qwen 3.6 35B A3B Opus 4.7 (MLX 5-bit)") — mlx accepts the model
id it was started with, not arbitrary labels.

Set MLX_MAX_INPUT_TOKENS=0 to disable the check (not recommended for huge prompts).

Environment:
  MLX_LISTEN_HOST, MLX_LISTEN_PORT  — bind address for this proxy (default 127.0.0.1:8080)
  MLX_BACKEND_HOST, MLX_BACKEND_PORT — mlx_lm server (default 127.0.0.1:18080)
  MLX_MODEL_ID — Hugging Face id used only to load the tokenizer for the token-cap
      check (default lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)
  MLX_MAX_INPUT_TOKENS — max prompt tokens allowed (default 32768)
  MLX_CHAT_TEMPLATE_ARGS — JSON object, passed through to apply_chat_template (default {})
  MLX_CHAT_MODEL_ID — model id mlx_lm is serving (default matches the Qwen3.6 launcher;
      for local MLX directories the launcher exports the absolute path here).
  MLX_CHAT_MODEL_REWRITE — set to 0 to disable rewriting the model field (default 1)
"""

from __future__ import annotations

import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse


def _normalize_request_path(request_path: str) -> str:
    """
    Path mlx_lm uses for routing (no query string; mlx matches exact strings).
    Handles absolute-form targets, percent-encoding, and stray CR/LF/tab.
    """
    if not request_path:
        return "/"
    raw = request_path.strip()
    parsed = urlparse(raw)
    # Absolute URL in request line: POST https://host/v1/chat/completions
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
    else:
        path = parsed.path or "/"
    path = unquote(path)
    path = path.strip("\r\n\t ") or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path


def _load_tokenizer():
    from transformers import AutoTokenizer

    model_id = os.environ.get(
        "MLX_MODEL_ID",
        "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled",
    )
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _chat_template_kwargs() -> Dict[str, Any]:
    raw = os.environ.get("MLX_CHAT_TEMPLATE_ARGS", "{}").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _count_chat_tokens(
    tokenizer, body: Dict[str, Any], template_kw: Dict[str, Any]
) -> int:
    messages = body.get("messages")
    if not messages:
        raise ValueError("missing messages")
    tools = body.get("tools")
    kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        **template_kw,
    }
    if tools:
        kwargs["tools"] = tools
    ids = tokenizer.apply_chat_template(messages, **kwargs)
    return len(ids)


def _count_completion_tokens(tokenizer, body: Dict[str, Any]) -> int:
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        # OpenAI legacy: array of integers or strings — best-effort
        if prompt and isinstance(prompt[0], int):
            return len(prompt)
        raise ValueError("unsupported prompt format for token cap")
    return len(tokenizer.encode(prompt))


def _pump_backend_to_client(handler, resp: http.client.HTTPResponse) -> None:
    """Stream body from mlx_lm to the IDE. Catch disconnects (compact chat, cancel, tab switch)."""
    try:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                handler.log_message(
                    "client closed connection mid-response (e.g. stream cancel or Compact conversation)"
                )
                break
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _rewrite_openai_model_field(body: bytes, backend_path: str, log) -> bytes:
    """
    Cursor custom models often use a display name as `model`, but mlx_lm expects a
    valid Hugging Face repo id for the loaded weights.
    """
    if os.environ.get("MLX_CHAT_MODEL_REWRITE", "1") == "0":
        return body
    if backend_path not in ("/v1/chat/completions", "/v1/completions", "/chat/completions"):
        return body
    target = os.environ.get(
        "MLX_CHAT_MODEL_ID",
        "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled",
    ).strip()
    if not target:
        return body
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(data, dict):
        return body
    old = data.get("model", "")
    if not isinstance(old, str):
        return body
    if old == target:
        return body
    # mlx_lm is started with one specific model id / path; anything else in the
    # client payload is a display name, stale id, or a mismatched HF repo — mlx
    # would reject it. Always normalize to the configured target.
    data["model"] = target
    if data.get("repetition_penalty", 0) == 0 and data.get("frequency_penalty", 0) == 0:
        data.setdefault("repetition_penalty", 1.15)
        data.setdefault("repetition_context_size", 256)
    log("rewrote JSON model field %r -> %r", old, target)
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def _error_payload(msg: str, code: str = "context_length_exceeded") -> bytes:
    err = {
        "error": {
            "message": msg,
            "type": "invalid_request_error",
            "code": code,
        }
    }
    return json.dumps(err).encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    tokenizer = None
    template_kw: Dict[str, Any] = {}
    max_input: int = 32768

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _max_allowed(self) -> Optional[int]:
        v = int(os.environ.get("MLX_MAX_INPUT_TOKENS", "32768"))
        return None if v <= 0 else v

    def _should_check(self, path_only: str) -> bool:
        return path_only in ("/v1/chat/completions", "/v1/completions")

    def _check_body(self, path_only: str, body: bytes) -> Tuple[bool, Optional[bytes]]:
        max_t = self._max_allowed()
        if max_t is None:
            return True, None
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return True, None
        try:
            if path_only == "/v1/chat/completions":
                n = _count_chat_tokens(self.tokenizer, data, self.template_kw)
            else:
                n = _count_completion_tokens(self.tokenizer, data)
        except Exception as e:
            self.log_message("token count failed: %s", e)
            return False, _error_payload(
                f"Could not measure prompt length: {e}", "invalid_request_error"
            )
        if n > max_t:
            msg = (
                f"Prompt length is {n} tokens; maximum allowed is {max_t} "
                f"(set MLX_MAX_INPUT_TOKENS to raise, or 0 to disable). "
                "In Continue: use Compact conversation, Cmd/Ctrl+L for a new session, fewer @-files, "
                "and match defaultCompletionOptions.contextLength to this cap. "
                "Otherwise Metal can OOM during prefill on local MLX."
            )
            self.log_message("rejecting request: %s", msg.replace("\n", " "))
            return False, _error_payload(msg)
        return True, None

    def _forward(self) -> None:
        backend_host = os.environ.get("MLX_BACKEND_HOST", "127.0.0.1")
        backend_port = int(os.environ.get("MLX_BACKEND_PORT", "18080"))
        backend_path = _normalize_request_path(self.path)

        body: Optional[bytes] = None
        if self.command in ("POST", "PUT", "PATCH"):
            length = self.headers.get("Content-Length")
            if length:
                body = self.rfile.read(int(length))
            elif self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                chunks = []
                while True:
                    line = self.rfile.readline().strip()
                    chunk_size = int(line, 16)
                    if chunk_size == 0:
                        self.rfile.readline()
                        break
                    chunks.append(self.rfile.read(chunk_size))
                    self.rfile.readline()
                body = b"".join(chunks)
            else:
                body = b""
            body = _rewrite_openai_model_field(body, backend_path, self.log_message)

        if body is not None and self._should_check(backend_path):
            ok, err_body = self._check_body(backend_path, body)
            if not ok:
                assert err_body is not None
                self.send_response(413)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

        out_headers: Dict[str, str] = {}
        skip = {"host", "connection", "proxy-connection", "content-length", "transfer-encoding"}
        for k, v in self.headers.items():
            if k.lower() in skip:
                continue
            out_headers[k] = v
        out_headers["Host"] = f"{backend_host}:{backend_port}"
        if body is not None and self.command in ("POST", "PUT", "PATCH"):
            out_headers["Content-Length"] = str(len(body))

        conn = http.client.HTTPConnection(backend_host, backend_port, timeout=None)
        try:
            try:
                conn.request(self.command, backend_path, body=body, headers=out_headers)
                resp = conn.getresponse()
                # mlx: 404 = unknown path OR handle_completion caught an exception (JSON body).
                if resp.status == 404:
                    err_body = resp.read()
                    preview = err_body[:1500].decode("utf-8", "replace").replace("\n", " ")
                    self.log_message(
                        "mlx 404: path_out=%r path_in=%r body=%s",
                        backend_path,
                        self.path,
                        preview[:700],
                    )
                    self.send_response(404)
                    for hk, hv in resp.getheaders():
                        if hk.lower() in ("transfer-encoding", "connection"):
                            continue
                        self.send_header(hk, hv)
                    self.end_headers()
                    try:
                        self.wfile.write(err_body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    try:
                        resp.close()
                    except Exception:
                        pass
                    return
            except (ConnectionRefusedError, OSError) as e:
                self.log_message("backend unreachable: %s", e)
                err = _error_payload(
                    f"MLX backend at {backend_host}:{backend_port} is not accepting connections ({e!r}). "
                    "The mlx_lm process likely crashed (e.g. Metal OOM). Restart "
                    "scripts/run_mlx_server_qwen36_opus47.sh. If OOM persists, lower "
                    "MLX_PREFILL_STEP_SIZE and MLX_MAX_INPUT_TOKENS.",
                    code="mlx_backend_unavailable",
                )
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                try:
                    self.wfile.write(err)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(resp.status)
            for hk, hv in resp.getheaders():
                # Hop-by-hop headers
                if hk.lower() in ("transfer-encoding", "connection"):
                    continue
                self.send_header(hk, hv)
            self.end_headers()
            _pump_backend_to_client(self, resp)
        finally:
            conn.close()

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def do_OPTIONS(self) -> None:
        self._forward()


def main() -> None:
    listen_host = os.environ.get("MLX_LISTEN_HOST", "127.0.0.1")
    listen_port = int(os.environ.get("MLX_LISTEN_PORT", "8080"))

    ProxyHandler.tokenizer = _load_tokenizer()
    ProxyHandler.template_kw = _chat_template_kwargs()
    max_t = os.environ.get("MLX_MAX_INPUT_TOKENS", "32768")
    sys.stderr.write(
        f"mlx_openai_input_limit_proxy: tokenizer ok, MLX_MAX_INPUT_TOKENS={max_t}\n"
    )
    sys.stderr.write(
        f"Forwarding to http://{os.environ.get('MLX_BACKEND_HOST', '127.0.0.1')}:"
        f"{os.environ.get('MLX_BACKEND_PORT', '18080')}\n"
    )

    httpd = ThreadingHTTPServer((listen_host, listen_port), ProxyHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
