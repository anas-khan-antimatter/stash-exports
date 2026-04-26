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

import hashlib
import http.client
import json
import os
import sys
import threading
import time
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


import re as _re

_LOOP_WINDOW = int(os.environ.get("MLX_LOOP_WINDOW", "200"))
_LOOP_NGRAM = int(os.environ.get("MLX_LOOP_NGRAM", "8"))
_LOOP_THRESHOLD = int(os.environ.get("MLX_LOOP_THRESHOLD", "4"))


def _detect_degenerate_loop(text: str) -> bool:
    """
    Detect degenerate generation via:
      1. N-gram repetition: any 8-gram repeated 4+ times in the last 200 words.
      2. Word-salad / synonym chain: if the last 60 words contain no sentence
         boundary (period, question mark, exclamation, newline) the model is
         likely in free-association runaway.
      3. Single-word stutter: any word repeated 6+ times in the last 40 words.
    """
    words = text.split()
    tail = words[-_LOOP_WINDOW:] if len(words) > _LOOP_WINDOW else words
    if len(tail) < 30:
        return False

    # Check 1: N-gram repetition
    if len(tail) >= _LOOP_NGRAM * _LOOP_THRESHOLD:
        ngrams: Dict[str, int] = {}
        for i in range(len(tail) - _LOOP_NGRAM + 1):
            key = " ".join(tail[i : i + _LOOP_NGRAM])
            ngrams[key] = ngrams.get(key, 0) + 1
            if ngrams[key] >= _LOOP_THRESHOLD:
                return True

    # Check 2: no sentence-ending punctuation in last 80 words = word salad
    check2_len = min(80, len(tail))
    if check2_len >= 60:
        recent_text = " ".join(tail[-check2_len:])
        if not _re.search(r'[.!?\n]', recent_text):
            return True

    # Check 3: single word repeated 6+ times in last 40 words
    short_tail = tail[-40:]
    word_counts: Dict[str, int] = {}
    for w in short_tail:
        wl = w.lower().strip(".,!?;:-\"'")
        if len(wl) < 2:
            continue
        word_counts[wl] = word_counts.get(wl, 0) + 1
        if word_counts[wl] >= 6:
            return True

    # Check 4: any "word" over 80 chars is a merged/hyphenated degenerate chain
    for w in tail[-20:]:
        if len(w) > 80:
            return True

    # Check 5: high average word length signals word-concatenation degeneration
    recent = tail[-30:] if len(tail) >= 30 else tail
    if len(recent) >= 20:
        avg_len = sum(len(w) for w in recent) / len(recent)
        long_count = sum(1 for w in recent if len(w) > 15)
        if avg_len > 12 or long_count > len(recent) * 0.3:
            return True

    return False


def _pump_backend_to_client(handler, resp: http.client.HTTPResponse, is_stream: bool = False) -> None:
    """
    Stream body from mlx_lm to the IDE with loop detection.
    For SSE streams, accumulates token text and kills the response early
    if degenerate repetition is detected.
    """
    accumulated_text = ""
    check_every = 40
    token_count = 0
    try:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            if is_stream:
                for line in chunk.decode("utf-8", "replace").split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("data: ") and stripped != "data: [DONE]":
                        try:
                            evt = json.loads(stripped[6:])
                            delta = evt.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "") or delta.get("reasoning", "")
                            if content:
                                accumulated_text += content
                                token_count += 1
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
                if token_count > 0 and token_count % check_every == 0:
                    if _detect_degenerate_loop(accumulated_text):
                        handler.log_message(
                            "loop detector: degenerate repetition after ~%d tokens, terminating stream",
                            token_count,
                        )
                        try:
                            handler.wfile.write(b"data: [DONE]\n\n")
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return
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


_MLX_MAX_COMPLETION_TOKENS = int(os.environ.get("MLX_MAX_COMPLETION_TOKENS", "512"))
_MLX_REP_PENALTY = float(os.environ.get("MLX_REPETITION_PENALTY", "2.0"))
_MLX_REP_CONTEXT = int(os.environ.get("MLX_REPETITION_CONTEXT_SIZE", "1024"))
_MLX_FORCE_NONSTREAM = os.environ.get("MLX_FORCE_NONSTREAM", "0") != "0"


def _inject_generation_defaults(data: dict) -> None:
    """
    Reasoning-distilled models loop inside <think> blocks without strong
    repetition penalties and token caps.  Inject aggressive defaults.
    """
    if data.get("repetition_penalty", 0) == 0 and data.get("frequency_penalty", 0) == 0:
        data.setdefault("repetition_penalty", _MLX_REP_PENALTY)
        data.setdefault("repetition_context_size", _MLX_REP_CONTEXT)
    if _MLX_MAX_COMPLETION_TOKENS > 0:
        cur = data.get("max_tokens") or data.get("max_completion_tokens") or 999999
        cap = min(int(cur), _MLX_MAX_COMPLETION_TOKENS)
        data["max_tokens"] = cap
    if _MLX_FORCE_NONSTREAM:
        # Only force non-streaming when explicitly requested.
        data["stream"] = False
        data.pop("stream_options", None)


def _rewrite_openai_model_field(body: bytes, backend_path: str, log) -> bytes:
    """
    Cursor custom models often use a display name as `model`, but mlx_lm expects a
    valid Hugging Face repo id for the loaded weights.
    Also injects repetition_penalty defaults on chat/completion endpoints.
    """
    if backend_path not in ("/v1/chat/completions", "/v1/completions", "/chat/completions"):
        return body
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(data, dict):
        return body

    changed = False

    _inject_generation_defaults(data)

    # Continue can send tool definitions and expect structured tool-call JSON in
    # the response. This particular model frequently emits tool-like markup as
    # plain text (e.g. "tool ... BEGIN ARG ...") which causes Continue to try
    # JSON-parsing and fail. Default to stripping tool-calling fields.
    if os.environ.get("MLX_STRIP_TOOLS", "1") != "0":
        if "tools" in data:
            data.pop("tools", None)
            changed = True
        if "tool_choice" in data:
            data.pop("tool_choice", None)
            changed = True

    if os.environ.get("MLX_CHAT_MODEL_REWRITE", "1") != "0":
        target = os.environ.get(
            "MLX_CHAT_MODEL_ID",
            "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled",
        ).strip()
        old = data.get("model", "")
        if target and isinstance(old, str) and old != target:
            data["model"] = target
            log("rewrote JSON model field %r -> %r", old, target)
            changed = True

    return json.dumps(data, ensure_ascii=False).encode("utf-8")


_MIN_KEEP_CHARS = 200


def _truncate_at_degeneration(text: str) -> str:
    """
    Walk through the text in overlapping windows.  At the first window that
    triggers the loop detector, truncate back to the last sentence boundary
    before it.  Always keep at least _MIN_KEEP_CHARS of text.
    """
    words = text.split()
    if len(words) < 120:
        return text
    window = 120
    for end in range(window, len(words), 10):
        segment = " ".join(words[max(0, end - window) : end])
        if _detect_degenerate_loop(segment):
            cut_point = max(0, end - window)
            good_text = " ".join(words[:cut_point])
            if len(good_text) < _MIN_KEEP_CHARS:
                good_text = text[:_MIN_KEEP_CHARS]
            last_sentence = max(
                good_text.rfind(". "),
                good_text.rfind(".\n"),
                good_text.rfind("? "),
                good_text.rfind("! "),
            )
            if last_sentence > _MIN_KEEP_CHARS // 2:
                return good_text[: last_sentence + 1]
            return good_text
    return text


def _postprocess_nonstreaming(body: bytes, log) -> bytes:
    """
    For non-streaming chat/completion responses, detect and truncate degenerate
    content/reasoning fields before sending to the client.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(data, dict):
        return body
    changed = False
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        for field in ("content", "reasoning"):
            val = msg.get(field)
            if val and isinstance(val, str) and len(val) > 200:
                truncated = _truncate_at_degeneration(val)
                if len(truncated) < len(val):
                    msg[field] = truncated
                    choice["finish_reason"] = "stop"
                    changed = True
                    log(
                        "post-process: truncated %s from %d to %d chars (degeneration detected)",
                        field, len(val), len(truncated),
                    )
        # Some backends put almost everything in `reasoning`. Continue expects
        # `message.content` to be present to render anything.
        content = msg.get("content")
        reasoning = msg.get("reasoning")
        if (not content or (isinstance(content, str) and not content.strip())) and isinstance(reasoning, str) and reasoning.strip():
            msg["content"] = _truncate_at_degeneration(reasoning)
            msg["reasoning"] = ""
            choice["finish_reason"] = "stop"
            changed = True
    if changed:
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    return body


def _error_payload(msg: str, code: str = "context_length_exceeded") -> bytes:
    err = {
        "error": {
            "message": msg,
            "type": "invalid_request_error",
            "code": code,
        }
    }
    return json.dumps(err).encode("utf-8")


_DEDUP_TTL = float(os.environ.get("MLX_DEDUP_TTL", "10"))


class _ResponseCache:
    """Prevents client retry storms by caching recent responses by request hash."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, int, Dict[str, str], bytes]] = {}

    def _key(self, body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()[:32]

    def get(self, body: bytes) -> Optional[Tuple[int, Dict[str, str], bytes]]:
        if _DEDUP_TTL <= 0:
            return None
        key = self._key(body)
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.monotonic() - entry[0]) < _DEDUP_TTL:
                return entry[1], entry[2], entry[3]
            return None

    def put(self, body: bytes, status: int, headers: Dict[str, str], resp_body: bytes) -> None:
        if _DEDUP_TTL <= 0:
            return
        key = self._key(body)
        with self._lock:
            self._cache[key] = (time.monotonic(), status, headers, resp_body)
            cutoff = time.monotonic() - _DEDUP_TTL * 3
            stale = [k for k, v in self._cache.items() if v[0] < cutoff]
            for k in stale:
                del self._cache[k]


_response_cache = _ResponseCache()


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

        is_stream = False
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
            try:
                req_data = json.loads(body.decode("utf-8"))
                is_stream = bool(req_data.get("stream", False))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

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

        if (not is_stream and body is not None
                and backend_path in ("/v1/chat/completions", "/v1/completions", "/chat/completions")):
            cached = _response_cache.get(body)
            if cached is not None:
                c_status, c_headers, c_body = cached
                self.log_message("dedup: returning cached response (%d bytes, ttl=%ss)", len(c_body), _DEDUP_TTL)
                self.send_response(c_status)
                # Continue expects JSON for /v1/chat/completions. Ensure Content-Type.
                if not any(k.lower() == "content-type" for k in c_headers):
                    self.send_header("Content-Type", "application/json")
                for hk, hv in c_headers.items():
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(c_body)))
                self.end_headers()
                try:
                    self.wfile.write(c_body)
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
            if not is_stream and backend_path in ("/v1/chat/completions", "/v1/completions", "/chat/completions"):
                raw_body = resp.read()
                resp.close()
                cleaned = _postprocess_nonstreaming(raw_body, self.log_message)
                resp_headers: Dict[str, str] = {}
                for hk, hv in resp.getheaders():
                    if hk.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    resp_headers[hk] = hv
                if not any(k.lower() == "content-type" for k in resp_headers):
                    resp_headers["Content-Type"] = "application/json"
                if body is not None:
                    _response_cache.put(body, resp.status, resp_headers, cleaned)
                self.send_response(resp.status)
                for hk, hv in resp_headers.items():
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(cleaned)))
                self.end_headers()
                try:
                    self.wfile.write(cleaned)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(resp.status)
                for hk, hv in resp.getheaders():
                    if hk.lower() in ("transfer-encoding", "connection"):
                        continue
                    self.send_header(hk, hv)
                self.end_headers()
                _pump_backend_to_client(self, resp, is_stream=is_stream)
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
        f"  anti-loop: repetition_penalty={_MLX_REP_PENALTY}, "
        f"context_size={_MLX_REP_CONTEXT}, "
        f"max_completion_tokens={_MLX_MAX_COMPLETION_TOKENS}, "
        f"loop_detect=window:{_LOOP_WINDOW}/ngram:{_LOOP_NGRAM}/threshold:{_LOOP_THRESHOLD}\n"
    )
    sys.stderr.write(
        f"Forwarding to http://{os.environ.get('MLX_BACKEND_HOST', '127.0.0.1')}:"
        f"{os.environ.get('MLX_BACKEND_PORT', '18080')}\n"
    )

    httpd = ThreadingHTTPServer((listen_host, listen_port), ProxyHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
