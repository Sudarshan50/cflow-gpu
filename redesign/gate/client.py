"""Minimal OpenAI-compatible client for the gate. Stdlib only."""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_seconds: float


class CompletionError(RuntimeError):
    pass


class GateClient:
    def __init__(
        self,
        base_url: str,
        model: str = "default",
        api_key: str | None = None,
        timeout: int = 900,
    ) -> None:
        parsed = urlparse(base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = 1234,
    ) -> Completion:
        """Greedy by default. Every check depends on temperature 0."""
        import time

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started = time.monotonic()
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps(payload).encode(), headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status != 200:
                raise CompletionError(
                    f"HTTP {response.status}: {raw[:500].decode('utf-8', 'replace')}"
                )
        except OSError as exc:
            raise CompletionError(f"transport: {exc}") from exc
        finally:
            conn.close()

        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            usage = body.get("usage", {})
            # /v1/completions returns "text"; chat returns "message".content.
            # Reading either outside this guard turns a malformed response into
            # a crashed gate run instead of a failed check.
            if "message" in choice:
                text = choice["message"].get("content") or ""
            else:
                text = choice.get("text") or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise CompletionError(f"malformed response: {exc}") from exc

        return Completion(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            latency_seconds=time.monotonic() - started,
        )
