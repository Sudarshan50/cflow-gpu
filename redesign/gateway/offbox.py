"""Off-box OpenAI-compatible client and outage-fallback policy."""

from __future__ import annotations

from .engine import EngineClient, ProxyResponse
from .models import Priority

FALLBACK_PRIORITIES = (Priority.INTERACTIVE, Priority.SHORT_CHAT)


class OffBoxClient:
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        model: str = "offbox",
    ) -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.engine = EngineClient(url, extra_headers=headers, snapshot_ttl=0.0)
        self.model = model
        self.url = url

    def proxy(self, path: str, payload: dict, stream: bool) -> ProxyResponse:
        forwarded = dict(payload)
        forwarded["model"] = self.model
        forwarded.pop("priority", None)
        return self.engine.proxy(path, forwarded, stream=stream)

    def healthy(self) -> bool:
        return self.engine.healthy()


def should_fallback(priority: Priority) -> bool:
    return priority in FALLBACK_PRIORITIES
