"""P3 distillation lane against the local engine."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from redesign.gateway.backpressure import CircuitBreaker
from redesign.gateway.classification import BATCH
from redesign.gateway.engine import EngineClient


@dataclass(frozen=True)
class DistillSample:
    sample_id: str
    prompt: str


@dataclass(frozen=True)
class DistillResult:
    sample_id: str
    ok: bool
    text: str = ""
    error: str = ""


class Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.completed: set[str] = set()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("ok"):
                    self.completed.add(row["sample_id"])

    def record(self, result: DistillResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result)) + "\n")
        if result.ok:
            self.completed.add(result.sample_id)


def load_prompts(path: Path) -> list[DistillSample]:
    samples = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        samples.append(DistillSample(str(row.get("id", line_no)), row["prompt"]))
    return samples


class DistillRunner:
    def __init__(
        self,
        engine: EngineClient,
        breaker: CircuitBreaker,
        checkpoint: Checkpoint,
        output: Path,
        priority: int = 3,
        poll_seconds: float = 5.0,
        model: str = "default",
    ) -> None:
        self.engine = engine
        self.breaker = breaker
        self.checkpoint = checkpoint
        self.output = output
        self.priority = priority
        self.poll_seconds = poll_seconds
        self.model = model
        self.model = model

    def run(self, samples: list[DistillSample]) -> int:
        written = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", encoding="utf-8") as handle:
            for sample in samples:
                if sample.sample_id in self.checkpoint.completed:
                    continue
                self._wait_until_quiet()
                result = self._one(sample)
                self.checkpoint.record(result)
                handle.write(json.dumps(asdict(result)) + "\n")
                handle.flush()
                written += 1
        return written

    def _wait_until_quiet(self) -> None:
        while True:
            verdict = self.breaker.should_shed(BATCH)
            if not verdict.distressed:
                return
            time.sleep(self.poll_seconds)

    def _one(self, sample: DistillSample) -> DistillResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": sample.prompt}],
            "priority": self.priority,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        try:
            response = self.engine.proxy("/v1/chat/completions", payload, stream=False)
            body = b"".join(response.body)
            response.close()
            if response.status >= 400:
                return DistillResult(sample.sample_id, False, error=f"HTTP {response.status}")
            parsed = json.loads(body)
            text = parsed["choices"][0]["message"]["content"]
            return DistillResult(sample.sample_id, True, text=text)
        except (OSError, KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            return DistillResult(sample.sample_id, False, error=str(exc))
