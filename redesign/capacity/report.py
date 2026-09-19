"""Assembles a capacity report. Data only; rendering lives in renderers.py."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import deployment, hypothesis, scenarios
from .model import CapacityModel
from .scenarios import ScenarioResult

SWEEP_LENGTHS = (1_000, 5_000, 10_000, 30_000, 50_000, 100_000, 200_000, 262_144)


@dataclass(frozen=True)
class SweepRow:
    tokens: int
    concurrency_by_scenario: dict[str, int]


@dataclass(frozen=True)
class CapacityReport:
    model: CapacityModel
    mean_prompt_tokens: float
    hypothesis: hypothesis.HypothesisResult
    scenarios: list[ScenarioResult]
    sweep: list[SweepRow] = field(default_factory=list)

    configured_max_num_seqs: int = deployment.CONFIGURED_MAX_NUM_SEQS
    observed_running: tuple[int, int] = deployment.OBSERVED_RUNNING
    observed_queued: tuple[int, int] = deployment.OBSERVED_QUEUED

    @property
    def modelled_concurrency(self) -> int:
        return self.model.max_concurrency(int(self.mean_prompt_tokens))

    @property
    def overcommit_factor(self) -> float:
        return self.configured_max_num_seqs / self.modelled_concurrency


_SWEEP_VARIANTS = {
    "today": {},
    "fp8": {"fp8_kv": True},
    "dedup": {"deduplicated": True},
    "dedup+fp8": {"deduplicated": True, "fp8_kv": True},
}


def _build_sweep(base: CapacityModel) -> list[SweepRow]:
    rows = []
    for tokens in SWEEP_LENGTHS:
        rows.append(
            SweepRow(
                tokens=tokens,
                concurrency_by_scenario={
                    name: base.with_variant(**variant).max_concurrency(tokens)
                    for name, variant in _SWEEP_VARIANTS.items()
                },
            )
        )
    return rows


def build(include_sweep: bool = False) -> CapacityReport:
    base = deployment.production_model()
    mean_tokens = deployment.OBSERVED_PROMPTS.mean_tokens

    return CapacityReport(
        model=base,
        mean_prompt_tokens=mean_tokens,
        hypothesis=hypothesis.evaluate(base, deployment.REPORTED_POOL_TOKENS),
        scenarios=scenarios.evaluate(base, int(mean_tokens)),
        sweep=_build_sweep(base) if include_sweep else [],
    )
