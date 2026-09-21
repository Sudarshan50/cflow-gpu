"""Optimisation scenarios and their evaluated capacities."""

from __future__ import annotations

from dataclasses import dataclass

from .model import CapacityModel, PrefixSharing

CPU_TIER_512GB = 512 * 10**9


@dataclass(frozen=True)
class Scenario:
    label: str
    register_ids: tuple[str, ...]
    deduplicated: bool = False
    fp8_kv: bool = False
    host_tier_bytes: int = 0

    def apply(self, base: CapacityModel) -> CapacityModel:
        return base.with_variant(
            deduplicated=self.deduplicated,
            fp8_kv=self.fp8_kv,
            host_tier_bytes=self.host_tier_bytes,
        )


REGISTRY: tuple[Scenario, ...] = (
    Scenario("Today (TP8 replicated, bf16 KV)", ()),
    Scenario("fp8 KV", ("A2",), fp8_kv=True),
    Scenario("512 GB CPU L2 tier", ("A3",), host_tier_bytes=CPU_TIER_512GB),
    Scenario("KV de-duplication", ("A1",), deduplicated=True),
    Scenario("de-duplication + fp8 KV", ("A1", "A2"), deduplicated=True, fp8_kv=True),
    Scenario(
        "de-duplication + fp8 + L2 tier",
        ("A1", "A2", "A3"),
        deduplicated=True,
        fp8_kv=True,
        host_tier_bytes=CPU_TIER_512GB,
    ),
)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    resident_tokens: int
    tier_tokens: int
    addressable_tokens: int
    multiplier: float
    max_concurrency: int
    recommended_admission: int
    bytes_per_token: int
    per_rank_resident_tokens: int
    aggregate_unique_tokens: int


def evaluate(
    base: CapacityModel,
    mean_tokens: int,
    observed_peak: int = 0,
    sharing: PrefixSharing | None = None,
) -> list[ScenarioResult]:
    baseline = base.addressable_tokens()
    results = []
    for scenario in REGISTRY:
        model = scenario.apply(base)
        addressable = model.addressable_tokens()
        floor = model.max_concurrency(mean_tokens, sharing=None)
        results.append(
            ScenarioResult(
                scenario=scenario,
                resident_tokens=model.resident_tokens(),
                tier_tokens=model.tier_tokens,
                addressable_tokens=addressable,
                multiplier=addressable / baseline,
                max_concurrency=floor,
                recommended_admission=model.recommended_admission(
                    mean_tokens, observed_peak, sharing
                ),
                bytes_per_token=model.bytes_per_token,
                per_rank_resident_tokens=model.per_rank_resident_tokens(),
                aggregate_unique_tokens=model.aggregate_unique_tokens(),
            )
        )
    return results
