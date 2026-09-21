"""Replication hypothesis against the engine-reported pool."""

from __future__ import annotations

from dataclasses import dataclass

from .model import GiB, KiB, CapacityModel

REPLICATED = "REPLICATED"
NOT_REPLICATED = "NOT REPLICATED"


@dataclass(frozen=True)
class HypothesisResult:
    measured_tokens: int
    predicted_replicated: int
    predicted_deduplicated: int
    error_replicated: float
    error_deduplicated: float
    residual_bytes: int
    residual_as_kda_sequences: float
    verdict: str
    scope: str = "per-rank"
    implied_kib_per_token: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.verdict == REPLICATED


@dataclass(frozen=True)
class ObservationFit:
    available_gib: float
    pool_tokens: int
    source: str
    implied_kib_per_token: float
    predicted_tokens: int
    error: float


def evaluate(base: CapacityModel, measured_tokens: int) -> HypothesisResult:
    # measured_tokens is one worker's pool. predicted_deduplicated is the
    # cluster unique total, not a per-rank line.
    replicated = base.with_variant(deduplicated=False)
    deduplicated = base.with_variant(deduplicated=True)

    predicted_replicated = replicated.per_rank_resident_tokens()
    predicted_deduplicated_aggregate = deduplicated.aggregate_unique_tokens()

    error_replicated = abs(predicted_replicated - measured_tokens) / measured_tokens
    error_deduplicated = (
        abs(predicted_deduplicated_aggregate - measured_tokens) / measured_tokens
    )

    residual = (
        replicated.allocation.hbm_bytes_per_rank
        - measured_tokens * replicated.bytes_per_token_per_rank
    )
    per_rank_hbm = replicated.allocation.hbm_bytes_per_rank
    implied = per_rank_hbm / measured_tokens / KiB if measured_tokens else 0.0

    return HypothesisResult(
        measured_tokens=measured_tokens,
        predicted_replicated=predicted_replicated,
        predicted_deduplicated=predicted_deduplicated_aggregate,
        error_replicated=error_replicated,
        error_deduplicated=error_deduplicated,
        residual_bytes=residual,
        residual_as_kda_sequences=residual / base.architecture.kda_state_bytes,
        verdict=REPLICATED if error_replicated < error_deduplicated else NOT_REPLICATED,
        scope="per-rank",
        implied_kib_per_token=implied,
    )


def fit_observation(
    available_gib: float,
    pool_tokens: int,
    source: str,
    bytes_per_token_per_rank: int,
) -> ObservationFit:
    predicted = int(available_gib * GiB) // bytes_per_token_per_rank
    implied = (available_gib * GiB) / pool_tokens / KiB if pool_tokens else 0.0
    error = abs(predicted - pool_tokens) / pool_tokens if pool_tokens else 1.0
    return ObservationFit(
        available_gib=available_gib,
        pool_tokens=pool_tokens,
        source=source,
        implied_kib_per_token=implied,
        predicted_tokens=predicted,
        error=error,
    )


def fit_known_observations(base: CapacityModel) -> tuple[ObservationFit, ...]:
    from . import deployment

    bytes_per = base.bytes_per_token_per_rank
    return tuple(
        fit_observation(available, pool, source, bytes_per)
        for available, pool, source in deployment.KNOWN_POOL_OBSERVATIONS
    )
