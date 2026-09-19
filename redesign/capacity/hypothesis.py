"""Tests the replication hypothesis against the engine-reported pool size.

docs/SYSTEM-DESIGN.md 2.2. This is the pre-GPU deliverable: it decides whether
A1 is worth chasing, without booting the box.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import CapacityModel

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

    @property
    def confirmed(self) -> bool:
        return self.verdict == REPLICATED


def evaluate(base: CapacityModel, measured_tokens: int) -> HypothesisResult:
    replicated = base.with_variant(deduplicated=False)
    deduplicated = base.with_variant(deduplicated=True)

    predicted_replicated = replicated.resident_tokens()
    predicted_deduplicated = deduplicated.resident_tokens()

    error_replicated = abs(predicted_replicated - measured_tokens) / measured_tokens
    error_deduplicated = abs(predicted_deduplicated - measured_tokens) / measured_tokens

    # Residual under the replicated hypothesis should be explainable as KDA
    # state the engine reserved up front. An implausible figure means the model
    # is wrong elsewhere, so it is reported rather than fitted away.
    residual = replicated.allocation.hbm_bytes - measured_tokens * replicated.bytes_per_token

    return HypothesisResult(
        measured_tokens=measured_tokens,
        predicted_replicated=predicted_replicated,
        predicted_deduplicated=predicted_deduplicated,
        error_replicated=error_replicated,
        error_deduplicated=error_deduplicated,
        residual_bytes=residual,
        residual_as_kda_sequences=residual / base.architecture.kda_state_bytes,
        verdict=REPLICATED if error_replicated < error_deduplicated else NOT_REPLICATED,
    )
