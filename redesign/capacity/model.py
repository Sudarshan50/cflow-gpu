"""KV capacity arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, replace

KiB = 1024
MiB = 1024**2
GiB = 1024**3


@dataclass(frozen=True)
class Architecture:
    total_layers: int
    kda_layers: int
    mla_layers: int
    kda_state_bytes: int
    mla_bytes_per_token_per_rank: int

    def __post_init__(self) -> None:
        if self.kda_layers + self.mla_layers != self.total_layers:
            raise ValueError("layer counts must sum to total_layers")


@dataclass(frozen=True)
class Parallelism:
    tp_size: int
    mla_kv_heads: int = 1
    deduplicated: bool = False

    @property
    def replication_factor(self) -> int:
        if self.deduplicated:
            return 1
        return max(1, self.tp_size // self.mla_kv_heads)


@dataclass(frozen=True)
class Precision:
    fp8_kv: bool = False

    @property
    def kv_divisor(self) -> int:
        return 2 if self.fp8_kv else 1


@dataclass(frozen=True)
class Allocation:
    # Cluster-wide HBM; per-rank views divide by ranks.
    hbm_bytes: int
    host_tier_bytes: int = 0
    ranks: int = 8

    @property
    def hbm_bytes_per_rank(self) -> int:
        return self.hbm_bytes // max(1, self.ranks)


@dataclass(frozen=True)
class PrefixSharing:
    # None means unknown; callers must not invent a fraction.
    unique_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.unique_fraction is None:
            return
        if not 0.0 < self.unique_fraction <= 1.0:
            raise ValueError("unique_fraction must be in (0, 1]")

    @property
    def modelled(self) -> bool:
        return self.unique_fraction is not None

    def billed_tokens(self, tokens: int) -> int:
        if self.unique_fraction is None:
            return tokens
        return max(1, int(tokens * self.unique_fraction))


@dataclass(frozen=True)
class PromptDistribution:
    # (upper_bound_tokens, cumulative_share)
    buckets: tuple[tuple[int, float], ...]

    @property
    def mean_tokens(self) -> float:
        total = 0.0
        prev_bound, prev_share = 0, 0.0
        for bound, cumulative in self.buckets:
            share = cumulative - prev_share
            total += share * (prev_bound + bound) / 2
            prev_bound, prev_share = bound, cumulative
        return total


class CapacityModel:
    def __init__(
        self,
        architecture: Architecture,
        parallelism: Parallelism,
        precision: Precision,
        allocation: Allocation,
    ) -> None:
        self.architecture = architecture
        self.parallelism = parallelism
        self.precision = precision
        self.allocation = allocation

    def with_variant(
        self,
        *,
        deduplicated: bool | None = None,
        fp8_kv: bool | None = None,
        host_tier_bytes: int | None = None,
    ) -> "CapacityModel":
        parallelism = self.parallelism
        if deduplicated is not None:
            parallelism = replace(parallelism, deduplicated=deduplicated)

        precision = self.precision
        if fp8_kv is not None:
            precision = replace(precision, fp8_kv=fp8_kv)

        allocation = self.allocation
        if host_tier_bytes is not None:
            allocation = replace(allocation, host_tier_bytes=host_tier_bytes)

        return CapacityModel(self.architecture, parallelism, precision, allocation)

    @property
    def bytes_per_token_per_rank(self) -> int:
        return self.architecture.mla_bytes_per_token_per_rank // self.precision.kv_divisor

    @property
    def bytes_per_token(self) -> int:
        # Cluster view. Under replication the 8× cancels aggregate HBM, so
        # resident_tokens() equals the per-rank pool; under de-duplication it
        # is the 8-rank unique capacity.
        return (
            self.bytes_per_token_per_rank * self.parallelism.replication_factor
        )

    @property
    def breakeven_tokens(self) -> float:
        return self.architecture.kda_state_bytes / self.bytes_per_token

    def sequence_cost(self, tokens: int) -> int:
        return self.architecture.kda_state_bytes + self.bytes_per_token * tokens

    def resident_tokens(self, reserved_sequences: int = 0) -> int:
        usable = self.allocation.hbm_bytes - reserved_sequences * self.architecture.kda_state_bytes
        return max(0, usable // self.bytes_per_token)

    def per_rank_resident_tokens(self, reserved_sequences: int = 0) -> int:
        # One worker's "GPU KV cache size" line. The 8× win is aggregate-only.
        usable = (
            self.allocation.hbm_bytes_per_rank
            - reserved_sequences * self.architecture.kda_state_bytes
        )
        return max(0, usable // self.bytes_per_token_per_rank)

    def aggregate_unique_tokens(self, reserved_sequences: int = 0) -> int:
        per_rank = self.per_rank_resident_tokens(reserved_sequences)
        if self.parallelism.deduplicated:
            return per_rank * self.parallelism.tp_size
        return per_rank

    @property
    def tier_tokens(self) -> int:
        return self.allocation.host_tier_bytes // self.bytes_per_token

    def addressable_tokens(self, reserved_sequences: int = 0) -> int:
        return self.resident_tokens(reserved_sequences) + self.tier_tokens

    def max_concurrency(
        self,
        mean_tokens: int,
        sharing: PrefixSharing | None = None,
    ) -> int:
        # Zero-sharing result is a floor, not an admission target.
        billed = sharing.billed_tokens(mean_tokens) if sharing else mean_tokens
        return self.allocation.hbm_bytes // self.sequence_cost(billed)

    def recommended_admission(
        self,
        mean_tokens: int,
        observed_peak: int,
        sharing: PrefixSharing | None = None,
    ) -> int:
        modelled = self.max_concurrency(mean_tokens, sharing)
        if sharing is not None and sharing.modelled:
            return modelled
        return max(modelled, observed_peak)

    def pool_share(self, tokens: int) -> float:
        return self.sequence_cost(tokens) / self.allocation.hbm_bytes
