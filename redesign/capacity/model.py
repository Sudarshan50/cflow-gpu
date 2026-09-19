"""Domain model for Kimi-K3 KV capacity. Pure arithmetic, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace

KiB = 1024
MiB = 1024**2
GiB = 1024**3


@dataclass(frozen=True)
class Architecture:
    """Per-layer memory characteristics of a hybrid attention model."""

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
    """Tensor-parallel layout and whether the KV latent is de-duplicated.

    MLA compresses KV into a single latent vector, so `mla_kv_heads` is 1 and
    tensor parallelism cannot shard it. The cache is duplicated
    `tp_size // mla_kv_heads` times unless a de-duplication mechanism
    (DCP, or SGLang DP attention) partitions it.
    """

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
    hbm_bytes: int
    host_tier_bytes: int = 0


@dataclass(frozen=True)
class PromptDistribution:
    """Bucketed prompt lengths as (upper_bound_tokens, cumulative_share)."""

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
    """Token capacity and concurrency for one parallelism/precision choice."""

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
    def bytes_per_token(self) -> int:
        base = self.architecture.mla_bytes_per_token_per_rank
        return base * self.parallelism.replication_factor // self.precision.kv_divisor

    @property
    def breakeven_tokens(self) -> float:
        """Length at which MLA KV cost equals the fixed KDA state cost.

        Below this, concurrency binds rather than context length.
        """
        return self.architecture.kda_state_bytes / self.bytes_per_token

    def sequence_cost(self, tokens: int) -> int:
        return self.architecture.kda_state_bytes + self.bytes_per_token * tokens

    def resident_tokens(self, reserved_sequences: int = 0) -> int:
        usable = self.allocation.hbm_bytes - reserved_sequences * self.architecture.kda_state_bytes
        return max(0, usable // self.bytes_per_token)

    @property
    def tier_tokens(self) -> int:
        return self.allocation.host_tier_bytes // self.bytes_per_token

    def addressable_tokens(self, reserved_sequences: int = 0) -> int:
        return self.resident_tokens(reserved_sequences) + self.tier_tokens

    def max_concurrency(self, mean_tokens: int) -> int:
        return self.allocation.hbm_bytes // self.sequence_cost(mean_tokens)

    def pool_share(self, tokens: int) -> float:
        return self.sequence_cost(tokens) / self.allocation.hbm_bytes
