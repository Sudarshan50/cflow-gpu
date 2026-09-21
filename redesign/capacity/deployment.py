"""Measured constants for the Kimi-K3 deployment."""

from __future__ import annotations

from .model import (
    GiB,
    KiB,
    MiB,
    Allocation,
    Architecture,
    CapacityModel,
    Parallelism,
    Precision,
    PrefixSharing,
    PromptDistribution,
)

KIMI_K3 = Architecture(
    total_layers=93,
    kda_layers=69,
    mla_layers=24,
    kda_state_bytes=54 * MiB,
    mla_bytes_per_token_per_rank=27 * KiB,
)

TP8 = Parallelism(tp_size=8, mla_kv_heads=1, deduplicated=False)

BF16_KV = Precision(fp8_kv=False)

# 61 GiB is per-rank; Allocation.hbm_bytes is the 8-GPU aggregate.
HBM_BYTES_PER_RANK = 61 * GiB
HBM_KV_ALLOCATION = Allocation(hbm_bytes=HBM_BYTES_PER_RANK * 8, ranks=8)

OBSERVED_PROMPTS = PromptDistribution(
    buckets=(
        (10_000, 0.53),
        (20_000, 0.69),
        (50_000, 0.81),
        (100_000, 0.93),
        (200_000, 0.97),
        (262_144, 1.00),
    )
)

# Worker-logged pool; per-rank, not cluster aggregate.
REPORTED_POOL_TOKENS = 2_295_266
REPORTED_POOL_SCOPE = "per-rank"

KNOWN_POOL_OBSERVATIONS = (
    (49.41, 1_169_808, "K3-DEPLOYMENT.md:111 FULL_AND_PIECEWISE"),
    (61.09, 1_696_278, "K3-DEPLOYMENT.md:521 max-model-len 12800"),
    (61.06, 2_295_266, "RUNBOOK.md:154 current"),
)

PREFIX_SHARING = PrefixSharing(unique_fraction=None)
PREFIX_SHARING_MODELLED = PREFIX_SHARING.modelled
OBSERVED_PEAK_CONCURRENCY = 427

CONFIGURED_MAX_NUM_SEQS = 512

OBSERVED_RUNNING = (46, 51)
OBSERVED_QUEUED = (22, 25)


def production_model() -> CapacityModel:
    return CapacityModel(KIMI_K3, TP8, BF16_KV, HBM_KV_ALLOCATION)
