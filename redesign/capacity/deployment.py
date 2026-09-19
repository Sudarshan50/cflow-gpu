"""Measured constants for the cflox.store Kimi-K3 deployment.

Every value is sourced. Changing one changes the plan, so each carries its
provenance rather than a justification.
"""

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
    PromptDistribution,
)

# LMSYS Kimi-K3 day-0 analysis; layer split corroborated by config.yaml.
KIMI_K3 = Architecture(
    total_layers=93,
    kda_layers=69,
    mla_layers=24,
    kda_state_bytes=54 * MiB,
    mla_bytes_per_token_per_rank=27 * KiB,
)

# config.yaml: tensor-parallel-size 8.
TP8 = Parallelism(tp_size=8, mla_kv_heads=1, deduplicated=False)

BF16_KV = Precision(fp8_kv=False)

# K3-DEPLOYMENT.md / FINDINGS.md: 61 GiB of HBM KV per GPU at
# gpu-memory-utilization 0.92, across 8 GPUs.
HBM_KV_ALLOCATION = Allocation(hbm_bytes=61 * GiB * 8)

# FINDINGS.md 1, from a 229-request sample. Provisional until Z3 replaces it
# with a full week of production traffic.
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

# config.yaml / FINDINGS.md 5: what the engine reports today.
REPORTED_POOL_TOKENS = 2_295_266

# config.yaml.
CONFIGURED_MAX_NUM_SEQS = 512

# FINDINGS.md 1: production steady state.
OBSERVED_RUNNING = (46, 51)
OBSERVED_QUEUED = (22, 25)


def production_model() -> CapacityModel:
    return CapacityModel(KIMI_K3, TP8, BF16_KV, HBM_KV_ALLOCATION)
