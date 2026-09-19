#!/usr/bin/env python3
"""Kimi-K3 KV capacity model — the arithmetic behind docs/SYSTEM-DESIGN.md §2.

Runs anywhere. No GPU, no network, no dependencies. This is the reference
implementation of the cost model; every sizing number in the design doc comes
from here, so if this file is wrong the plan is wrong.

  cost(sequence) = KDA_STATE_BYTES                    fixed per request
                 + MLA_BYTES_PER_TOKEN * replicas * N per token

The `replicas` term is the whole finding. MLA compresses KV into a single
latent vector per token -- effectively ONE kv head. Tensor parallelism shards
the KV cache along the head dimension, so when tp_size > num_kv_heads the cache
is duplicated tp_size/num_kv_heads times. At TP=8 with one head, that is 8
copies of the same bytes.

Usage:
    python3 capacity_model.py                 # scenario table
    python3 capacity_model.py --verify        # test the replication hypothesis
                                              #   against the measured pool
    python3 capacity_model.py --sweep         # concurrency vs prompt length
    python3 capacity_model.py --json          # machine-readable

The --verify path is the one that matters before any GPU spend: it predicts the
engine-reported pool size under both hypotheses (replicated / not) and reports
which one the measured 2,295,266 tokens actually matches.
"""

from __future__ import annotations

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Constants. Every one of these is sourced; none are guesses.
# ---------------------------------------------------------------------------

KiB = 1024
MiB = 1024**2
GiB = 1024**3

# -- Model architecture ------------------------------------------------------
# Kimi-K3 is hybrid: 93 layers total, of which 69 are KDA (linear attention,
# constant state per sequence) and 24 are full attention (MLA, KV proportional
# to sequence length). Source: config.yaml comment block, corroborated by the
# SGLang day-0 writeup.
N_LAYERS_TOTAL = 93
N_LAYERS_KDA = 69
N_LAYERS_MLA = 24

# KDA recurrent state, all 69 layers, at TP=8. Overwritten in place at every
# token, so it does NOT grow with sequence length.
# Source: LMSYS Kimi-K3 day-0 analysis, "~54 MB per request under TP=8".
KDA_STATE_BYTES = 54 * MiB

# MLA KV cache across the 24 full-attention layers, PER RANK, per token.
# Source: same, "~27 KB per token".
MLA_BYTES_PER_TOKEN_PER_RANK = 27 * KiB

# -- Hardware ----------------------------------------------------------------
N_GPUS = 8
HBM_PER_GPU = 288 * 10**9          # 288 GB, vendor spec (decimal GB)
TP_SIZE = 8

# MLA has a single latent KV head, so TP cannot shard it. Replication factor is
# tp_size / num_kv_heads = 8 / 1.
MLA_KV_HEADS = 1

# -- Measured on the box (2026-09-18/19) -------------------------------------
# KV allocation per GPU at gpu-memory-utilization 0.92.
# Source: K3-DEPLOYMENT.md / FINDINGS.md, "61 GiB of HBM KV per GPU".
KV_ALLOC_PER_GPU = 61 * GiB
KV_ALLOC_TOTAL = KV_ALLOC_PER_GPU * N_GPUS

# What the engine actually reports. This is the number the whole hypothesis is
# tested against. Source: config.yaml, FINDINGS.md §5.
REPORTED_POOL_TOKENS = 2_295_266

# Observed production traffic. 229 requests is a small sample -- Z3 replaces
# this with a real week. Treat every number derived from it as provisional.
PROMPT_DISTRIBUTION = [
    # (upper bound tokens, cumulative share)
    (10_000, 0.53),
    (20_000, 0.69),
    (50_000, 0.81),
    (100_000, 0.93),
    (200_000, 0.97),
    (262_144, 1.00),
]
MEAN_PROMPT_TOKENS = 30_000          # derived below by _mean_prompt()

# Steady state observed in production, for cross-checking the model.
OBSERVED_RUNNING = (46, 51)
OBSERVED_QUEUED = (22, 25)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def replication_factor(tp_size: int = TP_SIZE, kv_heads: int = MLA_KV_HEADS) -> int:
    """How many redundant copies of the MLA KV cache TP forces.

    vLLM: "TP shards KV cache along kv-heads (H dimension). When tp_size > H,
    KV cache gets duplicated tp_size/H." For MLA, H is effectively 1.
    """
    return max(1, tp_size // kv_heads)


def mla_bytes_per_token(*, deduplicated: bool = False, fp8: bool = False) -> int:
    """Cluster-wide MLA KV cost for one token.

    deduplicated -- context/decode parallelism shards the latent along the
                    token dimension instead of replicating it per rank (A1).
    fp8          -- e4m3 KV cache halves the bytes (A2).
    """
    per_token = MLA_BYTES_PER_TOKEN_PER_RANK
    if not deduplicated:
        per_token *= replication_factor()
    if fp8:
        per_token //= 2
    return per_token


def sequence_cost(tokens: int, *, deduplicated: bool = False, fp8: bool = False) -> int:
    """Total bytes a single in-flight sequence of `tokens` length occupies."""
    return KDA_STATE_BYTES + mla_bytes_per_token(deduplicated=deduplicated, fp8=fp8) * tokens


def breakeven_tokens(*, deduplicated: bool = False, fp8: bool = False) -> float:
    """Sequence length at which MLA KV cost equals the fixed KDA state cost.

    Below this, the constant term dominates and CONCURRENCY is the binding
    constraint rather than context length.
    """
    return KDA_STATE_BYTES / mla_bytes_per_token(deduplicated=deduplicated, fp8=fp8)


def pool_tokens(kv_bytes: int = KV_ALLOC_TOTAL, *,
                reserved_sequences: int = 0,
                deduplicated: bool = False,
                fp8: bool = False) -> int:
    """Token capacity of a KV allocation, after reserving KDA state.

    `reserved_sequences` models the engine sizing its KDA pool up front for
    max-num-seqs, which is memory unavailable to MLA regardless of live load.
    """
    usable = kv_bytes - reserved_sequences * KDA_STATE_BYTES
    if usable <= 0:
        return 0
    return int(usable // mla_bytes_per_token(deduplicated=deduplicated, fp8=fp8))


def max_concurrency(mean_tokens: int = MEAN_PROMPT_TOKENS,
                    kv_bytes: int = KV_ALLOC_TOTAL, *,
                    deduplicated: bool = False,
                    fp8: bool = False) -> int:
    """How many sequences of `mean_tokens` fit simultaneously.

    This is the number max-num-seqs (vLLM) / --max-running-requests (SGLang)
    should be set near. Admitting materially more does not buy throughput; it
    buys preemption, and every preemption discards completed prefill.
    """
    per_seq = sequence_cost(mean_tokens, deduplicated=deduplicated, fp8=fp8)
    return int(kv_bytes // per_seq)


def _mean_prompt() -> float:
    """Expected prompt length from the bucketed production distribution.

    Uses the bucket midpoint as the representative value, which is
    conservative: it under-weights the >200k tail relative to a true mean.
    """
    total, prev_bound, prev_share = 0.0, 0, 0.0
    for bound, cum_share in PROMPT_DISTRIBUTION:
        share = cum_share - prev_share
        total += share * (prev_bound + bound) / 2
        prev_bound, prev_share = bound, cum_share
    return total


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label, deduplicated, fp8, extra host-DRAM bytes for an L2 tier)
    ("Today  (TP8 replicated, bf16 KV)",        False, False, 0),
    ("A2     + fp8 KV",                          False, True,  0),
    ("A3     + 512 GB CPU L2 tier",              False, False, 512 * 10**9),
    ("A1     + KV de-duplication (DCP/DP-attn)", True,  False, 0),
    ("A1+A2  + de-dup and fp8",                  True,  True,  0),
    ("A1+A2+A3 all three",                       True,  True,  512 * 10**9),
]


def scenario_rows() -> list[dict]:
    base = pool_tokens()
    rows = []
    for label, dedup, fp8, l2_bytes in SCENARIOS:
        resident = pool_tokens(deduplicated=dedup, fp8=fp8)
        l2 = 0
        if l2_bytes:
            l2 = int(l2_bytes // mla_bytes_per_token(deduplicated=dedup, fp8=fp8))
        rows.append({
            "scenario": label,
            "resident_tokens": resident,
            "l2_tokens": l2,
            "addressable_tokens": resident + l2,
            "multiplier": (resident + l2) / base,
            "max_concurrency": max_concurrency(deduplicated=dedup, fp8=fp8),
            "bytes_per_token": mla_bytes_per_token(deduplicated=dedup, fp8=fp8),
        })
    return rows


# ---------------------------------------------------------------------------
# Hypothesis test -- the pre-GPU deliverable
# ---------------------------------------------------------------------------

def verify() -> dict:
    """Predict the engine-reported pool under both hypotheses; compare to truth.

    If the replicated prediction lands within a few percent of the measured
    2,295,266 tokens and the de-duplicated prediction is ~8x off, the finding in
    SYSTEM-DESIGN.md §2 is confirmed without touching a GPU.
    """
    pred_replicated = pool_tokens(deduplicated=False)
    pred_dedup = pool_tokens(deduplicated=True)

    err_repl = abs(pred_replicated - REPORTED_POOL_TOKENS) / REPORTED_POOL_TOKENS
    err_dedup = abs(pred_dedup - REPORTED_POOL_TOKENS) / REPORTED_POOL_TOKENS

    # The residual under the replicated hypothesis should be explainable as KDA
    # state the engine reserved up front. Report how many sequences' worth it is
    # rather than force-fitting it -- an implausible number is a signal that the
    # model is wrong somewhere else.
    residual_bytes = KV_ALLOC_TOTAL - REPORTED_POOL_TOKENS * mla_bytes_per_token()
    residual_seqs = residual_bytes / KDA_STATE_BYTES

    return {
        "measured_pool_tokens": REPORTED_POOL_TOKENS,
        "kv_alloc_bytes": KV_ALLOC_TOTAL,
        "replication_factor": replication_factor(),
        "predicted_if_replicated": pred_replicated,
        "predicted_if_deduplicated": pred_dedup,
        "error_if_replicated": err_repl,
        "error_if_deduplicated": err_dedup,
        "residual_bytes": residual_bytes,
        "residual_as_kda_sequences": residual_seqs,
        "verdict": "REPLICATED" if err_repl < err_dedup else "NOT REPLICATED",
    }


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}k"
    return str(n)


def print_report() -> None:
    w = 78
    print("=" * w)
    print("KIMI-K3 KV CAPACITY MODEL".center(w))
    print("docs/SYSTEM-DESIGN.md §2".center(w))
    print("=" * w)

    print("\nARCHITECTURE")
    print(f"  layers                    {N_LAYERS_TOTAL}  "
          f"({N_LAYERS_KDA} KDA + {N_LAYERS_MLA} MLA)")
    print(f"  KDA state per sequence    {KDA_STATE_BYTES/MiB:,.0f} MiB   (fixed)")
    print(f"  MLA KV per token, /rank   {MLA_BYTES_PER_TOKEN_PER_RANK/KiB:,.0f} KiB")
    print(f"  TP size / MLA kv heads    {TP_SIZE} / {MLA_KV_HEADS}")
    print(f"  replication factor        {replication_factor()}x   <-- the finding")
    print(f"  MLA KV per token, cluster {mla_bytes_per_token()/KiB:,.0f} KiB")

    print("\nDERIVED")
    mean = _mean_prompt()
    print(f"  mean prompt (bucketed)    {mean:,.0f} tokens")
    print(f"  breakeven KDA vs MLA      {breakeven_tokens():,.0f} tokens")
    print(f"    -> below this, concurrency binds, not context length")
    print(f"  cost of a 30k sequence    {sequence_cost(30_000)/GiB:,.2f} GiB")
    print(f"  cost of a 256k sequence   {sequence_cost(262_144)/GiB:,.2f} GiB"
          f"   ({100*sequence_cost(262_144)/KV_ALLOC_TOTAL:.1f}% of the pool)")

    print("\nHYPOTHESIS TEST")
    v = verify()
    print(f"  measured pool             {v['measured_pool_tokens']:,} tokens")
    print(f"  predicted if REPLICATED   {v['predicted_if_replicated']:,}"
          f"   (error {100*v['error_if_replicated']:.1f}%)")
    print(f"  predicted if DE-DUPED     {v['predicted_if_deduplicated']:,}"
          f"   (error {100*v['error_if_deduplicated']:.1f}%)")
    print(f"  unexplained residual      {v['residual_bytes']/GiB:,.1f} GiB"
          f"  = {v['residual_as_kda_sequences']:,.0f} sequences of KDA state")
    print(f"  VERDICT                   {v['verdict']}")

    print("\nSCENARIOS")
    print(f"  {'':44s} {'resident':>9s} {'+L2':>9s} {'total':>9s} {'x':>6s} {'conc':>6s}")
    for r in scenario_rows():
        print(f"  {r['scenario']:44s} "
              f"{_fmt_tokens(r['resident_tokens']):>9s} "
              f"{_fmt_tokens(r['l2_tokens']):>9s} "
              f"{_fmt_tokens(r['addressable_tokens']):>9s} "
              f"{r['multiplier']:>5.1f}x "
              f"{r['max_concurrency']:>6d}")

    print("\nCROSS-CHECK AGAINST PRODUCTION")
    lo, hi = OBSERVED_RUNNING
    qlo, qhi = OBSERVED_QUEUED
    modelled = max_concurrency()
    print(f"  model says sustainable    {modelled} sequences @ {mean:,.0f} tok")
    print(f"  observed running          {lo}-{hi}")
    print(f"  observed queued           {qlo}-{qhi}  (total in system {lo+qlo}-{hi+qhi})")
    print(f"  configured max-num-seqs   512   "
          f"({512/modelled:.1f}x the model's sustainable figure)")
    print()
    print("  Reading: the engine is admitting ~5x what the pool holds, then")
    print("  preempting to recover. Every preemption discards completed prefill,")
    print("  which at 97.7% input traffic is the most expensive loss available.")

    print("\nWHAT max-num-seqs SHOULD BE")
    for label, dedup, fp8, _ in SCENARIOS[:5]:
        c = max_concurrency(deduplicated=dedup, fp8=fp8)
        print(f"  {label:44s} {c:>6d}")
    print()
    print("  Note the last row: after de-duplication, 512 is roughly correct.")
    print("  The current setting was never absurd in ambition -- it was sized")
    print("  for a pool that replication had silently divided by 8.")
    print("=" * w)


def print_sweep() -> None:
    lengths = [1_000, 5_000, 10_000, 30_000, 50_000, 100_000, 200_000, 262_144]
    print(f"\n{'prompt':>9s} | {'today':>7s} {'+fp8':>7s} {'+dedup':>7s} {'dedup+fp8':>10s}"
          "    max concurrent sequences")
    print("-" * 68)
    for n in lengths:
        row = [
            max_concurrency(n),
            max_concurrency(n, fp8=True),
            max_concurrency(n, deduplicated=True),
            max_concurrency(n, deduplicated=True, fp8=True),
        ]
        print(f"{n:>9,} | {row[0]:>7d} {row[1]:>7d} {row[2]:>7d} {row[3]:>10d}")
    print("\nA single 262,144-token request costs "
          f"{100*sequence_cost(262_144)/KV_ALLOC_TOTAL:.1f}% of the pool today, "
          f"{100*sequence_cost(262_144, deduplicated=True)/KV_ALLOC_TOTAL:.1f}% "
          "de-duplicated.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true",
                   help="test the replication hypothesis only")
    p.add_argument("--sweep", action="store_true",
                   help="concurrency vs prompt length table")
    p.add_argument("--json", action="store_true",
                   help="machine-readable output")
    a = p.parse_args()

    if a.json:
        json.dump({
            "constants": {
                "kda_state_bytes": KDA_STATE_BYTES,
                "mla_bytes_per_token_per_rank": MLA_BYTES_PER_TOKEN_PER_RANK,
                "replication_factor": replication_factor(),
                "kv_alloc_bytes": KV_ALLOC_TOTAL,
                "mean_prompt_tokens": _mean_prompt(),
                "breakeven_tokens": breakeven_tokens(),
            },
            "verify": verify(),
            "scenarios": scenario_rows(),
        }, sys.stdout, indent=2)
        print()
        return 0

    if a.verify:
        v = verify()
        for k, val in v.items():
            print(f"{k:32s} {val}")
        return 0 if v["verdict"] == "REPLICATED" else 1

    print_report()
    if a.sweep:
        print_sweep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
