"""Renderers for a CapacityReport. Add a format by implementing Renderer."""

from __future__ import annotations

import dataclasses
import json
from typing import Protocol

from .model import GiB, KiB, MiB
from .report import CapacityReport

WIDTH = 78


class Renderer(Protocol):
    def render(self, report: CapacityReport) -> str: ...


def _tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _scenario_label(result) -> str:
    ids = "+".join(result.scenario.register_ids)
    return f"{ids:<10s}{result.scenario.label}"


class TextRenderer:
    def render(self, report: CapacityReport) -> str:
        sections = [
            self._header(),
            self._architecture(report),
            self._derived(report),
            self._hypothesis(report),
            self._scenarios(report),
            self._cross_check(report),
            self._admission(report),
        ]
        if report.sweep:
            sections.append(self._sweep(report))
        return "\n".join(sections) + "\n" + "=" * WIDTH

    def _header(self) -> str:
        return "\n".join([
            "=" * WIDTH,
            "KIMI-K3 KV CAPACITY MODEL".center(WIDTH),
            "docs/SYSTEM-DESIGN.md 2".center(WIDTH),
            "=" * WIDTH,
        ])

    def _architecture(self, report: CapacityReport) -> str:
        arch = report.model.architecture
        par = report.model.parallelism
        return "\n".join([
            "\nARCHITECTURE",
            f"  layers                    {arch.total_layers}"
            f"  ({arch.kda_layers} KDA + {arch.mla_layers} MLA)",
            f"  KDA state per sequence    {arch.kda_state_bytes / MiB:,.0f} MiB (fixed)",
            f"  MLA KV per token, /rank   {arch.mla_bytes_per_token_per_rank / KiB:,.0f} KiB",
            f"  TP size / MLA kv heads    {par.tp_size} / {par.mla_kv_heads}",
            f"  replication factor        {par.replication_factor}x",
            f"  MLA KV per token, cluster {report.model.bytes_per_token / KiB:,.0f} KiB",
        ])

    def _derived(self, report: CapacityReport) -> str:
        model = report.model
        return "\n".join([
            "\nDERIVED",
            f"  mean prompt (bucketed)    {report.mean_prompt_tokens:,.0f} tokens",
            f"  breakeven KDA vs MLA      {model.breakeven_tokens:,.0f} tokens",
            f"  cost of a 30k sequence    {model.sequence_cost(30_000) / GiB:,.2f} GiB",
            f"  cost of a 256k sequence   {model.sequence_cost(262_144) / GiB:,.2f} GiB"
            f"  ({100 * model.pool_share(262_144):.1f}% of the pool)",
        ])

    def _hypothesis(self, report: CapacityReport) -> str:
        h = report.hypothesis
        return "\n".join([
            "\nHYPOTHESIS TEST",
            f"  measured pool             {h.measured_tokens:,} tokens",
            f"  predicted if REPLICATED   {h.predicted_replicated:,}"
            f"  (error {100 * h.error_replicated:.1f}%)",
            f"  predicted if DE-DUPED     {h.predicted_deduplicated:,}"
            f"  (error {100 * h.error_deduplicated:.1f}%)",
            f"  unexplained residual      {h.residual_bytes / GiB:,.1f} GiB"
            f"  = {h.residual_as_kda_sequences:,.0f} sequences of KDA state",
            f"  VERDICT                   {h.verdict}",
        ])

    def _scenarios(self, report: CapacityReport) -> str:
        lines = [
            "\nSCENARIOS",
            f"  {'':44s} {'resident':>9s} {'+L2':>9s} {'total':>9s} {'x':>6s} {'conc':>6s}",
        ]
        for result in report.scenarios:
            lines.append(
                f"  {_scenario_label(result):44s} "
                f"{_tokens(result.resident_tokens):>9s} "
                f"{_tokens(result.tier_tokens):>9s} "
                f"{_tokens(result.addressable_tokens):>9s} "
                f"{result.multiplier:>5.1f}x "
                f"{result.max_concurrency:>6d}"
            )
        return "\n".join(lines)

    def _cross_check(self, report: CapacityReport) -> str:
        run_lo, run_hi = report.observed_running
        q_lo, q_hi = report.observed_queued
        return "\n".join([
            "\nCROSS-CHECK AGAINST PRODUCTION",
            f"  model says sustainable    {report.modelled_concurrency} sequences"
            f" @ {report.mean_prompt_tokens:,.0f} tok",
            f"  observed running          {run_lo}-{run_hi}",
            f"  observed queued           {q_lo}-{q_hi}"
            f"  (total in system {run_lo + q_lo}-{run_hi + q_hi})",
            f"  configured max-num-seqs   {report.configured_max_num_seqs}"
            f"  ({report.overcommit_factor:.1f}x sustainable)",
        ])

    def _admission(self, report: CapacityReport) -> str:
        lines = ["\nADMISSION CEILING BY SCENARIO"]
        for result in report.scenarios:
            lines.append(f"  {_scenario_label(result):44s} {result.max_concurrency:>6d}")
        return "\n".join(lines)

    def _sweep(self, report: CapacityReport) -> str:
        names = list(report.sweep[0].concurrency_by_scenario)
        header = f"\n{'prompt':>9s} | " + " ".join(f"{n:>9s}" for n in names)
        lines = ["\nCONCURRENCY VS PROMPT LENGTH", header, "-" * len(header)]
        for row in report.sweep:
            cells = " ".join(f"{row.concurrency_by_scenario[n]:>9d}" for n in names)
            lines.append(f"{row.tokens:>9,} | {cells}")
        return "\n".join(lines)


class JsonRenderer:
    def render(self, report: CapacityReport) -> str:
        model = report.model
        payload = {
            "architecture": dataclasses.asdict(model.architecture),
            "parallelism": {
                **dataclasses.asdict(model.parallelism),
                "replication_factor": model.parallelism.replication_factor,
            },
            "bytes_per_token": model.bytes_per_token,
            "breakeven_tokens": model.breakeven_tokens,
            "mean_prompt_tokens": report.mean_prompt_tokens,
            "modelled_concurrency": report.modelled_concurrency,
            "configured_max_num_seqs": report.configured_max_num_seqs,
            "overcommit_factor": report.overcommit_factor,
            "hypothesis": dataclasses.asdict(report.hypothesis),
            "scenarios": [
                {
                    "label": r.scenario.label,
                    "register_ids": list(r.scenario.register_ids),
                    "resident_tokens": r.resident_tokens,
                    "tier_tokens": r.tier_tokens,
                    "addressable_tokens": r.addressable_tokens,
                    "multiplier": r.multiplier,
                    "max_concurrency": r.max_concurrency,
                    "bytes_per_token": r.bytes_per_token,
                }
                for r in report.scenarios
            ],
            "sweep": [
                {"tokens": row.tokens, **row.concurrency_by_scenario}
                for row in report.sweep
            ],
        }
        return json.dumps(payload, indent=2)
