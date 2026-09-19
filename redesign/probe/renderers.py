"""Renderers for probe results."""

from __future__ import annotations

import dataclasses
import json
from typing import Protocol, Sequence

from .base import ProbeResult

WIDTH = 74


class Renderer(Protocol):
    def render(self, results: Sequence[ProbeResult]) -> str: ...


class TextRenderer:
    def render(self, results: Sequence[ProbeResult]) -> str:
        return "\n\n".join(self._one(r) for r in results)

    def _one(self, result: ProbeResult) -> str:
        lines = [
            "=" * WIDTH,
            f"  {result.engine.upper()}  --  A1 de-duplication capability probe",
            "=" * WIDTH,
        ]

        if not result.importable:
            lines.append(f"  NOT IMPORTABLE: {result.error}")
            lines.append("  (expected if this engine is absent from the current image)")
            return "\n".join(lines)

        lines += [
            f"  version           {result.version}",
            f"  platform          {result.platform}",
            f"  CLI flags seen    {result.cli_surface_size}",
        ]
        if result.cli_surface_error:
            lines.append(f"  parser error      {result.cli_surface_error}")

        lines.append("\n  FLAGS")
        for flag in result.flags:
            mark = "yes" if flag.present else " NO"
            lines.append(f"    [{mark}]  {flag.spec.name:32s} {flag.spec.buys}")

        lines.append("\n  HYBRID HINTS IN THE DE-DUPLICATION PATH")
        for scan in result.scans:
            if scan.status != "scanned":
                lines.append(f"    {scan.module:48s} {scan.status}")
                continue
            lines.append(f"    {scan.module:48s} {', '.join(scan.hints) or '(none)'}")

        lines += [
            "\n  VERDICT",
            f"    A1 ({' / '.join(result.a1_flags)}): "
            f"{'present in the CLI surface' if result.a1_available else 'ABSENT'}",
            f"    A2 (kv_cache_dtype): {'present' if result.a2_available else 'ABSENT'}",
            "",
            "    CLI presence proves the build accepts the flag. It does not prove",
            "    the flag works for a hybrid KDA+MLA model on ROCm; that is G2.",
            "=" * WIDTH,
        ]
        return "\n".join(lines)


class JsonRenderer:
    def render(self, results: Sequence[ProbeResult]) -> str:
        payload = []
        for result in results:
            entry = dataclasses.asdict(result)
            entry["a1_available"] = result.a1_available
            entry["a2_available"] = result.a2_available
            payload.append(entry)
        return json.dumps(payload, indent=2)
