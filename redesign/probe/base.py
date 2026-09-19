"""Abstractions for static engine capability probes.

A probe answers one question: does this build expose the flags that
docs/SYSTEM-DESIGN.md A1 depends on? It never starts an engine, allocates a
GPU, or downloads weights, so it is safe to run before any paid session.
"""

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field

# Substrings that indicate hybrid-model handling in a de-duplication code path.
# A hit is ambiguous: it may be support or an explicit refusal. The value is
# that it names the file a human should read next.
HYBRID_HINTS = ("mamba", "hybrid", "linear_attn", "is_hybrid", "supportshma", "kda")


@dataclass(frozen=True)
class FlagSpec:
    name: str
    buys: str


@dataclass(frozen=True)
class FlagResult:
    spec: FlagSpec
    present: bool


@dataclass(frozen=True)
class SourceScan:
    module: str
    status: str
    path: str | None = None
    hints: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    engine: str
    importable: bool
    error: str | None = None
    version: str = "unknown"
    platform: str = "unknown"
    cli_surface_size: int = 0
    cli_surface_error: str | None = None
    flags: tuple[FlagResult, ...] = ()
    scans: tuple[SourceScan, ...] = ()
    a1_flags: tuple[str, ...] = field(default_factory=tuple)

    def _present(self, name: str) -> bool:
        return any(f.present for f in self.flags if f.spec.name == name)

    @property
    def a1_available(self) -> bool:
        return any(self._present(name) for name in self.a1_flags)

    @property
    def a2_available(self) -> bool:
        return self._present("kv_cache_dtype")


def detect_platform() -> str:
    try:
        import torch
    except ImportError:
        return "unknown"
    if getattr(torch.version, "hip", None):
        return f"ROCm (hip {torch.version.hip})"
    if getattr(torch.version, "cuda", None):
        return f"CUDA {torch.version.cuda}"
    return "unknown"


class EngineProbe(abc.ABC):
    """Template for a probe. Subclass to add an engine; nothing else changes."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def flags(self) -> tuple[FlagSpec, ...]: ...

    @property
    @abc.abstractmethod
    def a1_flags(self) -> tuple[str, ...]: ...

    @property
    @abc.abstractmethod
    def scan_modules(self) -> tuple[str, ...]: ...

    @abc.abstractmethod
    def cli_surface(self) -> set[str]:
        """Argument destinations the build's real parser accepts.

        Introspecting the parser is version-proof in a way that importing
        internal config dataclasses is not: if a flag reaches the parser, the
        build accepts it whatever the docs say.
        """

    def run(self) -> ProbeResult:
        module = self._import(self.name)
        if module is None:
            return ProbeResult(
                engine=self.name,
                importable=False,
                error=self._last_error,
                a1_flags=self.a1_flags,
            )

        surface, surface_error = self._safe_cli_surface()
        return ProbeResult(
            engine=self.name,
            importable=True,
            version=getattr(module, "__version__", "unknown"),
            platform=detect_platform(),
            cli_surface_size=len(surface),
            cli_surface_error=surface_error,
            flags=tuple(FlagResult(spec, spec.name in surface) for spec in self.flags),
            scans=tuple(self._scan(name) for name in self.scan_modules),
            a1_flags=self.a1_flags,
        )

    def _safe_cli_surface(self) -> tuple[set[str], str | None]:
        try:
            return self.cli_surface(), None
        except Exception as exc:  # noqa: BLE001
            return set(), f"{type(exc).__name__}: {exc}"

    def _import(self, name: str):
        try:
            self._last_error = None
            return importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _scan(self, module_name: str) -> SourceScan:
        module = self._import(module_name)
        if module is None:
            return SourceScan(module_name, "unimportable", detail=self._last_error)

        path = getattr(module, "__file__", None)
        if not path:
            return SourceScan(module_name, "no source")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                source = handle.read().lower()
        except OSError as exc:
            return SourceScan(module_name, "unreadable", detail=str(exc))

        hits = tuple(sorted(h for h in HYBRID_HINTS if h in source))
        return SourceScan(module_name, "scanned", path=path, hints=hits)
