"""G2 interpretation of per-rank pool log lines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class G2PoolVerdict:
    per_rank: tuple[int, ...]
    aggregate: int
    expected_per_rank: int
    expected_aggregate: int
    pass_on_aggregate: bool
    occupancy_spread: float

    @property
    def passed(self) -> bool:
        return self.pass_on_aggregate


def interpret_g2_pools(
    per_rank_pools: list[int],
    expected_per_rank: int,
    expected_aggregate: int,
    tolerance: float = 0.25,
) -> G2PoolVerdict:
    if not per_rank_pools:
        raise ValueError("G2 needs at least one per-rank pool reading")
    aggregate = sum(per_rank_pools)
    peak = max(per_rank_pools)
    trough = min(per_rank_pools)
    spread = (peak - trough) / peak if peak else 0.0
    return G2PoolVerdict(
        per_rank=tuple(per_rank_pools),
        aggregate=aggregate,
        expected_per_rank=expected_per_rank,
        expected_aggregate=expected_aggregate,
        # Pass is on the cluster unique total; one rank still logs ~2.3M.
        pass_on_aggregate=abs(aggregate - expected_aggregate) / expected_aggregate
        <= tolerance,
        occupancy_spread=spread,
    )
