"""
Gate 2 SMC co-occurrence diagnostic.

Uses existing locked detections only:
- OB zones
- FVG zones
- Liquidity sweeps

No outcome, RR, cost, baseline, or parameter tuning.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


N_BARS = 10
PRICE_ATR_MULTIPLE = 1.0


@dataclass(frozen=True)
class Signal:
    kind: str
    signal_id: int
    direction: str
    index: int
    high: float
    low: float
    atr: float
    year: int
    session: str


def load_obs(path: Path) -> list[Signal]:
    rows: list[Signal] = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Signal(
                    kind="OB",
                    signal_id=int(row["zone_id"]),
                    direction=row["direction"],
                    index=int(row["impulse_end_index"]),
                    high=float(row["zone_high"]),
                    low=float(row["zone_low"]),
                    atr=float(row["frozen_atr"]),
                    year=int(row["year"]),
                    session=row["session"],
                )
            )
    return rows


def load_fvgs(path: Path) -> list[Signal]:
    rows: list[Signal] = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Signal(
                    kind="FVG",
                    signal_id=int(row["fvg_id"]),
                    direction=row["direction"],
                    index=int(row["candle3_index"]),
                    high=float(row["gap_high"]),
                    low=float(row["gap_low"]),
                    atr=float(row["frozen_atr"]),
                    year=int(row["year"]),
                    session=row["session"],
                )
            )
    return rows


def load_sweeps(path: Path) -> list[Signal]:
    rows: list[Signal] = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            level = float(row["swept_swing_level"])
            rows.append(
                Signal(
                    kind="Sweep",
                    signal_id=int(row["sweep_id"]),
                    direction=row["direction"],
                    index=int(row["rejection_index"]),
                    high=level,
                    low=level,
                    atr=float(row["frozen_atr"]),
                    year=int(row["year"]),
                    session=row["session"],
                )
            )
    return rows


def range_distance(a: Signal, b: Signal) -> float:
    if a.low <= b.high and b.low <= a.high:
        return 0.0
    if a.high < b.low:
        return b.low - a.high
    return a.low - b.high


def cooccurs(anchor: Signal, other: Signal) -> bool:
    if abs(anchor.index - other.index) > N_BARS:
        return False
    return range_distance(anchor, other) <= PRICE_ATR_MULTIPLE * anchor.atr


def candidates_by_time(anchor: Signal, others: list[Signal]) -> Iterable[Signal]:
    # Dataset is small enough for direct scan, but time prefilter keeps intent clear.
    for other in others:
        if abs(anchor.index - other.index) <= N_BARS:
            yield other


def matches(anchor: Signal, others: list[Signal]) -> list[Signal]:
    return [other for other in candidates_by_time(anchor, others) if cooccurs(anchor, other)]


def aligned(anchor: Signal, others: list[Signal]) -> list[Signal]:
    return [other for other in matches(anchor, others) if other.direction == anchor.direction]


def pairwise_report(name_a: str, a: list[Signal], name_b: str, b: list[Signal]) -> tuple[int, int]:
    any_count = 0
    aligned_count = 0
    for item in a:
        ms = matches(item, b)
        if ms:
            any_count += 1
        if any(other.direction == item.direction for other in ms):
            aligned_count += 1
    print(
        f"{name_a}_with_{name_b},"
        f"{len(a)},{any_count},{any_count / len(a):.2%},"
        f"{aligned_count},{aligned_count / len(a):.2%}"
    )
    return any_count, aligned_count


def triple_confluence(obs: list[Signal], fvgs: list[Signal], sweeps: list[Signal]) -> tuple[list[Signal], int]:
    confluence_obs: list[Signal] = []
    combo_count = 0
    for ob in obs:
        fvg_matches = [item for item in aligned(ob, fvgs)]
        sweep_matches = [item for item in aligned(ob, sweeps)]
        valid_for_ob = False
        for fvg in fvg_matches:
            for sweep in sweep_matches:
                if fvg.direction != sweep.direction:
                    continue
                if not cooccurs(fvg, sweep):
                    continue
                combo_count += 1
                valid_for_ob = True
        if valid_for_ob:
            confluence_obs.append(ob)
    return confluence_obs, combo_count


def print_triple_breakdowns(confluence: list[Signal]) -> None:
    by_year = Counter(item.year for item in confluence)
    by_session = Counter(item.session for item in confluence)

    print("\nTRIPLE_CONFLUENCE_BY_YEAR")
    print("year,count")
    for year in range(2016, 2027):
        print(f"{year},{by_year[year]}")

    print("\nTRIPLE_CONFLUENCE_BY_SESSION")
    print("session,count")
    for session in ("asian", "london", "ny_overlap", "off_session"):
        print(f"{session},{by_session[session]}")


def main() -> None:
    obs = load_obs(Path("research/order_block_zones.csv"))
    fvgs = load_fvgs(Path("research/fair_value_gap_fvgs.csv"))
    sweeps = load_sweeps(Path("research/liquidity_sweep_sweeps.csv"))

    print("SMC_GATE_2_COOCCURRENCE")
    print(f"rule: co-occurs if abs(index_a-index_b) <= {N_BARS} M15 bars and price range/level distance <= {PRICE_ATR_MULTIPLE:.1f} * anchor frozen_ATR")
    print("direction_aligned: exact same direction label: bullish with bullish, bearish with bearish")
    print(f"counts: OB={len(obs)}, FVG={len(fvgs)}, Sweep={len(sweeps)}")

    print("\nPAIRWISE_ANCHOR_RATES")
    print("anchor_pair,anchor_count,cooccur_count,cooccur_pct,direction_aligned_count,direction_aligned_pct")
    pairwise_report("OB", obs, "FVG", fvgs)
    pairwise_report("OB", obs, "Sweep", sweeps)
    pairwise_report("FVG", fvgs, "OB", obs)
    pairwise_report("FVG", fvgs, "Sweep", sweeps)
    pairwise_report("Sweep", sweeps, "OB", obs)
    pairwise_report("Sweep", sweeps, "FVG", fvgs)

    confluence_obs, combo_count = triple_confluence(obs, fvgs, sweeps)
    print("\nTRIPLE_CONFLUENCE")
    print(f"unique_OB_anchored_direction_aligned_triples={len(confluence_obs)}")
    print(f"raw_direction_aligned_OB_FVG_Sweep_combinations={combo_count}")
    print(f"pct_of_OBs={len(confluence_obs) / len(obs):.2%}")
    print_triple_breakdowns(confluence_obs)

    print("\nINTERPRETATION_GUIDE")
    print("Pairwise rates below ~20% imply mostly distinct triggers; high rates imply proxy-like overlap.")
    print("Triple unique OB-anchored count is the practical confluence sample size for a later pre-registered test.")
    if len(confluence_obs) < 100:
        print("UNDERPOWERED_FLAG: triple confluence has fewer than 100 unique OB-anchored instances across the decade.")
    elif len(confluence_obs) < 300:
        print("POWER_CAUTION: triple confluence sample is modest; later CI will likely be wide.")
    else:
        print("SAMPLE_OK_FOR_TESTING: triple confluence sample is large enough to justify a pre-registered outcome test.")


if __name__ == "__main__":
    main()
