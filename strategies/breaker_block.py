"""
Breaker Block detector.

Detector only:
- Consume existing locked Order Block detections from research/order_block_zones.csv.
- Confirm real-time breaks using completed M15 closes beyond the OB zone.
- Require clean separation before a valid retest.
- Expire breakers if no retest occurs within 50 completed M15 bars after break.
- No outcome, RR, baseline, cost, or PnL measurement.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional

from order_block import (
    GAP_MINUTES,
    SESSION_ORDER,
    Bar,
    Zone,
    classify_session,
    default_tick_path,
    fmt_ts,
    load_bars,
    parse_timestamp,
)


BREAK_ATR_THRESHOLD = 0.5
N_EXPIRY = 50


@dataclass(frozen=True)
class Breaker:
    breaker_id: int
    source_ob_id: int
    source_ob_direction: str
    flipped_direction: str
    zone_high: float
    zone_low: float
    ob_creation_time: datetime
    break_candle_time: datetime
    break_candle_index: int
    break_close_price: float
    frozen_atr: float
    displacement_atr: float
    separation_confirm_time: Optional[datetime]
    separation_confirm_index: Optional[int]
    retest_time: Optional[datetime]
    retest_index: Optional[int]
    expired: bool
    bars_ob_creation_to_break: int
    bars_break_to_separation: Optional[int]
    bars_break_to_retest: Optional[int]
    session: str
    year: int
    segment_id: int

    @property
    def valid_retest(self) -> bool:
        return self.retest_time is not None and not self.expired


def bool_value(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes"}


def parse_zones(path: Path) -> list[Zone]:
    zones: list[Zone] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            zones.append(
                Zone(
                    zone_id=int(row["zone_id"]),
                    direction=row["direction"],
                    zone_high=float(row["zone_high"]),
                    zone_low=float(row["zone_low"]),
                    zone_creation_time=parse_timestamp(row["zone_creation_time"]),
                    frozen_atr=float(row["frozen_atr"]),
                    displacement_atr=float(row["displacement_atr"]),
                    bos_swing_level=float(row["bos_swing_level"]),
                    fvg_present=bool_value(row["fvg_present"]),
                    session=row["session"],
                    year=int(row["year"]),
                    ob_candle_time=parse_timestamp(row["ob_candle_time"]),
                    ob_candle_open=float(row["ob_candle_open"]),
                    ob_candle_high=float(row["ob_candle_high"]),
                    ob_candle_low=float(row["ob_candle_low"]),
                    ob_candle_close=float(row["ob_candle_close"]),
                    ob_candle_index=int(row["ob_candle_index"]),
                    impulse_start_time=parse_timestamp(row["impulse_start_time"]),
                    impulse_end_time=parse_timestamp(row["impulse_end_time"]),
                    impulse_start_index=int(row["impulse_start_index"]),
                    impulse_end_index=int(row["impulse_end_index"]),
                    impulse_extreme=float(row["impulse_extreme"]),
                    segment_id=int(row["segment_id"]),
                )
            )
    return zones


def find_break(bars: list[Bar], zone: Zone) -> Optional[int]:
    threshold = BREAK_ATR_THRESHOLD * zone.frozen_atr
    for i in range(zone.impulse_end_index + 1, len(bars)):
        bar = bars[i]
        if bar.segment_id != zone.segment_id:
            return None
        if zone.direction == "bullish":
            if bar.close <= zone.zone_low - threshold:
                return i
        else:
            if bar.close >= zone.zone_high + threshold:
                return i
    return None


def separated(bar: Bar, zone: Zone) -> bool:
    if zone.direction == "bullish":
        return bar.high < zone.zone_low
    return bar.low > zone.zone_high


def touches_zone(bar: Bar, zone: Zone) -> bool:
    return bar.low <= zone.zone_high and bar.high >= zone.zone_low


def detect_breaker_for_zone(bars: list[Bar], zone: Zone, breaker_id: int) -> Optional[Breaker]:
    break_index = find_break(bars, zone)
    if break_index is None:
        return None

    break_bar = bars[break_index]
    separation_index: Optional[int] = None
    retest_index: Optional[int] = None
    expiry_end = min(len(bars) - 1, break_index + N_EXPIRY)

    for i in range(break_index + 1, expiry_end + 1):
        bar = bars[i]
        if bar.segment_id != zone.segment_id:
            break
        if separation_index is None:
            if separated(bar, zone):
                separation_index = i
            continue
        if touches_zone(bar, zone):
            retest_index = i
            break

    flipped = "bearish" if zone.direction == "bullish" else "bullish"
    sep_bar = bars[separation_index] if separation_index is not None else None
    retest_bar = bars[retest_index] if retest_index is not None else None

    return Breaker(
        breaker_id=breaker_id,
        source_ob_id=zone.zone_id,
        source_ob_direction=zone.direction,
        flipped_direction=flipped,
        zone_high=zone.zone_high,
        zone_low=zone.zone_low,
        ob_creation_time=zone.zone_creation_time,
        break_candle_time=break_bar.end,
        break_candle_index=break_index,
        break_close_price=break_bar.close,
        frozen_atr=zone.frozen_atr,
        displacement_atr=zone.displacement_atr,
        separation_confirm_time=sep_bar.end if sep_bar is not None else None,
        separation_confirm_index=separation_index,
        retest_time=retest_bar.end if retest_bar is not None else None,
        retest_index=retest_index,
        expired=retest_index is None,
        bars_ob_creation_to_break=break_index - zone.impulse_end_index,
        bars_break_to_separation=(
            separation_index - break_index if separation_index is not None else None
        ),
        bars_break_to_retest=retest_index - break_index if retest_index is not None else None,
        session=classify_session(break_bar.end),
        year=break_bar.end.year,
        segment_id=zone.segment_id,
    )


def detect_breakers(bars: list[Bar], zones: list[Zone]) -> list[Breaker]:
    breakers: list[Breaker] = []
    for zone in zones:
        breaker = detect_breaker_for_zone(bars, zone, len(breakers) + 1)
        if breaker is not None:
            breakers.append(breaker)
    return breakers


BREAKER_FIELDS = [
    "breaker_id",
    "source_ob_id",
    "source_ob_direction",
    "flipped_direction",
    "zone_high",
    "zone_low",
    "ob_creation_time",
    "break_candle_time",
    "break_candle_index",
    "break_close_price",
    "frozen_atr",
    "displacement_atr",
    "separation_confirm_time",
    "separation_confirm_index",
    "retest_time",
    "retest_index",
    "expired",
    "valid_retest",
    "bars_ob_creation_to_break",
    "bars_break_to_separation",
    "bars_break_to_retest",
    "session",
    "year",
    "segment_id",
]


FIRST10_FIELDS = [
    "breaker_id",
    "source_ob_id",
    "source_ob_direction",
    "flipped_direction",
    "zone_high",
    "zone_low",
    "ob_creation_time",
    "break_candle_time",
    "break_close_price",
    "frozen_atr",
    "displacement_atr",
    "separation_confirm_time",
    "retest_time",
    "bars_ob_creation_to_break",
    "bars_break_to_separation",
    "bars_break_to_retest",
]


def breaker_to_row(breaker: Breaker) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in BREAKER_FIELDS:
        value = getattr(breaker, field)
        if isinstance(value, datetime):
            row[field] = fmt_ts(value)
        elif isinstance(value, float):
            row[field] = f"{value:.6f}"
        elif value is None:
            row[field] = ""
        else:
            row[field] = value
    return row


def write_breakers(path: Path, breakers: list[Breaker]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BREAKER_FIELDS)
        writer.writeheader()
        for breaker in breakers:
            writer.writerow(breaker_to_row(breaker))


def write_first10(path: Path, breakers: list[Breaker]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [breaker for breaker in breakers if breaker.valid_retest]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIRST10_FIELDS)
        writer.writeheader()
        for breaker in valid[:10]:
            row = breaker_to_row(breaker)
            writer.writerow({field: row[field] for field in FIRST10_FIELDS})


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def print_distribution(label: str, values: list[int]) -> None:
    if not values:
        print(f"  {label}: n=0")
        return
    as_float = [float(item) for item in values]
    print(
        f"  {label}: n={len(values):,}, min={min(values):,}, "
        f"median={median(values):.1f}, p90={percentile(as_float, 0.90):.1f}, max={max(values):,}"
    )


def print_report(
    zones: list[Zone],
    breakers: list[Breaker],
    breakers_out: Path,
    first10_out: Path,
) -> None:
    valid = [breaker for breaker in breakers if breaker.valid_retest]
    expired = [breaker for breaker in breakers if breaker.expired]
    broke_source_ids = {breaker.source_ob_id for breaker in breakers}

    print("\nBreaker Block Detection Report")
    print("=" * 38)
    print("Detector only: no outcome, RR, baseline, cost, or PnL measurement.")
    print(f"All breakers CSV:          {breakers_out}")
    print(f"First 10 valid retests CSV: {first10_out}")
    print("\nCore Counts")
    print(f"  Source OBs loaded:        {len(zones):,}")
    print(f"  Source OBs became breaker:{len(broke_source_ids):,}")
    print(f"  Source OBs never broke:   {len(zones) - len(broke_source_ids):,}")
    print(f"  Breakers detected:        {len(breakers):,}")
    print(f"  Valid retest <= {N_EXPIRY}:       {len(valid):,}")
    print(f"  Expired / never retested: {len(expired):,}")

    print("\nFlipped Direction")
    direction_counts = Counter(breaker.flipped_direction for breaker in breakers)
    for direction in ("bullish", "bearish"):
        print(f"  {direction:<8} {direction_counts[direction]:,}")

    print("\nPer Session")
    session_counts = Counter(breaker.session for breaker in breakers)
    valid_session_counts = Counter(breaker.session for breaker in valid)
    for session in SESSION_ORDER:
        print(
            f"  {session:<11} breakers={session_counts[session]:>5,} "
            f"valid_retests={valid_session_counts[session]:>5,}"
        )

    print("\nPer Year")
    year_counts = Counter(breaker.year for breaker in breakers)
    valid_year_counts = Counter(breaker.year for breaker in valid)
    for year in range(2016, 2027):
        print(
            f"  {year}: breakers={year_counts[year]:>4,} "
            f"valid_retests={valid_year_counts[year]:>4,}"
        )

    print("\nBar Count Distributions")
    print_distribution(
        "OB creation -> break",
        [breaker.bars_ob_creation_to_break for breaker in breakers],
    )
    print_distribution(
        "break -> retest",
        [
            breaker.bars_break_to_retest
            for breaker in valid
            if breaker.bars_break_to_retest is not None
        ],
    )

    print("\nVisual Verification Gate")
    print("  Exported first 10 breakers with valid retests for M15 chart review.")
    print("  Stop here before any outcome, RR, cost, or baseline work.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None, help="Dukascopy XAUUSD tick CSV")
    parser.add_argument("--zones", type=Path, default=Path("research/order_block_zones.csv"))
    parser.add_argument("--breakers-out", type=Path, default=Path("research/breaker_block_breakers.csv"))
    parser.add_argument("--first10-out", type=Path, default=Path("research/breaker_block_first10_valid_retests.csv"))
    parser.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)
    parser.add_argument("--max-rows", type=int, default=None, help="development smoke-test row limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tick_path = args.ticks or default_tick_path()
    zones = parse_zones(args.zones)
    bars = load_bars(tick_path, gap_minutes=args.gap_minutes, max_rows=args.max_rows)
    breakers = detect_breakers(bars, zones)
    write_breakers(args.breakers_out, breakers)
    write_first10(args.first10_out, breakers)
    print_report(zones, breakers, args.breakers_out, args.first10_out)


if __name__ == "__main__":
    main()
