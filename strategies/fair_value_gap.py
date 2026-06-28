"""
Fair Value Gap detector.

Detector only:
- Resample Dukascopy bid/ask ticks to gap-aware completed M15 mid-price bars.
- Detect 3-candle FVGs at candle3 close using ATR(14) frozen at creation.
- Export all detected FVGs and the first 10 for visual chart verification.
- No outcome, RR, cost, baseline, or strategy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

from order_block import (
    ATR_PERIOD,
    GAP_MINUTES,
    SESSION_ORDER,
    SWING_LEFT,
    SWING_RIGHT,
    Bar,
    Swing,
    classify_session,
    compute_atr,
    default_tick_path,
    fmt_ts,
    is_swing_high,
    is_swing_low,
    load_bars,
)


FVG_ATR_THRESHOLD = 0.5


@dataclass(frozen=True)
class FairValueGap:
    fvg_id: int
    direction: str
    gap_high: float
    gap_low: float
    creation_time: datetime
    frozen_atr: float
    gap_size_atr: float
    bos_present: bool
    bos_swing_level: Optional[float]
    session: str
    year: int
    segment_id: int
    candle1_time: datetime
    candle1_open: float
    candle1_high: float
    candle1_low: float
    candle1_close: float
    candle1_index: int
    candle2_time: datetime
    candle2_open: float
    candle2_high: float
    candle2_low: float
    candle2_close: float
    candle2_index: int
    candle3_time: datetime
    candle3_open: float
    candle3_high: float
    candle3_low: float
    candle3_close: float
    candle3_index: int
    touched: bool
    first_touch_time: Optional[datetime]
    bars_to_touch: Optional[int]


def detect_confirmed_swings(bars: list[Bar]) -> tuple[list[Optional[Swing]], list[Optional[Swing]]]:
    latest_highs: list[Optional[Swing]] = [None] * len(bars)
    latest_lows: list[Optional[Swing]] = [None] * len(bars)
    latest_high: Optional[Swing] = None
    latest_low: Optional[Swing] = None

    for i, bar in enumerate(bars):
        pivot = i - SWING_RIGHT
        if pivot >= SWING_LEFT:
            pivot_bar = bars[pivot]
            if pivot_bar.segment_id == bar.segment_id:
                if is_swing_high(bars, pivot):
                    latest_high = Swing(
                        index=pivot,
                        confirmed_at_index=i,
                        timestamp=pivot_bar.start,
                        level=pivot_bar.high,
                    )
                if is_swing_low(bars, pivot):
                    latest_low = Swing(
                        index=pivot,
                        confirmed_at_index=i,
                        timestamp=pivot_bar.start,
                        level=pivot_bar.low,
                    )
        latest_highs[i] = latest_high
        latest_lows[i] = latest_low

    return latest_highs, latest_lows


def first_touch_after_creation(
    bars: list[Bar],
    *,
    gap_high: float,
    gap_low: float,
    candle3_index: int,
    segment_id: int,
) -> tuple[bool, Optional[datetime], Optional[int]]:
    for i in range(candle3_index + 1, len(bars)):
        bar = bars[i]
        if bar.segment_id != segment_id:
            return False, None, None
        if bar.low <= gap_high and bar.high >= gap_low:
            return True, bar.end, i - candle3_index
    return False, None, None


def make_fvg(
    *,
    fvg_id: int,
    direction: str,
    bars: list[Bar],
    candle1_index: int,
    candle2_index: int,
    candle3_index: int,
    gap_high: float,
    gap_low: float,
    frozen_atr: float,
    gap_size_atr: float,
    bos_present: bool,
    bos_swing_level: Optional[float],
) -> FairValueGap:
    candle1 = bars[candle1_index]
    candle2 = bars[candle2_index]
    candle3 = bars[candle3_index]
    touched, first_touch_time, bars_to_touch = first_touch_after_creation(
        bars,
        gap_high=gap_high,
        gap_low=gap_low,
        candle3_index=candle3_index,
        segment_id=candle3.segment_id,
    )
    return FairValueGap(
        fvg_id=fvg_id,
        direction=direction,
        gap_high=gap_high,
        gap_low=gap_low,
        creation_time=candle3.end,
        frozen_atr=frozen_atr,
        gap_size_atr=gap_size_atr,
        bos_present=bos_present,
        bos_swing_level=bos_swing_level,
        session=classify_session(candle3.end),
        year=candle3.end.year,
        segment_id=candle3.segment_id,
        candle1_time=candle1.end,
        candle1_open=candle1.open,
        candle1_high=candle1.high,
        candle1_low=candle1.low,
        candle1_close=candle1.close,
        candle1_index=candle1.index,
        candle2_time=candle2.end,
        candle2_open=candle2.open,
        candle2_high=candle2.high,
        candle2_low=candle2.low,
        candle2_close=candle2.close,
        candle2_index=candle2.index,
        candle3_time=candle3.end,
        candle3_open=candle3.open,
        candle3_high=candle3.high,
        candle3_low=candle3.low,
        candle3_close=candle3.close,
        candle3_index=candle3.index,
        touched=touched,
        first_touch_time=first_touch_time,
        bars_to_touch=bars_to_touch,
    )


def detect_fvgs(bars: list[Bar], atr: list[Optional[float]]) -> list[FairValueGap]:
    fvgs: list[FairValueGap] = []
    latest_highs, latest_lows = detect_confirmed_swings(bars)

    for i in range(2, len(bars)):
        candle1 = bars[i - 2]
        candle2 = bars[i - 1]
        candle3 = bars[i]
        if not (
            candle1.segment_id == candle2.segment_id == candle3.segment_id
            and candle1.index + 1 == candle2.index
            and candle2.index + 1 == candle3.index
        ):
            continue

        frozen_atr = atr[i]
        if frozen_atr is None or frozen_atr <= 0:
            continue

        if candle1.high < candle3.low:
            gap_low = candle1.high
            gap_high = candle3.low
            gap_size_atr = (gap_high - gap_low) / frozen_atr
            if gap_size_atr >= FVG_ATR_THRESHOLD:
                swing = latest_highs[i - 1]
                bos_present = (
                    swing is not None
                    and swing.confirmed_at_index <= i - 1
                    and swing.index < i - 1
                    and candle2.high > swing.level
                    and bars[swing.index].segment_id == candle2.segment_id
                )
                fvgs.append(
                    make_fvg(
                        fvg_id=len(fvgs) + 1,
                        direction="bullish",
                        bars=bars,
                        candle1_index=i - 2,
                        candle2_index=i - 1,
                        candle3_index=i,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        frozen_atr=frozen_atr,
                        gap_size_atr=gap_size_atr,
                        bos_present=bos_present,
                        bos_swing_level=swing.level if bos_present and swing is not None else None,
                    )
                )

        if candle1.low > candle3.high:
            gap_low = candle3.high
            gap_high = candle1.low
            gap_size_atr = (gap_high - gap_low) / frozen_atr
            if gap_size_atr >= FVG_ATR_THRESHOLD:
                swing = latest_lows[i - 1]
                bos_present = (
                    swing is not None
                    and swing.confirmed_at_index <= i - 1
                    and swing.index < i - 1
                    and candle2.low < swing.level
                    and bars[swing.index].segment_id == candle2.segment_id
                )
                fvgs.append(
                    make_fvg(
                        fvg_id=len(fvgs) + 1,
                        direction="bearish",
                        bars=bars,
                        candle1_index=i - 2,
                        candle2_index=i - 1,
                        candle3_index=i,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        frozen_atr=frozen_atr,
                        gap_size_atr=gap_size_atr,
                        bos_present=bos_present,
                        bos_swing_level=swing.level if bos_present and swing is not None else None,
                    )
                )

    return fvgs


FVG_FIELDS = [
    "fvg_id",
    "direction",
    "gap_high",
    "gap_low",
    "creation_time",
    "frozen_atr",
    "gap_size_atr",
    "bos_present",
    "bos_swing_level",
    "session",
    "year",
    "segment_id",
    "candle1_time",
    "candle1_open",
    "candle1_high",
    "candle1_low",
    "candle1_close",
    "candle1_index",
    "candle2_time",
    "candle2_open",
    "candle2_high",
    "candle2_low",
    "candle2_close",
    "candle2_index",
    "candle3_time",
    "candle3_open",
    "candle3_high",
    "candle3_low",
    "candle3_close",
    "candle3_index",
    "touched",
    "first_touch_time",
    "bars_to_touch",
]


FIRST10_FIELDS = [
    "fvg_id",
    "direction",
    "creation_time",
    "gap_high",
    "gap_low",
    "frozen_atr",
    "gap_size_atr",
    "bos_present",
    "bos_swing_level",
    "candle1_time",
    "candle1_open",
    "candle1_high",
    "candle1_low",
    "candle1_close",
    "candle2_time",
    "candle2_open",
    "candle2_high",
    "candle2_low",
    "candle2_close",
    "candle3_time",
    "candle3_open",
    "candle3_high",
    "candle3_low",
    "candle3_close",
]


def fvg_to_row(fvg: FairValueGap) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in FVG_FIELDS:
        value = getattr(fvg, field)
        if isinstance(value, datetime):
            row[field] = fmt_ts(value)
        elif isinstance(value, float):
            row[field] = f"{value:.6f}"
        elif value is None:
            row[field] = ""
        else:
            row[field] = value
    return row


def write_fvgs(path: Path, fvgs: list[FairValueGap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FVG_FIELDS)
        writer.writeheader()
        for fvg in fvgs:
            writer.writerow(fvg_to_row(fvg))


def write_first10(path: Path, fvgs: list[FairValueGap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIRST10_FIELDS)
        writer.writeheader()
        for fvg in fvgs[:10]:
            row = fvg_to_row(fvg)
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


def print_report(
    fvgs: list[FairValueGap],
    bars: list[Bar],
    fvgs_out: Path,
    first10_out: Path,
) -> None:
    by_year = Counter(item.year for item in fvgs)
    by_session = Counter(item.session for item in fvgs)
    by_direction = Counter(item.direction for item in fvgs)
    gap_sizes = [item.gap_size_atr for item in fvgs]
    bos_count = sum(1 for item in fvgs if item.bos_present)
    touched_count = sum(1 for item in fvgs if item.touched)

    print("\nFair Value Gap Phase A Detection Report")
    print("=" * 43)
    print("Detector only: no outcomes, RR, cost, or baseline measured.")
    print(f"Completed M15 bars: {len(bars):,}")
    print(f"Detected FVGs:       {len(fvgs):,}")
    print(f"FVGs CSV:            {fvgs_out}")
    print(f"First 10 CSV:        {first10_out}")
    print(f"Gap filter:          >= {FVG_ATR_THRESHOLD:.1f} * ATR({ATR_PERIOD}) frozen at candle3 close")

    print("\nDirection")
    for direction in ("bullish", "bearish"):
        print(f"  {direction:<8} {by_direction[direction]:,}")

    print("\nPer Year")
    for year in range(2016, 2027):
        print(f"  {year}: {by_year[year]:,}")

    print("\nPer Session")
    for session in SESSION_ORDER:
        print(f"  {session:<11} {by_session[session]:,}")

    print("\nGap Size ATR")
    if gap_sizes:
        print(f"  min:    {min(gap_sizes):.3f}")
        print(f"  median: {median(gap_sizes):.3f}")
        print(f"  p90:    {percentile(gap_sizes, 0.90):.3f}")
        print(f"  max:    {max(gap_sizes):.3f}")
    else:
        print("  n/a")

    print("\nBOS Present")
    print(f"  {bos_count:,} / {len(fvgs):,} ({(bos_count / len(fvgs) if fvgs else 0):.2%})")

    print("\nMitigation Tracking")
    print(f"  First later touch observed: {touched_count:,} / {len(fvgs):,} ({(touched_count / len(fvgs) if fvgs else 0):.2%})")
    print("  Descriptive only; no reaction/outcome measured.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None, help="Dukascopy XAUUSD tick CSV")
    parser.add_argument("--fvgs-out", type=Path, default=Path("research/fair_value_gap_fvgs.csv"))
    parser.add_argument("--first10-out", type=Path, default=Path("research/fair_value_gap_first10.csv"))
    parser.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)
    parser.add_argument("--max-rows", type=int, default=None, help="development smoke-test row limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tick_path = args.ticks or default_tick_path()
    bars = load_bars(tick_path, gap_minutes=args.gap_minutes, max_rows=args.max_rows)
    atr = compute_atr(bars)
    fvgs = detect_fvgs(bars, atr)
    write_fvgs(args.fvgs_out, fvgs)
    write_first10(args.first10_out, fvgs)
    print_report(fvgs, bars, args.fvgs_out, args.first10_out)


if __name__ == "__main__":
    main()
