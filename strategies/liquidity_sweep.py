"""
Liquidity Sweep detector.

Detector only:
- Uses gap-aware completed M15 mid-price bars from Dukascopy bid/ask ticks.
- Uses the same 10/10 confirmed fractal swing logic as Order Block.
- Detects ATR-relative breaches of confirmed swings followed by close-back-inside
  rejection within 1-3 completed M15 bars.
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


BREACH_ATR_THRESHOLD = 0.25
REJECTION_WINDOW_BARS = 3


@dataclass(frozen=True)
class LiquiditySweep:
    sweep_id: int
    direction: str
    side: str
    swept_swing_level: float
    swing_index: int
    swing_time: datetime
    swing_confirm_time: datetime
    breach_time: datetime
    breach_index: int
    breach_extreme: float
    breach_distance_atr: float
    rejection_close_time: datetime
    rejection_index: int
    rejection_close_price: float
    bars_breach_to_rejection: int
    frozen_atr: float
    session: str
    year: int
    segment_id: int


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


def detect_sweeps(bars: list[Bar], atr: list[Optional[float]]) -> list[LiquiditySweep]:
    sweeps: list[LiquiditySweep] = []
    latest_high: Optional[Swing] = None
    latest_low: Optional[Swing] = None
    consumed_highs: set[int] = set()
    consumed_lows: set[int] = set()

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

        frozen_atr = atr[i]
        if frozen_atr is None or frozen_atr <= 0:
            continue

        threshold = BREACH_ATR_THRESHOLD * frozen_atr

        if (
            latest_low is not None
            and latest_low.confirmed_at_index < i
            and latest_low.index not in consumed_lows
            and bars[latest_low.index].segment_id == bar.segment_id
            and bar.low <= latest_low.level - threshold
        ):
            consumed_lows.add(latest_low.index)
            rejection = find_sell_side_rejection(bars, i, latest_low.level)
            if rejection is not None:
                rejection_bar = bars[rejection]
                sweeps.append(
                    LiquiditySweep(
                        sweep_id=len(sweeps) + 1,
                        direction="bullish",
                        side="sell_side",
                        swept_swing_level=latest_low.level,
                        swing_index=latest_low.index,
                        swing_time=bars[latest_low.index].end,
                        swing_confirm_time=bars[latest_low.confirmed_at_index].end,
                        breach_time=bar.end,
                        breach_index=i,
                        breach_extreme=bar.low,
                        breach_distance_atr=(latest_low.level - bar.low) / frozen_atr,
                        rejection_close_time=rejection_bar.end,
                        rejection_index=rejection,
                        rejection_close_price=rejection_bar.close,
                        bars_breach_to_rejection=rejection - i,
                        frozen_atr=frozen_atr,
                        session=classify_session(rejection_bar.end),
                        year=rejection_bar.end.year,
                        segment_id=bar.segment_id,
                    )
                )

        if (
            latest_high is not None
            and latest_high.confirmed_at_index < i
            and latest_high.index not in consumed_highs
            and bars[latest_high.index].segment_id == bar.segment_id
            and bar.high >= latest_high.level + threshold
        ):
            consumed_highs.add(latest_high.index)
            rejection = find_buy_side_rejection(bars, i, latest_high.level)
            if rejection is not None:
                rejection_bar = bars[rejection]
                sweeps.append(
                    LiquiditySweep(
                        sweep_id=len(sweeps) + 1,
                        direction="bearish",
                        side="buy_side",
                        swept_swing_level=latest_high.level,
                        swing_index=latest_high.index,
                        swing_time=bars[latest_high.index].end,
                        swing_confirm_time=bars[latest_high.confirmed_at_index].end,
                        breach_time=bar.end,
                        breach_index=i,
                        breach_extreme=bar.high,
                        breach_distance_atr=(bar.high - latest_high.level) / frozen_atr,
                        rejection_close_time=rejection_bar.end,
                        rejection_index=rejection,
                        rejection_close_price=rejection_bar.close,
                        bars_breach_to_rejection=rejection - i,
                        frozen_atr=frozen_atr,
                        session=classify_session(rejection_bar.end),
                        year=rejection_bar.end.year,
                        segment_id=bar.segment_id,
                    )
                )

    return sweeps


def find_sell_side_rejection(bars: list[Bar], breach_index: int, swing_level: float) -> Optional[int]:
    segment_id = bars[breach_index].segment_id
    for offset in range(1, REJECTION_WINDOW_BARS + 1):
        i = breach_index + offset
        if i >= len(bars) or bars[i].segment_id != segment_id:
            return None
        if bars[i].close > swing_level:
            return i
    return None


def find_buy_side_rejection(bars: list[Bar], breach_index: int, swing_level: float) -> Optional[int]:
    segment_id = bars[breach_index].segment_id
    for offset in range(1, REJECTION_WINDOW_BARS + 1):
        i = breach_index + offset
        if i >= len(bars) or bars[i].segment_id != segment_id:
            return None
        if bars[i].close < swing_level:
            return i
    return None


SWEEP_FIELDS = [
    "sweep_id",
    "direction",
    "side",
    "swept_swing_level",
    "swing_index",
    "swing_time",
    "swing_confirm_time",
    "breach_time",
    "breach_index",
    "breach_extreme",
    "breach_distance_atr",
    "rejection_close_time",
    "rejection_index",
    "rejection_close_price",
    "bars_breach_to_rejection",
    "frozen_atr",
    "session",
    "year",
    "segment_id",
]


FIRST10_FIELDS = [
    "sweep_id",
    "direction",
    "swept_swing_level",
    "swing_confirm_time",
    "breach_time",
    "breach_extreme",
    "rejection_close_time",
    "rejection_close_price",
    "breach_distance_atr",
    "frozen_atr",
    "bars_breach_to_rejection",
]


def sweep_to_row(sweep: LiquiditySweep) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in SWEEP_FIELDS:
        value = getattr(sweep, field)
        if isinstance(value, datetime):
            row[field] = fmt_ts(value)
        elif isinstance(value, float):
            row[field] = f"{value:.6f}"
        else:
            row[field] = value
    return row


def write_sweeps(path: Path, sweeps: list[LiquiditySweep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for sweep in sweeps:
            writer.writerow(sweep_to_row(sweep))


def write_first10(path: Path, sweeps: list[LiquiditySweep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIRST10_FIELDS)
        writer.writeheader()
        for sweep in sweeps[:10]:
            row = sweep_to_row(sweep)
            writer.writerow({field: row[field] for field in FIRST10_FIELDS})


def print_report(sweeps: list[LiquiditySweep], bars: list[Bar], sweeps_out: Path, first10_out: Path) -> None:
    by_year = Counter(item.year for item in sweeps)
    by_session = Counter(item.session for item in sweeps)
    by_direction = Counter(item.direction for item in sweeps)
    breach_values = [item.breach_distance_atr for item in sweeps]
    rejection_offsets = [item.bars_breach_to_rejection for item in sweeps]

    print("\nLiquidity Sweep Phase A Detection Report")
    print("=" * 43)
    print("Detector only: no outcomes, RR, cost, or baseline measured.")
    print(f"Completed M15 bars: {len(bars):,}")
    print(f"Detected sweeps:    {len(sweeps):,}")
    print(f"Sweeps CSV:         {sweeps_out}")
    print(f"First 10 CSV:       {first10_out}")
    print(
        f"Locked filters: breach >= {BREACH_ATR_THRESHOLD:.2f} * ATR({ATR_PERIOD}), "
        f"rejection close within 1-{REJECTION_WINDOW_BARS} bars"
    )

    print("\nDirection")
    print(f"  bullish-reversal {by_direction['bullish']:,}")
    print(f"  bearish-reversal {by_direction['bearish']:,}")

    print("\nPer Year")
    for year in range(2016, 2027):
        print(f"  {year}: {by_year[year]:,}")

    print("\nPer Session")
    for session in SESSION_ORDER:
        print(f"  {session:<11} {by_session[session]:,}")

    print("\nBreach Distance ATR")
    if breach_values:
        print(f"  min:    {min(breach_values):.3f}")
        print(f"  median: {median(breach_values):.3f}")
        print(f"  p90:    {percentile(breach_values, 0.90):.3f}")
        print(f"  max:    {max(breach_values):.3f}")
    else:
        print("  n/a")

    print("\nBars Breach -> Rejection Close")
    offset_counts = Counter(rejection_offsets)
    for offset in range(1, REJECTION_WINDOW_BARS + 1):
        print(f"  {offset}: {offset_counts[offset]:,}")

    print("\nVisual Verification Gate")
    print("  Exported first 10 sweeps for chart review.")
    print("  Stop here before any outcome, RR, cost, or baseline work.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None, help="Dukascopy XAUUSD tick CSV")
    parser.add_argument("--sweeps-out", type=Path, default=Path("research/liquidity_sweep_sweeps.csv"))
    parser.add_argument("--first10-out", type=Path, default=Path("research/liquidity_sweep_first10.csv"))
    parser.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)
    parser.add_argument("--max-rows", type=int, default=None, help="development smoke-test row limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tick_path = args.ticks or default_tick_path()
    bars = load_bars(tick_path, gap_minutes=args.gap_minutes, max_rows=args.max_rows)
    atr = compute_atr(bars)
    sweeps = detect_sweeps(bars, atr)
    write_sweeps(args.sweeps_out, sweeps)
    write_first10(args.first10_out, sweeps)
    print_report(sweeps, bars, args.sweeps_out, args.first10_out)


if __name__ == "__main__":
    main()
