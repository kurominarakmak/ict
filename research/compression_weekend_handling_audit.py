"""
Weekend/session-gap handling audit for XAUUSD compression breakout-following.

Findings this script is designed to expose:
- The current simple breakout simulator does NOT step across segment gaps.
  `segment_end_index()` force-closes at the last available bar before any
  market gap, including daily 1h pauses and Friday->Sunday weekend gaps.
- A hypothetical hold-across-gaps mode is included only to quantify what would
  happen if Friday and Sunday/Monday bars were treated as adjacent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=base.timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=base.timezone.utc)
SPREAD = 0.20
RR_VARIANTS = ("rr_1_5", "rr_2")
HORIZON = 10


@dataclass(frozen=True)
class WeekendTrade:
    mode: str
    rr: str
    entry_time: datetime
    net_r: float
    exit_reason: str
    truncated_by_gap: bool
    weekend_gap_in_horizon: bool


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def first_gap_index(bars: list[base.DeltaBar], entry_index: int, horizon: int) -> int | None:
    segment = bars[entry_index].segment_id
    for i in range(entry_index + 1, min(len(bars), entry_index + horizon + 1)):
        if bars[i].segment_id != segment:
            return i
    return None


def gap_is_weekend(bars: list[base.DeltaBar], gap_index: int) -> bool:
    prev = bars[gap_index - 1]
    curr = bars[gap_index]
    hours = (curr.start - prev.end).total_seconds() / 3600
    return hours >= 24 or prev.end.weekday() == 4


def end_index_for_mode(bars: list[base.DeltaBar], event: simple.BreakoutEvent, horizon: int, mode: str) -> tuple[int, bool, bool]:
    entry = event.breakout_index
    natural = min(len(bars) - 1, entry + horizon)
    gap_idx = first_gap_index(bars, entry, horizon)
    weekend = gap_idx is not None and gap_is_weekend(bars, gap_idx)
    if mode == "current_segment_close":
        end = simple.segment_end_index(bars, entry + 1, horizon)
        return end, gap_idx is not None, weekend
    if mode == "hypothetical_hold_across_gaps":
        return natural, False, weekend
    raise ValueError(mode)


def simulate(bars: list[base.DeltaBar], event: simple.BreakoutEvent, rr_name: str, mode: str) -> WeekendTrade | None:
    risk = bars[event.setup_end].atr14
    if risk is None or risk <= 0:
        return None
    rr = {"rr_1_5": 1.5, "rr_2": 2.0}[rr_name]
    direction = event.breakout_direction
    entry = event.range_high if direction == 1 else event.range_low
    stop = entry - direction * risk
    target = entry + direction * rr * risk
    end_index, truncated, weekend = end_index_for_mode(bars, event, HORIZON, mode)
    gross = 0.0
    reason = "force_close"
    for i in range(event.breakout_index + 1, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross = direction * (fill - entry) / risk
            reason = "stop"
            break
        if target_hit:
            gross = rr
            reason = "target"
            break
    else:
        gross = direction * (bars[end_index].close - entry) / risk
    return WeekendTrade(mode, rr_name, bars[event.breakout_index].start, gross - SPREAD / risk, reason, truncated, weekend)


def period(rows: list[WeekendTrade], name: str) -> list[WeekendTrade]:
    if name == "all":
        return rows
    if name == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if name == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(name)


def main() -> None:
    bars = simple.load_symbol_bars("XAUUSD", Path("data/2026.6.15XAUUSD-TICK-No Session.csv"), Path("data/xauusd_m15_delta_bars.csv"))
    events = simple.detect_compression_breakouts(bars)
    rows: list[WeekendTrade] = []
    for event in events:
        for rr in RR_VARIANTS:
            for mode in ("current_segment_close", "hypothetical_hold_across_gaps"):
                trade = simulate(bars, event, rr, mode)
                if trade:
                    rows.append(trade)

    affected_events = set()
    affected_weekend = set()
    for event in events:
        gap_idx = first_gap_index(bars, event.breakout_index, HORIZON)
        if gap_idx is not None:
            affected_events.add(event.event_id)
            if gap_is_weekend(bars, gap_idx):
                affected_weekend.add(event.event_id)

    print("WEEKEND_HANDLING_CONTEXT")
    print(f"events={len(events)}")
    print(f"entry_windows_cross_any_segment_gap_if_not_truncated={len(affected_events)}")
    print(f"entry_windows_cross_weekend_gap_if_not_truncated={len(affected_weekend)}")
    print("current_backtest_behavior=segment_end_index stops at last available bar before any segment gap; it does not hold across Friday->Sunday weekend gaps")

    print("\nWEEKEND_RETURNS")
    print("period,mode,rr,n,net_mean,ci_low,ci_high,win_rate,truncated_pct,weekend_gap_pct,target,stop,force")
    for per in ("all", "train", "test"):
        per_rows = period(rows, per)
        for mode in ("current_segment_close", "hypothetical_hold_across_gaps"):
            for rr in RR_VARIANTS:
                subset = [r for r in per_rows if r.mode == mode and r.rr == rr]
                vals = [r.net_r for r in subset]
                n, m, lo, hi = ci(vals)
                print(
                    f"{per},{mode},{rr},{n},{m:.6f},{lo:.6f},{hi:.6f},"
                    f"{(sum(v > 0 for v in vals) / n if n else math.nan):.2%},"
                    f"{(sum(r.truncated_by_gap for r in subset) / n if n else math.nan):.2%},"
                    f"{(sum(r.weekend_gap_in_horizon for r in subset) / n if n else math.nan):.2%},"
                    f"{sum(r.exit_reason == 'target' for r in subset)},"
                    f"{sum(r.exit_reason == 'stop' for r in subset)},"
                    f"{sum(r.exit_reason == 'force_close' for r in subset)}"
                )


if __name__ == "__main__":
    main()
