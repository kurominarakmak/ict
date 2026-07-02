"""
H-2026-REV-01: failed-breakout reversal audit.

Research-only. Does not touch the live bot.

Pre-registered mechanism:
Compression breakouts that stop out may be liquidity grabs / stop hunts. The
stop-hit bar of the validated base strategy becomes a look-ahead-free signal to
enter the opposite direction on the next bar open.

No threshold search, no parameter variants.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
RR = 1.5
HORIZON = 10
SPREAD = 0.20
BOOT_N = 1000
SEED = 20260702


@dataclass(frozen=True)
class BaseStopSignal:
    event_id: int
    failed_direction: int
    stop_index: int
    stop_time: datetime
    bars_to_stop: int


@dataclass(frozen=True)
class ReversalTrade:
    label: str
    signal_id: int
    entry_time: datetime
    year: int
    session: str
    direction: int
    gross_r: float
    net_r: float
    win: bool
    exit_reason: str
    bars_held: int
    bars_to_stop: int | None = None


def q(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(vals: list[float], seed: str) -> tuple[float, float]:
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(BOOT_N):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N)]


def detect_base_stop_signals(bars: list[base.DeltaBar]) -> list[BaseStopSignal]:
    signals: list[BaseStopSignal] = []
    for event in ablate.detect_compression(bars):
        risk = ablate.risk_at_setup_end(bars, event)
        if risk is None:
            continue
        entry_index = event.breakout_index
        eval_start = entry_index + 1
        if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
            continue
        entry = event.range_high if event.direction == 1 else event.range_low
        stop = entry - event.direction * risk
        target = entry + event.direction * RR * risk
        end_index = simple.segment_end_index(bars, eval_start, HORIZON)
        for i in range(eval_start, end_index + 1):
            bar = bars[i]
            stop_hit = bar.low <= stop if event.direction == 1 else bar.high >= stop
            target_hit = bar.high >= target if event.direction == 1 else bar.low <= target
            # Same intrabar convention as the validated base audit: stop first.
            if stop_hit:
                signals.append(BaseStopSignal(event.event_id, event.direction, i, bar.start, i - entry_index))
                break
            if target_hit:
                break
    return signals


def simulate_entry(
    bars: list[base.DeltaBar],
    signal_id: int,
    entry_index: int,
    direction: int,
    label: str,
    bars_to_stop: int | None = None,
) -> ReversalTrade | None:
    if entry_index >= len(bars):
        return None
    entry_bar = bars[entry_index]
    risk = entry_bar.atr14
    if risk is None or risk <= 0:
        return None
    entry = entry_bar.open
    stop = entry - direction * risk
    target = entry + direction * RR * risk
    end_index = simple.segment_end_index(bars, entry_index, HORIZON)
    gross_r = direction * (bars[end_index].close - entry) / risk
    exit_reason = "force_close"
    exit_index = end_index
    for i in range(entry_index, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r = direction * (fill - entry) / risk
            exit_reason = "stop"
            exit_index = i
            break
        if target_hit:
            gross_r = RR
            exit_reason = "target"
            exit_index = i
            break
    net_r = gross_r - SPREAD / risk
    return ReversalTrade(
        label=label,
        signal_id=signal_id,
        entry_time=entry_bar.start,
        year=entry_bar.start.year,
        session=entry_bar.session,
        direction=direction,
        gross_r=gross_r,
        net_r=net_r,
        win=net_r > 0,
        exit_reason=exit_reason,
        bars_held=exit_index - entry_index,
        bars_to_stop=bars_to_stop,
    )


def build_reversal_and_controls(bars: list[base.DeltaBar], signals: list[BaseStopSignal]) -> list[ReversalTrade]:
    rng = random.Random(SEED)
    rows: list[ReversalTrade] = []
    for sig in signals:
        entry_index = sig.stop_index + 1
        if entry_index >= len(bars) or bars[entry_index].segment_id != bars[sig.stop_index].segment_id:
            continue
        specs = [
            ("reversal", -sig.failed_direction),
            ("same_direction_rebreak", sig.failed_direction),
            ("matched_random_direction", 1 if rng.random() >= 0.5 else -1),
        ]
        for label, direction in specs:
            trade = simulate_entry(bars, sig.event_id, entry_index, direction, label, sig.bars_to_stop)
            if trade is not None:
                rows.append(trade)
    return rows


def valid_random_indices(bars: list[base.DeltaBar], start_year: int, end_year: int) -> list[int]:
    out = []
    for i, bar in enumerate(bars):
        if not (start_year <= bar.start.year <= end_year):
            continue
        if bar.atr14 is None or bar.atr14 <= 0:
            continue
        if simple.segment_end_index(bars, i, HORIZON) <= i:
            continue
        out.append(i)
    return out


def build_unconditional_random(bars: list[base.DeltaBar], reversal_rows: list[ReversalTrade]) -> list[ReversalTrade]:
    rng = random.Random(f"{SEED}-unconditional")
    rows: list[ReversalTrade] = []
    period_counts = {
        "train": sum(1 for r in reversal_rows if r.entry_time <= TRAIN_END),
        "test": sum(1 for r in reversal_rows if r.entry_time >= TEST_START),
    }
    period_ranges = {"train": (2016, 2021), "test": (2022, 2026)}
    sid = 1
    for period, count in period_counts.items():
        candidates = valid_random_indices(bars, *period_ranges[period])
        for entry_index in rng.sample(candidates, min(count, len(candidates))):
            direction = 1 if rng.random() >= 0.5 else -1
            trade = simulate_entry(bars, sid, entry_index, direction, "unconditional_random_entry")
            sid += 1
            if trade is not None:
                rows.append(trade)
    return rows


def period_filter(rows: list[ReversalTrade], period: str) -> list[ReversalTrade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def summary_line(label: str, period: str, rows: list[ReversalTrade]) -> str:
    subset = period_filter(rows, period)
    vals = [r.net_r for r in subset]
    gross = [r.gross_r for r in subset]
    lo, hi = bootstrap_ci(vals, f"{SEED}-{label}-{period}")
    return (
        f"{label},{period},{len(subset)},{sum(r.win for r in subset)/len(subset):.2%},"
        f"{mean(gross):.4f},{mean(vals):.4f},{lo:.4f},{hi:.4f}"
        if subset
        else f"{label},{period},0,n/a,n/a,n/a,n/a,n/a"
    )


def tercile_buckets(vals: list[int]) -> tuple[float, float]:
    return q([float(v) for v in vals], 1 / 3), q([float(v) for v in vals], 2 / 3)


def diagnostics(rows: list[ReversalTrade]) -> list[str]:
    rev = [r for r in rows if r.label == "reversal" and r.bars_to_stop is not None]
    lo, hi = tercile_buckets([int(r.bars_to_stop or 0) for r in rev])
    lines = ["diagnostic,type,bucket,n,win_rate,mean_net_r,ci_low,ci_high"]
    for name, subset in (
        ("fast_stop", [r for r in rev if float(r.bars_to_stop or 0) <= lo]),
        ("medium_stop", [r for r in rev if lo < float(r.bars_to_stop or 0) <= hi]),
        ("slow_stop", [r for r in rev if float(r.bars_to_stop or 0) > hi]),
    ):
        vals = [r.net_r for r in subset]
        ci = bootstrap_ci(vals, f"{SEED}-diag-bars-{name}") if subset else (math.nan, math.nan)
        lines.append(f"bars_to_stop,{name},{name},{len(subset)},{sum(r.win for r in subset)/len(subset):.2%},{mean(vals):.4f},{ci[0]:.4f},{ci[1]:.4f}" if subset else f"bars_to_stop,{name},{name},0,n/a,n/a,n/a,n/a")
    for session in sorted(set(r.session for r in rev)):
        subset = [r for r in rev if r.session == session]
        vals = [r.net_r for r in subset]
        ci = bootstrap_ci(vals, f"{SEED}-diag-session-{session}")
        lines.append(f"session,{session},{session},{len(subset)},{sum(r.win for r in subset)/len(subset):.2%},{mean(vals):.4f},{ci[0]:.4f},{ci[1]:.4f}")
    return lines


def yearly(rows: list[ReversalTrade]) -> list[str]:
    rev = [r for r in rows if r.label == "reversal"]
    out = ["year,n,win_rate,mean_net_r,total_net_r"]
    for year in sorted(set(r.year for r in rev)):
        subset = [r for r in rev if r.year == year]
        out.append(f"{year},{len(subset)},{sum(r.win for r in subset)/len(subset):.2%},{mean([r.net_r for r in subset]):.4f},{sum(r.net_r for r in subset):.2f}")
    return out


def update_registry(path: Path, result: str | None = None) -> None:
    base_line = (
        "- 2026-07-02: H-2026-REV-01 registered. Failed-breakout reversal hypothesis: "
        "validated compression trades that hit 1R stop become opposite-direction next-bar-open "
        "signals with 1R SL, 1.5R TP, 10-bar force close, $0.20 spread; no parameter tuning.\n"
    )
    text = path.read_text() if path.exists() else "# Hypothesis Registry\n\n"
    if base_line.strip() not in text:
        with path.open("a") as handle:
            if not path.exists() or path.stat().st_size == 0:
                handle.write("# Hypothesis Registry\n\n")
            handle.write(base_line)
    if result:
        result_line = f"- 2026-07-02: H-2026-REV-01 result: {result}\n"
        text = path.read_text()
        if result_line.strip() not in text:
            with path.open("a") as handle:
                handle.write(result_line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--registry", type=Path, default=Path("research/hypothesis_registry.md"))
    args = parser.parse_args()

    update_registry(args.registry)
    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    signals = detect_base_stop_signals(bars)
    rows = build_reversal_and_controls(bars, signals)
    reversal = [r for r in rows if r.label == "reversal"]
    rows.extend(build_unconditional_random(bars, reversal))

    print("FAILED_BREAKOUT_REVERSAL_AUDIT")
    print("rules=base validated compression stop-outs -> opposite next-bar-open reversal; ATR at reversal entry; 1R SL; 1.5R TP; 10-bar segment-aware force close; $0.20 spread")
    print(f"base_stop_signals={len(signals)},executable_reversal_signals={len(reversal)}")
    print("\nSUMMARY")
    print("label,period,n,win_rate,mean_gross_r,mean_net_r,ci_low,ci_high")
    for label in ("reversal", "matched_random_direction", "same_direction_rebreak", "unconditional_random_entry"):
        label_rows = [r for r in rows if r.label == label]
        for period in ("full", "train", "test"):
            print(summary_line(label, period, label_rows))

    print("\nEXPLORATORY_DIAGNOSTICS")
    for line in diagnostics(rows):
        print(line)

    print("\nYEARLY_REVERSAL")
    for line in yearly(rows):
        print(line)

    rev_train = period_filter(reversal, "train")
    rev_test = period_filter(reversal, "test")
    controls = {label: [r for r in rows if r.label == label] for label in ("matched_random_direction", "same_direction_rebreak")}
    passes = (
        mean([r.net_r for r in rev_train]) > 0
        and mean([r.net_r for r in rev_test]) > 0
        and bootstrap_ci([r.net_r for r in rev_train], "gate-train")[0] > 0
        and bootstrap_ci([r.net_r for r in rev_test], "gate-test")[0] > 0
        and all(mean([r.net_r for r in reversal]) > mean([r.net_r for r in control]) for control in controls.values())
    )
    if passes:
        result = "PASS under pre-registered gates; reversal positive train/test CI and beats matched random plus same-direction controls. Candidate second strategy only, requires separate demo."
    else:
        result = "FAIL under pre-registered gates; reversal does not clear train/test net-of-cost CI and/or does not beat required controls. Hypothesis closed unless re-registered with new data."
    print("\nVERDICT")
    print(result)
    update_registry(args.registry, result)


if __name__ == "__main__":
    main()
