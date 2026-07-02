"""
H-2026-EXIT-01: compression exit trailing audit.

Research-only. Does not touch the live bot.

This is the backtest audit for the trailing hypothesis whose live shadow
tracker was already started. Config C mirrors
research/live_shadow_trail_tracker.py::simulate_config_c semantics:
- initial stop 1.0 ATR
- no fixed TP
- arm when a closed bar's favorable extreme reaches +1.0R
- after arming, trail 1.0 ATR behind the best closed-bar favorable extreme
- 10-bar force close
- active stop is checked before the current bar updates the trail
- stop fills use the same gap-through conservative convention as the shadow
  tracker and prior audits.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
HORIZON = 10
SPREAD = 0.20
BOOT_N = 1000
SEED = 20260702


@dataclass(frozen=True)
class ExitTrade:
    config: str
    event_id: int
    entry_time: datetime
    year: int
    net_r: float
    gross_r: float
    win: bool
    exit_reason: str
    bars_held: int
    mfe_r: float


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


def max_drawdown(vals: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def mfe_for_path(bars: list[base.DeltaBar], start: int, end: int, entry: float, direction: int, risk: float) -> float:
    mfe = 0.0
    for i in range(start, end + 1):
        bar = bars[i]
        if direction == 1:
            mfe = max(mfe, bar.high - entry)
        else:
            mfe = max(mfe, entry - bar.low)
    return mfe / risk


def simulate_fixed(
    bars: list[base.DeltaBar],
    event: ablate.Event,
    rr: float,
    config: str,
) -> ExitTrade | None:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return None
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    direction = event.direction
    entry = event.range_high if direction == 1 else event.range_low
    stop = entry - direction * risk
    target = entry + direction * rr * risk
    end_index = simple.segment_end_index(bars, eval_start, HORIZON)
    gross_r = direction * (bars[end_index].close - entry) / risk
    exit_reason = "force_close"
    exit_index = end_index
    for i in range(eval_start, end_index + 1):
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
            gross_r = rr
            exit_reason = "target"
            exit_index = i
            break
    net_r = gross_r - SPREAD / risk
    mfe = mfe_for_path(bars, eval_start, end_index, entry, direction, risk)
    return ExitTrade(config, event.event_id, bars[entry_index].start, bars[entry_index].start.year, net_r, gross_r, net_r > 0, exit_reason, exit_index - entry_index, mfe)


def simulate_trail(
    bars: list[base.DeltaBar],
    event: ablate.Event,
    trail_mult: float,
    config: str,
) -> ExitTrade | None:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return None
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    d = event.direction
    entry = event.range_high if d == 1 else event.range_low
    stop = entry - d * risk
    arm_level = entry + d * risk
    armed = False
    best = entry
    end_index = simple.segment_end_index(bars, eval_start, HORIZON)
    exit_price = bars[end_index].close
    exit_reason = "force_close"
    exit_index = end_index

    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if d == 1 else bar.high >= stop
        if stop_hit:
            exit_price = min(stop, bar.low) if d == 1 else max(stop, bar.high)
            exit_reason = "trail_hit" if armed else "initial_stop"
            exit_index = i
            break

        arm_hit = bar.high >= arm_level if d == 1 else bar.low <= arm_level
        if arm_hit:
            armed = True

        if armed:
            if d == 1:
                best = max(best, bar.high)
                stop = max(stop, best - trail_mult * risk)
            else:
                best = min(best, bar.low)
                stop = min(stop, best + trail_mult * risk)

        if i == end_index:
            exit_price = bar.close
            exit_reason = "force_close"

    gross_r = d * (exit_price - entry) / risk
    net_r = gross_r - SPREAD / risk
    mfe = mfe_for_path(bars, eval_start, end_index, entry, d, risk)
    return ExitTrade(config, event.event_id, bars[entry_index].start, bars[entry_index].start.year, net_r, gross_r, net_r > 0, exit_reason, exit_index - entry_index, mfe)


def build_trades(bars: list[base.DeltaBar]) -> list[ExitTrade]:
    rows: list[ExitTrade] = []
    for event in ablate.detect_compression(bars):
        specs = [
            simulate_fixed(bars, event, 1.5, "A_fixed_1p5R"),
            simulate_fixed(bars, event, 2.0, "B_fixed_2R"),
            simulate_trail(bars, event, 1.0, "C_trail_1ATR"),
            simulate_trail(bars, event, 1.5, "Cprime_trail_1p5ATR"),
        ]
        if all(t is not None for t in specs):
            rows.extend([t for t in specs if t is not None])
    return rows


def period_rows(rows: list[ExitTrade], period: str) -> list[ExitTrade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def exit_dist(rows: list[ExitTrade]) -> str:
    total = len(rows)
    parts = []
    for reason in sorted(set(r.exit_reason for r in rows)):
        n = sum(1 for r in rows if r.exit_reason == reason)
        parts.append(f"{reason}:{n}/{total}={n/total:.1%}")
    return "|".join(parts)


def summary_line(config: str, period: str, rows: list[ExitTrade]) -> str:
    subset = [r for r in period_rows(rows, period) if r.config == config]
    vals = [r.net_r for r in subset]
    lo, hi = bootstrap_ci(vals, f"{SEED}-{config}-{period}")
    capture = [r.net_r / r.mfe_r for r in subset if r.mfe_r > 0]
    ordered = sorted(subset, key=lambda r: (r.entry_time, r.event_id))
    return (
        f"{config},{period},{len(subset)},{sum(r.win for r in subset)/len(subset):.2%},"
        f"{mean(vals):.4f},{lo:.4f},{hi:.4f},{mean([r.bars_held for r in subset]):.2f},"
        f"{mean(capture):.4f},{max_drawdown([r.net_r for r in ordered]):.2f},{exit_dist(subset)}"
    )


def yearly_lines(rows: list[ExitTrade]) -> list[str]:
    out = ["year,A_net_R,C_net_R,C_minus_A,A_total_R,C_total_R"]
    years = sorted(set(r.year for r in rows))
    for year in years:
        a = [r.net_r for r in rows if r.config == "A_fixed_1p5R" and r.year == year]
        c = [r.net_r for r in rows if r.config == "C_trail_1ATR" and r.year == year]
        if a and c:
            out.append(f"{year},{mean(a):.4f},{mean(c):.4f},{mean(c)-mean(a):.4f},{sum(a):.2f},{sum(c):.2f}")
    return out


def pass_gate(rows: list[ExitTrade]) -> tuple[bool, str]:
    train_a = [r.net_r for r in rows if r.config == "A_fixed_1p5R" and r.entry_time <= TRAIN_END]
    test_a = [r.net_r for r in rows if r.config == "A_fixed_1p5R" and r.entry_time >= TEST_START]
    train_c = [r.net_r for r in rows if r.config == "C_trail_1ATR" and r.entry_time <= TRAIN_END]
    test_c = [r.net_r for r in rows if r.config == "C_trail_1ATR" and r.entry_time >= TEST_START]
    train_c_ci = bootstrap_ci(train_c, f"{SEED}-gate-train-c")
    test_c_ci = bootstrap_ci(test_c, f"{SEED}-gate-test-c")
    a_train_dd = abs(max_drawdown([r.net_r for r in sorted([x for x in rows if x.config == "A_fixed_1p5R" and x.entry_time <= TRAIN_END], key=lambda r: r.entry_time)]))
    a_test_dd = abs(max_drawdown([r.net_r for r in sorted([x for x in rows if x.config == "A_fixed_1p5R" and x.entry_time >= TEST_START], key=lambda r: r.entry_time)]))
    c_train_dd = abs(max_drawdown([r.net_r for r in sorted([x for x in rows if x.config == "C_trail_1ATR" and x.entry_time <= TRAIN_END], key=lambda r: r.entry_time)]))
    c_test_dd = abs(max_drawdown([r.net_r for r in sorted([x for x in rows if x.config == "C_trail_1ATR" and x.entry_time >= TEST_START], key=lambda r: r.entry_time)]))
    beats_a = mean(train_c) > mean(train_a) and mean(test_c) > mean(test_a)
    clears_zero = train_c_ci[0] > 0 and test_c_ci[0] > 0
    dd_ok = c_train_dd <= 1.25 * a_train_dd and c_test_dd <= 1.25 * a_test_dd
    passed = beats_a and clears_zero and dd_ok
    detail = (
        f"C_train={mean(train_c):.4f} vs A_train={mean(train_a):.4f}; "
        f"C_test={mean(test_c):.4f} vs A_test={mean(test_a):.4f}; "
        f"C_CI_train=[{train_c_ci[0]:.4f},{train_c_ci[1]:.4f}], "
        f"C_CI_test=[{test_c_ci[0]:.4f},{test_c_ci[1]:.4f}]; "
        f"DD_train C/A={c_train_dd:.2f}/{a_train_dd:.2f}, DD_test C/A={c_test_dd:.2f}/{a_test_dd:.2f}"
    )
    if passed:
        return True, "PASS: C beats A in train/test, C CI clears zero in train/test, and C maxDD is within 25% of A. " + detail
    return False, "FAIL: C does not satisfy all pre-registered gates. " + detail


def update_registry(path: Path, result: str | None = None) -> None:
    registered = (
        "- 2026-07-02: H-2026-EXIT-01 registered. Compression trailing exit hypothesis: "
        "Config C initial 1ATR stop, arm at closed-bar +1R, trail 1ATR behind best closed-bar "
        "favorable extreme, no fixed TP, 10-bar force close; pass only if C beats A in train/test, "
        "C CI clears zero in train/test, and maxDD is not >25% worse than A.\n"
    )
    text = path.read_text() if path.exists() else "# Hypothesis Registry\n\n"
    if registered.strip() not in text:
        with path.open("a") as handle:
            if not path.exists() or path.stat().st_size == 0:
                handle.write("# Hypothesis Registry\n\n")
            handle.write(registered)
    if result:
        result_line = f"- 2026-07-02: H-2026-EXIT-01 result: {result}\n"
        text = path.read_text()
        if result_line.strip() not in text:
            with path.open("a") as handle:
                handle.write(result_line)


def build_report(rows: list[ExitTrade], result: str) -> str:
    configs = ("A_fixed_1p5R", "B_fixed_2R", "C_trail_1ATR", "Cprime_trail_1p5ATR")
    lines = [
        "COMPRESSION_EXIT_TRAILING_AUDIT",
        "rules=validated compression entries; setup-end ATR; range-edge entry; $0.20 spread; 10-bar segment-aware force close",
        "config_C=mirrors research/live_shadow_trail_tracker.py simulate_config_c semantics",
        "",
        "CONFIG_TABLE",
        "config,period,n,win_rate,mean_net_r,ci_low,ci_high,avg_bars_held,mean_mfe_capture,max_drawdown_R,exit_distribution",
    ]
    for config in configs:
        for period in ("train", "test"):
            lines.append(summary_line(config, period, rows))
    lines.extend(["", "YEARLY_C_VS_A"])
    lines.extend(yearly_lines(rows))
    lines.extend(["", "PASS_GATE", result])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--results", type=Path, default=Path("research/compression_exit_trailing_results.txt"))
    parser.add_argument("--registry", type=Path, default=Path("research/hypothesis_registry.md"))
    args = parser.parse_args()

    update_registry(args.registry)
    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    rows = build_trades(bars)
    passed, result = pass_gate(rows)
    report = build_report(rows, result)
    args.results.write_text(report)
    update_registry(args.registry, result)
    print(report, end="")


if __name__ == "__main__":
    main()
