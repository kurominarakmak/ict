"""
Gross/net simple compression-breakout audit with matched random baseline.

Purpose:
- Separate raw expansion edge from spread drag.
- Classify replication as:
  A: mean <= 0 and not better than matched random baseline.
  B: mean > 0 and better than random, but CI crosses zero.
  C: mean > 0, better than random, and CI clears zero.

The strategy is simple-follow only:
- Same compression definition as the prior audit.
- Entry at broken compression range edge in breakout direction.
- Stop = 1.0 * ATR(14) at breakout bar.
- Fixed targets: 1R, 1.5R, 2R.
- Horizons: 10 and 20 M15 bars.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=base.timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=base.timezone.utc)
RR_VARIANTS = ("rr_1", "rr_1_5", "rr_2")
HORIZONS = (10, 20)
RANDOM_SEED = 20260629


@dataclass(frozen=True)
class DirectionTrade:
    symbol: str
    signal: str
    exit_variant: str
    horizon: int
    event_id: int
    entry_time: datetime
    gross_r: float
    net_r: float


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def period_filter(rows: list[DirectionTrade], period: str) -> list[DirectionTrade]:
    if period == "all":
        return rows
    if period == "train_2016_2021":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def simulate_direction(
    symbol: str,
    signal: str,
    bars: list[base.DeltaBar],
    event: simple.BreakoutEvent,
    exit_variant: str,
    horizon: int,
    direction: int,
    spread: float,
) -> DirectionTrade | None:
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    risk = bars[entry_index].atr14
    if risk is None or risk <= 0:
        return None
    # Same breakout-edge price as the simple-follow strategy. Random direction
    # tests directional value after the breakout edge is known.
    entry = event.range_high if event.breakout_direction == 1 else event.range_low
    stop = entry - direction * risk
    rr = {"rr_1": 1.0, "rr_1_5": 1.5, "rr_2": 2.0}[exit_variant]
    target = entry + direction * rr * risk
    end_index = simple.segment_end_index(bars, eval_start, horizon)
    gross_r = 0.0
    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r = direction * (fill - entry) / risk
            break
        if target_hit:
            gross_r = rr
            break
    else:
        gross_r = direction * (bars[end_index].close - entry) / risk
    return DirectionTrade(
        symbol=symbol,
        signal=signal,
        exit_variant=exit_variant,
        horizon=horizon,
        event_id=event.event_id,
        entry_time=bars[entry_index].start,
        gross_r=gross_r,
        net_r=gross_r - spread / risk,
    )


def build_asset_rows(symbol: str, bars: list[base.DeltaBar], spread: float) -> list[DirectionTrade]:
    events = simple.detect_compression_breakouts(bars)
    rng = random.Random(f"{RANDOM_SEED}-{symbol}")
    rows: list[DirectionTrade] = []
    for event in events:
        random_dir = 1 if rng.random() >= 0.5 else -1
        directions = {
            "simple_follow": event.breakout_direction,
            "random": random_dir,
        }
        for signal, direction in directions.items():
            for exit_variant in RR_VARIANTS:
                for horizon in HORIZONS:
                    row = simulate_direction(symbol, signal, bars, event, exit_variant, horizon, direction, spread)
                    if row is not None:
                        rows.append(row)
    return rows


def summarize(rows: list[DirectionTrade], value_field: str) -> dict[str, float]:
    vals = [getattr(r, value_field) for r in rows]
    n, m, lo, hi = ci(vals)
    return {
        "n": n,
        "mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "win_rate": sum(v > 0 for v in vals) / n if n else math.nan,
    }


def grouped(rows: list[DirectionTrade], period: str) -> dict[tuple[str, str, str, int], list[DirectionTrade]]:
    out: dict[tuple[str, str, str, int], list[DirectionTrade]] = {}
    for row in period_filter(rows, period):
        out.setdefault((row.symbol, row.signal, row.exit_variant, row.horizon), []).append(row)
    return out


def classify(strategy: dict[str, float], random_baseline: dict[str, float]) -> str:
    beats_random = strategy["mean"] > random_baseline["mean"]
    if strategy["mean"] <= 0 and not beats_random:
        return "A_absent"
    if strategy["mean"] > 0 and beats_random and strategy["ci_low"] > 0:
        return "C_confirmed"
    if strategy["mean"] > 0 and beats_random:
        return "B_present_unproven"
    return "A_absent"


def print_matrix(rows: list[DirectionTrade], period: str) -> None:
    g = grouped(rows, period)
    print(f"\nGROSS_NET_RANDOM_{period.upper()}")
    print("period,symbol,exit,horizon,signal,n,gross_mean,gross_ci_low,gross_ci_high,gross_win_rate,net_mean,net_ci_low,net_ci_high,net_win_rate,beats_random_gross,beats_random_net,gross_class,net_class")
    symbols = sorted({r.symbol for r in rows})
    for symbol in symbols:
        for exit_variant in RR_VARIANTS:
            for horizon in HORIZONS:
                random_rows = g.get((symbol, "random", exit_variant, horizon), [])
                rand_gross = summarize(random_rows, "gross_r")
                rand_net = summarize(random_rows, "net_r")
                for signal in ("simple_follow", "random"):
                    signal_rows = g.get((symbol, signal, exit_variant, horizon), [])
                    gross = summarize(signal_rows, "gross_r")
                    net = summarize(signal_rows, "net_r")
                    beats_gross = gross["mean"] > rand_gross["mean"] if signal != "random" else False
                    beats_net = net["mean"] > rand_net["mean"] if signal != "random" else False
                    gross_class = classify(gross, rand_gross) if signal != "random" else "baseline"
                    net_class = classify(net, rand_net) if signal != "random" else "baseline"
                    print(
                        f"{period},{symbol},{exit_variant},{horizon},{signal},{int(gross['n'])},"
                        f"{gross['mean']:.6f},{gross['ci_low']:.6f},{gross['ci_high']:.6f},{gross['win_rate']:.2%},"
                        f"{net['mean']:.6f},{net['ci_low']:.6f},{net['ci_high']:.6f},{net['win_rate']:.2%},"
                        f"{beats_gross},{beats_net},{gross_class},{net_class}"
                    )


def available_tick_files() -> list[Path]:
    return sorted(Path("data").glob("*-TICK-No Session.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xag-ticks", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--xag-cache", type=Path, default=Path("data/xagusd_m15_delta_bars.csv"))
    parser.add_argument("--xau-spread", type=float, default=0.20)
    parser.add_argument("--xag-spread", type=float, default=0.02)
    args = parser.parse_args()

    assets = [
        ("XAUUSD", args.xau_ticks, args.xau_cache, args.xau_spread),
        ("XAGUSD", args.xag_ticks, args.xag_cache, args.xag_spread),
    ]
    rows: list[DirectionTrade] = []
    contexts = []
    for symbol, tick_path, cache_path, spread in assets:
        bars = simple.load_symbol_bars(symbol, tick_path, cache_path)
        events = simple.detect_compression_breakouts(bars)
        rows.extend(build_asset_rows(symbol, bars, spread))
        contexts.append((symbol, tick_path, len(bars), bars[0].start, bars[-1].end, len(events), spread))

    print("GROSS_RANDOM_AUDIT_CONTEXT")
    print(f"available_tick_files={';'.join(str(p) for p in available_tick_files())}")
    print("additional_assets_run=none; only XAUUSD and XAGUSD tick files exist under data/")
    print("strategy=compressed range breakout simple-follow, entry at broken range edge, stop=1*ATR14")
    print(f"random_seed={RANDOM_SEED}")
    print("classification=A_absent if mean<=0 and not better than random; B_present_unproven if mean>0 and beats random but CI crosses zero; C_confirmed if mean>0, beats random, and CI clears zero")
    for symbol, tick_path, bars_n, start, end, events_n, spread in contexts:
        print(f"symbol_context={symbol},tick_file={tick_path},bars={bars_n},date_range={start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC,compression_events={events_n},spread={spread:.4f}")

    for period in ("all", "train_2016_2021", "test_2022_2026"):
        print_matrix(rows, period)


if __name__ == "__main__":
    main()
