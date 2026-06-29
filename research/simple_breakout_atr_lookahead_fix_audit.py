"""
Look-ahead audit for simple compression breakout-following.

Compares:
- BEFORE: risk = ATR14 on breakout bar (old backtest; look-ahead for range-edge
  intrabar entry).
- AFTER:  risk = ATR14 on compression-end bar (known when pending range-edge
  stop is placed; matches live bot).

Strategy:
- Same compression events.
- Entry at broken range edge.
- Stop = 1R, TP = 1/1.5/2R, force close at 10/20 bars.
- Simple-follow direction vs matched random baseline.
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
ATR_MODES = ("before_breakout_atr", "after_compression_atr")
RANDOM_SEED = 20260629


@dataclass(frozen=True)
class AtrModeTrade:
    symbol: str
    atr_mode: str
    signal: str
    exit_variant: str
    horizon: int
    event_id: int
    entry_time: datetime
    gross_r: float
    net_r: float
    risk_usd_oz: float
    spread_r: float
    exit_reason: str
    unresolved: bool
    false_breakout: bool


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def period_filter(rows: list[AtrModeTrade], period: str) -> list[AtrModeTrade]:
    if period == "all":
        return rows
    if period == "train_2016_2021":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def risk_for_mode(bars: list[base.DeltaBar], event: simple.BreakoutEvent, atr_mode: str) -> float | None:
    if atr_mode == "before_breakout_atr":
        atr = bars[event.breakout_index].atr14
    elif atr_mode == "after_compression_atr":
        atr = bars[event.setup_end].atr14
    else:
        raise ValueError(atr_mode)
    if atr is None or atr <= 0:
        return None
    return atr


def simulate(
    symbol: str,
    bars: list[base.DeltaBar],
    event: simple.BreakoutEvent,
    atr_mode: str,
    signal: str,
    direction: int,
    exit_variant: str,
    horizon: int,
    spread: float,
) -> AtrModeTrade | None:
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    risk = risk_for_mode(bars, event, atr_mode)
    if risk is None:
        return None
    entry = event.range_high if event.breakout_direction == 1 else event.range_low
    stop = entry - direction * risk
    rr = {"rr_1": 1.0, "rr_1_5": 1.5, "rr_2": 2.0}[exit_variant]
    target = entry + direction * rr * risk
    end_index = simple.segment_end_index(bars, eval_start, horizon)
    gross_r = 0.0
    exit_reason = "force_close"
    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r = direction * (fill - entry) / risk
            exit_reason = "stop"
            break
        if target_hit:
            gross_r = rr
            exit_reason = "target"
            break
    else:
        gross_r = direction * (bars[end_index].close - entry) / risk
    return AtrModeTrade(
        symbol=symbol,
        atr_mode=atr_mode,
        signal=signal,
        exit_variant=exit_variant,
        horizon=horizon,
        event_id=event.event_id,
        entry_time=bars[entry_index].start,
        gross_r=gross_r,
        net_r=gross_r - spread / risk,
        risk_usd_oz=risk,
        spread_r=spread / risk,
        exit_reason=exit_reason,
        unresolved=exit_reason == "force_close",
        false_breakout=event.false_breakout,
    )


def build_rows(symbol: str, bars: list[base.DeltaBar], spread: float) -> list[AtrModeTrade]:
    events = simple.detect_compression_breakouts(bars)
    rng = random.Random(f"{RANDOM_SEED}-{symbol}")
    rows: list[AtrModeTrade] = []
    for event in events:
        random_dir = 1 if rng.random() >= 0.5 else -1
        directions = {
            "simple_follow": event.breakout_direction,
            "random": random_dir,
        }
        for atr_mode in ATR_MODES:
            for signal, direction in directions.items():
                for exit_variant in RR_VARIANTS:
                    for horizon in HORIZONS:
                        row = simulate(symbol, bars, event, atr_mode, signal, direction, exit_variant, horizon, spread)
                        if row is not None:
                            rows.append(row)
    return rows


def summarize(rows: list[AtrModeTrade], field: str) -> dict[str, float]:
    vals = [getattr(r, field) for r in rows]
    n, m, lo, hi = ci(vals)
    return {
        "n": n,
        "mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "win_rate": sum(v > 0 for v in vals) / n if n else math.nan,
        "unresolved": sum(r.unresolved for r in rows) / n if n else math.nan,
        "false_breakout_pct": sum(r.false_breakout for r in rows) / n if n else math.nan,
        "avg_risk": mean([r.risk_usd_oz for r in rows]) if rows else math.nan,
        "avg_spread_r": mean([r.spread_r for r in rows]) if rows else math.nan,
        "target_count": sum(r.exit_reason == "target" for r in rows),
        "stop_count": sum(r.exit_reason == "stop" for r in rows),
        "force_count": sum(r.exit_reason == "force_close" for r in rows),
    }


def grouped(rows: list[AtrModeTrade], period: str) -> dict[tuple[str, str, str, str, int], list[AtrModeTrade]]:
    out: dict[tuple[str, str, str, str, int], list[AtrModeTrade]] = {}
    for row in period_filter(rows, period):
        out.setdefault((row.symbol, row.atr_mode, row.signal, row.exit_variant, row.horizon), []).append(row)
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


def print_matrix(rows: list[AtrModeTrade], period: str) -> None:
    g = grouped(rows, period)
    print(f"\nATR_LOOKAHEAD_{period.upper()}")
    print("period,symbol,atr_mode,exit,horizon,signal,n,gross_mean,gross_ci_low,gross_ci_high,net_mean,net_ci_low,net_ci_high,win_rate,unresolved_pct,false_breakout_pct,avg_risk,avg_spread_r,beats_random_gross,beats_random_net,gross_class,net_class,target_count,stop_count,force_count")
    symbols = sorted({r.symbol for r in rows})
    for symbol in symbols:
        for atr_mode in ATR_MODES:
            for exit_variant in RR_VARIANTS:
                for horizon in HORIZONS:
                    rand_rows = g.get((symbol, atr_mode, "random", exit_variant, horizon), [])
                    rand_gross = summarize(rand_rows, "gross_r")
                    rand_net = summarize(rand_rows, "net_r")
                    for signal in ("simple_follow", "random"):
                        signal_rows = g.get((symbol, atr_mode, signal, exit_variant, horizon), [])
                        gross = summarize(signal_rows, "gross_r")
                        net = summarize(signal_rows, "net_r")
                        stats = summarize(signal_rows, "net_r")
                        beats_gross = gross["mean"] > rand_gross["mean"] if signal != "random" else False
                        beats_net = net["mean"] > rand_net["mean"] if signal != "random" else False
                        gross_class = classify(gross, rand_gross) if signal != "random" else "baseline"
                        net_class = classify(net, rand_net) if signal != "random" else "baseline"
                        print(
                            f"{period},{symbol},{atr_mode},{exit_variant},{horizon},{signal},{int(gross['n'])},"
                            f"{gross['mean']:.6f},{gross['ci_low']:.6f},{gross['ci_high']:.6f},"
                            f"{net['mean']:.6f},{net['ci_low']:.6f},{net['ci_high']:.6f},"
                            f"{net['win_rate']:.2%},{stats['unresolved']:.2%},{stats['false_breakout_pct']:.2%},"
                            f"{stats['avg_risk']:.6f},{stats['avg_spread_r']:.6f},"
                            f"{beats_gross},{beats_net},{gross_class},{net_class},"
                            f"{int(stats['target_count'])},{int(stats['stop_count'])},{int(stats['force_count'])}"
                        )


def print_delta(rows: list[AtrModeTrade]) -> None:
    g = grouped(rows, "all")
    print("\nBEFORE_AFTER_DELTA_ALL_SIMPLE_FOLLOW")
    print("symbol,exit,horizon,before_net,after_net,delta_after_minus_before,before_spread_r,after_spread_r,before_unresolved,after_unresolved,before_win,after_win")
    symbols = sorted({r.symbol for r in rows})
    for symbol in symbols:
        for exit_variant in RR_VARIANTS:
            for horizon in HORIZONS:
                before = summarize(g.get((symbol, "before_breakout_atr", "simple_follow", exit_variant, horizon), []), "net_r")
                after = summarize(g.get((symbol, "after_compression_atr", "simple_follow", exit_variant, horizon), []), "net_r")
                print(
                    f"{symbol},{exit_variant},{horizon},"
                    f"{before['mean']:.6f},{after['mean']:.6f},{after['mean'] - before['mean']:.6f},"
                    f"{before['avg_spread_r']:.6f},{after['avg_spread_r']:.6f},"
                    f"{before['unresolved']:.2%},{after['unresolved']:.2%},"
                    f"{before['win_rate']:.2%},{after['win_rate']:.2%}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xag-ticks", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--xag-cache", type=Path, default=Path("data/xagusd_m15_delta_bars.csv"))
    parser.add_argument("--xau-spread", type=float, default=0.20)
    parser.add_argument("--xag-spread", type=float, default=0.02)
    args = parser.parse_args()

    rows: list[AtrModeTrade] = []
    contexts = []
    for symbol, ticks, cache, spread in (
        ("XAUUSD", args.xau_ticks, args.xau_cache, args.xau_spread),
        ("XAGUSD", args.xag_ticks, args.xag_cache, args.xag_spread),
    ):
        bars = simple.load_symbol_bars(symbol, ticks, cache)
        events = simple.detect_compression_breakouts(bars)
        rows.extend(build_rows(symbol, bars, spread))
        contexts.append((symbol, ticks, len(bars), bars[0].start, bars[-1].end, len(events), spread))

    print("ATR_LOOKAHEAD_FIX_CONTEXT")
    print("before_breakout_atr=old look-ahead mode: risk from breakout bar ATR14")
    print("after_compression_atr=correct live-matched mode: risk from compression-end bar ATR14")
    print("entry=range edge; stop=1R; targets=1/1.5/2R; force close=10/20 bars")
    for symbol, ticks, n, start, end, events_n, spread in contexts:
        print(f"symbol_context={symbol},tick_file={ticks},bars={n},date_range={start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC,compression_events={events_n},spread={spread:.4f}")
    for period in ("all", "train_2016_2021", "test_2022_2026"):
        print_matrix(rows, period)
    print_delta(rows)


if __name__ == "__main__":
    main()
