"""
Wyckoff RSI compression-breakout entry/RR matrix audit.

This keeps the prior compression definition and Wyckoff RSI direction logic, but
tests two corrected entry approaches and simple single-target RR exits:

Approach A:
    Enter immediately at compression-end close in the signal direction.

Approach B:
    Wait for breakout and enter at the broken range edge in the breakout
    direction, filtered by whether the signal agreed with the breakout.

Stops:
    fixed_10: 1R = $10/oz
    atr_1x:   1R = 1.0 * ATR(14) at entry

Targets:
    1:1 and 1:2 single full-position target.

Horizons:
    force close at 10 or 20 M15 bars, or earlier at segment end.

The event universe is the same detector used by the prior compression audit:
2,800 compression->breakout episodes, with 916 Wyckoff RSI calls.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import volatility_compression_breakout_audit as base
from delta_signal_audit import IUX_XAUUSD_ROUNDTRIP_SPREAD, default_tick_path


COIN_FLIP_SEED = 20260629
TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=base.timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=base.timezone.utc)

APPROACHES = ("A_immediate_compression_close", "B_breakout_edge_filter")
STOP_MODES = ("fixed_10", "atr_1x")
RR_TARGETS = (1.0, 2.0)
HORIZONS = (10, 20)
SIGNALS = ("wyckoff_rsi", "coin_flip", "prior_momentum", "simple_follow")


@dataclass(frozen=True)
class MatrixTrade:
    approach: str
    stop_mode: str
    rr_target: float
    horizon: int
    signal: str
    event_id: int
    entry_index: int
    eval_start_index: int
    entry_time: datetime
    direction: int
    breakout_direction: int
    direction_correct: bool
    entry_price: float
    risk_usd_oz: float
    net_r: float
    gross_r: float
    exit_reason: str
    unresolved_forced: bool
    bars_held: int


def ci(vals: list[float]) -> tuple[int, float, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se, sd


def segment_end_index(bars: list[base.DeltaBar], start_index: int, horizon: int) -> int:
    end = min(len(bars) - 1, start_index + horizon)
    segment = bars[start_index].segment_id
    j = start_index
    while j + 1 <= end and bars[j + 1].segment_id == segment:
        j += 1
    return j


def risk_for_entry(bars: list[base.DeltaBar], entry_index: int, stop_mode: str) -> float | None:
    if stop_mode == "fixed_10":
        return 10.0
    if stop_mode == "atr_1x":
        atr = bars[entry_index].atr14
        if atr is None or atr <= 0:
            return None
        return atr
    raise ValueError(stop_mode)


def simulate_single_target(
    bars: list[base.DeltaBar],
    event: base.CompressionEvent,
    approach: str,
    stop_mode: str,
    rr_target: float,
    horizon: int,
    signal: str,
    direction: int,
    entry_index: int,
    entry_price: float,
    eval_start_index: int,
) -> MatrixTrade | None:
    risk = risk_for_entry(bars, entry_index, stop_mode)
    if risk is None:
        return None
    if approach == "A_immediate_compression_close":
        next_move = bars[eval_start_index].close - entry_price
        direction_correct = (next_move * direction) > 0
    else:
        direction_correct = direction == event.breakout_direction
    stop = entry_price - direction * risk
    target = entry_price + direction * rr_target * risk
    end_index = segment_end_index(bars, eval_start_index, horizon)
    gross_r = 0.0
    exit_reason = "force_close"
    exit_index = end_index

    for i in range(eval_start_index, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        # Conservative M15 OHLC convention: if both are possible in the same
        # bar, the stop is counted first. Stop fill uses the adverse extreme as
        # slippage because the tick path is not retained in the bar cache.
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r = direction * (fill - entry_price) / risk
            exit_reason = "stop"
            exit_index = i
            break
        if target_hit:
            gross_r = rr_target
            exit_reason = "target"
            exit_index = i
            break
    else:
        close = bars[end_index].close
        gross_r = direction * (close - entry_price) / risk

    net_r = gross_r - (IUX_XAUUSD_ROUNDTRIP_SPREAD / risk)
    return MatrixTrade(
        approach=approach,
        stop_mode=stop_mode,
        rr_target=rr_target,
        horizon=horizon,
        signal=signal,
        event_id=event.event_id,
        entry_index=entry_index,
        eval_start_index=eval_start_index,
        entry_time=bars[entry_index].start,
        direction=direction,
        breakout_direction=event.breakout_direction,
        direction_correct=direction_correct,
        entry_price=entry_price,
        risk_usd_oz=risk,
        net_r=net_r,
        gross_r=gross_r,
        exit_reason=exit_reason,
        unresolved_forced=exit_reason == "force_close",
        bars_held=exit_index - entry_index,
    )


def directions_for_event(event: base.CompressionEvent, coin_dir: int) -> dict[str, int | None]:
    return {
        "wyckoff_rsi": event.predictions["method_3_wyckoff_rsi"],
        "coin_flip": coin_dir,
        "prior_momentum": event.predictions["baseline_prior_momentum"],
        # Approach A cannot know this at compression end. It is retained as an
        # oracle comparator so the requested baseline is visible, not as a
        # tradable immediate-entry signal.
        "simple_follow": event.breakout_direction,
    }


def build_matrix_trades(bars: list[base.DeltaBar], events: list[base.CompressionEvent]) -> list[MatrixTrade]:
    rng = random.Random(COIN_FLIP_SEED)
    trades: list[MatrixTrade] = []
    for event in events:
        wyckoff = event.predictions["method_3_wyckoff_rsi"]
        if wyckoff is None:
            continue
        coin_dir = 1 if rng.random() >= 0.5 else -1
        dirs = directions_for_event(event, coin_dir)
        for approach in APPROACHES:
            for signal, raw_direction in dirs.items():
                if raw_direction is None:
                    continue
                if approach == "A_immediate_compression_close":
                    direction = int(raw_direction)
                    entry_index = event.setup_end
                    entry_price = bars[entry_index].close
                    eval_start = entry_index + 1
                else:
                    # Breakout-edge filter: enter in breakout direction only
                    # when the signal agrees with that direction. Simple-follow
                    # always agrees by construction.
                    if int(raw_direction) != event.breakout_direction:
                        continue
                    direction = event.breakout_direction
                    entry_index = event.breakout_index
                    entry_price = event.range_high if direction == 1 else event.range_low
                    eval_start = entry_index + 1
                if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
                    continue
                for stop_mode in STOP_MODES:
                    for rr_target in RR_TARGETS:
                        for horizon in HORIZONS:
                            trade = simulate_single_target(
                                bars,
                                event,
                                approach,
                                stop_mode,
                                rr_target,
                                horizon,
                                signal,
                                direction,
                                entry_index,
                                entry_price,
                                eval_start,
                            )
                            if trade is not None:
                                trades.append(trade)
    return trades


def breakeven_win_rate(rr_target: float, avg_spread_r: float) -> float:
    # Solve p * (RR - spread) + (1-p) * (-1 - spread) = 0.
    # Spread cancels from the denominator, but raises required p.
    return (1.0 + avg_spread_r) / (1.0 + rr_target)


def summarize(trades: list[MatrixTrade]) -> dict[str, float]:
    vals = [t.net_r for t in trades]
    n, m, lo, hi, sd = ci(vals)
    wins = sum(t.net_r > 0 for t in trades)
    avg_spread_r = mean([IUX_XAUUSD_ROUNDTRIP_SPREAD / t.risk_usd_oz for t in trades]) if trades else math.nan
    rr = trades[0].rr_target if trades else math.nan
    be = breakeven_win_rate(rr, avg_spread_r) if trades else math.nan
    return {
        "n": n,
        "direction_correct": sum(t.direction_correct for t in trades) / n if n else math.nan,
        "win_rate": wins / n if n else math.nan,
        "breakeven": be,
        "win_minus_be": wins / n - be if n else math.nan,
        "expectancy": m,
        "ci_low": lo,
        "ci_high": hi,
        "unresolved": sum(t.unresolved_forced for t in trades) / n if n else math.nan,
        "target": sum(t.exit_reason == "target" for t in trades),
        "stop": sum(t.exit_reason == "stop" for t in trades),
        "force": sum(t.exit_reason == "force_close" for t in trades),
        "avg_risk": mean([t.risk_usd_oz for t in trades]) if trades else math.nan,
        "avg_spread_r": avg_spread_r,
        "sd": sd,
    }


def filter_period(trades: list[MatrixTrade], period: str) -> list[MatrixTrade]:
    if period == "all":
        return trades
    if period == "train_2016_2021":
        return [t for t in trades if t.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [t for t in trades if t.entry_time >= TEST_START]
    raise ValueError(period)


def group_trades(trades: list[MatrixTrade], period: str) -> dict[tuple[str, str, float, int, str], list[MatrixTrade]]:
    grouped: dict[tuple[str, str, float, int, str], list[MatrixTrade]] = {}
    for t in filter_period(trades, period):
        key = (t.approach, t.stop_mode, t.rr_target, t.horizon, t.signal)
        grouped.setdefault(key, []).append(t)
    return grouped


def print_matrix(trades: list[MatrixTrade], period: str) -> None:
    grouped = group_trades(trades, period)
    print(f"\nMATRIX_{period.upper()}")
    print("period,approach,stop,rr,horizon,signal,n,direction_correct,win_rate,breakeven_win_rate,win_minus_breakeven,net_expectancy_r,ci_low,ci_high,unresolved_pct,target_count,stop_count,force_count,avg_risk_usd_oz,avg_spread_r,ci_clears_zero")
    for approach in APPROACHES:
        for stop_mode in STOP_MODES:
            for rr in RR_TARGETS:
                for horizon in HORIZONS:
                    for signal in SIGNALS:
                        key = (approach, stop_mode, rr, horizon, signal)
                        s = summarize(grouped.get(key, []))
                        print(
                            f"{period},{approach},{stop_mode},{rr:.0f},{horizon},{signal},"
                            f"{int(s['n'])},{s['direction_correct']:.2%},{s['win_rate']:.2%},"
                            f"{s['breakeven']:.2%},{s['win_minus_be']:.2%},"
                            f"{s['expectancy']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},"
                            f"{s['unresolved']:.2%},{int(s['target'])},{int(s['stop'])},{int(s['force'])},"
                            f"{s['avg_risk']:.6f},{s['avg_spread_r']:.6f},"
                            f"{s['ci_low'] > 0 if math.isfinite(s['ci_low']) else False}"
                        )


def best_by_signal(trades: list[MatrixTrade], signal: str, period: str) -> tuple[tuple[str, str, float, int, str] | None, dict[str, float]]:
    grouped = group_trades(trades, period)
    candidates = [(key, summarize(vals)) for key, vals in grouped.items() if key[-1] == signal]
    if not candidates:
        return None, summarize([])
    return max(candidates, key=lambda item: item[1]["expectancy"])


def consistency_check(trades: list[MatrixTrade]) -> list[str]:
    grouped_all = group_trades(trades, "all")
    grouped_train = group_trades(trades, "train_2016_2021")
    grouped_test = group_trades(trades, "test_2022_2026")
    passed: list[str] = []
    for key, vals in grouped_all.items():
        if key[-1] != "wyckoff_rsi":
            continue
        all_s = summarize(vals)
        train_s = summarize(grouped_train.get(key, []))
        test_s = summarize(grouped_test.get(key, []))
        if not (all_s["ci_low"] > 0 and train_s["ci_low"] > 0 and test_s["ci_low"] > 0):
            continue
        approach, stop_mode, rr, horizon, _ = key
        beats_all = True
        for baseline in ("coin_flip", "prior_momentum", "simple_follow"):
            b_key = (approach, stop_mode, rr, horizon, baseline)
            if all_s["expectancy"] <= summarize(grouped_all.get(b_key, []))["expectancy"]:
                beats_all = False
            if train_s["expectancy"] <= summarize(grouped_train.get(b_key, []))["expectancy"]:
                beats_all = False
            if test_s["expectancy"] <= summarize(grouped_test.get(b_key, []))["expectancy"]:
                beats_all = False
        if beats_all:
            passed.append(f"{approach}|{stop_mode}|rr={rr:.0f}|h={horizon}")
    return passed


def print_verdict(trades: list[MatrixTrade]) -> None:
    passed = consistency_check(trades)
    a_wy = [t for t in trades if t.approach == "A_immediate_compression_close" and t.signal == "wyckoff_rsi"]
    b_wy = [t for t in trades if t.approach == "B_breakout_edge_filter" and t.signal == "wyckoff_rsi"]
    a_dir = sum(t.direction_correct for t in a_wy) / len(a_wy) if a_wy else math.nan
    b_dir = sum(t.direction_correct for t in b_wy) / len(b_wy) if b_wy else math.nan
    best_wy_key, best_wy = best_by_signal(trades, "wyckoff_rsi", "all")
    best_simple_key, best_simple = best_by_signal(trades, "simple_follow", "all")

    print("\nVERDICT")
    print(f"approach_a_wyckoff_direction_correct={a_dir:.2%}")
    print(f"approach_b_wyckoff_direction_correct={b_dir:.2%}")
    print(f"best_wyckoff_cell={best_wy_key},expectancy_r={best_wy['expectancy']:.6f},ci_low={best_wy['ci_low']:.6f},ci_high={best_wy['ci_high']:.6f}")
    print(f"best_simple_follow_cell={best_simple_key},expectancy_r={best_simple['expectancy']:.6f},ci_low={best_simple['ci_low']:.6f},ci_high={best_simple['ci_high']:.6f}")
    print(f"wyckoff_cells_passing_all_train_test_and_baseline_checks={';'.join(passed) if passed else 'none'}")
    if passed:
        print("interpretation=At least one Wyckoff cell clears the strict all/train/test/baseline screen, but this is a large matrix and still requires out-of-sample confirmation.")
    else:
        print("interpretation=No Wyckoff cell clears zero in all sample, train, and test while also beating same-exit baselines. Treat isolated positive cells as noise.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None)
    parser.add_argument("--bar-cache", type=Path, default=base.DEFAULT_BAR_CACHE)
    args = parser.parse_args()

    tick_path = args.ticks or default_tick_path()
    base.ensure_bar_cache(tick_path, args.bar_cache)
    bars = base.load_cached_bars(args.bar_cache)
    base.bars_global = bars
    events = base.detect_events(bars)
    trades = build_matrix_trades(bars, events)
    wyckoff_events = [e for e in events if e.predictions["method_3_wyckoff_rsi"] is not None]

    print("WYCKOFF_RSI_ENTRY_RR_MATRIX_CONTEXT")
    print(f"tick_file={tick_path}")
    print(f"bars={len(bars)}")
    print(f"date_range={bars[0].start:%Y-%m-%d %H:%M:%S} to {bars[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print(f"compression_events_total={len(events)}")
    print(f"wyckoff_covered_events={len(wyckoff_events)}")
    print("approach_a=enter immediately at compression-end close in signal direction")
    print("approach_b=enter broken range edge in breakout direction only when signal agrees")
    print("approach_a_simple_follow=oracle eventual breakout direction, included only as requested comparator")
    print(f"spread_usd_oz={IUX_XAUUSD_ROUNDTRIP_SPREAD:.2f}")
    print("fill_rule=TP limit at target; SL at adverse OHLC extreme; if same bar can hit both, stop first")
    print("approach_b_intrabar_note=edge entry uses next bar for TP/SL evaluation because breakout-bar tick order is unavailable")
    print(f"train_period=2016-01-03 through {TRAIN_END:%Y-%m-%d}; test_period={TEST_START:%Y-%m-%d} through 2026-06-15")

    for period in ("all", "train_2016_2021", "test_2022_2026"):
        print_matrix(trades, period)
    print_verdict(trades)


if __name__ == "__main__":
    main()
