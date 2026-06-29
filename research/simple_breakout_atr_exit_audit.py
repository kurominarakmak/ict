"""
Simple compression-breakout following audit with ATR-scaled exits.

Hypothesis:
    Compression predicts volatility expansion. Do not predict direction; enter
    in whichever direction the compression range breaks.

Validation gates:
    - XAUUSD full/train/test with CI clearing zero.
    - Consistency across 1:1, 1:1.5, 1:2, and 1.5*ATR trailing exits.
    - False breakouts are included.
    - Compare compression breakouts to rolling range breakouts without the
      compression filter.
    - Replicate on XAGUSD with the same rules.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

import volatility_compression_breakout_audit as base
from delta_signal_audit import IUX_XAUUSD_ROUNDTRIP_SPREAD


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=base.timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=base.timezone.utc)

ENTRY_RANGE_WINDOW = base.COMPRESSION_WINDOW
DETECT_SKIP_BARS = base.EXIT_HORIZON
FALSE_BREAKOUT_BARS = base.FALSE_BREAKOUT_BARS
RR_VARIANTS = ("rr_1", "rr_1_5", "rr_2", "trail_1_5atr")
HORIZONS = (10, 20)
TRAIL_ATR_MULT = 1.5


@dataclass(frozen=True)
class BreakoutEvent:
    event_id: int
    setup_start: int
    setup_end: int
    breakout_index: int
    range_high: float
    range_low: float
    breakout_direction: int
    false_breakout: bool
    compressed: bool


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    universe: str
    exit_variant: str
    horizon: int
    event_id: int
    entry_time: datetime
    net_r: float
    gross_r: float
    exit_reason: str
    unresolved: bool
    false_breakout: bool
    risk_usd_oz: float
    spread_r: float
    bars_held: int


@dataclass(frozen=True)
class ExpansionResult:
    symbol: str
    universe: str
    horizon: int
    event_id: int
    entry_time: datetime
    false_breakout: bool
    mfe_r: float
    mae_r: float
    risk_usd_oz: float


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


def false_breakout(bars: list[base.DeltaBar], event: BreakoutEvent) -> bool:
    for k in range(event.breakout_index + 1, min(event.breakout_index + FALSE_BREAKOUT_BARS, len(bars) - 1) + 1):
        if bars[k].segment_id != bars[event.breakout_index].segment_id:
            break
        if event.breakout_direction == 1 and bars[k].close < event.range_low:
            return True
        if event.breakout_direction == -1 and bars[k].close > event.range_high:
            return True
    return False


def detect_compression_breakouts(bars: list[base.DeltaBar]) -> list[BreakoutEvent]:
    events: list[BreakoutEvent] = []
    i = base.ATR_TRAIL + base.COMPRESSION_WINDOW
    while i < len(bars) - DETECT_SKIP_BARS:
        if not base.is_compression_end(bars, i):
            i += 1
            continue
        start = i - base.COMPRESSION_WINDOW + 1
        range_high = max(b.high for b in bars[start : i + 1])
        range_low = min(b.low for b in bars[start : i + 1])
        breakout = None
        for j in range(i + 1, len(bars) - DETECT_SKIP_BARS):
            if bars[j].segment_id != bars[i].segment_id:
                break
            if bars[j].close > range_high:
                breakout = (j, 1)
                break
            if bars[j].close < range_low:
                breakout = (j, -1)
                break
        if breakout is None:
            i += 1
            continue
        bidx, direction = breakout
        event = BreakoutEvent(len(events) + 1, start, i, bidx, range_high, range_low, direction, False, True)
        event = BreakoutEvent(event.event_id, event.setup_start, event.setup_end, event.breakout_index, event.range_high, event.range_low, event.breakout_direction, false_breakout(bars, event), True)
        events.append(event)
        i = bidx + DETECT_SKIP_BARS + 1
    return events


def detect_plain_range_breakouts(bars: list[base.DeltaBar]) -> list[BreakoutEvent]:
    events: list[BreakoutEvent] = []
    i = ENTRY_RANGE_WINDOW - 1
    while i < len(bars) - DETECT_SKIP_BARS:
        start = i - ENTRY_RANGE_WINDOW + 1
        if not base.same_segment(bars, start, i):
            i += 1
            continue
        range_high = max(b.high for b in bars[start : i + 1])
        range_low = min(b.low for b in bars[start : i + 1])
        breakout = None
        for j in range(i + 1, len(bars) - DETECT_SKIP_BARS):
            if bars[j].segment_id != bars[i].segment_id:
                break
            if bars[j].close > range_high:
                breakout = (j, 1)
                break
            if bars[j].close < range_low:
                breakout = (j, -1)
                break
        if breakout is None:
            i += 1
            continue
        bidx, direction = breakout
        event = BreakoutEvent(len(events) + 1, start, i, bidx, range_high, range_low, direction, False, False)
        event = BreakoutEvent(event.event_id, event.setup_start, event.setup_end, event.breakout_index, event.range_high, event.range_low, event.breakout_direction, false_breakout(bars, event), False)
        events.append(event)
        i = bidx + DETECT_SKIP_BARS + 1
    return events


def simulate_trade(
    symbol: str,
    universe: str,
    bars: list[base.DeltaBar],
    event: BreakoutEvent,
    exit_variant: str,
    horizon: int,
    spread_usd_oz: float,
) -> TradeResult | None:
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    risk = bars[entry_index].atr14
    if risk is None or risk <= 0:
        return None
    direction = event.breakout_direction
    entry = event.range_high if direction == 1 else event.range_low
    initial_stop = entry - direction * risk
    end_index = segment_end_index(bars, eval_start, horizon)
    gross_r = 0.0
    exit_reason = "force_close"
    exit_index = end_index

    if exit_variant.startswith("rr_"):
        rr = {"rr_1": 1.0, "rr_1_5": 1.5, "rr_2": 2.0}[exit_variant]
        target = entry + direction * rr * risk
        for i in range(eval_start, end_index + 1):
            bar = bars[i]
            stop_hit = bar.low <= initial_stop if direction == 1 else bar.high >= initial_stop
            target_hit = bar.high >= target if direction == 1 else bar.low <= target
            if stop_hit:
                fill = min(initial_stop, bar.low) if direction == 1 else max(initial_stop, bar.high)
                gross_r = direction * (fill - entry) / risk
                exit_reason = "stop"
                exit_index = i
                break
            if target_hit:
                gross_r = rr
                exit_reason = "target"
                exit_index = i
                break
        else:
            gross_r = direction * (bars[end_index].close - entry) / risk
    else:
        trail_distance = TRAIL_ATR_MULT * risk
        best = entry
        stop = initial_stop
        for i in range(eval_start, end_index + 1):
            bar = bars[i]
            stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
            if stop_hit:
                fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
                gross_r = direction * (fill - entry) / risk
                exit_reason = "trail_stop"
                exit_index = i
                break
            if direction == 1:
                best = max(best, bar.high)
                stop = max(stop, best - trail_distance)
            else:
                best = min(best, bar.low)
                stop = min(stop, best + trail_distance)
        else:
            gross_r = direction * (bars[end_index].close - entry) / risk

    net_r = gross_r - (spread_usd_oz / risk)
    return TradeResult(
        symbol,
        universe,
        exit_variant,
        horizon,
        event.event_id,
        bars[entry_index].start,
        net_r,
        gross_r,
        exit_reason,
        exit_reason == "force_close",
        event.false_breakout,
        risk,
        spread_usd_oz / risk,
        exit_index - entry_index,
    )


def expansion_for_event(
    symbol: str,
    universe: str,
    bars: list[base.DeltaBar],
    event: BreakoutEvent,
    horizon: int,
) -> ExpansionResult | None:
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    risk = bars[entry_index].atr14
    if risk is None or risk <= 0:
        return None
    direction = event.breakout_direction
    entry = event.range_high if direction == 1 else event.range_low
    end_index = segment_end_index(bars, eval_start, horizon)
    mfe = 0.0
    mae = 0.0
    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        if direction == 1:
            mfe = max(mfe, bar.high - entry)
            mae = max(mae, entry - bar.low)
        else:
            mfe = max(mfe, entry - bar.low)
            mae = max(mae, bar.high - entry)
    return ExpansionResult(
        symbol=symbol,
        universe=universe,
        horizon=horizon,
        event_id=event.event_id,
        entry_time=bars[entry_index].start,
        false_breakout=event.false_breakout,
        mfe_r=mfe / risk,
        mae_r=mae / risk,
        risk_usd_oz=risk,
    )


def build_results(
    symbol: str,
    universe: str,
    bars: list[base.DeltaBar],
    events: list[BreakoutEvent],
    spread_usd_oz: float,
) -> list[TradeResult]:
    out: list[TradeResult] = []
    for event in events:
        for exit_variant in RR_VARIANTS:
            for horizon in HORIZONS:
                result = simulate_trade(symbol, universe, bars, event, exit_variant, horizon, spread_usd_oz)
                if result is not None:
                    out.append(result)
    return out


def build_expansion_results(
    symbol: str,
    universe: str,
    bars: list[base.DeltaBar],
    events: list[BreakoutEvent],
) -> list[ExpansionResult]:
    out: list[ExpansionResult] = []
    for event in events:
        for horizon in HORIZONS:
            result = expansion_for_event(symbol, universe, bars, event, horizon)
            if result is not None:
                out.append(result)
    return out


def filter_period(results: list[TradeResult], period: str) -> list[TradeResult]:
    if period == "all":
        return results
    if period == "train_2016_2021":
        return [r for r in results if r.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [r for r in results if r.entry_time >= TEST_START]
    raise ValueError(period)


def summarize(results: list[TradeResult]) -> dict[str, float]:
    vals = [r.net_r for r in results]
    n, m, lo, hi, sd = ci(vals)
    return {
        "n": n,
        "win_rate": sum(r.net_r > 0 for r in results) / n if n else math.nan,
        "expectancy": m,
        "ci_low": lo,
        "ci_high": hi,
        "unresolved": sum(r.unresolved for r in results) / n if n else math.nan,
        "false_breakouts": sum(r.false_breakout for r in results),
        "false_breakout_pct": sum(r.false_breakout for r in results) / n if n else math.nan,
        "target": sum(r.exit_reason == "target" for r in results),
        "stop": sum(r.exit_reason == "stop" for r in results),
        "trail_stop": sum(r.exit_reason == "trail_stop" for r in results),
        "force": sum(r.exit_reason == "force_close" for r in results),
        "avg_risk": mean([r.risk_usd_oz for r in results]) if results else math.nan,
        "avg_spread_r": mean([r.spread_r for r in results]) if results else math.nan,
        "avg_bars": mean([r.bars_held for r in results]) if results else math.nan,
        "sd": sd,
    }


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_expansion(results: list[ExpansionResult]) -> dict[str, float]:
    mfes = [r.mfe_r for r in results]
    maes = [r.mae_r for r in results]
    n = len(results)
    return {
        "n": n,
        "mfe_p25": percentile(mfes, 0.25),
        "mfe_median": median(mfes) if mfes else math.nan,
        "mfe_p75": percentile(mfes, 0.75),
        "mfe_p90": percentile(mfes, 0.90),
        "mae_p25": percentile(maes, 0.25),
        "mae_median": median(maes) if maes else math.nan,
        "mae_p75": percentile(maes, 0.75),
        "mae_p90": percentile(maes, 0.90),
        "reach_1r": sum(v >= 1.0 for v in mfes) / n if n else math.nan,
        "reach_1_5r": sum(v >= 1.5 for v in mfes) / n if n else math.nan,
        "reach_2r": sum(v >= 2.0 for v in mfes) / n if n else math.nan,
        "reach_3r": sum(v >= 3.0 for v in mfes) / n if n else math.nan,
        "avg_risk": mean([r.risk_usd_oz for r in results]) if results else math.nan,
    }


def grouped(results: list[TradeResult], period: str) -> dict[tuple[str, str, str, int], list[TradeResult]]:
    out: dict[tuple[str, str, str, int], list[TradeResult]] = {}
    for r in filter_period(results, period):
        key = (r.symbol, r.universe, r.exit_variant, r.horizon)
        out.setdefault(key, []).append(r)
    return out


def print_matrix(results: list[TradeResult], period: str) -> None:
    g = grouped(results, period)
    print(f"\nSIMPLE_FOLLOW_{period.upper()}")
    print("period,symbol,universe,exit,horizon,n,win_rate,net_expectancy_r,ci_low,ci_high,unresolved_pct,false_breakout_pct,false_breakout_count,target_count,stop_count,trail_stop_count,force_count,avg_atr_risk,avg_spread_r,avg_bars,ci_clears_zero")
    for symbol in ("XAUUSD", "XAGUSD"):
        for universe in ("compression", "plain_range"):
            for exit_variant in RR_VARIANTS:
                for horizon in HORIZONS:
                    s = summarize(g.get((symbol, universe, exit_variant, horizon), []))
                    print(
                        f"{period},{symbol},{universe},{exit_variant},{horizon},"
                        f"{int(s['n'])},{s['win_rate']:.2%},{s['expectancy']:.6f},"
                        f"{s['ci_low']:.6f},{s['ci_high']:.6f},{s['unresolved']:.2%},"
                        f"{s['false_breakout_pct']:.2%},{int(s['false_breakouts'])},"
                        f"{int(s['target'])},{int(s['stop'])},{int(s['trail_stop'])},{int(s['force'])},"
                        f"{s['avg_risk']:.6f},{s['avg_spread_r']:.6f},{s['avg_bars']:.2f},"
                        f"{s['ci_low'] > 0 if math.isfinite(s['ci_low']) else False}"
                    )


def grouped_expansions(results: list[ExpansionResult], period: str) -> dict[tuple[str, str, int, str], list[ExpansionResult]]:
    if period == "all":
        period_results = results
    elif period == "train_2016_2021":
        period_results = [r for r in results if r.entry_time <= TRAIN_END]
    elif period == "test_2022_2026":
        period_results = [r for r in results if r.entry_time >= TEST_START]
    else:
        raise ValueError(period)
    out: dict[tuple[str, str, int, str], list[ExpansionResult]] = {}
    for r in period_results:
        for fakeout_filter in ("all_breakouts", "non_false_only"):
            if fakeout_filter == "non_false_only" and r.false_breakout:
                continue
            key = (r.symbol, r.universe, r.horizon, fakeout_filter)
            out.setdefault(key, []).append(r)
    return out


def print_expansion_matrix(results: list[ExpansionResult], period: str) -> None:
    g = grouped_expansions(results, period)
    print(f"\nEXPANSION_DISTRIBUTION_{period.upper()}")
    print("period,symbol,universe,horizon,filter,n,mfe_p25_r,mfe_median_r,mfe_p75_r,mfe_p90_r,mae_p25_r,mae_median_r,mae_p75_r,mae_p90_r,pct_reach_1r,pct_reach_1_5r,pct_reach_2r,pct_reach_3r,avg_atr_risk")
    for symbol in ("XAUUSD", "XAGUSD"):
        for universe in ("compression", "plain_range"):
            for horizon in HORIZONS:
                for fakeout_filter in ("all_breakouts", "non_false_only"):
                    s = summarize_expansion(g.get((symbol, universe, horizon, fakeout_filter), []))
                    print(
                        f"{period},{symbol},{universe},{horizon},{fakeout_filter},{int(s['n'])},"
                        f"{s['mfe_p25']:.6f},{s['mfe_median']:.6f},{s['mfe_p75']:.6f},{s['mfe_p90']:.6f},"
                        f"{s['mae_p25']:.6f},{s['mae_median']:.6f},{s['mae_p75']:.6f},{s['mae_p90']:.6f},"
                        f"{s['reach_1r']:.2%},{s['reach_1_5r']:.2%},{s['reach_2r']:.2%},{s['reach_3r']:.2%},"
                        f"{s['avg_risk']:.6f}"
                    )


def print_capture_matrix(trades: list[TradeResult], expansions: list[ExpansionResult], period: str) -> None:
    expansion_by_key = {
        (r.symbol, r.universe, r.horizon, r.event_id): r.mfe_r
        for r in expansions
        if (period == "all")
        or (period == "train_2016_2021" and r.entry_time <= TRAIN_END)
        or (period == "test_2022_2026" and r.entry_time >= TEST_START)
    }
    period_trades = filter_period(trades, period)
    grouped_trades: dict[tuple[str, str, str, int], list[TradeResult]] = {}
    for t in period_trades:
        grouped_trades.setdefault((t.symbol, t.universe, t.exit_variant, t.horizon), []).append(t)
    print(f"\nREALIZED_CAPTURE_{period.upper()}")
    print("period,symbol,universe,exit,horizon,n,avg_mfe_potential_r,avg_gross_capture_r,avg_net_capture_r,gross_capture_over_mfe,net_capture_over_mfe,target_or_trail_count,force_count")
    for symbol in ("XAUUSD", "XAGUSD"):
        for universe in ("compression", "plain_range"):
            for exit_variant in RR_VARIANTS:
                for horizon in HORIZONS:
                    rows = grouped_trades.get((symbol, universe, exit_variant, horizon), [])
                    pairs = [
                        (t, expansion_by_key.get((t.symbol, t.universe, t.horizon, t.event_id)))
                        for t in rows
                    ]
                    clean = [(t, mfe) for t, mfe in pairs if mfe is not None]
                    if not clean:
                        print(f"{period},{symbol},{universe},{exit_variant},{horizon},0,nan,nan,nan,nan,nan,0,0")
                        continue
                    mfe_vals = [float(mfe) for _, mfe in clean]
                    gross_vals = [t.gross_r for t, _ in clean]
                    net_vals = [t.net_r for t, _ in clean]
                    denom = sum(mfe_vals)
                    favorable_exits = sum(t.exit_reason in ("target", "trail_stop") for t, _ in clean)
                    force_count = sum(t.exit_reason == "force_close" for t, _ in clean)
                    print(
                        f"{period},{symbol},{universe},{exit_variant},{horizon},{len(clean)},"
                        f"{mean(mfe_vals):.6f},{mean(gross_vals):.6f},{mean(net_vals):.6f},"
                        f"{(sum(gross_vals) / denom) if denom > 0 else math.nan:.2%},"
                        f"{(sum(net_vals) / denom) if denom > 0 else math.nan:.2%},"
                        f"{favorable_exits},{force_count}"
                    )


def strict_passes(results: list[TradeResult]) -> list[str]:
    out: list[str] = []
    all_g = grouped(results, "all")
    train_g = grouped(results, "train_2016_2021")
    test_g = grouped(results, "test_2022_2026")
    for key in all_g:
        symbol, universe, exit_variant, horizon = key
        if universe != "compression":
            continue
        all_s = summarize(all_g.get(key, []))
        train_s = summarize(train_g.get(key, []))
        test_s = summarize(test_g.get(key, []))
        if all_s["ci_low"] > 0 and train_s["ci_low"] > 0 and test_s["ci_low"] > 0:
            out.append(f"{symbol}|{exit_variant}|h={horizon}")
    return sorted(out)


def load_symbol_bars(symbol: str, tick_path: Path, cache_path: Path) -> list[base.DeltaBar]:
    if not cache_path.exists() or cache_path.stat().st_mtime < tick_path.stat().st_mtime:
        source = Path("research/fast_m15_delta_bars.cpp")
        binary = Path("/tmp/fast_m15_delta_bars")
        print(f"Building {symbol} M15 cache: {cache_path}", flush=True)
        subprocess.run(["c++", "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(tick_path), str(cache_path)], check=True)
    return base.load_cached_bars(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xag-ticks", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--xag-cache", type=Path, default=Path("data/xagusd_m15_delta_bars.csv"))
    parser.add_argument("--xau-spread", type=float, default=IUX_XAUUSD_ROUNDTRIP_SPREAD)
    parser.add_argument("--xag-spread", type=float, default=0.02)
    args = parser.parse_args()

    all_results: list[TradeResult] = []
    all_expansions: list[ExpansionResult] = []
    contexts = []
    for symbol, tick_path, cache_path, spread in (
        ("XAUUSD", args.xau_ticks, args.xau_cache, args.xau_spread),
        ("XAGUSD", args.xag_ticks, args.xag_cache, args.xag_spread),
    ):
        bars = load_symbol_bars(symbol, tick_path, cache_path)
        base.bars_global = bars
        compression_events = detect_compression_breakouts(bars)
        plain_events = detect_plain_range_breakouts(bars)
        all_results.extend(build_results(symbol, "compression", bars, compression_events, spread))
        all_results.extend(build_results(symbol, "plain_range", bars, plain_events, spread))
        all_expansions.extend(build_expansion_results(symbol, "compression", bars, compression_events))
        all_expansions.extend(build_expansion_results(symbol, "plain_range", bars, plain_events))
        contexts.append((symbol, tick_path, len(bars), bars[0].start, bars[-1].end, len(compression_events), len(plain_events), spread))

    print("SIMPLE_BREAKOUT_ATR_EXIT_CONTEXT")
    print(f"compression_window={base.COMPRESSION_WINDOW},atr_rule=bottom tercile of prior {base.ATR_TRAIL} valid ATR14 values,min_fraction={base.COMPRESSION_MIN_FRACTION:.2f}")
    print("entry=broken range edge in breakout direction; no direction prediction")
    print("stop=1.0*ATR14 at breakout bar; spread deducted in R")
    print("fill_rule=TP limit at target; SL/trail at adverse OHLC extreme; same-bar stop before target")
    print("plain_range_baseline=rolling 16-bar range breakout with same non-overlap skip, no compression filter")
    print(f"train_period=2016-01-03 through {TRAIN_END:%Y-%m-%d}; test_period={TEST_START:%Y-%m-%d} through final data")
    for symbol, tick_path, bars_n, start, end, comp_n, plain_n, spread in contexts:
        print(f"symbol_context={symbol},tick_file={tick_path},bars={bars_n},date_range={start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC,compression_events={comp_n},plain_range_events={plain_n},spread={spread:.4f}")

    for period in ("all", "train_2016_2021", "test_2022_2026"):
        print_expansion_matrix(all_expansions, period)
        print_matrix(all_results, period)
        print_capture_matrix(all_results, all_expansions, period)

    passes = strict_passes(all_results)
    print("\nVERDICT_INPUTS")
    print(f"compression_cells_with_full_train_test_ci_above_zero={';'.join(passes) if passes else 'none'}")


if __name__ == "__main__":
    main()
