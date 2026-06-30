"""
Full gauntlet for BB/Keltner squeeze breakout-following.

This is a fair follow-up to the compression ablation. It uses the same exits,
costs, period splits, random matching, regime decomposition, and head-to-head
metrics as the corrected compression breakout audit.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
RRS = (1.5, 2.0)
HORIZONS = (10, 20)
RANDOM_SEED = 20260630
GATE_THRESHOLD = 0.10
SQUEEZE_PERIOD = 20
SQUEEZE_MIN_RUN = 5
KELTNER_MULT = 1.5


@dataclass(frozen=True)
class Event:
    event_id: int
    family: str
    setup_start: int
    setup_end: int
    fire_index: int
    breakout_index: int
    range_high: float
    range_low: float
    direction: int


@dataclass(frozen=True)
class Trade:
    symbol: str
    family: str
    signal: str
    rr: float
    horizon: int
    gated: bool
    event_id: int
    setup_end: int
    breakout_index: int
    entry_time: datetime
    direction: int
    gross_r: float
    net_r: float
    exit_reason: str
    unresolved: bool
    bars_held: int
    risk: float
    spread_r: float


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def period_filter(rows: list[Trade], period: str) -> list[Trade]:
    if period == "all":
        return rows
    if period == "train_2016_2021":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def summarize(rows: list[Trade], field: str = "net_r") -> dict[str, float]:
    vals = [getattr(r, field) for r in rows]
    n, m, lo, hi = ci(vals)
    return {
        "n": n,
        "mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "win_rate": sum(v > 0 for v in vals) / n if n else math.nan,
        "unresolved": sum(r.unresolved for r in rows) / n if n else math.nan,
        "avg_bars": mean([r.bars_held for r in rows]) if rows else math.nan,
        "avg_risk": mean([r.risk for r in rows]) if rows else math.nan,
        "avg_spread_r": mean([r.spread_r for r in rows]) if rows else math.nan,
        "total_r": sum(vals),
    }


def equity_stats(rows: list[Trade]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda r: (r.entry_time, r.event_id))
    vals = [r.net_r for r in ordered]
    n = len(vals)
    sd = pstdev(vals) if n > 1 else math.nan
    sharpe = mean(vals) / sd * math.sqrt(n) if n > 1 and sd > 0 else math.nan
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    if not ordered:
        years = math.nan
    else:
        years = (ordered[-1].entry_time - ordered[0].entry_time).days / 365.25
    return {
        "sharpe_trade_order": sharpe,
        "max_drawdown_r": max_dd,
        "trades_per_year": n / years if years and years > 0 else math.nan,
        "total_r": sum(vals),
    }


def squeeze_flags(bars: list[base.DeltaBar]) -> list[bool]:
    closes = [b.close for b in bars]
    flags = [False] * len(bars)
    for i, bar in enumerate(bars):
        if i < SQUEEZE_PERIOD - 1 or bar.atr14 is None:
            continue
        if not ablate.same_segment(bars, i - SQUEEZE_PERIOD + 1, i):
            continue
        window = closes[i - SQUEEZE_PERIOD + 1 : i + 1]
        sma = mean(window)
        sd = pstdev(window)
        bb_upper = sma + 2.0 * sd
        bb_lower = sma - 2.0 * sd
        kc_upper = sma + KELTNER_MULT * bar.atr14
        kc_lower = sma - KELTNER_MULT * bar.atr14
        flags[i] = bb_upper < kc_upper and bb_lower > kc_lower
    return flags


def find_range_break(
    bars: list[base.DeltaBar],
    start: int,
    setup_end: int,
    range_high: float,
    range_low: float,
) -> tuple[int, int] | None:
    for j in range(start, len(bars) - ablate.DETECT_SKIP_BARS):
        if bars[j].segment_id != bars[setup_end].segment_id:
            return None
        if bars[j].close > range_high:
            return j, 1
        if bars[j].close < range_low:
            return j, -1
    return None


def detect_squeeze_events(bars: list[base.DeltaBar]) -> list[Event]:
    flags = squeeze_flags(bars)
    events: list[Event] = []
    i = 0
    while i < len(bars) - ablate.DETECT_SKIP_BARS - 1:
        if not flags[i]:
            i += 1
            continue
        run_start = i
        while i + 1 < len(bars) and flags[i + 1] and bars[i + 1].segment_id == bars[run_start].segment_id:
            i += 1
        run_end = i
        fire_index = run_end + 1
        if (
            run_end - run_start + 1 < SQUEEZE_MIN_RUN
            or fire_index >= len(bars) - ablate.DETECT_SKIP_BARS
            or bars[fire_index].segment_id != bars[run_end].segment_id
            or flags[fire_index]
            or bars[run_end].atr14 is None
        ):
            i += 1
            continue
        range_high = max(b.high for b in bars[run_start : run_end + 1])
        range_low = min(b.low for b in bars[run_start : run_end + 1])
        breakout = find_range_break(bars, fire_index, run_end, range_high, range_low)
        if breakout is None:
            i += 1
            continue
        breakout_index, direction = breakout
        events.append(Event(len(events) + 1, "squeeze", run_start, run_end, fire_index, breakout_index, range_high, range_low, direction))
        i = breakout_index + ablate.DETECT_SKIP_BARS + 1
    return events


def compression_events(bars: list[base.DeltaBar]) -> list[Event]:
    out: list[Event] = []
    for e in ablate.detect_compression(bars):
        out.append(Event(e.event_id, "compression", e.setup_start, e.setup_end, e.setup_end + 1, e.breakout_index, e.range_high, e.range_low, e.direction))
    return out


def simulate(
    symbol: str,
    bars: list[base.DeltaBar],
    event: Event,
    signal: str,
    direction: int,
    rr: float,
    horizon: int,
    spread: float,
    gated: bool = False,
) -> Trade | None:
    risk = bars[event.setup_end].atr14
    if risk is None or risk <= 0:
        return None
    if gated and spread / risk > GATE_THRESHOLD:
        return None
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    entry = event.range_high if event.direction == 1 else event.range_low
    stop = entry - direction * risk
    target = entry + direction * rr * risk
    end_index = simple.segment_end_index(bars, eval_start, horizon)
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
    net_r = gross_r - spread / risk
    return Trade(
        symbol=symbol,
        family=event.family,
        signal=signal,
        rr=rr,
        horizon=horizon,
        gated=gated,
        event_id=event.event_id,
        setup_end=event.setup_end,
        breakout_index=event.breakout_index,
        entry_time=bars[entry_index].start,
        direction=direction,
        gross_r=gross_r,
        net_r=net_r,
        exit_reason=exit_reason,
        unresolved=exit_reason == "force_close",
        bars_held=exit_index - entry_index,
        risk=risk,
        spread_r=spread / risk,
    )


def build_trades(symbol: str, bars: list[base.DeltaBar], events: list[Event], spread: float) -> list[Trade]:
    rng = random.Random(f"{RANDOM_SEED}-{symbol}-{events[0].family if events else 'none'}")
    rows: list[Trade] = []
    for event in events:
        random_dir = 1 if rng.random() >= 0.5 else -1
        for signal, direction in (("breakout", event.direction), ("matched_random", random_dir)):
            for rr in RRS:
                for horizon in HORIZONS:
                    trade = simulate(symbol, bars, event, signal, direction, rr, horizon, spread, False)
                    if trade is not None:
                        rows.append(trade)
                    if signal == "breakout" and horizon == 10:
                        gated_trade = simulate(symbol, bars, event, "breakout", direction, rr, horizon, spread, True)
                        if gated_trade is not None:
                            rows.append(gated_trade)
    return rows


def trailing_atr_env(bars: list[base.DeltaBar], i: int) -> float | None:
    vals: list[float] = []
    j = i - 1
    while j >= 0 and len(vals) < base.ATR_TRAIL:
        if bars[j].atr14 is not None:
            vals.append(bars[j].atr14)
        j -= 1
    return mean(vals) if len(vals) == base.ATR_TRAIL else None


def percentile(vals: list[float], q: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def regime_map(bars: list[base.DeltaBar], events: list[Event]) -> dict[int, str]:
    env_by_event = {e.event_id: trailing_atr_env(bars, e.setup_end) for e in events}
    envs = [v for v in env_by_event.values() if v is not None]
    low = percentile(envs, 1 / 3)
    high = percentile(envs, 2 / 3)
    out = {}
    for eid, env in env_by_event.items():
        if env is None:
            continue
        out[eid] = "low_vol" if env <= low else ("mid_vol" if env <= high else "high_vol")
    return out


def monthly_returns(rows: list[Trade]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for r in rows:
        key = (r.entry_time.year, r.entry_time.month)
        out[key] = out.get(key, 0.0) + r.net_r
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def print_matrix(rows: list[Trade]) -> None:
    print("\nSQUEEZE_GAUNTLET_MATRIX")
    print("period,symbol,family,signal,rr,horizon,gated,n,gross_r,gross_ci_low,gross_ci_high,net_r,net_ci_low,net_ci_high,win_rate,unresolved_pct,avg_bars,avg_risk,avg_spread_r,total_net_r,ci_clears_zero")
    keys = sorted({(r.symbol, r.family, r.signal, r.rr, r.horizon, r.gated) for r in rows})
    for period in ("all", "train_2016_2021", "test_2022_2026"):
        period_rows = period_filter(rows, period)
        for symbol, family, signal, rr, horizon, gated in keys:
            subset = [r for r in period_rows if (r.symbol, r.family, r.signal, r.rr, r.horizon, r.gated) == (symbol, family, signal, rr, horizon, gated)]
            gross = summarize(subset, "gross_r")
            net = summarize(subset, "net_r")
            print(
                f"{period},{symbol},{family},{signal},{rr:.1f},{horizon},{gated},{int(net['n'])},"
                f"{gross['mean']:.6f},{gross['ci_low']:.6f},{gross['ci_high']:.6f},"
                f"{net['mean']:.6f},{net['ci_low']:.6f},{net['ci_high']:.6f},"
                f"{net['win_rate']:.2%},{net['unresolved']:.2%},{net['avg_bars']:.2f},{net['avg_risk']:.6f},"
                f"{net['avg_spread_r']:.6f},{net['total_r']:.6f},{net['ci_low'] > 0 if math.isfinite(net['ci_low']) else False}"
            )


def print_matched_random(rows: list[Trade]) -> None:
    print("\nMATCHED_RANDOM_DELTA")
    print("period,symbol,family,rr,horizon,gated,breakout_net,random_net,delta_net,breakout_gross,random_gross,delta_gross,beats_random_net,beats_random_gross")
    for period in ("all", "train_2016_2021", "test_2022_2026"):
        period_rows = period_filter(rows, period)
        keys = sorted({(r.symbol, r.family, r.rr, r.horizon, r.gated) for r in period_rows if r.signal == "breakout"})
        for symbol, family, rr, horizon, gated in keys:
            b = [r for r in period_rows if (r.symbol, r.family, r.signal, r.rr, r.horizon, r.gated) == (symbol, family, "breakout", rr, horizon, gated)]
            rnd = [r for r in period_rows if (r.symbol, r.family, r.signal, r.rr, r.horizon, r.gated) == (symbol, family, "matched_random", rr, horizon, gated)]
            bs, rs = summarize(b, "net_r"), summarize(rnd, "net_r")
            bg, rg = summarize(b, "gross_r"), summarize(rnd, "gross_r")
            print(f"{period},{symbol},{family},{rr:.1f},{horizon},{gated},{bs['mean']:.6f},{rs['mean']:.6f},{bs['mean'] - rs['mean']:.6f},{bg['mean']:.6f},{rg['mean']:.6f},{bg['mean'] - rg['mean']:.6f},{bs['mean'] > rs['mean']},{bg['mean'] > rg['mean']}")


def print_regimes(rows: list[Trade], regimes: dict[tuple[str, str], dict[int, str]]) -> None:
    print("\nREGIME_BY_ATR_TERCILE")
    print("symbol,family,rr,horizon,regime,n,net_r,ci_low,ci_high,win_rate,total_r")
    for symbol in sorted({r.symbol for r in rows}):
        for family in ("squeeze", "compression"):
            reg = regimes.get((symbol, family), {})
            for rr in RRS:
                for horizon in (10,):
                    base_rows = [r for r in rows if r.symbol == symbol and r.family == family and r.signal == "breakout" and r.rr == rr and r.horizon == horizon and not r.gated]
                    for regime in ("low_vol", "mid_vol", "high_vol"):
                        subset = [r for r in base_rows if reg.get(r.event_id) == regime]
                        s = summarize(subset)
                        print(f"{symbol},{family},{rr:.1f},{horizon},{regime},{int(s['n'])},{s['mean']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{s['win_rate']:.2%},{s['total_r']:.6f}")


def print_year(rows: list[Trade]) -> None:
    print("\nREGIME_BY_YEAR")
    print("symbol,family,rr,horizon,year,n,net_r,ci_low,ci_high,win_rate,total_r")
    for symbol in sorted({r.symbol for r in rows}):
        for family in ("squeeze", "compression"):
            for rr in RRS:
                subset_base = [r for r in rows if r.symbol == symbol and r.family == family and r.signal == "breakout" and r.rr == rr and r.horizon == 10 and not r.gated]
                for year in range(2016, 2027):
                    subset = [r for r in subset_base if r.entry_time.year == year]
                    s = summarize(subset)
                    print(f"{symbol},{family},{rr:.1f},10,{year},{int(s['n'])},{s['mean']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{s['win_rate']:.2%},{s['total_r']:.6f}")


def print_head_to_head(rows: list[Trade], events_by_key: dict[tuple[str, str], list[Event]]) -> None:
    print("\nHEAD_TO_HEAD")
    print("period,symbol,rr,horizon,family,n,net_r,ci_low,ci_high,sharpe,max_drawdown_r,trades_per_year,total_r")
    for period in ("all", "train_2016_2021", "test_2022_2026"):
        for symbol in sorted({r.symbol for r in rows}):
            for rr in RRS:
                for horizon in (10,):
                    for family in ("compression", "squeeze"):
                        subset = [r for r in period_filter(rows, period) if r.symbol == symbol and r.family == family and r.signal == "breakout" and r.rr == rr and r.horizon == horizon and not r.gated]
                        s = summarize(subset)
                        e = equity_stats(subset)
                        print(f"{period},{symbol},{rr:.1f},{horizon},{family},{int(s['n'])},{s['mean']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{e['sharpe_trade_order']:.6f},{e['max_drawdown_r']:.6f},{e['trades_per_year']:.2f},{e['total_r']:.6f}")
    print("\nEVENT_OVERLAP")
    print("symbol,compression_events,squeeze_events,exact_breakout_overlap,exact_pct_of_compression,exact_pct_of_squeeze,within_4bar_overlap,pct_compression_within4,pct_squeeze_within4,monthly_net_corr_rr_1_5_h10")
    for symbol in sorted({k[0] for k in events_by_key}):
        comp = events_by_key[(symbol, "compression")]
        sq = events_by_key[(symbol, "squeeze")]
        comp_idx = {e.breakout_index for e in comp}
        sq_idx = {e.breakout_index for e in sq}
        exact = len(comp_idx & sq_idx)
        within = 0
        sq_sorted = sorted(sq_idx)
        for idx in comp_idx:
            if any(abs(idx - s) <= 4 for s in sq_sorted):
                within += 1
        comp_rows = [r for r in rows if r.symbol == symbol and r.family == "compression" and r.signal == "breakout" and r.rr == 1.5 and r.horizon == 10 and not r.gated]
        sq_rows = [r for r in rows if r.symbol == symbol and r.family == "squeeze" and r.signal == "breakout" and r.rr == 1.5 and r.horizon == 10 and not r.gated]
        cm, sm = monthly_returns(comp_rows), monthly_returns(sq_rows)
        months = sorted(set(cm) | set(sm))
        corr = pearson([cm.get(m, 0.0) for m in months], [sm.get(m, 0.0) for m in months])
        print(f"{symbol},{len(comp)},{len(sq)},{exact},{exact / len(comp):.2%},{exact / len(sq):.2%},{within},{within / len(comp):.2%},{within / len(sq):.2%},{corr:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xag-ticks", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--xag-cache", type=Path, default=Path("data/xagusd_m15_delta_bars.csv"))
    parser.add_argument("--xau-spread", type=float, default=0.20)
    parser.add_argument("--xag-spread", type=float, default=0.02)
    args = parser.parse_args()

    rows: list[Trade] = []
    events_by_key: dict[tuple[str, str], list[Event]] = {}
    regimes: dict[tuple[str, str], dict[int, str]] = {}
    contexts = []
    for symbol, ticks, cache, spread in (
        ("XAUUSD", args.xau_ticks, args.xau_cache, args.xau_spread),
        ("XAGUSD", args.xag_ticks, args.xag_cache, args.xag_spread),
    ):
        bars = simple.load_symbol_bars(symbol, ticks, cache)
        sq_events = detect_squeeze_events(bars)
        comp_events = compression_events(bars)
        events_by_key[(symbol, "squeeze")] = sq_events
        events_by_key[(symbol, "compression")] = comp_events
        regimes[(symbol, "squeeze")] = regime_map(bars, sq_events)
        regimes[(symbol, "compression")] = regime_map(bars, comp_events)
        rows.extend(build_trades(symbol, bars, sq_events, spread))
        rows.extend(build_trades(symbol, bars, comp_events, spread))
        contexts.append((symbol, ticks, len(bars), bars[0].start, bars[-1].end, spread, len(sq_events), len(comp_events)))

    print("SQUEEZE_GAUNTLET_CONTEXT")
    print("squeeze=BB(20,2.0) inside Keltner(20,1.5*ATR14); fire=first bar after inside state exits Keltner")
    print("entry=range edge of squeeze period; risk=ATR14 at squeeze-end/compression-end pre-breakout bar; no breakout-bar ATR")
    print("exits=1R stop; TP 1.5R/2R; horizons 10/20; same-bar stop before target; segment-aware force close")
    print("costs=XAUUSD $0.20; XAGUSD $0.02; spread_atr_gate=spread/pre-breakout_ATR <= 0.10")
    print(f"train_period=through {TRAIN_END:%Y-%m-%d}; test_period={TEST_START:%Y-%m-%d} through final data")
    for symbol, ticks, n, start, end, spread, sq_n, comp_n in contexts:
        print(f"symbol_context={symbol},tick_file={ticks},bars={n},date_range={start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC,spread={spread:.4f},squeeze_events={sq_n},compression_events={comp_n}")
    print_matrix(rows)
    print_matched_random(rows)
    print_regimes(rows, regimes)
    print_year(rows)
    print_head_to_head(rows, events_by_key)


if __name__ == "__main__":
    main()
