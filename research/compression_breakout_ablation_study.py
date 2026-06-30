"""
Ablation study for the corrected compression breakout strategy.

Baseline under test:
- XAUUSD compression breakout following.
- Compression end ATR for risk.
- Entry at broken range edge.
- 1R stop, 1.5R target, 10-bar segment-aware force close.
- Round-trip spread deducted in R.

The report intentionally includes unfavorable ablations. It is meant to locate
the source of the edge, not optimize a final configuration.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from statistics import mean, pstdev

import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

BASE_RR = 1.5
BASE_HORIZON = 10
DETECT_SKIP_BARS = base.EXIT_HORIZON
DONCHIAN_WINDOW = 20
SQUEEZE_PERIOD = 20
SQUEEZE_MIN_RUN = 5
SQUEEZE_KELTNER_MULT = 1.5
SPREAD_ATR_THRESHOLDS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


@dataclass(frozen=True)
class Event:
    event_id: int
    family: str
    setup_start: int
    setup_end: int
    breakout_index: int
    range_high: float
    range_low: float
    direction: int


@dataclass(frozen=True)
class Trade:
    symbol: str
    ablation: str
    family: str
    event_id: int
    entry_time: datetime
    net_r: float
    gross_r: float
    win: bool
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


def same_segment(bars: list[base.DeltaBar], start: int, end: int) -> bool:
    return start >= 0 and end < len(bars) and bars[start].segment_id == bars[end].segment_id and all(
        bars[i].segment_id == bars[start].segment_id for i in range(start, end + 1)
    )


def segment_last_index(bars: list[base.DeltaBar], start_index: int) -> int:
    segment = bars[start_index].segment_id
    i = start_index
    while i + 1 < len(bars) and bars[i + 1].segment_id == segment:
        i += 1
    return i


def risk_at_setup_end(bars: list[base.DeltaBar], event: Event) -> float | None:
    risk = bars[event.setup_end].atr14
    if risk is None or risk <= 0:
        return None
    return risk


def make_event(
    events: list[Event],
    family: str,
    setup_start: int,
    setup_end: int,
    breakout_index: int,
    range_high: float,
    range_low: float,
    direction: int,
) -> None:
    events.append(Event(len(events) + 1, family, setup_start, setup_end, breakout_index, range_high, range_low, direction))


def find_first_close_breakout(
    bars: list[base.DeltaBar],
    setup_end: int,
    range_high: float,
    range_low: float,
) -> tuple[int, int] | None:
    for j in range(setup_end + 1, len(bars) - DETECT_SKIP_BARS):
        if bars[j].segment_id != bars[setup_end].segment_id:
            return None
        if bars[j].close > range_high:
            return j, 1
        if bars[j].close < range_low:
            return j, -1
    return None


def detect_compression(bars: list[base.DeltaBar]) -> list[Event]:
    events: list[Event] = []
    i = base.ATR_TRAIL + base.COMPRESSION_WINDOW
    while i < len(bars) - DETECT_SKIP_BARS:
        if not base.is_compression_end(bars, i):
            i += 1
            continue
        start = i - base.COMPRESSION_WINDOW + 1
        range_high = max(b.high for b in bars[start : i + 1])
        range_low = min(b.low for b in bars[start : i + 1])
        breakout = find_first_close_breakout(bars, i, range_high, range_low)
        if breakout is None:
            i += 1
            continue
        bidx, direction = breakout
        make_event(events, "compression", start, i, bidx, range_high, range_low, direction)
        i = bidx + DETECT_SKIP_BARS + 1
    return events


def detect_donchian(bars: list[base.DeltaBar]) -> list[Event]:
    events: list[Event] = []
    i = DONCHIAN_WINDOW - 1
    while i < len(bars) - DETECT_SKIP_BARS:
        start = i - DONCHIAN_WINDOW + 1
        if not same_segment(bars, start, i) or bars[i].atr14 is None:
            i += 1
            continue
        range_high = max(b.high for b in bars[start : i + 1])
        range_low = min(b.low for b in bars[start : i + 1])
        breakout = find_first_close_breakout(bars, i, range_high, range_low)
        if breakout is None:
            i += 1
            continue
        bidx, direction = breakout
        make_event(events, "donchian20", start, i, bidx, range_high, range_low, direction)
        i = bidx + DETECT_SKIP_BARS + 1
    return events


def detect_asian_session(bars: list[base.DeltaBar]) -> list[Event]:
    events: list[Event] = []
    by_day: dict[tuple[int, date], list[int]] = {}
    for i, bar in enumerate(bars):
        by_day.setdefault((bar.segment_id, bar.start.date()), []).append(i)
    for _, idxs in sorted(by_day.items(), key=lambda item: item[1][0]):
        asian = [i for i in idxs if time(0, 0) <= bars[i].start.time() < time(7, 0)]
        if len(asian) < 8:
            continue
        setup_start, setup_end = asian[0], asian[-1]
        if bars[setup_end].atr14 is None:
            continue
        range_high = max(bars[i].high for i in asian)
        range_low = min(bars[i].low for i in asian)
        for j in idxs:
            if j <= setup_end or bars[j].start.time() >= time(17, 0):
                continue
            if bars[j].close > range_high:
                make_event(events, "asian_session_range", setup_start, setup_end, j, range_high, range_low, 1)
                break
            if bars[j].close < range_low:
                make_event(events, "asian_session_range", setup_start, setup_end, j, range_high, range_low, -1)
                break
    return events


def rolling_sma(values: list[float], end: int, period: int) -> float | None:
    if end - period + 1 < 0:
        return None
    window = values[end - period + 1 : end + 1]
    return mean(window)


def squeeze_flags(bars: list[base.DeltaBar]) -> list[bool]:
    closes = [b.close for b in bars]
    flags = [False] * len(bars)
    for i, bar in enumerate(bars):
        if i < SQUEEZE_PERIOD - 1 or bar.atr14 is None:
            continue
        if not same_segment(bars, i - SQUEEZE_PERIOD + 1, i):
            continue
        window = closes[i - SQUEEZE_PERIOD + 1 : i + 1]
        sma = mean(window)
        sd = pstdev(window)
        bb_upper = sma + 2.0 * sd
        bb_lower = sma - 2.0 * sd
        kc_mid = sma
        kc_upper = kc_mid + SQUEEZE_KELTNER_MULT * bar.atr14
        kc_lower = kc_mid - SQUEEZE_KELTNER_MULT * bar.atr14
        flags[i] = bb_upper < kc_upper and bb_lower > kc_lower
    return flags


def detect_squeeze(bars: list[base.DeltaBar]) -> list[Event]:
    flags = squeeze_flags(bars)
    events: list[Event] = []
    i = 0
    while i < len(bars) - DETECT_SKIP_BARS:
        if not flags[i]:
            i += 1
            continue
        run_start = i
        while i + 1 < len(bars) and flags[i + 1] and bars[i + 1].segment_id == bars[run_start].segment_id:
            i += 1
        run_end = i
        if run_end - run_start + 1 < SQUEEZE_MIN_RUN or bars[run_end].atr14 is None:
            i += 1
            continue
        range_high = max(b.high for b in bars[run_start : run_end + 1])
        range_low = min(b.low for b in bars[run_start : run_end + 1])
        breakout = find_first_close_breakout(bars, run_end, range_high, range_low)
        if breakout is None:
            i += 1
            continue
        bidx, direction = breakout
        make_event(events, "bb_inside_keltner_squeeze", run_start, run_end, bidx, range_high, range_low, direction)
        i = bidx + DETECT_SKIP_BARS + 1
    return events


def simulate(
    symbol: str,
    bars: list[base.DeltaBar],
    event: Event,
    ablation: str,
    rr: float = BASE_RR,
    horizon: int | None = BASE_HORIZON,
    spread: float = 0.20,
    entry_mode: str = "range_edge",
) -> Trade | None:
    risk = risk_at_setup_end(bars, event)
    if risk is None:
        return None
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    direction = event.direction
    entry = bars[entry_index].close if entry_mode == "close_confirmation" else (event.range_high if direction == 1 else event.range_low)
    stop = entry - direction * risk
    target = entry + direction * rr * risk
    if horizon is None:
        end_index = segment_last_index(bars, eval_start)
        default_reason = "segment_gap_unresolved"
    else:
        end_index = simple.segment_end_index(bars, eval_start, horizon)
        default_reason = "force_close"
    gross_r = direction * (bars[end_index].close - entry) / risk
    exit_reason = default_reason
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
        ablation=ablation,
        family=event.family,
        event_id=event.event_id,
        entry_time=bars[entry_index].start,
        net_r=net_r,
        gross_r=gross_r,
        win=net_r > 0,
        exit_reason=exit_reason,
        unresolved=exit_reason in ("force_close", "segment_gap_unresolved"),
        bars_held=exit_index - entry_index,
        risk=risk,
        spread_r=spread / risk,
    )


def period_rows(rows: list[Trade], period: str) -> list[Trade]:
    if period == "all":
        return rows
    if period == "train_2016_2021":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def summarize(rows: list[Trade]) -> dict[str, float]:
    vals = [r.net_r for r in rows]
    n, m, lo, hi = ci(vals)
    return {
        "n": n,
        "net": m,
        "ci_low": lo,
        "ci_high": hi,
        "win_rate": sum(r.win for r in rows) / n if n else math.nan,
        "unresolved": sum(r.unresolved for r in rows) / n if n else math.nan,
        "avg_bars": mean([r.bars_held for r in rows]) if rows else math.nan,
        "avg_risk": mean([r.risk for r in rows]) if rows else math.nan,
        "avg_spread_r": mean([r.spread_r for r in rows]) if rows else math.nan,
        "total_r": sum(vals),
    }


def build_core_rows(symbol: str, bars: list[base.DeltaBar], spread: float) -> tuple[list[Trade], dict[str, int]]:
    compression = detect_compression(bars)
    donchian = detect_donchian(bars)
    asian = detect_asian_session(bars)
    squeeze = detect_squeeze(bars)
    rows: list[Trade] = []
    for event in compression:
        specs = [
            ("baseline_range_edge_1_5r_10bar", BASE_RR, BASE_HORIZON, spread, "range_edge"),
            ("no_force_close_tp_sl_only", BASE_RR, None, spread, "range_edge"),
            ("close_confirmation_entry", BASE_RR, BASE_HORIZON, spread, "close_confirmation"),
            ("tp_3r", 3.0, BASE_HORIZON, spread, "range_edge"),
            ("cost_1_5x_spread_030", BASE_RR, BASE_HORIZON, 0.30 if symbol == "XAUUSD" else spread * 1.5, "range_edge"),
            ("cost_2x_spread_040", BASE_RR, BASE_HORIZON, 0.40 if symbol == "XAUUSD" else spread * 2.0, "range_edge"),
        ]
        for spec in specs:
            trade = simulate(symbol, bars, event, *spec)
            if trade is not None:
                rows.append(trade)
    for ablation, events in (
        ("donchian20_no_compression", donchian),
        ("asian_session_range", asian),
        ("bb_inside_keltner_squeeze", squeeze),
    ):
        for event in events:
            trade = simulate(symbol, bars, event, ablation, BASE_RR, BASE_HORIZON, spread, "range_edge")
            if trade is not None:
                rows.append(trade)
    counts = {
        "compression": len(compression),
        "donchian20": len(donchian),
        "asian_session_range": len(asian),
        "bb_inside_keltner_squeeze": len(squeeze),
    }
    return rows, counts


def gate_rows(symbol: str, bars: list[base.DeltaBar], spread: float) -> list[Trade]:
    events = detect_compression(bars)
    out: list[Trade] = []
    for threshold in SPREAD_ATR_THRESHOLDS:
        ablation = f"spread_atr_gate_le_{threshold:.3f}".replace(".", "p")
        for event in events:
            risk = risk_at_setup_end(bars, event)
            if risk is None or spread / risk > threshold:
                continue
            trade = simulate(symbol, bars, event, ablation, BASE_RR, BASE_HORIZON, spread, "range_edge")
            if trade is not None:
                out.append(trade)
    return out


def print_table(rows: list[Trade], title: str) -> None:
    print(f"\n{title}")
    print("period,symbol,ablation,n,net_r,ci_low,ci_high,win_rate,unresolved_pct,avg_bars,avg_risk,avg_spread_r,total_r,ci_clears_zero")
    keys = sorted({(r.symbol, r.ablation) for r in rows})
    for period in ("all", "train_2016_2021", "test_2022_2026"):
        for symbol, ablation in keys:
            subset = [r for r in period_rows(rows, period) if r.symbol == symbol and r.ablation == ablation]
            s = summarize(subset)
            print(
                f"{period},{symbol},{ablation},{int(s['n'])},{s['net']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},"
                f"{s['win_rate']:.2%},{s['unresolved']:.2%},{s['avg_bars']:.2f},{s['avg_risk']:.6f},"
                f"{s['avg_spread_r']:.6f},{s['total_r']:.6f},{s['ci_low'] > 0 if math.isfinite(s['ci_low']) else False}"
            )


def verdict_against_baseline(rows: list[Trade], symbol: str, ablation: str) -> str:
    base_rows = [r for r in rows if r.symbol == symbol and r.ablation == "baseline_range_edge_1_5r_10bar"]
    test_rows = [r for r in rows if r.symbol == symbol and r.ablation == ablation]
    b = summarize(base_rows)
    s = summarize(test_rows)
    if not math.isfinite(s["net"]):
        return "no data"
    delta = s["net"] - b["net"]
    if s["ci_high"] < 0:
        return "kills edge"
    if s["ci_low"] <= 0 and s["net"] < b["net"]:
        return "hurts / unproven"
    if delta > 0.025:
        return "improves"
    if delta < -0.025:
        return "hurts"
    return "neutral"


def cost_breakeven(rows: list[Trade], symbol: str) -> float:
    base_rows = [r for r in rows if r.symbol == symbol and r.ablation == "baseline_range_edge_1_5r_10bar"]
    gross_mean = mean([r.gross_r for r in base_rows])
    avg_inv_risk = mean([1 / r.risk for r in base_rows])
    return gross_mean / avg_inv_risk if avg_inv_risk > 0 else math.nan


def print_summary(rows: list[Trade], gate: list[Trade]) -> None:
    print("\nLOAD_BEARING_VERDICTS")
    print("symbol,component_or_alternative,verdict,one_line")
    checks = [
        ("no_force_close_tp_sl_only", "10-bar force close"),
        ("close_confirmation_entry", "range-edge entry vs close-confirmation"),
        ("tp_3r", "1.5R target vs 3R target"),
        ("donchian20_no_compression", "ATR-tercile compression vs Donchian 20"),
        ("asian_session_range", "ATR-tercile compression vs Asian range"),
        ("bb_inside_keltner_squeeze", "ATR-tercile compression vs BB/Keltner squeeze"),
        ("cost_1_5x_spread_030", "cost robustness at 1.5x"),
        ("cost_2x_spread_040", "cost robustness at 2x"),
    ]
    for symbol in sorted({r.symbol for r in rows}):
        for ablation, label in checks:
            verdict = verdict_against_baseline(rows, symbol, ablation)
            print(f"{symbol},{label},{verdict},{ablation}")
    print("\nCOST_BREAKEVEN")
    print("symbol,baseline_rr,baseline_horizon,estimated_spread_breakeven_usd_per_oz")
    for symbol in sorted({r.symbol for r in rows}):
        print(f"{symbol},{BASE_RR:.1f},{BASE_HORIZON},{cost_breakeven(rows, symbol):.6f}")

    print("\nSPREAD_ATR_GATE_TRAIN_SELECTED")
    print("symbol,selected_threshold,train_net_r,test_net_r,test_ci_low,test_ci_high,test_n,test_win_rate,test_unresolved")
    for symbol in sorted({r.symbol for r in gate}):
        symbol_rows = [r for r in gate if r.symbol == symbol]
        candidates = sorted({r.ablation for r in symbol_rows})
        best = None
        best_train = -math.inf
        for ablation in candidates:
            train = summarize([r for r in period_rows(symbol_rows, "train_2016_2021") if r.ablation == ablation])
            if train["n"] >= 100 and train["net"] > best_train:
                best_train = train["net"]
                best = ablation
        if best is None:
            print(f"{symbol},none,nan,nan,nan,nan,0,nan,nan")
            continue
        test = summarize([r for r in period_rows(symbol_rows, "test_2022_2026") if r.ablation == best])
        threshold = best.removeprefix("spread_atr_gate_le_").replace("p", ".")
        print(
            f"{symbol},{threshold},{best_train:.6f},{test['net']:.6f},{test['ci_low']:.6f},"
            f"{test['ci_high']:.6f},{int(test['n'])},{test['win_rate']:.2%},{test['unresolved']:.2%}"
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

    rows: list[Trade] = []
    gate: list[Trade] = []
    contexts = []
    for symbol, ticks, cache, spread in (
        ("XAUUSD", args.xau_ticks, args.xau_cache, args.xau_spread),
        ("XAGUSD", args.xag_ticks, args.xag_cache, args.xag_spread),
    ):
        bars = simple.load_symbol_bars(symbol, ticks, cache)
        symbol_rows, counts = build_core_rows(symbol, bars, spread)
        rows.extend(symbol_rows)
        gate.extend(gate_rows(symbol, bars, spread))
        contexts.append((symbol, ticks, len(bars), bars[0].start, bars[-1].end, spread, counts))

    print("COMPRESSION_BREAKOUT_ABLATION_CONTEXT")
    print("baseline=range-edge compression breakout; compression-end ATR risk; stop 1R; target 1.5R; 10-bar segment-aware force close")
    print(f"event_deoverlap_skip_bars={DETECT_SKIP_BARS}; exit_horizon_bars={BASE_HORIZON}")
    print("cost=XAUUSD $0.20 baseline spread; XAGUSD $0.02 baseline spread for spread/ATR gate diagnostic")
    print("standard_breakouts=Donchian 20, Asian session range 00:00-07:00 UTC broken before 17:00 UTC, BB(20,2) inside Keltner(20,1.5ATR) squeeze")
    print(f"train_period=through {TRAIN_END:%Y-%m-%d}; test_period={TEST_START:%Y-%m-%d} through final data")
    for symbol, ticks, n, start, end, spread, counts in contexts:
        count_text = ";".join(f"{k}={v}" for k, v in counts.items())
        print(f"symbol_context={symbol},tick_file={ticks},bars={n},date_range={start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC,spread={spread:.4f},{count_text}")
    print_table(rows, "ABLATION_TABLE")
    print_table(gate, "SPREAD_ATR_GATE_ALL_THRESHOLDS")
    print_summary(rows, gate)


if __name__ == "__main__":
    main()
