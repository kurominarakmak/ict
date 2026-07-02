"""
V-2026-FILL-01: entry-fill realism audit for compression breakout.

Research-only. This tests whether the validated close-confirmed range-edge
fill is executable, or whether its edge depends on an optimistic fill.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
from delta_signal_audit import IUX_XAUUSD_ROUNDTRIP_SPREAD


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
RESULTS_PATH = Path("research/entry_fill_realism_results.txt")
REGISTRY_PATH = Path("research/hypothesis_registry.md")
RR = 1.5
HORIZON = 10


@dataclass(frozen=True)
class FillTrade:
    model: str
    event_id: int
    setup_end: int
    entry_index: int
    entry_time: datetime
    direction: int
    entry: float
    risk: float
    gross_r: float
    net_r: float
    exit_reason: str
    bars_held: int
    skipped: bool = False


def q(vals: list[float], frac: float) -> float:
    if not vals:
        return math.nan
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * frac
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def ci(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    m = mean(vals)
    sd = pstdev(vals)
    se = sd / math.sqrt(len(vals))
    return m - 1.96 * se, m + 1.96 * se


def period(rows: list[FillTrade], name: str) -> list[FillTrade]:
    if name == "full":
        return rows
    if name == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if name == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(name)


def summarize(rows: list[FillTrade]) -> dict[str, float]:
    vals = [r.net_r for r in rows if not r.skipped]
    lo, hi = ci(vals)
    return {
        "n": len(vals),
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "net": mean(vals) if vals else math.nan,
        "ci_low": lo,
        "ci_high": hi,
        "skipped": sum(1 for r in rows if r.skipped),
    }


def segment_end_index(bars, start_index: int, horizon: int) -> int:
    return simple.segment_end_index(bars, start_index, horizon)


def simulate_from_entry(
    bars,
    event: ablate.Event,
    model: str,
    entry_index: int,
    direction: int,
    entry: float,
    risk: float,
    spread: float,
) -> FillTrade | None:
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    stop = entry - direction * risk
    target = entry + direction * RR * risk
    end_index = segment_end_index(bars, eval_start, HORIZON)
    exit_index = end_index
    exit_reason = "force_close"
    gross_r = direction * (bars[end_index].close - entry) / risk
    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r = direction * (fill - entry) / risk
            exit_index = i
            exit_reason = "stop"
            break
        if target_hit:
            gross_r = RR
            exit_index = i
            exit_reason = "target"
            break
    return FillTrade(
        model=model,
        event_id=event.event_id,
        setup_end=event.setup_end,
        entry_index=entry_index,
        entry_time=bars[entry_index].start,
        direction=direction,
        entry=entry,
        risk=risk,
        gross_r=gross_r,
        net_r=gross_r - spread / risk,
        exit_reason=exit_reason,
        bars_held=exit_index - entry_index,
    )


def idealized_range_edge(bars, event: ablate.Event, spread: float) -> FillTrade | None:
    trade = ablate.simulate("XAUUSD", bars, event, "research_idealized", RR, HORIZON, spread, "range_edge")
    if trade is None:
        return None
    entry = event.range_high if event.direction == 1 else event.range_low
    return FillTrade(
        "research_idealized_range_edge",
        event.event_id,
        event.setup_end,
        event.breakout_index,
        bars[event.breakout_index].start,
        event.direction,
        entry,
        trade.risk,
        trade.gross_r,
        trade.net_r,
        trade.exit_reason,
        trade.bars_held,
    )


def close_then_market(bars, event: ablate.Event, spread: float) -> FillTrade | None:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return None
    return simulate_from_entry(
        bars,
        event,
        "close_then_market",
        event.breakout_index,
        event.direction,
        bars[event.breakout_index].close,
        risk,
        spread,
    )


def stop_order_first_touch(bars, event: ablate.Event, spread: float) -> FillTrade | None:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return None
    for i in range(event.setup_end + 1, len(bars)):
        if bars[i].segment_id != bars[event.setup_end].segment_id:
            return None
        bar = bars[i]
        buy_hit = bar.high >= event.range_high
        sell_hit = bar.low <= event.range_low
        if not buy_hit and not sell_hit:
            continue
        if buy_hit and sell_hit:
            # Unknown intrabar order. Use the bar close side if it broke, else
            # conservative fixed tie-break to short for determinism.
            if bar.close > event.range_high:
                direction = 1
            elif bar.close < event.range_low:
                direction = -1
            else:
                direction = -1
        else:
            direction = 1 if buy_hit else -1
        entry = event.range_high if direction == 1 else event.range_low
        return simulate_from_entry(bars, event, "stop_order_first_touch", i, direction, entry, risk, spread)
    return None


def close_then_limit_at_edge(bars, event: ablate.Event, spread: float) -> FillTrade:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return FillTrade("close_then_limit_at_edge", event.event_id, event.setup_end, event.breakout_index, bars[event.breakout_index].start, event.direction, math.nan, math.nan, math.nan, math.nan, "skipped_no_risk", 0, True)
    edge = event.range_high if event.direction == 1 else event.range_low
    end = segment_end_index(bars, event.breakout_index + 1, HORIZON)
    for i in range(event.breakout_index + 1, end + 1):
        if bars[i].segment_id != bars[event.breakout_index].segment_id:
            break
        pulls_back = bars[i].low <= edge if event.direction == 1 else bars[i].high >= edge
        if pulls_back:
            trade = simulate_from_entry(bars, event, "close_then_limit_at_edge", i, event.direction, edge, risk, spread)
            if trade is not None:
                return trade
            break
    return FillTrade("close_then_limit_at_edge", event.event_id, event.setup_end, event.breakout_index, bars[event.breakout_index].start, event.direction, edge, risk, math.nan, math.nan, "skipped_no_pullback", 0, True)


def close_beyond_edge_r(bars, event: ablate.Event) -> float | None:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return None
    edge = event.range_high if event.direction == 1 else event.range_low
    return event.direction * (bars[event.breakout_index].close - edge) / risk


def wick_reconciliation(bars, events: list[ablate.Event], ideal: dict[int, FillTrade]) -> dict[str, float]:
    out = {
        "events": len(events),
        "prior_same_edge_touch": 0,
        "prior_opposite_edge_touch": 0,
        "same_bar_touch_only": 0,
        "ideal_wins_with_prior_same_edge_touch": 0,
        "ideal_losses_with_prior_same_edge_touch": 0,
        "same_bar_dual_touch": 0,
    }
    for event in events:
        prior_same = False
        prior_opp = False
        for i in range(event.setup_end + 1, event.breakout_index):
            if bars[i].segment_id != bars[event.setup_end].segment_id:
                break
            same = bars[i].high >= event.range_high if event.direction == 1 else bars[i].low <= event.range_low
            opp = bars[i].low <= event.range_low if event.direction == 1 else bars[i].high >= event.range_high
            prior_same = prior_same or same
            prior_opp = prior_opp or opp
        if prior_same:
            out["prior_same_edge_touch"] += 1
            row = ideal.get(event.event_id)
            if row is not None and row.net_r > 0:
                out["ideal_wins_with_prior_same_edge_touch"] += 1
            elif row is not None:
                out["ideal_losses_with_prior_same_edge_touch"] += 1
        if prior_opp:
            out["prior_opposite_edge_touch"] += 1
        bbar = bars[event.breakout_index]
        same_bar_touch = bbar.high >= event.range_high if event.direction == 1 else bbar.low <= event.range_low
        if same_bar_touch and not prior_same:
            out["same_bar_touch_only"] += 1
        if bbar.high >= event.range_high and bbar.low <= event.range_low:
            out["same_bar_dual_touch"] += 1
    return out


def append_registry(verdict: str, close_market: dict[str, float]) -> None:
    existing = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else "# Hypothesis Registry\n"
    lines = [line for line in existing.rstrip().splitlines() if "V-2026-FILL-01" not in line]
    lines.append("- 2026-07-02: V-2026-FILL-01 registered. Entry-fill realism audit: test whether close-confirmed range-edge fills are achievable; compare ideal range-edge, first-touch stop order, close-then-market, and close-then-limit-at-edge without changing the live bot.")
    lines.append(
        "- 2026-07-02: V-2026-FILL-01 result: "
        f"{verdict}; close_then_market_train={close_market['train_net']:.4f} "
        f"[{close_market['train_lo']:.4f},{close_market['train_hi']:.4f}], "
        f"test={close_market['test_net']:.4f} [{close_market['test_lo']:.4f},{close_market['test_hi']:.4f}]."
    )
    REGISTRY_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--spread", type=float, default=IUX_XAUUSD_ROUNDTRIP_SPREAD)
    args = parser.parse_args()

    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    events = ablate.detect_compression(bars)

    distances = [v for event in events if (v := close_beyond_edge_r(bars, event)) is not None]
    models: dict[str, list[FillTrade]] = {
        "research_idealized_range_edge": [],
        "stop_order_first_touch": [],
        "close_then_market": [],
        "close_then_limit_at_edge": [],
    }
    for event in events:
        for trade in (
            idealized_range_edge(bars, event, args.spread),
            stop_order_first_touch(bars, event, args.spread),
            close_then_market(bars, event, args.spread),
            close_then_limit_at_edge(bars, event, args.spread),
        ):
            if trade is not None:
                models[trade.model].append(trade)

    ideal_by_event = {t.event_id: t for t in models["research_idealized_range_edge"]}
    wick = wick_reconciliation(bars, events, ideal_by_event)
    close_train = summarize(period(models["close_then_market"], "train"))
    close_test = summarize(period(models["close_then_market"], "test"))
    close_pass = close_train["ci_low"] > 0 and close_test["ci_low"] > 0
    verdict = "PASS_EDGE_SURVIVES_CLOSE_MARKET" if close_pass else "FAIL_EDGE_DOES_NOT_SURVIVE_CLOSE_MARKET"

    lines: list[str] = []
    lines.append("V_2026_FILL_01_ENTRY_FILL_REALISM_AUDIT")
    lines.append("symbol,XAUUSD")
    lines.append("rr,1.5")
    lines.append("horizon,10")
    lines.append(f"spread,{args.spread}")
    lines.append(f"events,{len(events)}")
    lines.append("")
    lines.append("CLOSE_BEYOND_EDGE_DISTRIBUTION_R")
    lines.append("n,median,p75,p90,mean")
    lines.append(f"{len(distances)},{q(distances,0.50):.4f},{q(distances,0.75):.4f},{q(distances,0.90):.4f},{mean(distances):.4f}")
    lines.append("")
    lines.append("WICK_CATCH_RECONCILIATION")
    lines.append("metric,value")
    for key, value in wick.items():
        lines.append(f"{key},{value}")
    lines.append("")
    lines.append("MODEL_SUMMARY")
    lines.append("model,period,n,skipped,win_rate,net_r,ci_low,ci_high")
    for model, rows in models.items():
        for pname in ("full", "train", "test"):
            s = summarize(period(rows, pname))
            lines.append(
                f"{model},{pname},{s['n']},{s['skipped']},{s['win']:.4f},"
                f"{s['net']:.4f},{s['ci_low']:.4f},{s['ci_high']:.4f}"
            )
    lines.append("")
    lines.append("VERDICT")
    if close_pass:
        lines.append("Close-then-market keeps positive train/test CIs; the edge survives achievable close-confirm execution.")
        lines.append("Research implication: switch the executable candidate to close-confirmation market entry, then re-audit/live-shadow it.")
    else:
        lines.append("Close-then-market does not clear train/test CIs; the validated range-edge result depends on an optimistic/unachievable fill.")
        lines.append("Research implication: do not promote the live bot; compression edge must go back to research before prop-firm use.")
    report = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(report)
    print(report, end="")
    append_registry(
        verdict,
        {
            "train_net": close_train["net"],
            "train_lo": close_train["ci_low"],
            "train_hi": close_train["ci_high"],
            "test_net": close_test["net"],
            "test_lo": close_test["ci_low"],
            "test_hi": close_test["ci_high"],
        },
    )
    print(f"results_file={RESULTS_PATH}")


if __name__ == "__main__":
    main()
