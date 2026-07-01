"""
Fixed-vs-adaptive TP/SL slippage audit for XAUUSD compression RR1.5.

Research-only. The live bot is not imported or modified. Its relevant behavior
is mirrored from inspection: pending orders use absolute SL/TP from the intended
range-edge entry.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


RR = 1.5
HORIZON = 10
SPREAD = 0.20
MC_N = 250
SEED = 20260701
PROBS = (0.00, 0.01, 0.03, 0.05, 0.10, 0.25)
MAGS = (0.25, 0.50, 1.00)
FULL_PROBS = (0.01, 0.03, 0.05, 0.10, 0.25)
PERIODS = (("2016_2019", 2016, 2019), ("2020_2022", 2020, 2022), ("2023_2026", 2023, 2026))


@dataclass(frozen=True)
class Setup:
    event_id: int
    entry_index: int
    eval_start: int
    intended_entry: float
    original_sl: float
    original_tp: float
    direction: int
    risk: float
    year: int


@dataclass(frozen=True)
class Outcome:
    mode: str
    actual_entry: float
    stop: float
    target: float
    exit_reason: str
    exit_index: int
    exit_price_before_slip: float
    net_r: float
    ambiguous: bool


def q(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def draw_slip(rng: random.Random, spike_prob: float, spike_mean: float) -> float:
    if spike_mean <= 0:
        return 0.0
    if spike_prob > 0 and rng.random() < spike_prob:
        sigma = 0.45
        mu = math.log(spike_mean) - 0.5 * sigma * sigma
        return min(rng.lognormvariate(mu, sigma), spike_mean * 4.0)
    return rng.uniform(0.0, 0.05)


def max_dd(vals: list[float]) -> float:
    equity = peak = dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def build_setups(bars: list[base.DeltaBar]) -> list[Setup]:
    out: list[Setup] = []
    for event in ablate.detect_compression(bars):
        risk = ablate.risk_at_setup_end(bars, event)
        if risk is None or risk <= 0:
            continue
        entry_index = event.breakout_index
        eval_start = entry_index + 1
        if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
            continue
        direction = event.direction
        entry = event.range_high if direction == 1 else event.range_low
        out.append(
            Setup(
                event.event_id,
                entry_index,
                eval_start,
                entry,
                entry - direction * risk,
                entry + direction * RR * risk,
                direction,
                risk,
                bars[entry_index].start.year,
            )
        )
    return out


def simulate(
    bars: list[base.DeltaBar],
    setup: Setup,
    entry_slip: float,
    exit_slip: float,
    mode: str,
    same_bar_policy: str,
) -> Outcome | None:
    d = setup.direction
    actual_entry = setup.intended_entry + entry_slip if d == 1 else setup.intended_entry - entry_slip
    if mode == "fixed":
        stop = setup.original_sl
        target = setup.original_tp
    elif mode == "adaptive":
        stop = actual_entry - d * setup.risk
        target = actual_entry + d * RR * setup.risk
    else:
        raise ValueError(mode)

    end_index = simple.segment_end_index(bars, setup.eval_start, HORIZON)
    exit_reason = "force_close"
    exit_index = end_index
    exit_price = bars[end_index].close
    ambiguous = False
    for i in range(setup.eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if d == 1 else bar.high >= stop
        target_hit = bar.high >= target if d == 1 else bar.low <= target
        if stop_hit and target_hit:
            ambiguous = True
            if same_bar_policy == "exclude":
                return None
            if same_bar_policy == "tp_first":
                exit_reason = "target"
                exit_price = target
            else:
                exit_reason = "stop"
                exit_price = min(stop, bar.low) if d == 1 else max(stop, bar.high)
            exit_index = i
            break
        if stop_hit:
            exit_reason = "stop"
            exit_price = min(stop, bar.low) if d == 1 else max(stop, bar.high)
            exit_index = i
            break
        if target_hit:
            exit_reason = "target"
            exit_price = target
            exit_index = i
            break

    slipped_exit = exit_price - exit_slip if d == 1 else exit_price + exit_slip
    net_r = d * (slipped_exit - actual_entry) / setup.risk - SPREAD / setup.risk
    return Outcome(mode, actual_entry, stop, target, exit_reason, exit_index, exit_price, net_r, ambiguous)


def metrics(rows: list[Outcome]) -> dict[str, float]:
    vals = [r.net_r for r in rows]
    return {
        "net": mean(vals),
        "win": sum(v > 0 for v in vals) / len(vals),
        "tp": sum(r.exit_reason == "target" for r in rows) / len(rows),
        "sl": sum(r.exit_reason == "stop" for r in rows) / len(rows),
        "to": sum(r.exit_reason == "force_close" for r in rows) / len(rows),
        "dd": max_dd(vals),
    }


def run_pair_path(
    bars: list[base.DeltaBar],
    setups: list[Setup],
    rng: random.Random,
    entry_prob: float,
    entry_mag: float,
    exit_prob: float,
    exit_mag: float,
    same_bar_policy: str,
) -> tuple[list[Outcome], list[Outcome]]:
    fixed: list[Outcome] = []
    adaptive: list[Outcome] = []
    for setup in setups:
        entry_slip = draw_slip(rng, entry_prob, entry_mag)
        exit_slip = draw_slip(rng, exit_prob, exit_mag)
        f = simulate(bars, setup, entry_slip, exit_slip, "fixed", same_bar_policy)
        a = simulate(bars, setup, entry_slip, exit_slip, "adaptive", same_bar_policy)
        if f is not None and a is not None:
            fixed.append(f)
            adaptive.append(a)
    return fixed, adaptive


def summarize_mc(label: str, pairs: list[tuple[list[Outcome], list[Outcome]]]) -> str:
    fixed_nets = [metrics(f)["net"] for f, _ in pairs]
    adap_nets = [metrics(a)["net"] for _, a in pairs]
    fixed_wins = [metrics(f)["win"] for f, _ in pairs]
    adap_wins = [metrics(a)["win"] for _, a in pairs]
    fixed_tps = [metrics(f)["tp"] for f, _ in pairs]
    adap_tps = [metrics(a)["tp"] for _, a in pairs]
    fixed_sls = [metrics(f)["sl"] for f, _ in pairs]
    adap_sls = [metrics(a)["sl"] for _, a in pairs]
    fixed_tos = [metrics(f)["to"] for f, _ in pairs]
    adap_tos = [metrics(a)["to"] for _, a in pairs]
    fixed_dds = [metrics(f)["dd"] for f, _ in pairs]
    adap_dds = [metrics(a)["dd"] for _, a in pairs]
    deltas = [a - f for f, a in zip(fixed_nets, adap_nets)]
    return (
        f"{label},{median(fixed_nets):.4f},{median(adap_nets):.4f},{median(deltas):.4f},"
        f"{sum(v > 0 for v in fixed_nets)/len(fixed_nets):.2%},{sum(v > 0 for v in adap_nets)/len(adap_nets):.2%},"
        f"{median(fixed_wins):.2%},{median(adap_wins):.2%},{median(fixed_tps):.2%},{median(adap_tps):.2%},"
        f"{median(fixed_sls):.2%},{median(adap_sls):.2%},{median(fixed_tos):.2%},{median(adap_tos):.2%},"
        f"{median(fixed_dds):.2f},{median(adap_dds):.2f}"
    )


def mc_pairs(bars: list[base.DeltaBar], setups: list[Setup], label: str, entry_prob: float, entry_mag: float, exit_prob: float, exit_mag: float, policy: str) -> list[tuple[list[Outcome], list[Outcome]]]:
    out = []
    for i in range(MC_N):
        rng = random.Random(f"{SEED}-{label}-{i}")
        out.append(run_pair_path(bars, setups, rng, entry_prob, entry_mag, exit_prob, exit_mag, policy))
    return out


def period_label(year: int) -> str:
    for label, start, end in PERIODS:
        if start <= year <= end:
            return label
    return "other"


def path_change_audit(bars: list[base.DeltaBar], setups: list[Setup]) -> tuple[dict[str, int], list[str]]:
    rng = random.Random(f"{SEED}-changes")
    counts = {
        "tp_not_tp": 0,
        "sl_not_sl": 0,
        "tp_to_timeout": 0,
        "tp_to_sl": 0,
        "timeout_to_tp": 0,
        "sl_to_tp": 0,
    }
    examples: list[str] = []
    for setup in setups:
        entry_slip = draw_slip(rng, 0.25, 1.0)
        f = simulate(bars, setup, entry_slip, 0.0, "fixed", "sl_first")
        a = simulate(bars, setup, entry_slip, 0.0, "adaptive", "sl_first")
        if f is None or a is None or f.exit_reason == a.exit_reason:
            continue
        if f.exit_reason == "target" and a.exit_reason != "target":
            counts["tp_not_tp"] += 1
        if f.exit_reason == "stop" and a.exit_reason != "stop":
            counts["sl_not_sl"] += 1
        key = f"{f.exit_reason}_to_{a.exit_reason}"
        if key == "target_to_force_close":
            counts["tp_to_timeout"] += 1
        elif key == "target_to_stop":
            counts["tp_to_sl"] += 1
        elif key == "force_close_to_target":
            counts["timeout_to_tp"] += 1
        elif key == "stop_to_target":
            counts["sl_to_tp"] += 1
        if len(examples) < 20:
            shifted_sl = setup.original_sl + setup.direction * entry_slip
            shifted_tp = setup.original_tp + setup.direction * entry_slip
            examples.append(
                f"{setup.event_id},{'long' if setup.direction==1 else 'short'},{setup.intended_entry:.2f},{a.actual_entry:.2f},"
                f"{entry_slip:.2f},{setup.original_tp:.2f},{setup.original_sl:.2f},{shifted_tp:.2f},{shifted_sl:.2f},"
                f"{f.exit_reason},{a.exit_reason},{f.net_r:.4f},{a.net_r:.4f},{setup.risk:.4f},{f.exit_index},{a.exit_index}"
            )
    return counts, examples


def ambiguous_count(bars: list[base.DeltaBar], setups: list[Setup], policy: str) -> tuple[int, float]:
    rng = random.Random(f"{SEED}-amb-{policy}")
    rows = []
    ambiguous = 0
    for setup in setups:
        entry_slip = draw_slip(rng, 0.25, 1.0)
        marker = simulate(bars, setup, entry_slip, 0.0, "adaptive", "sl_first")
        if marker is not None and marker.ambiguous:
            ambiguous += 1
        out = simulate(bars, setup, entry_slip, 0.0, "adaptive", policy)
        if out is not None:
            rows.append(out)
    return ambiguous, mean(r.net_r for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    args = parser.parse_args()
    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    setups = build_setups(bars)

    print("ADAPTIVE_TP_SLIPPAGE_AUDIT")
    print(f"symbol=XAUUSD,strategy=compression,rr={RR},horizon={HORIZON},mc_runs={MC_N},spread={SPREAD}")
    fixed0, adap0 = run_pair_path(bars, setups, random.Random("sanity"), 0.0, 0.0, 0.0, 0.0, "sl_first")
    print("\n1_NO_SLIPPAGE_SANITY")
    print(f"fixed_mean_net_R={metrics(fixed0)['net']:.4f},adaptive_mean_net_R={metrics(adap0)['net']:.4f},expected=0.2230")
    if abs(metrics(fixed0)["net"] - 0.2230) > 0.0005 or abs(metrics(adap0)["net"] - 0.2230) > 0.0005:
        raise SystemExit("No-slippage sanity failed; stopping.")

    print("\n2_ENTRY_SLIPPAGE_ONLY_GRID")
    print("label,fixed_median_R,adaptive_median_R,adaptive_minus_fixed,P_fixed_positive,P_adaptive_positive,fixed_win,adaptive_win,fixed_tp,adaptive_tp,fixed_sl,adaptive_sl,fixed_timeout,adaptive_timeout,fixed_maxdd,adaptive_maxdd")
    entry_results = []
    for mag in MAGS:
        for prob in PROBS:
            label = f"entry_only_p{prob:.0%}_mag{mag:.2f}"
            pairs = mc_pairs(bars, setups, label, prob, mag, 0.0, mag, "sl_first")
            line = summarize_mc(label, pairs)
            entry_results.append(line)
            print(line)

    print("\n3_FULL_ENTRY_EXIT_SLIPPAGE")
    print("label,fixed_median_R,adaptive_median_R,adaptive_minus_fixed,P_fixed_positive,P_adaptive_positive,fixed_win,adaptive_win,fixed_tp,adaptive_tp,fixed_sl,adaptive_sl,fixed_timeout,adaptive_timeout,fixed_maxdd,adaptive_maxdd")
    for prob in FULL_PROBS:
        label = f"full_p{prob:.0%}_mag1.00"
        print(summarize_mc(label, mc_pairs(bars, setups, label, prob, 1.0, prob, 1.0, "sl_first")))

    worse = [line.split(",") for line in entry_results if float(line.split(",")[3]) < 0]
    print("\n4_BREAKING_POINT_ADAPTIVE_VS_FIXED")
    if worse:
        first = worse[0]
        print(f"adaptive_first_worse_than_fixed={first[0]},delta={first[3]}")
    else:
        print("adaptive_first_worse_than_fixed=not_found_in_entry_grid")

    print("\n5_PATH_CHANGE_AUDIT_ENTRY25_MAG1")
    counts, examples = path_change_audit(bars, setups)
    print(",".join(f"{k}={v}" for k, v in counts.items()))
    print("event_id,direction,intended_entry,actual_fill,entry_slip,original_tp,original_sl,shifted_tp,shifted_sl,fixed_exit,adaptive_exit,fixed_net_R,adaptive_net_R,ATR,fixed_exit_index,adaptive_exit_index")
    for ex in examples:
        print(ex)

    print("\n6_SAME_BAR_AMBIGUITY_ADAPTIVE_ENTRY25_MAG1")
    print("policy,ambiguous_count,mean_net_R")
    for policy in ("sl_first", "tp_first", "exclude"):
        count, net = ambiguous_count(bars, setups, policy)
        print(f"{policy},{count},{net:.4f}")

    print("\n7_PERIOD_BREAKDOWN_ENTRY03_MAG1")
    print("period,n,median_atr,fixed_median_net_R,adaptive_median_net_R,tp_rate_diff,timeout_diff,win_rate_diff")
    rng = random.Random("period")
    fixed_rows: list[tuple[Setup, Outcome]] = []
    adap_rows: list[tuple[Setup, Outcome]] = []
    for setup in setups:
        slip = draw_slip(rng, 0.03, 1.0)
        f = simulate(bars, setup, slip, 0.0, "fixed", "sl_first")
        a = simulate(bars, setup, slip, 0.0, "adaptive", "sl_first")
        if f and a:
            fixed_rows.append((setup, f))
            adap_rows.append((setup, a))
    for period, _, _ in PERIODS:
        fs = [(s, o) for s, o in fixed_rows if period_label(s.year) == period]
        ads = [(s, o) for s, o in adap_rows if period_label(s.year) == period]
        fouts = [o for _, o in fs]
        aouts = [o for _, o in ads]
        print(
            f"{period},{len(fs)},{median([s.risk for s,_ in fs]):.4f},"
            f"{median([o.net_r for o in fouts]):.4f},{median([o.net_r for o in aouts]):.4f},"
            f"{metrics(aouts)['tp']-metrics(fouts)['tp']:.2%},{metrics(aouts)['to']-metrics(fouts)['to']:.2%},{metrics(aouts)['win']-metrics(fouts)['win']:.2%}"
        )

    print("\n8_RECENT_2023_2026_ENTRY_AND_FULL")
    recent = [s for s in setups if s.year >= 2023]
    print("label,fixed_median_R,adaptive_median_R,adaptive_minus_fixed,P_fixed_positive,P_adaptive_positive,fixed_win,adaptive_win,fixed_tp,adaptive_tp,fixed_sl,adaptive_sl,fixed_timeout,adaptive_timeout,fixed_maxdd,adaptive_maxdd")
    for label, ep, xp in (("recent_entry_only_p03", 0.03, 0.0), ("recent_full_p03", 0.03, 0.03), ("recent_full_p05", 0.05, 0.05)):
        pairs = []
        for i in range(MC_N):
            rng2 = random.Random(f"{SEED}-{label}-{i}")
            pairs.append(run_pair_path(bars, recent, rng2, ep, 1.0, xp, 1.0, "sl_first"))
        print(summarize_mc(label, pairs))


if __name__ == "__main__":
    main()
