"""
Audit-grade stochastic slippage simulation for XAUUSD compression breakout.

Research-only. This does not touch the demo/live bot.

The live bot places pending stops with absolute SL/TP computed from the
intended range-edge order price. Therefore the primary execution model here is
fixed absolute SL/TP:
    intended entry -> original SL/TP -> adverse entry fill -> realized R from
    actual fill to actual exit.

Adaptive SL/TP after actual fill is reported separately as an execution-mode
test, not as the live-like default.
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


MC_N = 1000
SEED = 20260701
SPREAD_BASE = 0.20
HORIZON = 10
SPIKE_PROBS = (0.00, 0.01, 0.03, 0.05, 0.10, 0.25)
PERIODS = (
    ("2016_2019", 2016, 2019),
    ("2020_2022", 2020, 2022),
    ("2023_2026", 2023, 2026),
)


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
    gross_r_before_slippage: float
    entry_slippage_r: float
    exit_slippage_r: float
    spread_r: float
    final_net_r: float
    exit_reason: str
    exit_index: int


def quantile(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def pct_summary(vals: list[float]) -> dict[str, float]:
    return {
        "p05": quantile(vals, 0.05),
        "p25": quantile(vals, 0.25),
        "median": median(vals),
        "p75": quantile(vals, 0.75),
        "p95": quantile(vals, 0.95),
        "mean": mean(vals),
        "sd": pstdev(vals) if len(vals) > 1 else 0.0,
        "p_positive": sum(v > 0 for v in vals) / len(vals),
    }


def max_drawdown(vals: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def period_name(year: int) -> str:
    for name, start, end in PERIODS:
        if start <= year <= end:
            return name
    return "other"


def draw_lognormal_spike(rng: random.Random, mean_value: float) -> float:
    sigma = 0.45
    mu = math.log(max(mean_value, 1e-12)) - 0.5 * sigma * sigma
    return min(rng.lognormvariate(mu, sigma), mean_value * 4.0)


def draw_usd_slip(rng: random.Random, spike_prob: float, spike_mean_usd: float, near_zero_max: float = 0.05) -> float:
    if spike_prob <= 0 or rng.random() >= spike_prob:
        return rng.uniform(0.0, near_zero_max)
    return draw_lognormal_spike(rng, spike_mean_usd)


def draw_r_slip(rng: random.Random, spike_prob: float, spike_mean_r: float, near_zero_max_r: float) -> float:
    if spike_prob <= 0 or rng.random() >= spike_prob:
        return rng.uniform(0.0, near_zero_max_r)
    return draw_lognormal_spike(rng, spike_mean_r)


def build_setups(bars: list[base.DeltaBar], rr: float) -> list[Setup]:
    setups: list[Setup] = []
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
        setups.append(
            Setup(
                event_id=event.event_id,
                entry_index=entry_index,
                eval_start=eval_start,
                intended_entry=entry,
                original_sl=entry - direction * risk,
                original_tp=entry + direction * rr * risk,
                direction=direction,
                risk=risk,
                year=bars[entry_index].start.year,
            )
        )
    return setups


def simulate_one(
    bars: list[base.DeltaBar],
    setup: Setup,
    rr: float,
    entry_slip_usd: float,
    exit_slip_usd: float,
    spread_usd: float,
    mode: str,
    same_bar_policy: str,
) -> Outcome | None:
    direction = setup.direction
    actual_entry = setup.intended_entry + entry_slip_usd if direction == 1 else setup.intended_entry - entry_slip_usd
    if mode == "fixed":
        stop = setup.original_sl
        target = setup.original_tp
    elif mode == "adaptive":
        stop = actual_entry - direction * setup.risk
        target = actual_entry + direction * rr * setup.risk
    else:
        raise ValueError(mode)

    end_index = simple.segment_end_index(bars, setup.eval_start, HORIZON)
    exit_price = bars[end_index].close
    exit_reason = "force_close"
    exit_index = end_index
    for i in range(setup.eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit and target_hit:
            if same_bar_policy == "exclude":
                return None
            if same_bar_policy == "tp_first":
                exit_price = target
                exit_reason = "target"
            else:
                exit_price = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
                exit_reason = "stop"
            exit_index = i
            break
        if stop_hit:
            exit_price = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            exit_reason = "stop"
            exit_index = i
            break
        if target_hit:
            exit_price = target
            exit_reason = "target"
            exit_index = i
            break

    if exit_slip_usd:
        if direction == 1:
            exit_price -= exit_slip_usd
        else:
            exit_price += exit_slip_usd

    gross_r = direction * (exit_price - actual_entry) / setup.risk
    entry_slip_r = entry_slip_usd / setup.risk
    exit_slip_r = exit_slip_usd / setup.risk
    spread_r = spread_usd / setup.risk
    return Outcome(
        gross_r_before_slippage=gross_r + exit_slip_r,
        entry_slippage_r=entry_slip_r,
        exit_slippage_r=exit_slip_r,
        spread_r=spread_r,
        final_net_r=gross_r - spread_r,
        exit_reason=exit_reason,
        exit_index=exit_index,
    )


def slippage_draws(
    rng: random.Random,
    setup: Setup,
    model: str,
    entry_prob: float,
    exit_prob: float,
    recent_median_atr: float,
    period_median_atr: dict[str, float],
) -> tuple[float, float]:
    if model == "none":
        return 0.0, 0.0
    if model == "absolute":
        return draw_usd_slip(rng, entry_prob, 1.0), draw_usd_slip(rng, exit_prob, 1.0)
    if model == "atr_scaled":
        spike_mean_r = 1.0 / recent_median_atr
        near_zero_r = 0.05 / recent_median_atr
        return (
            draw_r_slip(rng, entry_prob, spike_mean_r, near_zero_r) * setup.risk,
            draw_r_slip(rng, exit_prob, spike_mean_r, near_zero_r) * setup.risk,
        )
    if model == "period_conditioned":
        scale = period_median_atr.get(period_name(setup.year), recent_median_atr) / recent_median_atr
        return (
            draw_usd_slip(rng, entry_prob, 1.0 * scale, 0.05 * scale),
            draw_usd_slip(rng, exit_prob, 1.0 * scale, 0.05 * scale),
        )
    raise ValueError(model)


def run_path(
    bars: list[base.DeltaBar],
    setups: list[Setup],
    rr: float,
    rng: random.Random,
    mode: str,
    slip_model: str,
    entry_prob: float,
    exit_prob: float,
    recent_median_atr: float,
    period_median_atr: dict[str, float],
    same_bar_policy: str,
) -> list[tuple[Setup, Outcome]]:
    out: list[tuple[Setup, Outcome]] = []
    for setup in setups:
        entry_slip, exit_slip = slippage_draws(rng, setup, slip_model, entry_prob, exit_prob, recent_median_atr, period_median_atr)
        outcome = simulate_one(bars, setup, rr, entry_slip, exit_slip, SPREAD_BASE, mode, same_bar_policy)
        if outcome is not None:
            out.append((setup, outcome))
    return out


def run_mc(
    bars: list[base.DeltaBar],
    setups: list[Setup],
    rr: float,
    label: str,
    mode: str,
    slip_model: str,
    entry_prob: float,
    exit_prob: float,
    recent_median_atr: float,
    period_median_atr: dict[str, float],
    same_bar_policy: str,
) -> list[list[tuple[Setup, Outcome]]]:
    paths = []
    for i in range(MC_N):
        rng = random.Random(f"{SEED}-{label}-{i}")
        paths.append(
            run_path(
                bars,
                setups,
                rr,
                rng,
                mode,
                slip_model,
                entry_prob,
                exit_prob,
                recent_median_atr,
                period_median_atr,
                same_bar_policy,
            )
        )
    return paths


def summarize_paths(label: str, paths: list[list[tuple[Setup, Outcome]]]) -> str:
    means = [mean(o.final_net_r for _, o in p) for p in paths if p]
    dds = [max_drawdown([o.final_net_r for _, o in p]) for p in paths if p]
    wins = [sum(o.final_net_r > 0 for _, o in p) / len(p) for p in paths if p]
    tps = [sum(o.exit_reason == "target" for _, o in p) / len(p) for p in paths if p]
    sls = [sum(o.exit_reason == "stop" for _, o in p) / len(p) for p in paths if p]
    tos = [sum(o.exit_reason == "force_close" for _, o in p) / len(p) for p in paths if p]
    s = pct_summary(means)
    return (
        f"{label},{s['p05']:.4f},{s['p25']:.4f},{s['median']:.4f},{s['p75']:.4f},{s['p95']:.4f},"
        f"{s['mean']:.4f},{s['sd']:.4f},{s['p_positive']:.2%},{median(dds):.2f},{quantile(dds,0.05):.2f},"
        f"{median(wins):.2%},{median(tps):.2%},{median(sls):.2%},{median(tos):.2%}"
    )


def point_summary(rows: list[tuple[Setup, Outcome]]) -> dict[str, float]:
    vals = [o.final_net_r for _, o in rows]
    return {
        "n": len(rows),
        "net": mean(vals) if vals else math.nan,
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "tp": sum(o.exit_reason == "target" for _, o in rows) / len(rows) if rows else math.nan,
        "sl": sum(o.exit_reason == "stop" for _, o in rows) / len(rows) if rows else math.nan,
        "timeout": sum(o.exit_reason == "force_close" for _, o in rows) / len(rows) if rows else math.nan,
        "avg_atr": mean(s.risk for s, _ in rows) if rows else math.nan,
        "avg_entry_slip_r": mean(o.entry_slippage_r for _, o in rows) if rows else math.nan,
        "avg_exit_slip_r": mean(o.exit_slippage_r for _, o in rows) if rows else math.nan,
        "median_gross_before_slip": median(o.gross_r_before_slippage for _, o in rows) if rows else math.nan,
        "median_spread_r": median(o.spread_r for _, o in rows) if rows else math.nan,
        "median_final_net_r": median(vals) if vals else math.nan,
    }


def print_year_breakdown(label: str, rows: list[tuple[Setup, Outcome]]) -> None:
    print(label)
    print("year,n,avg_atr,avg_entry_slip_R,avg_exit_slip_R,median_gross_R_before_slip,median_spread_R,median_final_net_R,mean_net_R")
    for year in sorted({s.year for s, _ in rows}):
        subset = [(s, o) for s, o in rows if s.year == year]
        ps = point_summary(subset)
        print(
            f"{year},{ps['n']},{ps['avg_atr']:.4f},{ps['avg_entry_slip_r']:.4f},{ps['avg_exit_slip_r']:.4f},"
            f"{ps['median_gross_before_slip']:.4f},{ps['median_spread_r']:.4f},{ps['median_final_net_r']:.4f},{ps['net']:.4f}"
        )


def sensitivity(
    bars: list[base.DeltaBar],
    setups: list[Setup],
    rr: float,
    mode: str,
    slip_model: str,
    recent_median_atr: float,
    period_median_atr: dict[str, float],
    same_bar_policy: str,
    varying: str,
) -> list[str]:
    lines = []
    breaking = None
    for p in SPIKE_PROBS:
        entry_prob = p if varying == "entry" else 0.0
        exit_prob = p if varying == "exit" else 0.0
        paths = run_mc(
            bars,
            setups,
            rr,
            f"sens-{mode}-{slip_model}-{varying}-{p}",
            mode,
            slip_model,
            entry_prob,
            exit_prob,
            recent_median_atr,
            period_median_atr,
            same_bar_policy,
        )
        means = [mean(o.final_net_r for _, o in path) for path in paths]
        dds = [max_drawdown([o.final_net_r for _, o in path]) for path in paths]
        s = pct_summary(means)
        status = "alive" if s["median"] > 0 else "dead"
        if breaking is None and s["median"] <= 0:
            breaking = (p, s["median"])
        lines.append(f"{varying},{p:.0%},{s['median']:.4f},{s['p05']:.4f},{s['p95']:.4f},{s['p_positive']:.2%},{median(dds):.2f},{status}")
    if breaking is None:
        lines.append(f"{varying}_breaking_point,not_found")
    else:
        lines.append(f"{varying}_breaking_point,{breaking[0]:.0%},median_net_r={breaking[1]:.4f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--rr", type=float, default=1.5)
    parser.add_argument("--same-bar-policy", choices=("sl_first", "tp_first", "exclude"), default="sl_first")
    args = parser.parse_args()

    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    setups = build_setups(bars, args.rr)
    recent_median_atr = median(s.risk for s in setups if s.year >= 2023)
    period_median_atr = {
        name: median(s.risk for s in setups if start <= s.year <= end)
        for name, start, end in PERIODS
    }

    def deterministic(mode: str, subset: list[Setup] | None = None) -> list[tuple[Setup, Outcome]]:
        rng = random.Random(f"{SEED}-det-{mode}")
        return run_path(
            bars,
            subset or setups,
            args.rr,
            rng,
            mode,
            "none",
            0.0,
            0.0,
            recent_median_atr,
            period_median_atr,
            args.same_bar_policy,
        )

    base_fixed = deterministic("fixed")
    base_adaptive = deterministic("adaptive")

    print("COMPRESSION_SLIPPAGE_TP_AUDIT")
    print(f"symbol=XAUUSD,rr={args.rr},horizon={HORIZON},mc_runs={MC_N},same_bar_policy={args.same_bar_policy}")
    print("bot_reference=live pending orders set absolute SL/TP from intended range-edge entry; bot file was read only, not edited")
    print(f"setups={len(setups)},recent_2023_2026_median_atr={recent_median_atr:.4f}")
    print("period_median_atr=" + ";".join(f"{k}:{v:.4f}" for k, v in period_median_atr.items()))

    print("\n1_ORIGINAL_EXPANSION_TP_BACKTEST")
    for label, rows in (("fixed_absolute_tp_sl_no_slippage", base_fixed), ("adaptive_rr_no_slippage", base_adaptive)):
        ps = point_summary(rows)
        print(f"{label},n={ps['n']},mean_net_R={ps['net']:.4f},win={ps['win']:.2%},tp={ps['tp']:.2%},sl={ps['sl']:.2%},timeout={ps['timeout']:.2%}")

    print("\n2_MODEL_SUMMARY_DISTRIBUTIONS")
    print("model,net_p05,net_p25,net_median,net_p75,net_p95,net_mean,net_sd,p_positive,median_maxdd_R,p95_worst_maxdd_R,win_rate,tp_rate,sl_rate,timeout_rate")
    scenarios = [
        ("fixed_abs_entry25_exit25_stress", "fixed", "absolute", 0.25, 0.25),
        ("adaptive_abs_entry25_exit25_stress", "adaptive", "absolute", 0.25, 0.25),
        ("fixed_atr_scaled_entry25_exit25", "fixed", "atr_scaled", 0.25, 0.25),
        ("fixed_period_conditioned_entry25_exit25", "fixed", "period_conditioned", 0.25, 0.25),
        ("fixed_abs_entry03_exit03_recent_like", "fixed", "absolute", 0.03, 0.03),
        ("fixed_atr_scaled_entry03_exit03_recent_like", "fixed", "atr_scaled", 0.03, 0.03),
    ]
    stored_paths: dict[str, list[list[tuple[Setup, Outcome]]]] = {}
    for label, mode, slip_model, entry_prob, exit_prob in scenarios:
        paths = run_mc(bars, setups, args.rr, label, mode, slip_model, entry_prob, exit_prob, recent_median_atr, period_median_atr, args.same_bar_policy)
        stored_paths[label] = paths
        print(summarize_paths(label, paths))

    print("\n3_YEAR_BY_YEAR_BASELINE")
    print_year_breakdown("original_fixed_no_slippage", base_fixed)

    print("\n4_YEAR_BY_YEAR_ABSOLUTE_25PCT_STRESS_SAMPLE")
    print_year_breakdown("fixed_abs_entry25_exit25_first_mc_path", stored_paths["fixed_abs_entry25_exit25_stress"][0])

    print("\n5_YEAR_BY_YEAR_ATR_SCALED_25PCT_SAMPLE")
    print_year_breakdown("fixed_atr_scaled_entry25_exit25_first_mc_path", stored_paths["fixed_atr_scaled_entry25_exit25"][0])

    print("\n6_RECENT_2023_2026_RESULT")
    recent = [s for s in setups if s.year >= 2023]
    print("model,net_p05,net_p25,net_median,net_p75,net_p95,net_mean,net_sd,p_positive,median_maxdd_R,p95_worst_maxdd_R,win_rate,tp_rate,sl_rate,timeout_rate")
    for label, slip_model, entry_prob, exit_prob in (
        ("recent_fixed_abs_entry03_exit03", "absolute", 0.03, 0.03),
        ("recent_fixed_abs_entry05_exit05", "absolute", 0.05, 0.05),
        ("recent_fixed_atr_scaled_entry03_exit03", "atr_scaled", 0.03, 0.03),
        ("recent_fixed_period_conditioned_entry03_exit03", "period_conditioned", 0.03, 0.03),
    ):
        paths = run_mc(bars, recent, args.rr, label, "fixed", slip_model, entry_prob, exit_prob, recent_median_atr, period_median_atr, args.same_bar_policy)
        print(summarize_paths(label, paths))

    print("\n7_BREAKING_POINT_SENSITIVITY_FIXED_ABSOLUTE")
    print("varying,spike_probability,median_net_r,p05_net_r,p95_net_r,p_positive,median_maxdd_R,status")
    for line in sensitivity(bars, setups, args.rr, "fixed", "absolute", recent_median_atr, period_median_atr, args.same_bar_policy, "entry"):
        print(line)
    for line in sensitivity(bars, setups, args.rr, "fixed", "absolute", recent_median_atr, period_median_atr, args.same_bar_policy, "exit"):
        print(line)

    print("\n8_AUDIT_VERDICT_INPUTS")
    print("previous_-0.9358R_used_adaptive_tp_sl_and_25pct_absolute_2026_dollar_slips_across_all_years=true")
    print("correct_live_like_primary_mode=fixed_absolute_strategy_tp_sl")
    print("absolute_dollar_slippage_is_reported_as_stress_test=true")
    print("atr_scaled_and_period_conditioned_models_reduce_old_year_over_penalty=true")


if __name__ == "__main__":
    main()
