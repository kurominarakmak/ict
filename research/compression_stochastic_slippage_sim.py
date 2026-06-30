"""
Stochastic slippage simulation for the validated XAUUSD compression strategy.

Uses the existing trade-level compression-only XAUUSD series from
hybrid_compression_squeeze_trades.csv. This is research-only and does not
change the trading bot.

Key modeling distinction:
- Entry slippage with adaptive SL/TP preserves planned R:R, so it should not
  materially change net R.
- Exit slippage is adverse fill slippage at TP/SL/force close and directly
  degrades realized R.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev


MC_N = 1000
SEED = 20260630
BACKTEST_SPREAD = 0.20


@dataclass(frozen=True)
class Trade:
    rr: float
    entry_time: datetime
    gross_r: float
    net_r: float
    risk: float
    spread_r: float
    exit_reason: str


def parse_dt(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def load_trades(path: Path, rr: float = 1.5) -> list[Trade]:
    out = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["symbol"] != "XAUUSD" or row["strategy"] != "compression_only" or row["gate"] != "none":
                continue
            if abs(float(row["rr"]) - rr) > 1e-9:
                continue
            out.append(
                Trade(
                    rr=float(row["rr"]),
                    entry_time=parse_dt(row["entry_time"]),
                    gross_r=float(row["gross_r"]),
                    net_r=float(row["net_r"]),
                    risk=float(row["risk"]),
                    spread_r=float(row["spread_r"]),
                    exit_reason=row["exit_reason"],
                )
            )
    return sorted(out, key=lambda t: t.entry_time)


def q(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def pct_summary(vals: list[float]) -> dict[str, float]:
    return {
        "p05": q(vals, 0.05),
        "p25": q(vals, 0.25),
        "median": median(vals),
        "p75": q(vals, 0.75),
        "p95": q(vals, 0.95),
        "p_positive": sum(v > 0 for v in vals) / len(vals),
    }


def normal_ci(vals: list[float]) -> tuple[float, float, float]:
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


def max_drawdown(vals: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def draw_observed_shape(rng: random.Random, spike_prob: float = 0.25, spike_mean: float = 1.0) -> float:
    if rng.random() >= spike_prob:
        return rng.uniform(0.0, 0.05)
    # Lognormal centered around spike_mean, clipped to avoid one impossible
    # draw dominating the whole experiment.
    sigma = 0.45
    mu = math.log(max(spike_mean, 1e-9)) - 0.5 * sigma * sigma
    return min(rng.lognormvariate(mu, sigma), spike_mean * 4.0)


def simulate_exit_slippage(
    trades: list[Trade],
    rng: random.Random,
    spike_prob: float = 0.25,
    spike_mean: float = 1.0,
    volatility_correlated: bool = False,
    variable_spread: bool = False,
) -> list[float]:
    risks = [t.risk for t in trades]
    lo = q(risks, 0.10)
    hi = q(risks, 0.90)
    vals = []
    for t in trades:
        if volatility_correlated:
            vol_rank = min(1.0, max(0.0, (t.risk - lo) / (hi - lo if hi > lo else 1.0)))
            p = min(0.80, 0.05 + 0.40 * vol_rank)
            mag = spike_mean * (0.50 + 1.50 * vol_rank)
            slip = draw_observed_shape(rng, p, mag)
        else:
            slip = draw_observed_shape(rng, spike_prob, spike_mean)
        spread = 0.40 if variable_spread and rng.random() < 0.15 else BACKTEST_SPREAD
        vals.append(t.gross_r - (spread / t.risk) - (slip / t.risk))
    return vals


def run_model_distribution(label: str, trades: list[Trade], fn) -> str:
    means = []
    total_rs = []
    dds = []
    for i in range(MC_N):
        rng = random.Random(f"{SEED}-{label}-{i}")
        vals = fn(rng)
        means.append(mean(vals))
        total_rs.append(sum(vals))
        dds.append(max_drawdown(vals))
    s = pct_summary(means)
    d = pct_summary(dds)
    return (
        f"{label},{s['p05']:.4f},{s['p25']:.4f},{s['median']:.4f},{s['p75']:.4f},{s['p95']:.4f},"
        f"{s['p_positive']:.2%},{median(total_rs):.2f},{d['median']:.2f},{q(dds,0.05):.2f}"
    )


def stress_grid(trades: list[Trade]) -> list[str]:
    lines = []
    probs = [0.10, 0.25, 0.50, 0.75]
    mags = [0.30, 0.50, 1.00, 2.00]
    for p in probs:
        for mag in mags:
            means = []
            for i in range(MC_N):
                rng = random.Random(f"{SEED}-stress-{p}-{mag}-{i}")
                vals = simulate_exit_slippage(trades, rng, spike_prob=p, spike_mean=mag)
                means.append(mean(vals))
            s = pct_summary(means)
            status = "alive" if s["median"] > 0 else "dead"
            lines.append(f"{p:.0%},{mag:.2f},{s['median']:.4f},{s['p05']:.4f},{s['p95']:.4f},{s['p_positive']:.2%},{status}")
    return lines


def entry_slippage_sanity(trades: list[Trade]) -> list[str]:
    adaptive = []
    fixed_wrong = []
    for i in range(MC_N):
        rng = random.Random(f"{SEED}-entry-{i}")
        adaptive_vals = []
        fixed_vals = []
        for t in trades:
            entry_slip = draw_observed_shape(rng, spike_prob=0.25, spike_mean=1.0)
            # Correct adaptive model: SL/TP are recomputed from actual fill, so
            # R multiple is unchanged except ordinary spread already in net_r.
            adaptive_vals.append(t.net_r)
            # Wrong fixed-order model: adverse entry slip consumes room to SL/TP.
            fixed_vals.append(t.net_r - entry_slip / t.risk)
        adaptive.append(mean(adaptive_vals))
        fixed_wrong.append(mean(fixed_vals))
    a = pct_summary(adaptive)
    f = pct_summary(fixed_wrong)
    return [
        f"adaptive_sl_tp,{a['median']:.4f},{a['p05']:.4f},{a['p95']:.4f},{a['p_positive']:.2%}",
        f"fixed_sl_tp_wrong_model,{f['median']:.4f},{f['p05']:.4f},{f['p95']:.4f},{f['p_positive']:.2%}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=Path("research/hybrid_compression_squeeze_trades.csv"))
    parser.add_argument("--rr", type=float, default=1.5)
    args = parser.parse_args()
    trades = load_trades(args.trades, args.rr)
    base_vals = [t.net_r for t in trades]
    base_mean, base_lo, base_hi = normal_ci(base_vals)

    print("COMPRESSION_STOCHASTIC_SLIPPAGE_SIM")
    print(f"symbol=XAUUSD,strategy=compression_only,rr={args.rr},trades={len(trades)},mc_runs={MC_N}")
    print("live_anchor_entry_slippage_usd=[0.03,0.0,0.0,1.0]; live_anchor_exit_slippage_usd=[0,0,0,0]")
    print(f"base_backtest_net_r={base_mean:.4f},ci_low={base_lo:.4f},ci_high={base_hi:.4f}")
    print("columns=model,net_p05,net_p25,net_median,net_p75,net_p95,p_positive,median_total_R,median_maxdd_R,p95_worst_maxdd_R")

    print("\nMODEL_1_REALISTIC_INDEPENDENT_EXIT_SLIPPAGE")
    print(run_model_distribution("realistic_independent", trades, lambda rng: simulate_exit_slippage(trades, rng, 0.25, 1.0)))

    print("\nMODEL_2_VOLATILITY_CORRELATED_EXIT_SLIPPAGE")
    print(run_model_distribution("volatility_correlated", trades, lambda rng: simulate_exit_slippage(trades, rng, 0.25, 1.0, True)))

    print("\nMODEL_3_STRESS_GRID")
    print("spike_probability,spike_mean_usd,median_net_r,p05_net_r,p95_net_r,p_positive,status")
    for line in stress_grid(trades):
        print(line)

    print("\nMODEL_4_ENTRY_SLIPPAGE_SANITY")
    print("entry_model,median_net_r,p05_net_r,p95_net_r,p_positive")
    for line in entry_slippage_sanity(trades):
        print(line)

    print("\nMODEL_5_FULL_COMBINED_REALISTIC")
    print(run_model_distribution("combined_realistic", trades, lambda rng: simulate_exit_slippage(trades, rng, 0.25, 1.0, True, True)))

    print("\nVERDICT_INPUTS")
    print("entry_slippage_adaptive_sl_tp_preserves_R=true")
    print("exit_slippage_degrades_R_by_slippage_usd_per_trade_ATR=true")
    print("variable_spread_model=15pct_trades_at_0.40_otherwise_0.20")


if __name__ == "__main__":
    main()
