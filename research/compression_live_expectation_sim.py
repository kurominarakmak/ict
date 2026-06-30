"""
Realistic live-expectation simulation for validated XAUUSD compression breakout.

Uses existing trade-level R data from hybrid_compression_squeeze_trades.csv:
- compression_only
- XAUUSD
- gate none
- corrected setup-end ATR risk
- range-edge entry
- 1.5R and 2R, 10-bar horizon

This is not an optimizer. It estimates path risk, execution-cost degradation,
and account-level compounding risk from the already validated trade series.
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


MC_N = 10_000
SEED = 20260630
TRADES_PER_YEAR = 272
START_EQUITY = 10_000.0
RISK_VARIANTS = (0.005, 0.01, 0.02)


@dataclass(frozen=True)
class Trade:
    rr: float
    entry_time: datetime
    net_r: float
    gross_r: float
    risk: float
    exit_reason: str


def parse_dt(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def load_trades(path: Path) -> dict[float, list[Trade]]:
    out: dict[float, list[Trade]] = {1.5: [], 2.0: []}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["symbol"] != "XAUUSD" or row["strategy"] != "compression_only" or row["gate"] != "none":
                continue
            rr = float(row["rr"])
            if rr not in out:
                continue
            out[rr].append(
                Trade(
                    rr=rr,
                    entry_time=parse_dt(row["entry_time"]),
                    net_r=float(row["net_r"]),
                    gross_r=float(row["gross_r"]),
                    risk=float(row["risk"]),
                    exit_reason=row["exit_reason"],
                )
            )
    return {rr: sorted(rows, key=lambda t: t.entry_time) for rr, rows in out.items()}


def ci(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


def q(vals: list[float], pct: float) -> float:
    if not vals:
        return math.nan
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def max_drawdown(path: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for r in path:
        equity += r
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def hit_total_limit(path: list[float], limit_r: float) -> bool:
    equity = 0.0
    for r in path:
        equity += r
        if equity <= -limit_r:
            return True
    return False


def daily_blocks(trades: list[Trade]) -> list[float]:
    blocks: dict[str, float] = {}
    for t in trades:
        key = t.entry_time.date().isoformat()
        blocks[key] = blocks.get(key, 0.0) + t.net_r
    return list(blocks.values())


def mc_paths(trades: list[Trade], horizon: int, mode: str, rng: random.Random) -> list[list[float]]:
    vals = [t.net_r for t in trades]
    paths = []
    for _ in range(MC_N):
        if mode == "bootstrap":
            paths.append([vals[rng.randrange(len(vals))] for _ in range(horizon)])
        elif mode == "shuffle_without_replacement":
            sample = rng.sample(vals, horizon)
            rng.shuffle(sample)
            paths.append(sample)
        else:
            raise ValueError(mode)
    return paths


def mc_summary(trades: list[Trade], rr: float, horizon_years: int, mode: str) -> list[str]:
    horizon = TRADES_PER_YEAR * horizon_years
    rng = random.Random(f"{SEED}-{rr}-{horizon_years}-{mode}")
    paths = mc_paths(trades, horizon, mode, rng)
    returns = [sum(p) for p in paths]
    dds = [max_drawdown(p) for p in paths]
    day_rng = random.Random(f"{SEED}-{rr}-{horizon_years}-{mode}-days")
    days = daily_blocks(trades)
    trading_days = 252 * horizon_years
    daily_limit_hits = 0
    for _ in range(MC_N):
        sample = [days[day_rng.randrange(len(days))] for _ in range(trading_days)]
        if min(sample) <= -5.0:
            daily_limit_hits += 1
    lines = []
    lines.append(
        f"{rr:.1f},{horizon_years}y,{mode},{q(returns,0.05):.2f},{q(returns,0.25):.2f},{median(returns):.2f},"
        f"{q(returns,0.75):.2f},{q(returns,0.95):.2f},{q(dds,0.50):.2f},{q(dds,0.05):.2f},"
        f"{sum(d <= -10 for d in dds)/MC_N:.2%},{sum(d <= -20 for d in dds)/MC_N:.2%},{sum(d <= -30 for d in dds)/MC_N:.2%},"
        f"{sum(r < 0 for r in returns)/MC_N:.2%},{sum(hit_total_limit(p,10.0) for p in paths)/MC_N:.2%},{daily_limit_hits/MC_N:.2%}"
    )
    return lines


def degraded_net(trades: list[Trade], spread: float, high_spread_frac: float = 0.0, entry_slip_r: float = 0.0, stop_slip_r: float = 0.0, commission: float = 0.0, seed: str = "") -> list[float]:
    rng = random.Random(seed)
    high_spread_ids = set(rng.sample(range(len(trades)), int(round(len(trades) * high_spread_frac)))) if high_spread_frac > 0 else set()
    vals = []
    for i, t in enumerate(trades):
        trade_spread = 0.40 if i in high_spread_ids else spread
        cost_r = trade_spread / t.risk + commission / t.risk + entry_slip_r
        if t.exit_reason == "stop":
            cost_r += stop_slip_r
        vals.append(t.gross_r - cost_r)
    return vals


def cost_ladder(trades: list[Trade], rr: float) -> list[str]:
    scenarios = [
        ("base_spread_020", dict(spread=0.20)),
        ("variable_15pct_at_040", dict(spread=0.20, high_spread_frac=0.15)),
        ("plus_mild_slip_0p10_entry_0p10_stop", dict(spread=0.20, high_spread_frac=0.15, entry_slip_r=0.10, stop_slip_r=0.10)),
        ("plus_medium_slip_0p20_entry_0p20_stop", dict(spread=0.20, high_spread_frac=0.15, entry_slip_r=0.20, stop_slip_r=0.20)),
        ("plus_severe_slip_0p30_entry_0p30_stop", dict(spread=0.20, high_spread_frac=0.15, entry_slip_r=0.30, stop_slip_r=0.30)),
        ("medium_slip_plus_commission_005", dict(spread=0.20, high_spread_frac=0.15, entry_slip_r=0.20, stop_slip_r=0.20, commission=0.05)),
    ]
    lines = []
    for name, kwargs in scenarios:
        vals = degraded_net(trades, seed=f"{SEED}-{rr}-{name}", **kwargs)
        m, lo, hi = ci(vals)
        lines.append(f"{rr:.1f},{name},{m:.4f},{lo:.4f},{hi:.4f},{sum(v > 0 for v in vals)/len(vals):.2%},{sum(vals):.2f}")
    return lines


def compounded_curve(trades: list[Trade], risk_pct: float) -> dict[str, float]:
    equity = START_EQUITY
    noncomp = START_EQUITY
    peak = equity
    max_dd = 0.0
    dd_start = None
    longest_recovery = 0
    current_dd_len = 0
    for i, t in enumerate(trades):
        noncomp += START_EQUITY * risk_pct * t.net_r
        equity *= 1.0 + risk_pct * t.net_r
        if equity >= peak:
            peak = equity
            if dd_start is not None:
                longest_recovery = max(longest_recovery, current_dd_len)
            dd_start = None
            current_dd_len = 0
        else:
            if dd_start is None:
                dd_start = i
                current_dd_len = 0
            current_dd_len += 1
            max_dd = min(max_dd, equity / peak - 1.0)
    longest_recovery = max(longest_recovery, current_dd_len)
    return {
        "risk_pct": risk_pct,
        "final_equity": equity,
        "noncomp_final": noncomp,
        "total_return_pct": equity / START_EQUITY - 1.0,
        "noncomp_return_pct": noncomp / START_EQUITY - 1.0,
        "max_dd_pct": max_dd,
        "longest_recovery_trades": longest_recovery,
    }


def mc_dd_by_risk(trades: list[Trade], rr: float, risk_pct: float) -> dict[str, float]:
    rng = random.Random(f"{SEED}-{rr}-{risk_pct}-risk")
    paths = mc_paths(trades, TRADES_PER_YEAR, "bootstrap", rng)
    dd_pct = [max_drawdown(p) * risk_pct for p in paths]
    return {
        "median_dd": q(dd_pct, 0.50),
        "p95_worst_dd": q(dd_pct, 0.05),
        "prob_breach_6": sum(d <= -0.06 for d in dd_pct) / MC_N,
        "prob_breach_10": sum(d <= -0.10 for d in dd_pct) / MC_N,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=Path("research/hybrid_compression_squeeze_trades.csv"))
    args = parser.parse_args()
    trades_by_rr = load_trades(args.trades)

    print("REALISTIC_COMPRESSION_LIVE_EXPECTATION")
    print("symbol=XAUUSD")
    print("strategy=validated compression only; setup-end ATR; range-edge entry; 10-bar segment-aware exit")
    print(f"mc_runs={MC_N},trades_per_year={TRADES_PER_YEAR},account={START_EQUITY:.0f}")
    print("percent_return_at_1pct_risk = R_return * 1%")

    print("\nPART_1_MONTE_CARLO_PATH_RISK")
    print("rr,horizon,mode,return_p05_R,return_p25_R,return_median_R,return_p75_R,return_p95_R,maxdd_median_R,maxdd_p95_worst_R,prob_dd_gt_10pct,prob_dd_gt_20pct,prob_dd_gt_30pct,prob_negative,prob_total_10pct_ruin,prob_daily_5pct_ruin")
    for rr, trades in trades_by_rr.items():
        for years in (1, 3):
            for mode in ("bootstrap", "shuffle_without_replacement"):
                for line in mc_summary(trades, rr, years, mode):
                    print(line)

    print("\nPART_2_COST_DEGRADATION")
    print("rr,scenario,net_r,ci_low,ci_high,win_rate,total_r")
    for rr, trades in trades_by_rr.items():
        for line in cost_ladder(trades, rr):
            print(line)

    print("\nPART_3_ACCOUNT_EQUITY_REALIZED_SEQUENCE")
    print("rr,risk_pct,final_equity,compounded_return,noncomp_final,noncomp_return,max_drawdown,longest_recovery_trades,mc_median_dd,mc_p95_worst_dd,mc_prob_breach_6pct,mc_prob_breach_10pct")
    for rr, trades in trades_by_rr.items():
        for risk in RISK_VARIANTS:
            c = compounded_curve(trades, risk)
            m = mc_dd_by_risk(trades, rr, risk)
            print(
                f"{rr:.1f},{risk:.2%},{c['final_equity']:.2f},{c['total_return_pct']:.2%},"
                f"{c['noncomp_final']:.2f},{c['noncomp_return_pct']:.2%},{c['max_dd_pct']:.2%},"
                f"{int(c['longest_recovery_trades'])},{m['median_dd']:.2%},{m['p95_worst_dd']:.2%},"
                f"{m['prob_breach_6']:.2%},{m['prob_breach_10']:.2%}"
            )

    print("\nFRAMING")
    print("These are still in-sample simulations from historical trades. Live/demo can be worse due to missed fills, downtime, regime shift, wider spreads, slippage, and execution errors.")
    print("For prop-firm use, the 95th-percentile drawdown and breach probabilities matter more than median return.")


if __name__ == "__main__":
    main()
