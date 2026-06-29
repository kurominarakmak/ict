"""
Validation diagnostics for fixed-point trend-following OB, 2024-2026.

Uses the same fixed SL/TP mechanics as research/fixed_point_trial.py:
- $10/oz SL, +$10/+20/+30 TP ladder.
- Edge-price entry, tick-ordered fills, actual tick stop slippage.
- Horizon exits are force-closed and included in expectancy.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategies"))

import fixed_point_trial as fp
from order_block import compute_atr, default_tick_path, load_bars


TRAIN_END = datetime(2025, 1, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 6, 15, tzinfo=timezone.utc)


def ci(vals: list[float]) -> tuple[float, float, float, float]:
    if not vals:
        return math.nan, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals)) if vals else math.nan
    return m, m - 1.96 * se, m + 1.96 * se, sd


def max_drawdown(vals: list[float]) -> tuple[float, int]:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_loss_streak = 0
    cur_loss_streak = 0
    for value in vals:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if value < 0:
            cur_loss_streak += 1
            max_loss_streak = max(max_loss_streak, cur_loss_streak)
        else:
            cur_loss_streak = 0
    return max_dd, max_loss_streak


def print_summary(label: str, trades: list[fp.Trade]) -> None:
    vals = [trade.net_r for trade in trades]
    m, lo, hi, sd = ci(vals)
    unresolved = sum(trade.horizon_exit for trade in trades) / len(trades) if trades else math.nan
    win = sum(v > 0 for v in vals) / len(vals) if vals else math.nan
    worst = min(vals) if vals else math.nan
    dd, streak = max_drawdown(vals)
    print(f"{label},{len(vals)},{m:.4f},{lo:.4f},{hi:.4f},{win:.2%},{unresolved:.2%},{sd:.4f},{worst:.4f},{dd:.4f},{streak}")


def equity_milestones(trades: list[fp.Trade]) -> list[tuple[int, str, float]]:
    out = []
    equity = 0.0
    ordered = sorted(trades, key=lambda t: t.entry_time or datetime.min.replace(tzinfo=timezone.utc))
    n = len(ordered)
    for i, trade in enumerate(ordered, start=1):
        equity += trade.net_r
        if i in {1, 5, 10, 20, 40, 60, n}:
            ts = (trade.entry_time or datetime.min.replace(tzinfo=timezone.utc)).strftime("%Y-%m-%d")
            out.append((i, ts, equity))
    return out


def main() -> None:
    tick_path = default_tick_path()
    print("Loading gap-aware M15 bars...", flush=True)
    bars = load_bars(tick_path, gap_minutes=30.0)
    atr = compute_atr(bars)
    obs = fp.load_ob_zones(Path("research/order_block_zones.csv"))
    adx = fp.compute_adx(bars)

    setups = fp.build_trend_ob_setups(bars, obs, atr)
    baseline = fp.build_baseline(bars, setups, "trend_ob_fixed_validation", atr)
    print(f"Prepared trend_fixed setups={len(setups)}, baseline={len(baseline)}", flush=True)
    results, skips = fp.run_ticks_many(tick_path, bars, [("trend", setups), ("baseline", baseline)])
    trades = results["trend"]
    base_trades = results["baseline"]

    print("\nFORCE_CLOSE_CHECK")
    print("Horizon exits are force-closed at the horizon bar close by fixed_point_trial.force_exit and are included below.")
    print("cohort,n,net_mean_R,ci_low,ci_high,win,force_closed_rate,std,worst_R,max_drawdown_R,max_loss_streak")
    print_summary("trend_all_force_closed", trades)
    print_summary("baseline_all_force_closed", base_trades)
    print(f"gap_skips_trend={skips['trend']},gap_skips_baseline={skips['baseline']}")

    train = [t for t in trades if t.entry_time and t.entry_time < TRAIN_END]
    test = [t for t in trades if t.entry_time and TRAIN_END <= t.entry_time < TEST_END]
    print("\nTRAIN_TEST_SPLIT")
    print("period,n,net_mean_R,ci_low,ci_high,win,force_closed_rate,std,worst_R,max_drawdown_R,max_loss_streak")
    print_summary("train_2024", train)
    print_summary("test_2025_to_2026_06_15", test)

    vals = [t.net_r for t in trades]
    profits = [v for v in vals if v > 0]
    total_profit = sum(profits)
    top5 = sorted(vals, reverse=True)[:5]
    without_top5 = sorted(vals, reverse=True)[5:]
    top5_share = sum(top5) / total_profit if total_profit > 0 else math.nan
    m2, lo2, hi2, _ = ci(without_top5)
    print("\nEDGE_CONCENTRATION")
    print(f"total_trades={len(vals)},profitable_trades={len(profits)},total_profit_R={total_profit:.4f}")
    print(f"top5_net_R={','.join(f'{v:.4f}' for v in top5)}")
    print(f"top5_profit_share={top5_share:.2%}")
    print(f"without_top5,n={len(without_top5)},net_mean_R={m2:.4f},ci_low={lo2:.4f},ci_high={hi2:.4f}")
    print("equity_curve_milestones_trade,date,cumulative_R")
    for i, ts, equity in equity_milestones(trades):
        print(f"{i},{ts},{equity:.4f}")

    print("\nREGIME_SPLIT_ADX14_THRESHOLD_25")
    print("regime,n,net_mean_R,ci_low,ci_high,win,force_closed_rate,std,worst_R,max_drawdown_R,max_loss_streak")
    trend_rows = [t for t in trades if (adx[t.setup.entry_bar] or 0.0) >= fp.ADX_TREND_THRESHOLD]
    range_rows = [t for t in trades if (adx[t.setup.entry_bar] or 0.0) < fp.ADX_TREND_THRESHOLD]
    print_summary("adx_trending", trend_rows)
    print_summary("adx_ranging", range_rows)


if __name__ == "__main__":
    main()
