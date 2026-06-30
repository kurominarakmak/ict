"""
Regime decomposition for XAUUSD compression breakout-following.

Uses the live-matched/corrected rules:
- compression definition from volatility_compression_breakout_audit.py
- entry at broken range edge
- risk = ATR14 at compression end, not breakout bar
- exits: 1.5R and 2.0R, 10-bar horizon
- spread = $0.20/oz

Reports year, ATR-regime terciles, and crisis-window contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


SPREAD = 0.20
HORIZON = 10
RR_VARIANTS = ("rr_1_5", "rr_2")

CRISIS_YEARS = {
    2020: "COVID shock / gold spike",
    2022: "Russia-Ukraine war / inflation shock",
    2025: "gold blowoff/crash regime",
}

CRISIS_WINDOWS = [
    ("covid_2020", datetime(2020, 2, 20, tzinfo=timezone.utc), datetime(2020, 8, 31, 23, 59, tzinfo=timezone.utc)),
    ("war_2022", datetime(2022, 2, 1, tzinfo=timezone.utc), datetime(2022, 5, 31, 23, 59, tzinfo=timezone.utc)),
    ("gold_blowoff_2025", datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)),
]


@dataclass(frozen=True)
class RegimeTrade:
    rr: str
    entry_time: datetime
    net_r: float
    win: bool
    atr_env: float
    atr_regime: str
    crisis_window: str | None


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def percentile(vals: list[float], q: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def trailing_atr_env(bars: list[base.DeltaBar], i: int) -> float | None:
    vals: list[float] = []
    j = i - 1
    while j >= 0 and len(vals) < base.ATR_TRAIL:
        if bars[j].atr14 is not None:
            vals.append(bars[j].atr14)
        j -= 1
    if len(vals) < base.ATR_TRAIL:
        return None
    return mean(vals)


def crisis_window_for(ts: datetime) -> str | None:
    for name, start, end in CRISIS_WINDOWS:
        if start <= ts <= end:
            return name
    return None


def simulate_trade(bars: list[base.DeltaBar], event: simple.BreakoutEvent, rr_name: str) -> tuple[float, bool] | None:
    risk = bars[event.setup_end].atr14
    if risk is None or risk <= 0:
        return None
    rr = {"rr_1_5": 1.5, "rr_2": 2.0}[rr_name]
    direction = event.breakout_direction
    entry = event.range_high if direction == 1 else event.range_low
    stop = entry - direction * risk
    target = entry + direction * rr * risk
    end_index = simple.segment_end_index(bars, event.breakout_index + 1, HORIZON)
    gross = 0.0
    for i in range(event.breakout_index + 1, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross = direction * (fill - entry) / risk
            break
        if target_hit:
            gross = rr
            break
    else:
        gross = direction * (bars[end_index].close - entry) / risk
    net = gross - SPREAD / risk
    return net, net > 0


def build_trades(bars: list[base.DeltaBar]) -> list[RegimeTrade]:
    events = simple.detect_compression_breakouts(bars)
    envs = [(event, trailing_atr_env(bars, event.setup_end)) for event in events]
    clean_envs = [env for _, env in envs if env is not None]
    low_cut = percentile(clean_envs, 1 / 3)
    high_cut = percentile(clean_envs, 2 / 3)

    def regime(env: float) -> str:
        if env <= low_cut:
            return "low_vol"
        if env <= high_cut:
            return "mid_vol"
        return "high_vol"

    out: list[RegimeTrade] = []
    for event, env in envs:
        if env is None:
            continue
        entry_time = bars[event.breakout_index].start
        for rr in RR_VARIANTS:
            result = simulate_trade(bars, event, rr)
            if result is None:
                continue
            net, win = result
            out.append(RegimeTrade(rr, entry_time, net, win, env, regime(env), crisis_window_for(entry_time)))
    return out


def summarize(rows: list[RegimeTrade]) -> tuple[int, float, float, float, float, float]:
    vals = [r.net_r for r in rows]
    n, m, lo, hi = ci(vals)
    win = sum(r.win for r in rows) / n if n else math.nan
    total = sum(vals) if vals else 0.0
    return n, m, lo, hi, win, total


def print_year(rows: list[RegimeTrade]) -> None:
    print("\nBY_YEAR")
    print("rr,year,regime_flag,n,net_mean,ci_low,ci_high,win_rate,total_r")
    for rr in RR_VARIANTS:
        for year in range(2016, 2027):
            subset = [r for r in rows if r.rr == rr and r.entry_time.year == year]
            n, m, lo, hi, win, total = summarize(subset)
            flag = CRISIS_YEARS.get(year, "normal/other")
            print(f"{rr},{year},{flag},{n},{m:.6f},{lo:.6f},{hi:.6f},{win:.2%},{total:.6f}")


def print_vol_regime(rows: list[RegimeTrade]) -> None:
    print("\nBY_ATR_REGIME_TERCILE")
    print("rr,atr_regime,n,net_mean,ci_low,ci_high,win_rate,total_r,avg_atr_env")
    for rr in RR_VARIANTS:
        for regime in ("low_vol", "mid_vol", "high_vol"):
            subset = [r for r in rows if r.rr == rr and r.atr_regime == regime]
            n, m, lo, hi, win, total = summarize(subset)
            avg_env = mean([r.atr_env for r in subset]) if subset else math.nan
            print(f"{rr},{regime},{n},{m:.6f},{lo:.6f},{hi:.6f},{win:.2%},{total:.6f},{avg_env:.6f}")


def print_concentration(rows: list[RegimeTrade]) -> None:
    print("\nCRISIS_WINDOW_CONCENTRATION")
    print("rr,window,n,net_mean,ci_low,ci_high,total_r,pct_of_total_r")
    for rr in RR_VARIANTS:
        rr_rows = [r for r in rows if r.rr == rr]
        total_all = sum(r.net_r for r in rr_rows)
        for name, _, _ in CRISIS_WINDOWS:
            subset = [r for r in rr_rows if r.crisis_window == name]
            n, m, lo, hi, _, total = summarize(subset)
            pct = total / total_all if total_all else math.nan
            print(f"{rr},{name},{n},{m:.6f},{lo:.6f},{hi:.6f},{total:.6f},{pct:.2%}")
        outside = [r for r in rr_rows if r.crisis_window is None]
        n, m, lo, hi, _, total = summarize(outside)
        pct = total / total_all if total_all else math.nan
        print(f"{rr},outside_crisis_windows,{n},{m:.6f},{lo:.6f},{hi:.6f},{total:.6f},{pct:.2%}")


def print_equity_by_year(rows: list[RegimeTrade]) -> None:
    print("\nYEARLY_TOTAL_R")
    print("rr,year,total_r,cumulative_r")
    for rr in RR_VARIANTS:
        cum = 0.0
        for year in range(2016, 2027):
            total = sum(r.net_r for r in rows if r.rr == rr and r.entry_time.year == year)
            cum += total
            print(f"{rr},{year},{total:.6f},{cum:.6f}")


def main() -> None:
    bars = simple.load_symbol_bars("XAUUSD", Path("data/2026.6.15XAUUSD-TICK-No Session.csv"), Path("data/xauusd_m15_delta_bars.csv"))
    rows = build_trades(bars)
    print("REGIME_DECOMPOSITION_CONTEXT")
    print("symbol=XAUUSD")
    print("rules=entry range edge; risk compression-end ATR; exits 1.5R/2R; horizon 10 bars; segment-gap close; spread $0.20")
    print(f"trades_per_rr={len(rows) // len(RR_VARIANTS)}")
    print_year(rows)
    print_vol_regime(rows)
    print_concentration(rows)
    print_equity_by_year(rows)


if __name__ == "__main__":
    main()
