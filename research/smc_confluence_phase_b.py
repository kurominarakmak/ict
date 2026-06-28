"""
SMC Confluence Phase B.

Locked hypothesis:
- Start from liquidity sweeps.
- Require aligned OB or FVG displacement zone within Gate 2 window.
- Entry is first pullback touch into that zone within 20 M15 bars.
- Tick-based scale-out execution with IUX spread overlay.

No re-detection and no parameter sweep.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategies"))
sys.path.insert(0, str(ROOT / "research"))

from cost_model import IUX_SPREAD_USD_OZ, SpreadMode, session_for_timestamp
from order_block import Bar, compute_atr, default_tick_path, load_bars, parse_timestamp


N_BARS = 10
PRICE_ATR_MULTIPLE = 1.0
ENTRY_EXPIRY_BARS = 20
REACTION_BARS = 20
SCALE_PLAN = ((1.0, 0.50), (2.0, 0.25), (3.0, 0.25))


@dataclass(frozen=True)
class Signal:
    kind: str
    signal_id: int
    direction: str
    index: int
    high: float
    low: float
    atr: float
    year: int
    session: str


@dataclass(frozen=True)
class Setup:
    setup_id: int
    kind: str
    sweep: Signal
    zone: Signal
    entry_bar_index: int
    horizon_bar_index: int
    entry_atr: float
    atr_bucket: str
    excluded_news_proxy: bool


@dataclass
class Trade:
    setup: Setup
    entry_found: bool = False
    entry_time: Optional[datetime] = None
    entry_mid: Optional[float] = None
    open_weight: float = 1.0
    stop_r: float = -1.0
    gross_r: float = 0.0
    net_r: float = 0.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    be_saved: bool = False
    resolved: bool = False
    horizon_exit: bool = False


def load_obs(path: Path) -> list[Signal]:
    out = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(Signal("OB", int(row["zone_id"]), row["direction"], int(row["impulse_end_index"]), float(row["zone_high"]), float(row["zone_low"]), float(row["frozen_atr"]), int(row["year"]), row["session"]))
    return out


def load_fvgs(path: Path) -> list[Signal]:
    out = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(Signal("FVG", int(row["fvg_id"]), row["direction"], int(row["candle3_index"]), float(row["gap_high"]), float(row["gap_low"]), float(row["frozen_atr"]), int(row["year"]), row["session"]))
    return out


def load_sweeps(path: Path) -> list[Signal]:
    out = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            level = float(row["swept_swing_level"])
            out.append(Signal("Sweep", int(row["sweep_id"]), row["direction"], int(row["rejection_index"]), level, level, float(row["frozen_atr"]), int(row["year"]), row["session"]))
    return out


def range_distance(a: Signal, b: Signal) -> float:
    if a.low <= b.high and b.low <= a.high:
        return 0.0
    if a.high < b.low:
        return b.low - a.high
    return a.low - b.high


def cooccurs(anchor: Signal, other: Signal) -> bool:
    return abs(anchor.index - other.index) <= N_BARS and range_distance(anchor, other) <= PRICE_ATR_MULTIPLE * anchor.atr


def aligned_zones(sweep: Signal, zones: list[Signal]) -> list[Signal]:
    return [zone for zone in zones if zone.direction == sweep.direction and cooccurs(sweep, zone)]


def choose_zone(sweep: Signal, obs: list[Signal], fvgs: list[Signal]) -> Optional[Signal]:
    ob_matches = aligned_zones(sweep, obs)
    if ob_matches:
        return min(ob_matches, key=lambda z: (abs(z.index - sweep.index), range_distance(sweep, z), z.signal_id))
    fvg_matches = aligned_zones(sweep, fvgs)
    if fvg_matches:
        return min(fvg_matches, key=lambda z: (abs(z.index - sweep.index), range_distance(sweep, z), z.signal_id))
    return None


def first_touch(bars: list[Bar], zone: Signal, after_index: int, expiry: int) -> Optional[int]:
    segment = bars[after_index].segment_id
    end = min(len(bars) - 1, after_index + expiry)
    for i in range(after_index + 1, end + 1):
        if bars[i].segment_id != segment:
            return None
        if bars[i].low <= zone.high and bars[i].high >= zone.low:
            return i
    return None


def quantile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def atr_bucket(value: float, low_cut: float, mid_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value <= mid_cut:
        return "medium"
    return "high"


def build_setups(bars: list[Bar], atr: list[Optional[float]], sweeps: list[Signal], obs: list[Signal], fvgs: list[Signal]) -> tuple[list[Setup], Counter]:
    matched_atrs = [s.atr for s in sweeps if choose_zone(s, obs, fvgs) is not None]
    low_cut, mid_cut = quantile(matched_atrs, 1 / 3), quantile(matched_atrs, 2 / 3)
    stats = Counter()
    setups = []
    for sweep in sweeps:
        zone = choose_zone(sweep, obs, fvgs)
        if zone is None:
            stats["no_aligned_zone"] += 1
            continue
        stats[f"zone_{zone.kind}"] += 1
        established = max(sweep.index, zone.index)
        if established >= len(bars):
            stats["bad_index"] += 1
            continue
        touch = first_touch(bars, zone, established, ENTRY_EXPIRY_BARS)
        if touch is None:
            stats["expired_no_pullback"] += 1
            continue
        if touch + REACTION_BARS >= len(bars):
            stats["no_full_horizon"] += 1
            continue
        if any(bars[i].segment_id != bars[touch].segment_id for i in range(touch + 1, touch + REACTION_BARS + 1)):
            stats["horizon_gap"] += 1
            continue
        entry_atr = atr[touch]
        if entry_atr is None or entry_atr <= 0:
            stats["missing_entry_atr"] += 1
            continue
        proxy_start = max(0, touch - 2)
        proxy_end = min(len(bars) - 1, touch + 2)
        news_proxy = any(bars[i].segment_id == bars[touch].segment_id and (bars[i].high - bars[i].low) > 3.0 * entry_atr for i in range(proxy_start, proxy_end + 1))
        setups.append(Setup(len(setups) + 1, "confluence", sweep, zone, touch, touch + REACTION_BARS, entry_atr, atr_bucket(entry_atr, low_cut, mid_cut), news_proxy))
    stats["atr_low_cut_x1e6"] = round(low_cut * 1_000_000)
    stats["atr_mid_cut_x1e6"] = round(mid_cut * 1_000_000)
    return setups, stats


def build_baseline_setups(bars: list[Bar], atr: list[Optional[float]], sweeps: list[Signal], confluence_sweep_ids: set[int], template_setups: list[Setup]) -> list[Setup]:
    atrs = [s.entry_atr for s in template_setups]
    low_cut, mid_cut = quantile(atrs, 1 / 3), quantile(atrs, 2 / 3)
    pool = []
    for sweep in sweeps:
        if sweep.signal_id in confluence_sweep_ids:
            continue
        touch = sweep.index
        if touch + REACTION_BARS >= len(bars):
            continue
        if any(bars[i].segment_id != bars[touch].segment_id for i in range(touch + 1, touch + REACTION_BARS + 1)):
            continue
        entry_atr = atr[touch]
        if entry_atr is None or entry_atr <= 0:
            continue
        level_zone = Signal("BaselineSweep", sweep.signal_id, sweep.direction, sweep.index, sweep.high, sweep.low, sweep.atr, sweep.year, sweep.session)
        proxy_start = max(0, touch - 2)
        proxy_end = min(len(bars) - 1, touch + 2)
        news_proxy = any(bars[i].segment_id == bars[touch].segment_id and (bars[i].high - bars[i].low) > 3.0 * entry_atr for i in range(proxy_start, proxy_end + 1))
        pool.append(Setup(len(pool) + 1, "baseline", sweep, level_zone, touch, touch + REACTION_BARS, entry_atr, atr_bucket(entry_atr, low_cut, mid_cut), news_proxy))
    # Deterministic matched sample: for each confluence setup, nearest prior same direction/session/ATR bucket baseline.
    selected = []
    used: set[int] = set()
    for setup in sorted(template_setups, key=lambda s: s.entry_bar_index):
        candidates = [b for b in pool if b.setup_id not in used and b.entry_bar_index < setup.entry_bar_index and b.sweep.direction == setup.sweep.direction and b.sweep.session == setup.sweep.session and b.atr_bucket == setup.atr_bucket]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda b: b.entry_bar_index)
        used.add(chosen.setup_id)
        selected.append(chosen)
    return selected


def spread_r(ts: datetime, atr: float) -> float:
    return IUX_SPREAD_USD_OZ[session_for_timestamp(ts)].value(SpreadMode.MEDIAN) / atr


def touches_entry(mid: float, setup: Setup) -> bool:
    return setup.zone.low <= mid <= setup.zone.high


def r_now(mid: float, trade: Trade) -> float:
    assert trade.entry_mid is not None
    if trade.setup.sweep.direction == "bullish":
        return (mid - trade.entry_mid) / trade.setup.entry_atr
    return (trade.entry_mid - mid) / trade.setup.entry_atr


def close_weight(trade: Trade, ts: datetime, weight: float, r_value: float) -> None:
    trade.gross_r += weight * r_value
    trade.net_r += weight * r_value
    trade.net_r -= weight * spread_r(ts, trade.setup.entry_atr) / 2.0
    trade.open_weight = max(0.0, trade.open_weight - weight)
    if trade.open_weight <= 1e-12:
        trade.resolved = True


def on_tick(trade: Trade, ts: datetime, mid: float) -> None:
    if trade.resolved:
        return
    if not trade.entry_found:
        if touches_entry(mid, trade.setup):
            trade.entry_found = True
            trade.entry_time = ts
            trade.entry_mid = mid
            trade.net_r -= spread_r(ts, trade.setup.entry_atr) / 2.0
        return
    value = r_now(mid, trade)
    if value <= trade.stop_r:
        if trade.stop_r == 0:
            trade.be_saved = True
        close_weight(trade, ts, trade.open_weight, trade.stop_r)
        return
    for target, weight in SCALE_PLAN:
        if target == 1.0 and trade.hit_1r:
            continue
        if target == 2.0 and trade.hit_2r:
            continue
        if target == 3.0 and trade.hit_3r:
            continue
        if value >= target:
            close_weight(trade, ts, weight, target)
            if target == 1.0:
                trade.hit_1r = True
                trade.stop_r = 0.0
            elif target == 2.0:
                trade.hit_2r = True
            elif target == 3.0:
                trade.hit_3r = True
            if trade.resolved:
                return


def force_exit(trade: Trade, bars: list[Bar]) -> None:
    if trade.resolved or not trade.entry_found:
        return
    bar = bars[trade.setup.horizon_bar_index]
    value = r_now(bar.close, trade)
    trade.horizon_exit = True
    close_weight(trade, bar.end, trade.open_weight, value)


def run_tick_model(tick_path: Path, bars: list[Bar], setups: list[Setup]) -> list[Trade]:
    trades = [Trade(s) for s in setups if not s.excluded_news_proxy]
    trades.sort(key=lambda t: bars[t.setup.entry_bar_index].start)
    if not trades:
        return []
    pending_idx = 0
    active: list[Trade] = []
    done: list[Trade] = []
    first_needed = min(bars[t.setup.entry_bar_index].start for t in trades)
    last_needed = max(bars[t.setup.horizon_bar_index].end for t in trades)
    with tick_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ti, bi, ai = header.index("DateTime"), header.index("Bid"), header.index("Ask")
        for row in reader:
            try:
                ts = parse_timestamp(row[ti])
                bid = float(row[bi])
                ask = float(row[ai])
            except (IndexError, ValueError):
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            if ts < first_needed:
                continue
            if ts > last_needed:
                break
            mid = (bid + ask) / 2.0
            while pending_idx < len(trades) and bars[trades[pending_idx].setup.entry_bar_index].start <= ts:
                active.append(trades[pending_idx])
                pending_idx += 1
            still = []
            for trade in active:
                if ts > bars[trade.setup.horizon_bar_index].end:
                    force_exit(trade, bars)
                    if trade.entry_found:
                        done.append(trade)
                    continue
                on_tick(trade, ts, mid)
                if trade.resolved:
                    done.append(trade)
                else:
                    still.append(trade)
            active = still
    for trade in active:
        force_exit(trade, bars)
        if trade.entry_found:
            done.append(trade)
    done.sort(key=lambda t: (t.entry_time or bars[t.setup.entry_bar_index].end, t.setup.setup_id))
    return done


def summarize(trades: list[Trade]) -> dict[str, float]:
    vals = [t.net_r for t in trades]
    if not vals:
        return {"n": 0, "mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "std": math.nan, "win": math.nan, "unresolved": math.nan, "worst": math.nan, "max_loss": 0}
    std = pstdev(vals) if len(vals) > 1 else 0.0
    se = std / math.sqrt(len(vals)) if vals else math.nan
    return {
        "n": len(vals),
        "mean": mean(vals),
        "ci_low": mean(vals) - 1.96 * se,
        "ci_high": mean(vals) + 1.96 * se,
        "std": std,
        "win": sum(1 for v in vals if v > 0) / len(vals),
        "unresolved": sum(1 for t in trades if t.horizon_exit) / len(trades),
        "worst": min(vals),
        "max_loss": max_consecutive_losses(vals),
    }


def max_consecutive_losses(vals: list[float]) -> int:
    best = cur = 0
    for v in vals:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def print_stats(label: str, trades: list[Trade]) -> None:
    s = summarize(trades)
    print(f"{label},{s['n']},{s['mean']:.4f},{s['ci_low']:.4f},{s['ci_high']:.4f},{s['win']:.2%},{s['unresolved']:.2%},{s['std']:.4f},{s['max_loss']},{s['worst']:.4f}")


def print_group(title: str, trades: list[Trade], groups: list[str], key) -> None:
    print(f"\n{title}")
    print("group,n,net_mean_R,ci_low,ci_high,win_rate,unresolved,std_R,max_consec_losses,worst_R")
    for g in groups:
        print_stats(str(g), [t for t in trades if str(key(t)) == str(g)])


def survivorship(bars: list[Bar], expired: list[Setup]) -> None:
    vals = []
    for s in expired:
        sweep = s.sweep
        future = bars[sweep.index + 1 : min(len(bars), sweep.index + REACTION_BARS + 1)]
        if not future or any(b.segment_id != bars[sweep.index].segment_id for b in future):
            continue
        if sweep.direction == "bullish":
            mfe = (max(b.high for b in future) - sweep.high) / sweep.atr
            close_r = (future[-1].close - sweep.high) / sweep.atr
        else:
            mfe = (sweep.low - min(b.low for b in future)) / sweep.atr
            close_r = (sweep.low - future[-1].close) / sweep.atr
        vals.append((mfe, close_r))
    print("\nSURVIVORSHIP_EXPIRED_CONFLUENCE")
    if not vals:
        print("n=0")
    else:
        print(f"n={len(vals)},mean_MFE_R={mean([v[0] for v in vals]):.3f},median_MFE_R={median([v[0] for v in vals]):.3f},mean_20bar_close_R={mean([v[1] for v in vals]):.3f},median_20bar_close_R={median([v[1] for v in vals]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None)
    args = parser.parse_args()
    tick_path = args.ticks or default_tick_path()
    obs = load_obs(Path("research/order_block_zones.csv"))
    fvgs = load_fvgs(Path("research/fair_value_gap_fvgs.csv"))
    sweeps = load_sweeps(Path("research/liquidity_sweep_sweeps.csv"))
    print("Loading gap-aware M15 bars...", flush=True)
    bars = load_bars(tick_path, gap_minutes=30.0)
    atr = compute_atr(bars)
    setups, stats = build_setups(bars, atr, sweeps, obs, fvgs)
    confluence_sweep_ids = {s.sweep.signal_id for s in setups}
    baseline = build_baseline_setups(bars, atr, sweeps, confluence_sweep_ids, setups)
    expired = []
    for sweep in sweeps:
        zone = choose_zone(sweep, obs, fvgs)
        if zone is not None and sweep.signal_id not in confluence_sweep_ids:
            # Minimal shell setup for survivorship anchored on sweep.
            expired.append(Setup(0, "expired", sweep, zone, sweep.index, min(len(bars) - 1, sweep.index + REACTION_BARS), sweep.atr, "na", False))
    print("Running tick execution...", flush=True)
    confluence_trades = run_tick_model(tick_path, bars, setups)
    baseline_trades = run_tick_model(tick_path, bars, baseline)
    excluded = sum(1 for s in setups if s.excluded_news_proxy)

    print("\nSMC_CONFLUENCE_PHASE_B_CONTEXT")
    print(f"sweeps_total={len(sweeps)}")
    print(f"sweeps_no_aligned_zone={stats['no_aligned_zone']}")
    print(f"aligned_zone_sweeps={len(sweeps) - stats['no_aligned_zone']}")
    print(f"zone_source_OB={stats['zone_OB']},zone_source_FVG_only={stats['zone_FVG']}")
    print(f"expired_no_pullback_or_unmeasurable={len(expired)}")
    print(f"entry_set_full_window={len(setups)}")
    print(f"news_proxy_excluded={excluded}")
    print(f"tick_measured_confluence_trades={len(confluence_trades)}")
    print(f"matched_baseline_trades={len(baseline_trades)}")
    print("rule: sweep + same-direction OB/FVG within <=10 bars and <=1.0*sweep_ATR price distance; OB bounds preferred if both exist")
    print("news_proxy: excludes entries within +/-30 min of any M15 bar range > 3*entry_ATR; no calendar wired")

    print("\nOVERALL_NET_EXPECTANCY")
    print("cohort,n,net_mean_R,ci_low,ci_high,win_rate,unresolved,std_R,max_consec_losses,worst_R")
    print_stats("confluence", confluence_trades)
    print_stats("matched_baseline", baseline_trades)

    print_group("CONFLUENCE_BY_ATR_REGIME", confluence_trades, ["low", "medium", "high"], lambda t: t.setup.atr_bucket)
    print_group("CONFLUENCE_BY_YEAR", confluence_trades, [str(y) for y in range(2016, 2027)], lambda t: t.entry_time.year if t.entry_time else bars[t.setup.entry_bar_index].end.year)

    survivorship(bars, expired)

    print("\nPRIOR_COMPARISON")
    print("bare_OB_edge=+1.46pp over matched baseline, 95% CI [-0.11,+3.03] percentage points; not significant")
    print("breaker_v2_mean_net_R=-0.1069 after spread")
    print(f"confluence_mean_net_R={summarize(confluence_trades)['mean']:.4f}, 95% CI [{summarize(confluence_trades)['ci_low']:.4f},{summarize(confluence_trades)['ci_high']:.4f}]")
    if summarize(confluence_trades)["ci_low"] <= 0 <= summarize(confluence_trades)["ci_high"]:
        print("CONCLUSION: CI crosses zero; edge is not demonstrated.")
    elif summarize(confluence_trades)["mean"] <= 0:
        print("CONCLUSION: net expectancy is negative; FAIL.")
    else:
        print("CONCLUSION: net expectancy is positive with CI above zero in-sample; requires separate out-of-sample validation.")


if __name__ == "__main__":
    main()
