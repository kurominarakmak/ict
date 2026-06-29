"""
Cross-asset validation: XAGUSD trend-following OB with fixed ~1 ATR SL/TP.

Pre-registered cross-asset test:
- Detect OBs on silver with the same locked M15 rules.
- Use FVG-present OBs only, trend filter, first retest within 33 bars.
- Choose fixed silver SL as the round point closest to 2024+ median M15 ATR.
- TP ladder: +1R 50% and BE, +2R 25%, +3R 25%.
- Two declared variants: fixed ladder horizon exit, and swing-trailing stop after TP1.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import trend_following_ob as tfo


START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2025, 1, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 6, 29, tzinfo=timezone.utc)
REACTION_BARS = 20
RETEST_BARS = 33
SCALE_PLAN = ((1.0, 0.50), (2.0, 0.25), (3.0, 0.25))
SILVER_SPREAD_NORMAL = 0.02
SILVER_SPREAD_STRESS = 0.04


@dataclass(frozen=True)
class Setup:
    setup_id: int
    cohort: str
    ob: tfo.OB
    direction: str
    entry_bar: int
    horizon_bar: int
    atr_bucket: str
    session: str
    news_proxy: bool


@dataclass
class Trade:
    setup: Setup
    variant: str
    entry_found: bool = False
    gap_skipped: bool = False
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    gross_r: float = 0.0
    net_r: float = 0.0
    net_stress_r: float = 0.0
    open_weight: float = 1.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    resolved: bool = False
    horizon_exit: bool = False
    next_bar_update: int = 0


def find_xag_path() -> Path:
    matches = sorted(Path("data").glob("*XAGUSD*.csv"))
    if not matches:
        raise SystemExit("No XAGUSD tick CSV found under data/")
    return matches[-1]


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


def round_silver_stop(median_atr: float) -> float:
    candidates = [0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    return min(candidates, key=lambda x: abs(x - median_atr))


def atr_bucket(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value <= hi:
        return "medium"
    return "high"


def entry_edge(ob: tfo.OB) -> float:
    return ob.high if ob.direction == "bullish" else ob.low


def gap_skip(mid: float, ob: tfo.OB) -> bool:
    return mid <= ob.low if ob.direction == "bullish" else mid >= ob.high


def touched(mid: float, ob: tfo.OB) -> bool:
    return mid <= ob.high if ob.direction == "bullish" else mid >= ob.low


def r_now(price: float, trade: Trade, stop_size: float) -> float:
    assert trade.entry_price is not None
    return (price - trade.entry_price) / stop_size if trade.setup.direction == "bullish" else (trade.entry_price - price) / stop_size


def spread_r(spread: float, stop_size: float) -> float:
    return spread / stop_size


def half_cost(stop_size: float, stress: bool = False) -> float:
    return spread_r(SILVER_SPREAD_STRESS if stress else SILVER_SPREAD_NORMAL, stop_size) / 2.0


def has_full_horizon(bars: list[tfo.Bar], i: int) -> bool:
    if i + REACTION_BARS >= len(bars):
        return False
    return all(bars[j].segment_id == bars[i].segment_id for j in range(i + 1, i + REACTION_BARS + 1))


def news_proxy(bars: list[tfo.Bar], i: int, atr_value: float) -> bool:
    return any(
        bars[j].segment_id == bars[i].segment_id and (bars[j].high - bars[j].low) > 3.0 * atr_value
        for j in range(max(0, i - 2), min(len(bars) - 1, i + 2) + 1)
    )


def build_silver_setups(bars: list[tfo.Bar], atr: list[Optional[float]], obs: list[tfo.OB], breaks: dict[int, str], avail: list[list[tfo.Swing]]) -> list[Setup]:
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    lo, hi = quantile(recent_atrs, 1 / 3), quantile(recent_atrs, 2 / 3)
    out: list[Setup] = []
    for ob in obs:
        if not ob.fvg_present or bars[ob.creation_index].end < START_DATE:
            continue
        if tfo.trend_at([s for s in avail[ob.creation_index] if s.confirmed_at <= ob.creation_index], {k: v for k, v in breaks.items() if k <= ob.creation_index}) != ob.direction:
            continue
        touch = None
        for i in range(ob.creation_index + 1, min(len(bars), ob.creation_index + RETEST_BARS + 1)):
            if bars[i].segment_id != bars[ob.creation_index].segment_id:
                break
            if bars[i].low <= ob.high and bars[i].high >= ob.low:
                touch = i
                break
        if touch is None or bars[touch].end < START_DATE or not has_full_horizon(bars, touch):
            continue
        a = atr[touch] or ob.atr
        out.append(
            Setup(
                len(out) + 1,
                "silver_trend_ob",
                ob,
                ob.direction,
                touch,
                touch + REACTION_BARS,
                atr_bucket(a, lo, hi),
                tfo.classify_session(bars[touch].end),
                news_proxy(bars, touch, a),
            )
        )
    return out


def build_baseline(bars: list[tfo.Bar], atr: list[Optional[float]], template: list[Setup]) -> list[Setup]:
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    lo, hi = quantile(recent_atrs, 1 / 3), quantile(recent_atrs, 2 / 3)
    pool: list[Setup] = []
    dummy_id = 1
    for i, bar in enumerate(bars):
        if bar.end < START_DATE or not has_full_horizon(bars, i):
            continue
        a = atr[i]
        if a is None:
            continue
        direction = "bullish" if i % 2 == 0 else "bearish"
        high, low = bar.high, bar.low
        ob = tfo.OB(dummy_id, direction, high, low, i, i, a, 0.0, True, tfo.classify_session(bar.end), bar.end.year)
        pool.append(Setup(dummy_id, "baseline", ob, direction, i, i + REACTION_BARS, atr_bucket(a, lo, hi), ob.session, news_proxy(bars, i, a)))
        dummy_id += 1
    selected: list[Setup] = []
    used: set[int] = set()
    for setup in sorted(template, key=lambda s: s.entry_bar):
        candidates = [
            p for p in pool
            if p.setup_id not in used
            and p.entry_bar < setup.entry_bar
            and p.direction == setup.direction
            and p.session == setup.session
            and p.atr_bucket == setup.atr_bucket
        ]
        if candidates:
            choice = max(candidates, key=lambda p: p.entry_bar)
            used.add(choice.setup_id)
            selected.append(choice)
    return selected


def close_weight(trade: Trade, ts: datetime, weight: float, r_value: float, stop_size: float) -> None:
    trade.gross_r += weight * r_value
    trade.net_r += weight * r_value - weight * half_cost(stop_size, False)
    trade.net_stress_r += weight * r_value - weight * half_cost(stop_size, True)
    trade.open_weight = max(0.0, trade.open_weight - weight)
    if trade.open_weight <= 1e-12:
        trade.resolved = True


def update_trailing(trade: Trade, bars: list[tfo.Bar], avail: list[list[tfo.Swing]], ts: datetime, mid: float) -> None:
    if trade.variant != "trailing" or not trade.hit_1r or trade.resolved or not trade.entry_found:
        return
    while trade.next_bar_update <= trade.setup.horizon_bar and bars[trade.next_bar_update].end <= ts:
        i = trade.next_bar_update
        for swing in avail[i]:
            if swing.confirmed_at != i:
                continue
            if trade.setup.direction == "bullish" and swing.kind == "low" and trade.stop_price is not None:
                trade.stop_price = max(trade.stop_price, swing.level)
            if trade.setup.direction == "bearish" and swing.kind == "high" and trade.stop_price is not None:
                trade.stop_price = min(trade.stop_price, swing.level)
        trade.next_bar_update += 1


def on_tick(trade: Trade, ts: datetime, mid: float, stop_size: float) -> None:
    if trade.resolved:
        return
    if not trade.entry_found:
        if touched(mid, trade.setup.ob):
            if gap_skip(mid, trade.setup.ob):
                trade.gap_skipped = True
                trade.resolved = True
                return
            trade.entry_found = True
            trade.entry_time = ts
            trade.entry_price = entry_edge(trade.setup.ob)
            trade.stop_price = trade.entry_price - stop_size if trade.setup.direction == "bullish" else trade.entry_price + stop_size
            trade.net_r -= half_cost(stop_size, False)
            trade.net_stress_r -= half_cost(stop_size, True)
        return
    assert trade.stop_price is not None
    value = r_now(mid, trade, stop_size)
    stop_r = r_now(trade.stop_price, trade, stop_size)
    if value <= stop_r:
        close_weight(trade, ts, trade.open_weight, value, stop_size)
        return
    for target, weight in SCALE_PLAN:
        if target == 1.0 and trade.hit_1r:
            continue
        if target == 2.0 and trade.hit_2r:
            continue
        if target == 3.0 and trade.hit_3r:
            continue
        if value >= target:
            close_weight(trade, ts, weight, target, stop_size)
            if target == 1.0:
                trade.hit_1r = True
                trade.stop_price = trade.entry_price
            elif target == 2.0:
                trade.hit_2r = True
            else:
                trade.hit_3r = True
            if trade.resolved:
                return


def force_exit(trade: Trade, bars: list[tfo.Bar], stop_size: float) -> None:
    if trade.resolved or not trade.entry_found:
        return
    bar = bars[trade.setup.horizon_bar]
    trade.horizon_exit = True
    close_weight(trade, bar.end, trade.open_weight, r_now(bar.close, trade, stop_size), stop_size)


def run_ticks_many(path: Path, bars: list[tfo.Bar], avail: list[list[tfo.Swing]], cohorts: list[tuple[str, list[Setup], str]], stop_size: float) -> tuple[dict[str, list[Trade]], dict[str, int]]:
    trades: list[Trade] = []
    labels: dict[int, str] = {}
    for label, setups, variant in cohorts:
        for setup in setups:
            if setup.news_proxy:
                continue
            trade = Trade(setup, variant, next_bar_update=setup.entry_bar)
            labels[id(trade)] = label
            trades.append(trade)
    out = {label: [] for label, _, _ in cohorts}
    skips = {label: 0 for label, _, _ in cohorts}
    if not trades:
        return out, skips
    trades.sort(key=lambda t: bars[t.setup.entry_bar].start)
    first = min(bars[t.setup.entry_bar].start for t in trades)
    last = max(bars[t.setup.horizon_bar].end for t in trades)
    active: list[Trade] = []
    p = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ti, bi, ai = header.index("DateTime"), header.index("Bid"), header.index("Ask")
        for row in reader:
            try:
                ts = tfo.parse_ts(row[ti]); bid = float(row[bi]); ask = float(row[ai])
            except Exception:
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            if ts < first:
                continue
            if ts > last:
                break
            mid = (bid + ask) / 2.0
            while p < len(trades) and bars[trades[p].setup.entry_bar].start <= ts:
                active.append(trades[p]); p += 1
            keep: list[Trade] = []
            for trade in active:
                label = labels[id(trade)]
                if ts > bars[trade.setup.horizon_bar].end:
                    force_exit(trade, bars, stop_size)
                else:
                    on_tick(trade, ts, mid, stop_size)
                    update_trailing(trade, bars, avail, ts, mid)
                    if trade.entry_found and not trade.resolved:
                        on_tick(trade, ts, mid, stop_size)
                if trade.resolved:
                    if trade.gap_skipped:
                        skips[label] += 1
                    elif trade.entry_found:
                        out[label].append(trade)
                else:
                    keep.append(trade)
            active = keep
    for trade in active:
        force_exit(trade, bars, stop_size)
        if trade.entry_found:
            out[labels[id(trade)]].append(trade)
    return out, skips


def ci(vals: list[float]) -> tuple[float, float, float, float]:
    if not vals:
        return math.nan, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se, sd


def max_losses(vals: list[float]) -> int:
    best = cur = 0
    for value in vals:
        if value < 0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def summarize(rows: list[Trade], stress: bool = False) -> dict[str, float]:
    vals = [t.net_stress_r if stress else t.net_r for t in rows]
    gross = [t.gross_r for t in rows]
    m, lo, hi, sd = ci(vals)
    return {
        "n": len(vals),
        "gross": mean(gross) if gross else math.nan,
        "net": m,
        "lo": lo,
        "hi": hi,
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "unresolved": sum(t.horizon_exit for t in rows) / len(rows) if rows else math.nan,
        "worst": min(vals) if vals else math.nan,
        "max_loss": max_losses(vals),
        "std": sd,
    }


def print_row(label: str, rows: list[Trade], skips: int, stress: bool = False) -> None:
    s = summarize(rows, stress)
    print(f"{label},{s['n']},{s['gross']:.4f},{s['net']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['win']:.2%},{s['unresolved']:.2%},{s['worst']:.4f},{s['max_loss']},{skips}")


def raw_spread_sample(path: Path, start: datetime) -> float:
    vals: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ti, bi, ai = header.index("DateTime"), header.index("Bid"), header.index("Ask")
        for row in reader:
            try:
                ts = tfo.parse_ts(row[ti])
                if ts < start:
                    continue
                bid = float(row[bi]); ask = float(row[ai])
            except Exception:
                continue
            if ask > bid > 0:
                vals.append(ask - bid)
                if len(vals) >= 250_000:
                    break
    return median(vals) if vals else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=Path, default=None)
    args = parser.parse_args()
    path = args.ticks or find_xag_path()
    print("Loading XAGUSD M15 bars...", flush=True)
    bars = tfo.load_bars(path, 15)
    atr = tfo.compute_atr(bars)
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    med_atr = median(recent_atrs)
    stop = round_silver_stop(med_atr)
    used = [b for b in bars if b.end >= START_DATE]
    print("Detecting silver OBs...", flush=True)
    _, avail = tfo.confirmed_swings(bars)
    obs, breaks, raw_count = tfo.detect_obs(bars, atr, avail)
    setups = build_silver_setups(bars, atr, obs, breaks, avail)
    baseline = build_baseline(bars, atr, setups)
    raw_spread = raw_spread_sample(path, START_DATE)
    print("Running silver tick execution...", flush=True)
    cohorts = [
        ("fixed", setups, "fixed"),
        ("trailing", setups, "trailing"),
        ("baseline_fixed", baseline, "fixed"),
        ("baseline_trailing", baseline, "trailing"),
    ]
    results, skips = run_ticks_many(path, bars, avail, cohorts, stop)

    print("\nSILVER_CROSS_ASSET_CONTEXT")
    print(f"tick_file={path}")
    print(f"bar_count_2024_2026={len(used)}")
    print(f"bar_date_range={used[0].start:%Y-%m-%d %H:%M:%S} to {used[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print(f"silver_median_M15_ATR_2024_2026={med_atr:.5f}")
    print(f"chosen_fixed_SL={stop:.5f} USD/oz, ratio_to_median_ATR={stop / med_atr:.2f}x")
    print(f"ob_raw_before_dedup={raw_count},ob_after_dedup={len(obs)},fvg_ob={sum(o.fvg_present for o in obs)}")
    print(f"trend_setups_pre_news={len(setups)},baseline_matched={len(baseline)}")
    print(f"silver_spread_proxy_normal={SILVER_SPREAD_NORMAL:.5f},stress={SILVER_SPREAD_STRESS:.5f},raw_dukascopy_median_spread_sample={raw_spread:.5f}")
    print(f"normal_roundtrip_cost_R={SILVER_SPREAD_NORMAL / stop:.4f},stress_roundtrip_cost_R={SILVER_SPREAD_STRESS / stop:.4f}")

    print("\nOVERALL_NORMAL_SPREAD")
    print("cohort,n,gross_R,net_R,ci_low,ci_high,win,unresolved,worst_net_R,max_loss,gap_skips")
    for label in ("fixed", "trailing", "baseline_fixed", "baseline_trailing"):
        print_row(label, results[label], skips[label])

    print("\nOVERALL_STRESS_SPREAD")
    print("cohort,n,gross_R,net_R,ci_low,ci_high,win,unresolved,worst_net_R,max_loss,gap_skips")
    for label in ("fixed", "trailing", "baseline_fixed", "baseline_trailing"):
        print_row(label, results[label], skips[label], True)

    print("\nTRAIN_TEST_NORMAL_SPREAD")
    print("cohort,period,n,gross_R,net_R,ci_low,ci_high,win,unresolved,worst_net_R,max_loss")
    for label in ("fixed", "trailing"):
        for period, rows in (
            ("train_2024", [t for t in results[label] if t.entry_time and t.entry_time < TRAIN_END]),
            ("test_2025_2026", [t for t in results[label] if t.entry_time and TRAIN_END <= t.entry_time < TEST_END]),
        ):
            s = summarize(rows)
            print(f"{label},{period},{s['n']},{s['gross']:.4f},{s['net']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['win']:.2%},{s['unresolved']:.2%},{s['worst']:.4f},{s['max_loss']}")

    print("\nEDGE_VS_BASELINE_NORMAL")
    print(f"fixed_edge_R={summarize(results['fixed'])['net'] - summarize(results['baseline_fixed'])['net']:.4f}")
    print(f"trailing_edge_R={summarize(results['trailing'])['net'] - summarize(results['baseline_trailing'])['net']:.4f}")


if __name__ == "__main__":
    main()
