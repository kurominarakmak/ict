"""
Strict vs less-strict trend-following OB fixed-SL comparison.

One pre-registered relaxation only:
- STRICT: trend-filtered OB + FVG present.
- LESS STRICT: trend-filtered OB, FVG not required.

Everything else is held constant:
- Same locked OB detection with overlap de-dup from trend_following_ob.py.
- Same ChoCh/HH-HL/LH-LL trend filter.
- Retest within 33 M15 bars.
- Entry at OB edge.
- Scale-out + BE ladder: +1R 50%, +2R 25%, +3R 25%.
- Stops fill at actual breaching tick, targets fill at target R.
- Unresolved force-closed at horizon.
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


@dataclass(frozen=True)
class Setup:
    setup_id: int
    label: str
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
    entry_found: bool = False
    gap_skipped: bool = False
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    gross_r: float = 0.0
    net_r: float = 0.0
    open_weight: float = 1.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    resolved: bool = False
    horizon_exit: bool = False


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bucket(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value <= hi:
        return "medium"
    return "high"


def round_silver_stop(median_atr: float) -> float:
    candidates = [0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    return min(candidates, key=lambda x: abs(x - median_atr))


def has_full_horizon(bars: list[tfo.Bar], i: int) -> bool:
    if i + REACTION_BARS >= len(bars):
        return False
    return all(bars[j].segment_id == bars[i].segment_id for j in range(i + 1, i + REACTION_BARS + 1))


def first_touch(bars: list[tfo.Bar], ob: tfo.OB) -> Optional[int]:
    for i in range(ob.creation_index + 1, min(len(bars), ob.creation_index + RETEST_BARS + 1)):
        if bars[i].segment_id != bars[ob.creation_index].segment_id:
            return None
        if bars[i].low <= ob.high and bars[i].high >= ob.low:
            return i
    return None


def news_proxy(bars: list[tfo.Bar], i: int, atr_value: float) -> bool:
    return any(
        bars[j].segment_id == bars[i].segment_id and (bars[j].high - bars[j].low) > 3.0 * atr_value
        for j in range(max(0, i - 2), min(len(bars) - 1, i + 2) + 1)
    )


def build_setups(
    bars: list[tfo.Bar],
    atr: list[Optional[float]],
    obs: list[tfo.OB],
    breaks: dict[int, str],
    avail: list[list[tfo.Swing]],
    require_fvg: bool,
    label: str,
) -> list[Setup]:
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    lo, hi = quantile(recent_atrs, 1 / 3), quantile(recent_atrs, 2 / 3)
    out: list[Setup] = []
    for ob in obs:
        if require_fvg and not ob.fvg_present:
            continue
        if bars[ob.creation_index].end < START_DATE:
            continue
        trend = tfo.trend_at(
            [s for s in avail[ob.creation_index] if s.confirmed_at <= ob.creation_index],
            {k: v for k, v in breaks.items() if k <= ob.creation_index},
        )
        if trend != ob.direction:
            continue
        touch = first_touch(bars, ob)
        if touch is None or bars[touch].end < START_DATE or not has_full_horizon(bars, touch):
            continue
        a = atr[touch] or ob.atr
        out.append(
            Setup(
                len(out) + 1,
                label,
                ob,
                ob.direction,
                touch,
                touch + REACTION_BARS,
                bucket(a, lo, hi),
                tfo.classify_session(bars[touch].end),
                news_proxy(bars, touch, a),
            )
        )
    return out


def build_baseline(bars: list[tfo.Bar], atr: list[Optional[float]], template: list[Setup], label: str) -> list[Setup]:
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    lo, hi = quantile(recent_atrs, 1 / 3), quantile(recent_atrs, 2 / 3)
    pool: list[Setup] = []
    setup_id = 1
    for i, bar in enumerate(bars):
        if bar.end < START_DATE or not has_full_horizon(bars, i):
            continue
        a = atr[i]
        if a is None:
            continue
        direction = "bullish" if i % 2 == 0 else "bearish"
        ob = tfo.OB(setup_id, direction, bar.high, bar.low, i, i, a, 0.0, True, tfo.classify_session(bar.end), bar.end.year)
        pool.append(Setup(setup_id, label, ob, direction, i, i + REACTION_BARS, bucket(a, lo, hi), ob.session, news_proxy(bars, i, a)))
        setup_id += 1
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
            choice = candidates[-1]
            used.add(choice.setup_id)
            selected.append(choice)
    return selected


def entry_edge(setup: Setup) -> float:
    return setup.ob.high if setup.direction == "bullish" else setup.ob.low


def touched(mid: float, setup: Setup) -> bool:
    return mid <= setup.ob.high if setup.direction == "bullish" else mid >= setup.ob.low


def gap_skip(mid: float, setup: Setup) -> bool:
    return mid <= setup.ob.low if setup.direction == "bullish" else mid >= setup.ob.high


def r_now(price: float, trade: Trade, stop_size: float) -> float:
    assert trade.entry_price is not None
    return (price - trade.entry_price) / stop_size if trade.setup.direction == "bullish" else (trade.entry_price - price) / stop_size


def close_weight(trade: Trade, ts: datetime, weight: float, r_value: float, spread: float, stop_size: float) -> None:
    trade.gross_r += weight * r_value
    trade.net_r += weight * r_value - weight * (spread / stop_size / 2.0)
    trade.open_weight = max(0.0, trade.open_weight - weight)
    if trade.open_weight <= 1e-12:
        trade.resolved = True


def on_tick(trade: Trade, ts: datetime, mid: float, stop_size: float, spread: float) -> None:
    if trade.resolved:
        return
    if not trade.entry_found:
        if touched(mid, trade.setup):
            if gap_skip(mid, trade.setup):
                trade.gap_skipped = True
                trade.resolved = True
                return
            trade.entry_found = True
            trade.entry_time = ts
            trade.entry_price = entry_edge(trade.setup)
            trade.net_r -= spread / stop_size / 2.0
        return
    value = r_now(mid, trade, stop_size)
    stop_r = 0.0 if trade.hit_1r else -1.0
    if value <= stop_r:
        close_weight(trade, ts, trade.open_weight, value, spread, stop_size)
        return
    for target, weight in SCALE_PLAN:
        if target == 1.0 and trade.hit_1r:
            continue
        if target == 2.0 and trade.hit_2r:
            continue
        if target == 3.0 and trade.hit_3r:
            continue
        if value >= target:
            close_weight(trade, ts, weight, target, spread, stop_size)
            if target == 1.0:
                trade.hit_1r = True
            elif target == 2.0:
                trade.hit_2r = True
            else:
                trade.hit_3r = True
            if trade.resolved:
                return


def force_exit(trade: Trade, bars: list[tfo.Bar], stop_size: float, spread: float) -> None:
    if trade.resolved or not trade.entry_found:
        return
    bar = bars[trade.setup.horizon_bar]
    trade.horizon_exit = True
    close_weight(trade, bar.end, trade.open_weight, r_now(bar.close, trade, stop_size), spread, stop_size)


def run_ticks_many(path: Path, bars: list[tfo.Bar], cohorts: list[tuple[str, list[Setup]]], stop_size: float, spread: float) -> tuple[dict[str, list[Trade]], dict[str, int]]:
    trades: list[Trade] = []
    labels: dict[int, str] = {}
    for label, setups in cohorts:
        for setup in setups:
            if setup.news_proxy:
                continue
            trade = Trade(setup)
            labels[id(trade)] = label
            trades.append(trade)
    out = {label: [] for label, _ in cohorts}
    skips = {label: 0 for label, _ in cohorts}
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
                    force_exit(trade, bars, stop_size, spread)
                else:
                    on_tick(trade, ts, mid, stop_size, spread)
                if trade.resolved:
                    if trade.gap_skipped:
                        skips[label] += 1
                    elif trade.entry_found:
                        out[label].append(trade)
                else:
                    keep.append(trade)
            active = keep
    for trade in active:
        force_exit(trade, bars, stop_size, spread)
        if trade.entry_found:
            out[labels[id(trade)]].append(trade)
    for rows in out.values():
        rows.sort(key=lambda t: (t.entry_time or datetime.min.replace(tzinfo=timezone.utc), t.setup.setup_id))
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
    for v in vals:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize(rows: list[Trade]) -> dict[str, float]:
    vals = [t.net_r for t in rows]
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
    }


def print_table(asset: str, results: dict[str, list[Trade]], skips: dict[str, int]) -> None:
    print(f"\n{asset}_STRICT_VS_LESS_STRICT")
    print("version,cohort,n,gross_R,net_R,ci_low,ci_high,win,unresolved,worst_R,max_loss,gap_skips")
    for version in ("strict", "less_strict"):
        for cohort in ("entry", "baseline"):
            label = version if cohort == "entry" else version + "_baseline"
            s = summarize(results[label])
            print(f"{version},{cohort},{s['n']},{s['gross']:.4f},{s['net']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['win']:.2%},{s['unresolved']:.2%},{s['worst']:.4f},{s['max_loss']},{skips[label]}")
        edge = summarize(results[version])["net"] - summarize(results[version + "_baseline"])["net"]
        print(f"{version},edge_vs_baseline,{edge:.4f}")


def print_gold_train_test(results: dict[str, list[Trade]]) -> None:
    print("\nGOLD_TRAIN_TEST")
    print("version,period,n,net_R,ci_low,ci_high,win,unresolved,worst_R,max_loss")
    for version in ("strict", "less_strict"):
        for period, rows in (
            ("train_2024", [t for t in results[version] if t.entry_time and t.entry_time < TRAIN_END]),
            ("test_2025_2026", [t for t in results[version] if t.entry_time and TRAIN_END <= t.entry_time < TEST_END]),
        ):
            s = summarize(rows)
            print(f"{version},{period},{s['n']},{s['net']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['win']:.2%},{s['unresolved']:.2%},{s['worst']:.4f},{s['max_loss']}")


def run_asset(asset: str, path: Path, stop_size: float, spread: float) -> tuple[dict[str, list[Trade]], dict[str, int], dict[str, float]]:
    print(f"Loading {asset} bars...", flush=True)
    bars = tfo.load_bars(path, 15)
    atr = tfo.compute_atr(bars)
    print(f"Detecting {asset} OBs...", flush=True)
    _, avail = tfo.confirmed_swings(bars)
    obs, breaks, raw_count = tfo.detect_obs(bars, atr, avail)
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    strict = build_setups(bars, atr, obs, breaks, avail, True, "strict")
    less = build_setups(bars, atr, obs, breaks, avail, False, "less_strict")
    strict_base = build_baseline(bars, atr, strict, "strict_baseline")
    less_base = build_baseline(bars, atr, less, "less_strict_baseline")
    print(f"Running {asset} tick execution...", flush=True)
    results, skips = run_ticks_many(
        path,
        bars,
        [
            ("strict", strict),
            ("strict_baseline", strict_base),
            ("less_strict", less),
            ("less_strict_baseline", less_base),
        ],
        stop_size,
        spread,
    )
    meta = {
        "bars_2024": sum(1 for b in bars if b.end >= START_DATE),
        "median_atr_2024": median(recent_atrs) if recent_atrs else math.nan,
        "raw_ob": raw_count,
        "dedup_ob": len(obs),
        "fvg_ob": sum(o.fvg_present for o in obs),
        "strict_pre_news": len(strict),
        "less_pre_news": len(less),
        "strict_base": len(strict_base),
        "less_base": len(less_base),
    }
    return results, skips, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--silver", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    args = parser.parse_args()

    gold_results, gold_skips, gold_meta = run_asset("GOLD", args.gold, 10.0, 0.20)
    silver_bars = tfo.load_bars(args.silver, 15)
    silver_atr = tfo.compute_atr(silver_bars)
    silver_recent = [a for i, a in enumerate(silver_atr) if a is not None and silver_bars[i].end >= START_DATE]
    silver_stop = round_silver_stop(median(silver_recent))
    # Re-run silver through the normal path after measuring stop; duplicated bar load is
    # intentional to keep run_asset self-contained and auditable.
    silver_results, silver_skips, silver_meta = run_asset("SILVER", args.silver, silver_stop, 0.02)

    print("\nFVG_RELAXATION_CONTEXT")
    print("one_relaxation=drop FVG-present requirement only; trend filter, pullback window, de-dup, fixed SL/TP, corrected execution unchanged")
    print(f"gold_stop=10.0000,gold_spread=0.2000,normal_cost_R={0.20 / 10.0:.4f}")
    print(f"silver_median_ATR_2024={silver_meta['median_atr_2024']:.5f},silver_stop={silver_stop:.5f},silver_spread_proxy=0.0200,normal_cost_R={0.02 / silver_stop:.4f}")
    print(f"gold_meta={gold_meta}")
    print(f"silver_meta={silver_meta}")

    print_table("GOLD", gold_results, gold_skips)
    print_gold_train_test(gold_results)
    print_table("SILVER", silver_results, silver_skips)


if __name__ == "__main__":
    main()
