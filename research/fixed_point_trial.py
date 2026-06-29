"""
Fixed-point SL/TP trial for 2024-2026.

New declared trial:
- SL = $10/oz fixed.
- TP ladder: +$10 closes 50% and moves stop to breakeven, +$20 closes 25%,
  +$30 closes final 25%.
- Entry families: bare OB reversal, SMC confluence, trend-following OB.
- Tick-ordered fills, edge-price entry, actual tick stop slippage, IUX $0.20/oz spread.

No parameter sweep.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategies"))
sys.path.insert(0, str(ROOT / "research"))

from cost_model import IUX_SPREAD_USD_OZ, SpreadMode, session_for_timestamp
from order_block import Bar, classify_session, compute_atr, default_tick_path, load_bars, parse_timestamp


START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
STOP_USD = 10.0
REACTION_BARS = 20
OB_RETEST_BARS = 33
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25.0
SWING_LEFT = 10
SWING_RIGHT = 10
SCALE_PLAN = ((1.0, 0.50), (2.0, 0.25), (3.0, 0.25))
N_BARS = 10
PRICE_ATR_MULTIPLE = 1.0
ENTRY_EXPIRY_BARS = 20


@dataclass(frozen=True)
class Zone:
    source: str
    signal_id: int
    direction: str
    high: float
    low: float
    index: int
    atr: float
    session: str
    segment_id: int
    fvg_present: bool = True


@dataclass(frozen=True)
class Swing:
    kind: str
    index: int
    confirmed_at: int
    level: float


@dataclass(frozen=True)
class Setup:
    setup_id: int
    family: str
    direction: str
    zone_high: float
    zone_low: float
    entry_bar: int
    horizon_bar: int
    session: str
    year: int
    atr_bucket: str
    news_proxy: bool
    baseline: bool = False


@dataclass
class Trade:
    setup: Setup
    entry_found: bool = False
    gap_skipped: bool = False
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    open_weight: float = 1.0
    gross_r: float = 0.0
    net_r: float = 0.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    resolved: bool = False
    horizon_exit: bool = False


def parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes"}


def load_ob_zones(path: Path) -> list[Zone]:
    out: list[Zone] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                Zone(
                    "OB",
                    int(row["zone_id"]),
                    row["direction"],
                    float(row["zone_high"]),
                    float(row["zone_low"]),
                    int(row["impulse_end_index"]),
                    float(row["frozen_atr"]),
                    row["session"],
                    int(row["segment_id"]),
                    parse_bool(row["fvg_present"]),
                )
            )
    return out


def load_fvgs(path: Path) -> list[Zone]:
    out: list[Zone] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(
                Zone(
                    "FVG",
                    int(row["fvg_id"]),
                    row["direction"],
                    float(row["gap_high"]),
                    float(row["gap_low"]),
                    int(row["candle3_index"]),
                    float(row["frozen_atr"]),
                    row["session"],
                    int(row["segment_id"]),
                    True,
                )
            )
    return out


def load_sweeps(path: Path) -> list[Zone]:
    out: list[Zone] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            level = float(row["swept_swing_level"])
            out.append(
                Zone(
                    "Sweep",
                    int(row["sweep_id"]),
                    row["direction"],
                    level,
                    level,
                    int(row["rejection_index"]),
                    float(row["frozen_atr"]),
                    row["session"],
                    int(row["segment_id"]),
                    True,
                )
            )
    return out


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


def first_touch(bars: list[Bar], zone: Zone, after: int, expiry: int) -> Optional[int]:
    segment = bars[after].segment_id
    for i in range(after + 1, min(len(bars), after + expiry + 1)):
        if bars[i].segment_id != segment:
            return None
        if bars[i].low <= zone.high and bars[i].high >= zone.low:
            return i
    return None


def has_full_horizon(bars: list[Bar], i: int) -> bool:
    if i + REACTION_BARS >= len(bars):
        return False
    return all(bars[j].segment_id == bars[i].segment_id for j in range(i + 1, i + REACTION_BARS + 1))


def news_proxy(bars: list[Bar], i: int) -> bool:
    start = max(0, i - 2)
    end = min(len(bars) - 1, i + 2)
    ranges = [bars[j].high - bars[j].low for j in range(max(0, i - 14), i + 1) if bars[j].segment_id == bars[i].segment_id]
    atr = sum(ranges) / len(ranges) if ranges else STOP_USD
    return any(bars[j].segment_id == bars[i].segment_id and (bars[j].high - bars[j].low) > 3.0 * atr for j in range(start, end + 1))


def range_distance(a: Zone, b: Zone) -> float:
    if a.low <= b.high and b.low <= a.high:
        return 0.0
    if a.high < b.low:
        return b.low - a.high
    return a.low - b.high


def cooccurs(anchor: Zone, other: Zone) -> bool:
    return abs(anchor.index - other.index) <= N_BARS and range_distance(anchor, other) <= PRICE_ATR_MULTIPLE * anchor.atr


def choose_zone(sweep: Zone, obs: list[Zone], fvgs: list[Zone]) -> Optional[Zone]:
    ob_matches = [z for z in obs if z.direction == sweep.direction and cooccurs(sweep, z)]
    if ob_matches:
        return min(ob_matches, key=lambda z: (abs(z.index - sweep.index), range_distance(sweep, z), z.signal_id))
    fvg_matches = [z for z in fvgs if z.direction == sweep.direction and cooccurs(sweep, z)]
    if fvg_matches:
        return min(fvg_matches, key=lambda z: (abs(z.index - sweep.index), range_distance(sweep, z), z.signal_id))
    return None


def compute_adx(bars: list[Bar]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(bars)
    trq: deque[float] = deque(maxlen=ADX_PERIOD)
    pdmq: deque[float] = deque(maxlen=ADX_PERIOD)
    ndmq: deque[float] = deque(maxlen=ADX_PERIOD)
    dxq: deque[float] = deque(maxlen=ADX_PERIOD)
    for i in range(1, len(bars)):
        b, p = bars[i], bars[i - 1]
        if b.segment_id != p.segment_id:
            trq.clear(); pdmq.clear(); ndmq.clear(); dxq.clear(); continue
        up = b.high - p.high
        down = p.low - b.low
        pdm = up if up > down and up > 0 else 0.0
        ndm = down if down > up and down > 0 else 0.0
        tr = max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
        trq.append(tr); pdmq.append(pdm); ndmq.append(ndm)
        if len(trq) == ADX_PERIOD and sum(trq) > 0:
            pdi = 100 * sum(pdmq) / sum(trq)
            ndi = 100 * sum(ndmq) / sum(trq)
            dxq.append(100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi else 0.0)
            if len(dxq) == ADX_PERIOD:
                out[i] = sum(dxq) / ADX_PERIOD
    return out


def is_high(bars: list[Bar], p: int) -> bool:
    if p - SWING_LEFT < 0 or p + SWING_RIGHT >= len(bars):
        return False
    w = bars[p - SWING_LEFT : p + SWING_RIGHT + 1]
    return all(x.segment_id == bars[p].segment_id for x in w) and all(bars[p].high > x.high for j, x in enumerate(w) if j != SWING_LEFT)


def is_low(bars: list[Bar], p: int) -> bool:
    if p - SWING_LEFT < 0 or p + SWING_RIGHT >= len(bars):
        return False
    w = bars[p - SWING_LEFT : p + SWING_RIGHT + 1]
    return all(x.segment_id == bars[p].segment_id for x in w) and all(bars[p].low < x.low for j, x in enumerate(w) if j != SWING_LEFT)


def swing_availability(bars: list[Bar]) -> tuple[list[list[Swing]], dict[int, str]]:
    available: list[list[Swing]] = [[] for _ in bars]
    cur: list[Swing] = []
    latest_h = latest_l = None
    broken_h: set[int] = set()
    broken_l: set[int] = set()
    break_dir: dict[int, str] = {}
    for i, bar in enumerate(bars):
        p = i - SWING_RIGHT
        if p >= SWING_LEFT and bars[p].segment_id == bar.segment_id:
            if is_high(bars, p):
                latest_h = Swing("high", p, i, bars[p].high); cur.append(latest_h)
            if is_low(bars, p):
                latest_l = Swing("low", p, i, bars[p].low); cur.append(latest_l)
        if latest_h and latest_h.index not in broken_h and bars[latest_h.index].segment_id == bar.segment_id and bar.high > latest_h.level:
            break_dir[i] = "bullish"; broken_h.add(latest_h.index)
        if latest_l and latest_l.index not in broken_l and bars[latest_l.index].segment_id == bar.segment_id and bar.low < latest_l.level:
            break_dir[i] = "bearish"; broken_l.add(latest_l.index)
        available[i] = list(cur)
    return available, break_dir


def trend_at(swings: list[Swing], break_dir: dict[int, str], i: int) -> str:
    highs = [s for s in swings if s.kind == "high" and s.confirmed_at <= i]
    lows = [s for s in swings if s.kind == "low" and s.confirmed_at <= i]
    if len(highs) < 2 or len(lows) < 2:
        return "none"
    last_break = None
    for idx in sorted(k for k in break_dir if k <= i):
        last_break = break_dir[idx]
    if highs[-1].level > highs[-2].level and lows[-1].level > lows[-2].level and last_break == "bullish":
        return "bullish"
    if highs[-1].level < highs[-2].level and lows[-1].level < lows[-2].level and last_break == "bearish":
        return "bearish"
    return "none"


def make_setup(next_id: int, family: str, direction: str, high: float, low: float, touch: int, bars: list[Bar], atr_value: float, lo: float, hi: float, baseline: bool = False) -> Setup:
    return Setup(next_id, family, direction, high, low, touch, touch + REACTION_BARS, classify_session(bars[touch].end), bars[touch].end.year, bucket(atr_value, lo, hi), news_proxy(bars, touch), baseline)


def build_bare_ob_setups(bars: list[Bar], obs: list[Zone], atr: list[Optional[float]]) -> list[Setup]:
    atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    lo, hi = quantile(atrs, 1 / 3), quantile(atrs, 2 / 3)
    out: list[Setup] = []
    for z in obs:
        if z.index >= len(bars) or bars[z.index].end < START_DATE:
            continue
        touch = first_touch(bars, z, z.index, 5000)
        if touch is None or bars[touch].end < START_DATE or not has_full_horizon(bars, touch):
            continue
        out.append(make_setup(len(out) + 1, "bare_ob", z.direction, z.high, z.low, touch, bars, atr[touch] or z.atr, lo, hi))
    return out


def build_confluence_setups(bars: list[Bar], sweeps: list[Zone], obs: list[Zone], fvgs: list[Zone], atr: list[Optional[float]]) -> tuple[list[Setup], int]:
    matched_atrs = [s.atr for s in sweeps if choose_zone(s, obs, fvgs) is not None]
    lo, hi = quantile(matched_atrs, 1 / 3), quantile(matched_atrs, 2 / 3)
    out: list[Setup] = []
    expired = 0
    for sweep in sweeps:
        zone = choose_zone(sweep, obs, fvgs)
        if zone is None:
            continue
        established = max(sweep.index, zone.index)
        if established >= len(bars):
            continue
        touch = first_touch(bars, zone, established, ENTRY_EXPIRY_BARS)
        if touch is None:
            expired += 1; continue
        if bars[touch].end < START_DATE or not has_full_horizon(bars, touch):
            continue
        out.append(make_setup(len(out) + 1, "confluence", sweep.direction, zone.high, zone.low, touch, bars, atr[touch] or sweep.atr, lo, hi))
    return out, expired


def build_trend_ob_setups(bars: list[Bar], obs: list[Zone], atr: list[Optional[float]]) -> list[Setup]:
    avail, breaks = swing_availability(bars)
    at = [z.atr for z in obs if z.fvg_present]
    lo, hi = quantile(at, 1 / 3), quantile(at, 2 / 3)
    out: list[Setup] = []
    for z in obs:
        if not z.fvg_present or z.index >= len(bars) or bars[z.index].end < START_DATE:
            continue
        if trend_at(avail[z.index], breaks, z.index) != z.direction:
            continue
        touch = first_touch(bars, z, z.index, OB_RETEST_BARS)
        if touch is None or bars[touch].end < START_DATE or not has_full_horizon(bars, touch):
            continue
        out.append(make_setup(len(out) + 1, "trend_ob", z.direction, z.high, z.low, touch, bars, atr[touch] or z.atr, lo, hi))
    return out


def build_baseline(bars: list[Bar], template: list[Setup], family: str, atr: list[Optional[float]]) -> list[Setup]:
    if not template:
        return []
    recent_atrs = [a for i, a in enumerate(atr) if a is not None and bars[i].end >= START_DATE]
    lo, hi = quantile(recent_atrs, 1 / 3), quantile(recent_atrs, 2 / 3)
    pool: list[Setup] = []
    next_id = 1
    for i, bar in enumerate(bars):
        if bar.end < START_DATE or i + REACTION_BARS >= len(bars) or not has_full_horizon(bars, i):
            continue
        a = atr[i]
        if a is None:
            continue
        direction = "bullish" if i % 2 == 0 else "bearish"
        pool.append(make_setup(next_id, family + "_baseline", direction, bar.high, bar.low, i, bars, a, lo, hi, True))
        next_id += 1
    selected: list[Setup] = []
    used: set[int] = set()
    for t in sorted(template, key=lambda x: x.entry_bar):
        candidates = [p for p in pool if p.setup_id not in used and p.entry_bar < t.entry_bar and p.direction == t.direction and p.session == t.session and p.atr_bucket == t.atr_bucket]
        if candidates:
            choice = max(candidates, key=lambda p: p.entry_bar)
            used.add(choice.setup_id)
            selected.append(choice)
    return selected


def entry_edge(setup: Setup) -> float:
    return setup.zone_high if setup.direction == "bullish" else setup.zone_low


def gap_skip(mid: float, setup: Setup) -> bool:
    if setup.zone_high <= setup.zone_low:
        return False
    return mid <= setup.zone_low if setup.direction == "bullish" else mid >= setup.zone_high


def touched(mid: float, setup: Setup) -> bool:
    return mid <= setup.zone_high if setup.direction == "bullish" else mid >= setup.zone_low


def r_now(price: float, trade: Trade) -> float:
    assert trade.entry_price is not None
    return (price - trade.entry_price) / STOP_USD if trade.setup.direction == "bullish" else (trade.entry_price - price) / STOP_USD


def half_spread_r(ts: datetime, mode: SpreadMode = SpreadMode.MEDIAN) -> float:
    return IUX_SPREAD_USD_OZ[session_for_timestamp(ts)].value(mode) / STOP_USD / 2.0


def close_weight(trade: Trade, ts: datetime, weight: float, r_value: float) -> None:
    trade.gross_r += weight * r_value
    trade.net_r += weight * r_value - weight * half_spread_r(ts)
    trade.open_weight = max(0.0, trade.open_weight - weight)
    if trade.open_weight <= 1e-12:
        trade.resolved = True


def on_tick(trade: Trade, ts: datetime, mid: float) -> None:
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
            trade.stop_price = trade.entry_price - STOP_USD if trade.setup.direction == "bullish" else trade.entry_price + STOP_USD
            trade.net_r -= half_spread_r(ts)
        return
    assert trade.stop_price is not None
    value = r_now(mid, trade)
    stop_r = r_now(trade.stop_price, trade)
    if value <= stop_r:
        close_weight(trade, ts, trade.open_weight, value)
        return
    for target, weight in SCALE_PLAN:
        if target == 1.0 and trade.hit_1r:
            continue
        if target == 2.0 and trade.hit_2r:
            continue
        if target == 3.0 and trade.hit_3r:
            continue
        if value >= target:
            close_weight(trade, ts, weight, value)
            if target == 1.0:
                trade.hit_1r = True
                trade.stop_price = trade.entry_price
            elif target == 2.0:
                trade.hit_2r = True
            else:
                trade.hit_3r = True
            if trade.resolved:
                return


def force_exit(trade: Trade, bars: list[Bar]) -> None:
    if trade.resolved or not trade.entry_found:
        return
    bar = bars[trade.setup.horizon_bar]
    trade.horizon_exit = True
    close_weight(trade, bar.end, trade.open_weight, r_now(bar.close, trade))


def run_ticks(tick_path: Path, bars: list[Bar], setups: list[Setup]) -> tuple[list[Trade], int]:
    trades = [Trade(s) for s in setups if not s.news_proxy]
    trades.sort(key=lambda t: bars[t.setup.entry_bar].start)
    if not trades:
        return [], 0
    first = min(bars[t.setup.entry_bar].start for t in trades)
    last = max(bars[t.setup.horizon_bar].end for t in trades)
    active: list[Trade] = []
    done: list[Trade] = []
    skips = 0
    p = 0
    with tick_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ti, bi, ai = header.index("DateTime"), header.index("Bid"), header.index("Ask")
        for row in reader:
            try:
                ts = parse_timestamp(row[ti]); bid = float(row[bi]); ask = float(row[ai])
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
                if ts > bars[trade.setup.horizon_bar].end:
                    force_exit(trade, bars)
                else:
                    on_tick(trade, ts, mid)
                if trade.resolved:
                    if trade.gap_skipped:
                        skips += 1
                    elif trade.entry_found:
                        done.append(trade)
                else:
                    keep.append(trade)
            active = keep
    for trade in active:
        force_exit(trade, bars)
        if trade.entry_found:
            done.append(trade)
    done.sort(key=lambda t: (t.entry_time or bars[t.setup.entry_bar].end, t.setup.setup_id))
    return done, skips


def run_ticks_many(tick_path: Path, bars: list[Bar], cohorts: list[tuple[str, list[Setup]]]) -> tuple[dict[str, list[Trade]], dict[str, int]]:
    trades: list[Trade] = []
    labels: dict[int, str] = {}
    for label, setups in cohorts:
        for setup in setups:
            if setup.news_proxy:
                continue
            trade = Trade(setup)
            labels[id(trade)] = label
            trades.append(trade)
    out: dict[str, list[Trade]] = {label: [] for label, _ in cohorts}
    skips: dict[str, int] = {label: 0 for label, _ in cohorts}
    if not trades:
        return out, skips
    trades.sort(key=lambda t: bars[t.setup.entry_bar].start)
    first = min(bars[t.setup.entry_bar].start for t in trades)
    last = max(bars[t.setup.horizon_bar].end for t in trades)
    active: list[Trade] = []
    p = 0
    with tick_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ti, bi, ai = header.index("DateTime"), header.index("Bid"), header.index("Ask")
        for row in reader:
            try:
                ts = parse_timestamp(row[ti]); bid = float(row[bi]); ask = float(row[ai])
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
                    force_exit(trade, bars)
                else:
                    on_tick(trade, ts, mid)
                if trade.resolved:
                    if trade.gap_skipped:
                        skips[label] += 1
                    elif trade.entry_found:
                        out[label].append(trade)
                else:
                    keep.append(trade)
            active = keep
    for trade in active:
        force_exit(trade, bars)
        if trade.entry_found:
            out[labels[id(trade)]].append(trade)
    for rows in out.values():
        rows.sort(key=lambda t: (t.entry_time or bars[t.setup.entry_bar].end, t.setup.setup_id))
    return out, skips


def max_losses(vals: list[float]) -> int:
    best = cur = 0
    for v in vals:
        if v < 0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def summarize(trades: list[Trade]) -> dict[str, float]:
    vals = [t.net_r for t in trades]
    gross = [t.gross_r for t in trades]
    if not vals:
        return {"n": 0, "gross": math.nan, "net": math.nan, "lo": math.nan, "hi": math.nan, "win": math.nan, "unresolved": math.nan, "worst": math.nan, "max_loss": 0, "cost": math.nan}
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return {
        "n": len(vals),
        "gross": mean(gross),
        "net": mean(vals),
        "lo": mean(vals) - 1.96 * se,
        "hi": mean(vals) + 1.96 * se,
        "win": sum(v > 0 for v in vals) / len(vals),
        "unresolved": sum(t.horizon_exit for t in trades) / len(trades),
        "worst": min(vals),
        "max_loss": max_losses(vals),
        "cost": mean([g - v for g, v in zip(gross, vals)]),
    }


def print_result(name: str, trades: list[Trade], baseline: list[Trade], skips: int, baseline_skips: int, prior: str) -> None:
    s = summarize(trades)
    b = summarize(baseline)
    print(f"\n{name}")
    print("cohort,n,gross_R,net_R,ci_low,ci_high,win,unresolved,worst_net_R,max_loss,mean_cost_R,gap_skips")
    print(f"entry,{s['n']},{s['gross']:.4f},{s['net']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['win']:.2%},{s['unresolved']:.2%},{s['worst']:.4f},{s['max_loss']},{s['cost']:.4f},{skips}")
    print(f"baseline,{b['n']},{b['gross']:.4f},{b['net']:.4f},{b['lo']:.4f},{b['hi']:.4f},{b['win']:.2%},{b['unresolved']:.2%},{b['worst']:.4f},{b['max_loss']},{b['cost']:.4f},{baseline_skips}")
    print(f"edge_vs_baseline_R={s['net'] - b['net']:.4f}")
    print(f"prior_ATR_relative={prior}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=Path, default=None)
    args = parser.parse_args()
    tick_path = args.ticks or default_tick_path()
    print("Loading gap-aware M15 bars...", flush=True)
    bars = load_bars(tick_path, gap_minutes=30.0)
    atr = compute_atr(bars)
    obs = load_ob_zones(Path("research/order_block_zones.csv"))
    fvgs = load_fvgs(Path("research/fair_value_gap_fvgs.csv"))
    sweeps = load_sweeps(Path("research/liquidity_sweep_sweeps.csv"))

    used = [b for b in bars if b.end >= START_DATE]
    print("\nFIXED_POINT_TRIAL_CONTEXT")
    print(f"date_filter_start={START_DATE:%Y-%m-%d %H:%M:%S} UTC")
    print(f"bar_count_2024_2026={len(used)}")
    print(f"bar_date_range={used[0].start:%Y-%m-%d %H:%M:%S} to {used[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print("fixed_scheme: SL=$10/oz, TP1=$10 50%+BE, TP2=$20 25%, TP3=$30 25%; R=$10/oz")
    print("cost: $0.20/oz normal spread = 0.0200R full round-trip on a full-size exit; half-spread per fill = 0.0100R.")

    bare = build_bare_ob_setups(bars, obs, atr)
    confluence, expired = build_confluence_setups(bars, sweeps, obs, fvgs, atr)
    trend = build_trend_ob_setups(bars, obs, atr)
    groups = [
        ("BARE_OB_REVERSAL_FIXED_2024_2026", bare, "ATR-relative bare OB: +1.46pp over baseline, CI crossed zero; no robust edge."),
        ("SMC_CONFLUENCE_FIXED_2024_2026", confluence, "Corrected ATR-relative confluence: -0.2448R, 95% CI [-0.3441,-0.1455]; fail."),
        ("TREND_FOLLOWING_OB_FIXED_2024_2026", trend, "Corrected trend OB trailing/structure: net negative, CIs crossed/below zero; fail."),
    ]
    print(f"confluence_expired_no_pullback_all_years={expired}")

    all_cohorts: list[tuple[str, list[Setup]]] = []
    baselines: dict[str, list[Setup]] = {}
    priors: dict[str, str] = {}
    for name, setups, prior in groups:
        baseline = build_baseline(bars, setups, name.lower(), atr)
        baselines[name] = baseline
        priors[name] = prior
        print(f"Prepared {name}: setups={len(setups)}, baseline={len(baseline)}", flush=True)
        all_cohorts.append((name, setups))
        all_cohorts.append((name + "_BASELINE", baseline))

    print("Running combined tick execution for all fixed-point cohorts...", flush=True)
    results, skips = run_ticks_many(tick_path, bars, all_cohorts)
    for name, _, _ in groups:
        print_result(name, results[name], results[name + "_BASELINE"], skips[name], skips[name + "_BASELINE"], priors[name])


if __name__ == "__main__":
    main()
