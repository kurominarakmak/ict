"""
Declared timeframe trials for SMC confluence.

Runs the locked Sweep + OB/FVG confluence hypothesis on a selected timeframe.
No parameter sweep: call once for H1 and once for H4 as declared trials.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from cost_model import IUX_SPREAD_USD_OZ, SpreadMode, session_for_timestamp


ATR_PERIOD = 14
SWING_LEFT = 10
SWING_RIGHT = 10
OB_DISPLACEMENT_ATR = 2.0
FVG_GAP_ATR = 0.5
SWEEP_BREACH_ATR = 0.25
SWEEP_REJECT_BARS = 3
COOC_BARS = 10
COOC_ATR = 1.0
ENTRY_EXPIRY_BARS = 20
REACTION_BARS = 20
SCALE_PLAN = ((1.0, 0.50), (2.0, 0.25), (3.0, 0.25))
GAP_MINUTES = 30.0


@dataclass(frozen=True)
class Bar:
    index: int
    segment_id: int
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Swing:
    index: int
    confirmed_at_index: int
    level: float


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
    net_median_r: float = 0.0
    net_p90_r: float = 0.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    horizon_exit: bool = False
    resolved: bool = False


def default_tick_path() -> Path:
    matches = sorted(Path("data").glob("*XAUUSD*.csv"))
    if not matches:
        raise SystemExit("No XAUUSD CSV found under data/")
    return matches[0]


def parse_timestamp(raw: str) -> datetime:
    raw = raw.strip()
    if len(raw) >= 17 and raw[8] == " ":
        microsecond = 0
        if len(raw) > 17 and raw[17] == ".":
            microsecond = int((raw[18:] + "000000")[:6])
        return datetime(int(raw[:4]), int(raw[4:6]), int(raw[6:8]), int(raw[9:11]), int(raw[12:14]), int(raw[15:17]), microsecond, tzinfo=timezone.utc)
    raise ValueError(raw)


def floor_tf(ts: datetime, minutes: int) -> datetime:
    total = ts.hour * 60 + ts.minute
    floored = (total // minutes) * minutes
    return ts.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)


def classify_session(ts: datetime) -> str:
    t = ts.time()
    if time(0, 0) <= t < time(7, 0):
        return "asian"
    if time(7, 0) <= t < time(12, 0):
        return "london"
    if time(12, 0) <= t < time(17, 0):
        return "ny_overlap"
    return "off_session"


def load_bars(path: Path, minutes: int) -> list[Bar]:
    bars: list[Bar] = []
    gap_threshold = timedelta(minutes=GAP_MINUTES)
    current = None
    previous_ts = None
    segment_id = 0

    def flush():
        nonlocal current
        if current is None:
            return
        start, o, h, l, c, invalid, seg = current
        current = None
        if not invalid:
            bars.append(Bar(len(bars), seg, start, start + timedelta(minutes=minutes), o, h, l, c))

    with path.open(newline="") as handle:
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
            bucket = floor_tf(ts, minutes)
            gap = previous_ts is not None and ts - previous_ts > gap_threshold
            mid = (bid + ask) / 2
            if gap:
                if current is not None and current[0] == bucket:
                    current = (*current[:5], True, current[6])
                flush()
                segment_id += 1
            if current is not None and current[0] != bucket:
                flush()
            if current is None:
                current = (bucket, mid, mid, mid, mid, False, segment_id)
            else:
                start, o, h, l, _, invalid, seg = current
                current = (start, o, max(h, mid), min(l, mid), mid, invalid, seg)
            previous_ts = ts
    flush()
    return bars


def true_range(bar: Bar, prev: Optional[Bar]) -> float:
    if prev is None or prev.segment_id != bar.segment_id:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close))


def compute_atr(bars: list[Bar]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(bars)
    win: deque[float] = deque(maxlen=ATR_PERIOD)
    prev = None
    prev_seg = None
    for i, bar in enumerate(bars):
        if prev_seg is None or bar.segment_id != prev_seg:
            win.clear()
            prev = None
        win.append(true_range(bar, prev))
        if len(win) == ATR_PERIOD:
            out[i] = sum(win) / ATR_PERIOD
        prev, prev_seg = bar, bar.segment_id
    return out


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


def is_swing_high(bars: list[Bar], p: int) -> bool:
    if p - SWING_LEFT < 0 or p + SWING_RIGHT >= len(bars):
        return False
    b = bars[p]
    window = bars[p - SWING_LEFT : p + SWING_RIGHT + 1]
    return all(x.segment_id == b.segment_id for x in window) and all(b.high > x.high for j, x in enumerate(window) if j != SWING_LEFT)


def is_swing_low(bars: list[Bar], p: int) -> bool:
    if p - SWING_LEFT < 0 or p + SWING_RIGHT >= len(bars):
        return False
    b = bars[p]
    window = bars[p - SWING_LEFT : p + SWING_RIGHT + 1]
    return all(x.segment_id == b.segment_id for x in window) and all(b.low < x.low for j, x in enumerate(window) if j != SWING_LEFT)


def find_last_opposite(bars: list[Bar], i: int, direction: str) -> Optional[int]:
    for j in range(i - 1, -1, -1):
        if bars[j].segment_id != bars[i].segment_id:
            return None
        bearish = bars[j].close < bars[j].open
        bullish = bars[j].close > bars[j].open
        if direction == "bullish" and bearish:
            return j
        if direction == "bearish" and bullish:
            return j
    return None


def detect_signals(bars: list[Bar], atr: list[Optional[float]]) -> tuple[list[Signal], list[Signal], list[Signal]]:
    obs: list[Signal] = []
    fvgs: list[Signal] = []
    sweeps: list[Signal] = []
    latest_high = latest_low = None
    broken_highs, broken_lows, consumed_highs, consumed_lows = set(), set(), set(), set()
    for i, bar in enumerate(bars):
        p = i - SWING_RIGHT
        if p >= SWING_LEFT and bars[p].segment_id == bar.segment_id:
            if is_swing_high(bars, p):
                latest_high = Swing(p, i, bars[p].high)
            if is_swing_low(bars, p):
                latest_low = Swing(p, i, bars[p].low)
        a = atr[i]
        if a is None or a <= 0:
            continue
        # OB
        if latest_high and latest_high.index not in broken_highs and latest_high.confirmed_at_index <= i and bars[latest_high.index].segment_id == bar.segment_id and bar.high > latest_high.level:
            ob = find_last_opposite(bars, i, "bullish")
            if ob is not None and (bar.high - bars[ob].low) / a >= OB_DISPLACEMENT_ATR:
                obs.append(Signal("OB", len(obs)+1, "bullish", i, bars[ob].high, bars[ob].low, a, bar.end.year, classify_session(bar.end)))
                broken_highs.add(latest_high.index)
        if latest_low and latest_low.index not in broken_lows and latest_low.confirmed_at_index <= i and bars[latest_low.index].segment_id == bar.segment_id and bar.low < latest_low.level:
            ob = find_last_opposite(bars, i, "bearish")
            if ob is not None and (bars[ob].high - bar.low) / a >= OB_DISPLACEMENT_ATR:
                obs.append(Signal("OB", len(obs)+1, "bearish", i, bars[ob].high, bars[ob].low, a, bar.end.year, classify_session(bar.end)))
                broken_lows.add(latest_low.index)
        # FVG
        if i >= 2 and bars[i-2].segment_id == bars[i-1].segment_id == bar.segment_id:
            c1 = bars[i-2]
            if c1.high < bar.low and (bar.low - c1.high) / a >= FVG_GAP_ATR:
                fvgs.append(Signal("FVG", len(fvgs)+1, "bullish", i, bar.low, c1.high, a, bar.end.year, classify_session(bar.end)))
            if c1.low > bar.high and (c1.low - bar.high) / a >= FVG_GAP_ATR:
                fvgs.append(Signal("FVG", len(fvgs)+1, "bearish", i, c1.low, bar.high, a, bar.end.year, classify_session(bar.end)))
        # Sweep
        th = SWEEP_BREACH_ATR * a
        if latest_low and latest_low.index not in consumed_lows and latest_low.confirmed_at_index < i and bars[latest_low.index].segment_id == bar.segment_id and bar.low <= latest_low.level - th:
            consumed_lows.add(latest_low.index)
            for off in range(1, SWEEP_REJECT_BARS+1):
                j = i + off
                if j >= len(bars) or bars[j].segment_id != bar.segment_id:
                    break
                if bars[j].close > latest_low.level:
                    sweeps.append(Signal("Sweep", len(sweeps)+1, "bullish", j, latest_low.level, latest_low.level, a, bars[j].end.year, classify_session(bars[j].end)))
                    break
        if latest_high and latest_high.index not in consumed_highs and latest_high.confirmed_at_index < i and bars[latest_high.index].segment_id == bar.segment_id and bar.high >= latest_high.level + th:
            consumed_highs.add(latest_high.index)
            for off in range(1, SWEEP_REJECT_BARS+1):
                j = i + off
                if j >= len(bars) or bars[j].segment_id != bar.segment_id:
                    break
                if bars[j].close < latest_high.level:
                    sweeps.append(Signal("Sweep", len(sweeps)+1, "bearish", j, latest_high.level, latest_high.level, a, bars[j].end.year, classify_session(bars[j].end)))
                    break
    return obs, fvgs, sweeps


def dist(a: Signal, b: Signal) -> float:
    if a.low <= b.high and b.low <= a.high:
        return 0
    if a.high < b.low:
        return b.low - a.high
    return a.low - b.high


def cooc(s: Signal, z: Signal) -> bool:
    return abs(s.index - z.index) <= COOC_BARS and dist(s, z) <= COOC_ATR * s.atr and s.direction == z.direction


def choose_zone(s: Signal, obs: list[Signal], fvgs: list[Signal]) -> Optional[Signal]:
    obs_m = [z for z in obs if cooc(s, z)]
    if obs_m:
        return min(obs_m, key=lambda z: (abs(z.index-s.index), dist(s,z), z.signal_id))
    fvg_m = [z for z in fvgs if cooc(s, z)]
    if fvg_m:
        return min(fvg_m, key=lambda z: (abs(z.index-s.index), dist(s,z), z.signal_id))
    return None


def first_touch(bars: list[Bar], zone: Signal, after: int) -> Optional[int]:
    seg = bars[after].segment_id
    for i in range(after + 1, min(len(bars), after + ENTRY_EXPIRY_BARS + 1)):
        if bars[i].segment_id != seg:
            return None
        if bars[i].low <= zone.high and bars[i].high >= zone.low:
            return i
    return None


def bucket(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value <= hi:
        return "medium"
    return "high"


def build_setups(bars: list[Bar], atr: list[Optional[float]], sweeps: list[Signal], obs: list[Signal], fvgs: list[Signal]) -> tuple[list[Setup], Counter]:
    matched = [(s, choose_zone(s, obs, fvgs)) for s in sweeps]
    at = [s.atr for s, z in matched if z is not None]
    lo, hi = quantile(at, 1/3), quantile(at, 2/3)
    setups = []
    stats = Counter()
    for s, z in matched:
        if z is None:
            stats["no_zone"] += 1
            continue
        stats[f"zone_{z.kind}"] += 1
        established = max(s.index, z.index)
        t = first_touch(bars, z, established)
        if t is None:
            stats["expired"] += 1
            continue
        if t + REACTION_BARS >= len(bars) or any(bars[i].segment_id != bars[t].segment_id for i in range(t+1, t+REACTION_BARS+1)):
            stats["no_horizon"] += 1
            continue
        ea = atr[t]
        if ea is None or ea <= 0:
            stats["no_atr"] += 1
            continue
        news = any(bars[i].segment_id == bars[t].segment_id and (bars[i].high - bars[i].low) > 3 * ea for i in range(max(0,t-2), min(len(bars)-1,t+2)+1))
        setups.append(Setup(len(setups)+1, s, z, t, t+REACTION_BARS, ea, bucket(ea, lo, hi), news))
    return setups, stats


def spread_r(ts: datetime, atr: float, mode: SpreadMode) -> float:
    return IUX_SPREAD_USD_OZ[session_for_timestamp(ts)].value(mode) / atr


def run_ticks(path: Path, bars: list[Bar], setups: list[Setup]) -> list[Trade]:
    trades = [Trade(s) for s in setups if not s.excluded_news_proxy]
    trades.sort(key=lambda t: bars[t.setup.entry_bar_index].start)
    if not trades:
        return []
    first = min(bars[t.setup.entry_bar_index].start for t in trades)
    last = max(bars[t.setup.horizon_bar_index].end for t in trades)
    active, done = [], []
    p = 0
    with path.open(newline="") as handle:
        r = csv.reader(handle)
        h = next(r)
        ti, bi, ai = h.index("DateTime"), h.index("Bid"), h.index("Ask")
        for row in r:
            try:
                ts = parse_timestamp(row[ti]); bid=float(row[bi]); ask=float(row[ai])
            except Exception:
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            if ts < first:
                continue
            if ts > last:
                break
            mid = (bid+ask)/2
            while p < len(trades) and bars[trades[p].setup.entry_bar_index].start <= ts:
                active.append(trades[p]); p += 1
            keep = []
            for tr in active:
                if ts > bars[tr.setup.horizon_bar_index].end:
                    force_exit(tr, bars); done.append(tr); continue
                on_tick(tr, ts, mid)
                if tr.resolved:
                    done.append(tr)
                else:
                    keep.append(tr)
            active = keep
    for tr in active:
        force_exit(tr, bars); done.append(tr)
    return [t for t in done if t.entry_found]


def on_tick(t: Trade, ts: datetime, mid: float) -> None:
    if not t.entry_found:
        if t.setup.zone.low <= mid <= t.setup.zone.high:
            t.entry_found = True; t.entry_time = ts; t.entry_mid = mid
            t.net_median_r -= spread_r(ts, t.setup.entry_atr, SpreadMode.MEDIAN)/2
            t.net_p90_r -= spread_r(ts, t.setup.entry_atr, SpreadMode.P90)/2
        return
    val = ((mid - t.entry_mid) / t.setup.entry_atr) if t.setup.sweep.direction == "bullish" else ((t.entry_mid - mid) / t.setup.entry_atr)
    if val <= t.stop_r:
        close(t, ts, t.open_weight, t.stop_r); return
    for target, weight in SCALE_PLAN:
        if (target == 1 and t.hit_1r) or (target == 2 and t.hit_2r) or (target == 3 and t.hit_3r):
            continue
        if val >= target:
            close(t, ts, weight, target)
            if target == 1: t.hit_1r = True; t.stop_r = 0
            if target == 2: t.hit_2r = True
            if target == 3: t.hit_3r = True
            if t.resolved: return


def close(t: Trade, ts: datetime, weight: float, val: float) -> None:
    t.gross_r += weight * val
    t.net_median_r += weight * val - weight * spread_r(ts, t.setup.entry_atr, SpreadMode.MEDIAN)/2
    t.net_p90_r += weight * val - weight * spread_r(ts, t.setup.entry_atr, SpreadMode.P90)/2
    t.open_weight = max(0, t.open_weight-weight)
    if t.open_weight <= 1e-12: t.resolved = True


def force_exit(t: Trade, bars: list[Bar]) -> None:
    if not t.entry_found or t.resolved:
        return
    mid = bars[t.setup.horizon_bar_index].close
    val = ((mid - t.entry_mid) / t.setup.entry_atr) if t.setup.sweep.direction == "bullish" else ((t.entry_mid - mid) / t.setup.entry_atr)
    t.horizon_exit = True
    close(t, bars[t.setup.horizon_bar_index].end, t.open_weight, val)


def stats(vals: list[float]) -> tuple[int,float,float,float,float]:
    if not vals: return 0, math.nan, math.nan, math.nan, math.nan
    m=mean(vals); sd=pstdev(vals) if len(vals)>1 else 0; se=sd/math.sqrt(len(vals))
    return len(vals), m, m-1.96*se, m+1.96*se, sd


def max_loss(vals: list[float]) -> int:
    best=cur=0
    for v in vals:
        if v<0: cur+=1; best=max(best,cur)
        else: cur=0
    return best


def print_report(tf: str, minutes: int, bars: list[Bar], obs, fvgs, sweeps, setups, trades, stats_counter):
    vals=[t.net_median_r for t in trades]; gross=[t.gross_r for t in trades]; p90=[t.net_p90_r for t in trades]
    n,m,lo,hi,sd=stats(vals)
    cost_vals=[spread_r(t.entry_time or bars[t.setup.entry_bar_index].end, t.setup.entry_atr, SpreadMode.MEDIAN) for t in trades]
    p90_cost=[spread_r(t.entry_time or bars[t.setup.entry_bar_index].end, t.setup.entry_atr, SpreadMode.P90) for t in trades]
    print(f"\\nTIMEFRAME_RESULT,{tf}")
    print(f"bars={len(bars)},OB={len(obs)},FVG={len(fvgs)},Sweep={len(sweeps)}")
    print(f"aligned_zone_sweeps={len(sweeps)-stats_counter['no_zone']},no_zone={stats_counter['no_zone']},zone_OB={stats_counter['zone_OB']},zone_FVG_only={stats_counter['zone_FVG']}")
    print(f"entry_set={len(setups)},news_proxy_excluded={sum(1 for s in setups if s.excluded_news_proxy)},tick_trades={len(trades)}")
    print("cohort,n,gross_mean_R,net_median_R,net_p90_R,ci_low,ci_high,win_rate,unresolved,std_R,max_loss_streak,worst_R,costR_median_mean,costR_p90_mean")
    print(f"{tf},{n},{mean(gross) if gross else math.nan:.4f},{m:.4f},{mean(p90) if p90 else math.nan:.4f},{lo:.4f},{hi:.4f},{sum(1 for v in vals if v>0)/n if n else math.nan:.2%},{sum(1 for t in trades if t.horizon_exit)/n if n else math.nan:.2%},{sd:.4f},{max_loss(vals)},{min(vals) if vals else math.nan:.4f},{mean(cost_vals) if cost_vals else math.nan:.4f},{mean(p90_cost) if p90_cost else math.nan:.4f}")
    print("BY_ATR")
    print("bucket,n,net_mean,ci_low,ci_high")
    for b in ("low","medium","high"):
        vv=[t.net_median_r for t in trades if t.setup.atr_bucket==b]; nn,mm,ll,hh,_=stats(vv); print(f"{b},{nn},{mm:.4f},{ll:.4f},{hh:.4f}")
    print("BY_YEAR")
    print("year,n,net_mean,ci_low,ci_high")
    for y in range(2016,2027):
        vv=[t.net_median_r for t in trades if (t.entry_time.year if t.entry_time else bars[t.setup.entry_bar_index].end.year)==y]; nn,mm,ll,hh,_=stats(vv); print(f"{y},{nn},{mm:.4f},{ll:.4f},{hh:.4f}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--timeframe", choices=("H1","H4"), required=True)
    ap.add_argument("--ticks", type=Path, default=None)
    args=ap.parse_args()
    minutes = {"H1":60,"H4":240}[args.timeframe]
    tick_path=args.ticks or default_tick_path()
    print(f"Loading {args.timeframe} bars...", flush=True)
    bars=load_bars(tick_path, minutes)
    atr=compute_atr(bars)
    print("Detecting signals...", flush=True)
    obs,fvgs,sweeps=detect_signals(bars,atr)
    setups, sc=build_setups(bars,atr,sweeps,obs,fvgs)
    print("Running tick fills...", flush=True)
    trades=run_ticks(tick_path,bars,setups)
    print_report(args.timeframe, minutes, bars, obs, fvgs, sweeps, setups, trades, sc)


if __name__ == "__main__":
    main()
