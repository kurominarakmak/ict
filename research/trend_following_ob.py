"""
Trend-following Order Block trial.

New declared hypothesis:
- FVG-present OBs only.
- Mechanical swing-structure trend filter.
- Retest within 33 bars.
- Entry at OB edge in trend direction.
- Structure SL at OB wick.
- TP1 at nearest prior opposing confirmed swing level.
- Two declared exits: structure-break and trailing swing stop.

No parameter sweep.
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
from statistics import mean, median, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from cost_model import IUX_SPREAD_USD_OZ, SpreadMode, session_for_timestamp

ATR_PERIOD = 14
SWING_LEFT = 10
SWING_RIGHT = 10
DISPLACEMENT_ATR = 2.0
RETEST_BARS = 33
MAX_HOLD_BARS = 100
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25.0


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
    kind: str
    index: int
    confirmed_at: int
    level: float


@dataclass(frozen=True)
class OB:
    ob_id: int
    direction: str
    high: float
    low: float
    creation_index: int
    ob_index: int
    atr: float
    displacement_atr: float
    fvg_present: bool
    session: str
    year: int


@dataclass(frozen=True)
class Setup:
    setup_id: int
    cohort: str
    ob: OB
    entry_bar: int
    entry_atr: float
    atr_bucket: str
    state: str
    tp1_level: float
    max_exit_bar: int
    news_proxy: bool


@dataclass
class Trade:
    setup: Setup
    variant: str
    active: bool = False
    entered: bool = False
    entry_time: Optional[datetime] = None
    entry_mid: Optional[float] = None
    stop_price: Optional[float] = None
    risk: Optional[float] = None
    open_weight: float = 1.0
    gross_r: float = 0.0
    net_median_r: float = 0.0
    net_p90_r: float = 0.0
    tp1_hit: bool = False
    exited: bool = False
    gap_skipped: bool = False
    horizon_exit: bool = False
    next_bar_update: int = 0


def default_tick_path() -> Path:
    matches = sorted(Path("data").glob("*XAUUSD*.csv"))
    if not matches:
        raise SystemExit("No XAUUSD CSV found under data/")
    return matches[0]


def parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    micro = 0
    if len(raw) > 17 and raw[17] == ".":
        micro = int((raw[18:] + "000000")[:6])
    return datetime(int(raw[:4]), int(raw[4:6]), int(raw[6:8]), int(raw[9:11]), int(raw[12:14]), int(raw[15:17]), micro, tzinfo=timezone.utc)


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
    current = None
    previous_ts = None
    segment = 0
    gap_threshold = timedelta(minutes=30)

    def flush() -> None:
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
                ts = parse_ts(row[ti]); bid = float(row[bi]); ask = float(row[ai])
            except Exception:
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            bucket = floor_tf(ts, minutes)
            mid = (bid + ask) / 2
            gap = previous_ts is not None and ts - previous_ts > gap_threshold
            if gap:
                if current is not None and current[0] == bucket:
                    current = (*current[:5], True, current[6])
                flush(); segment += 1
            if current is not None and current[0] != bucket:
                flush()
            if current is None:
                current = (bucket, mid, mid, mid, mid, False, segment)
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
    prev = None; prev_seg = None
    for i, bar in enumerate(bars):
        if prev_seg is None or bar.segment_id != prev_seg:
            win.clear(); prev = None
        win.append(true_range(bar, prev))
        if len(win) == ATR_PERIOD:
            out[i] = sum(win) / ATR_PERIOD
        prev = bar; prev_seg = bar.segment_id
    return out


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
        trq.append(true_range(b, p)); pdmq.append(pdm); ndmq.append(ndm)
        if len(trq) == ADX_PERIOD and sum(trq) > 0:
            pdi = 100 * sum(pdmq) / sum(trq)
            ndi = 100 * sum(ndmq) / sum(trq)
            dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0
            dxq.append(dx)
            if len(dxq) == ADX_PERIOD:
                out[i] = sum(dxq) / ADX_PERIOD
    return out


def is_high(bars: list[Bar], p: int) -> bool:
    if p - SWING_LEFT < 0 or p + SWING_RIGHT >= len(bars): return False
    b = bars[p]; w = bars[p-SWING_LEFT:p+SWING_RIGHT+1]
    return all(x.segment_id == b.segment_id for x in w) and all(b.high > x.high for j,x in enumerate(w) if j != SWING_LEFT)


def is_low(bars: list[Bar], p: int) -> bool:
    if p - SWING_LEFT < 0 or p + SWING_RIGHT >= len(bars): return False
    b = bars[p]; w = bars[p-SWING_LEFT:p+SWING_RIGHT+1]
    return all(x.segment_id == b.segment_id for x in w) and all(b.low < x.low for j,x in enumerate(w) if j != SWING_LEFT)


def confirmed_swings(bars: list[Bar]) -> tuple[list[Swing], list[list[Swing]]]:
    events: list[Swing] = []
    available: list[list[Swing]] = [[] for _ in bars]
    cur: list[Swing] = []
    for i, bar in enumerate(bars):
        p = i - SWING_RIGHT
        if p >= SWING_LEFT and bars[p].segment_id == bar.segment_id:
            if is_high(bars, p): cur.append(Swing("high", p, i, bars[p].high)); events.append(cur[-1])
            if is_low(bars, p): cur.append(Swing("low", p, i, bars[p].low)); events.append(cur[-1])
        available[i] = list(cur)
    return events, available


def trend_at(swings: list[Swing], break_dir: dict[int, str]) -> str:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "none"
    hh_hl = highs[-1].level > highs[-2].level and lows[-1].level > lows[-2].level
    lh_ll = highs[-1].level < highs[-2].level and lows[-1].level < lows[-2].level
    last_break = None
    for idx in sorted(break_dir):
        last_break = break_dir[idx]
    if hh_hl and last_break == "bullish": return "bullish"
    if lh_ll and last_break == "bearish": return "bearish"
    return "none"


def find_last_opposite(bars: list[Bar], i: int, direction: str) -> Optional[int]:
    for j in range(i-1, -1, -1):
        if bars[j].segment_id != bars[i].segment_id: return None
        if direction == "bullish" and bars[j].close < bars[j].open: return j
        if direction == "bearish" and bars[j].close > bars[j].open: return j
    return None


def has_fvg(bars: list[Bar], ob: int, imp: int, direction: str) -> bool:
    start = max(ob, imp - 2)
    for i in range(start + 2, imp + 1):
        if bars[i-2].segment_id != bars[i].segment_id: continue
        if direction == "bullish" and bars[i-2].high < bars[i].low: return True
        if direction == "bearish" and bars[i-2].low > bars[i].high: return True
    return False


def zone_overlap_ratio(new_ob: OB, old_ob: OB) -> float:
    overlap = max(0.0, min(new_ob.high, old_ob.high) - max(new_ob.low, old_ob.low))
    height = max(new_ob.high - new_ob.low, 1e-12)
    return overlap / height


def is_duplicate_ob(new_ob: OB, obs: list[OB], mitigated: set[int]) -> bool:
    for old in obs:
        if old.ob_id in mitigated or old.direction != new_ob.direction:
            continue
        if zone_overlap_ratio(new_ob, old) > 0.50:
            return True
    return False


def detect_obs(bars: list[Bar], atr: list[Optional[float]], avail_swings: list[list[Swing]]) -> tuple[list[OB], dict[int,str], int]:
    obs: list[OB] = []
    mitigated: set[int] = set()
    raw_count = 0
    broken_h=set(); broken_l=set(); break_dir={}
    latest_h=latest_l=None
    for i,b in enumerate(bars):
        for old in obs:
            if old.ob_id in mitigated or i <= old.creation_index or bars[old.creation_index].segment_id != b.segment_id:
                continue
            if b.low <= old.high and b.high >= old.low:
                mitigated.add(old.ob_id)
        for s in avail_swings[i]:
            if s.confirmed_at == i:
                if s.kind=="high": latest_h=s
                else: latest_l=s
        a=atr[i]
        if a is None or a<=0: continue
        if latest_h and latest_h.index not in broken_h and bars[latest_h.index].segment_id==b.segment_id and b.high>latest_h.level:
            break_dir[i]="bullish"; ob=find_last_opposite(bars,i,"bullish")
            if ob is not None and (b.high-bars[ob].low)/a>=DISPLACEMENT_ATR:
                raw_count += 1
                candidate = OB(len(obs)+1,"bullish",bars[ob].high,bars[ob].low,i,ob,a,(b.high-bars[ob].low)/a,has_fvg(bars,ob,i,"bullish"),classify_session(b.end),b.end.year)
                if not is_duplicate_ob(candidate, obs, mitigated):
                    obs.append(candidate)
                broken_h.add(latest_h.index)
        if latest_l and latest_l.index not in broken_l and bars[latest_l.index].segment_id==b.segment_id and b.low<latest_l.level:
            break_dir[i]="bearish"; ob=find_last_opposite(bars,i,"bearish")
            if ob is not None and (bars[ob].high-b.low)/a>=DISPLACEMENT_ATR:
                raw_count += 1
                candidate = OB(len(obs)+1,"bearish",bars[ob].high,bars[ob].low,i,ob,a,(bars[ob].high-b.low)/a,has_fvg(bars,ob,i,"bearish"),classify_session(b.end),b.end.year)
                if not is_duplicate_ob(candidate, obs, mitigated):
                    obs.append(candidate)
                broken_l.add(latest_l.index)
    return obs, break_dir, raw_count


def quantile(vals: list[float], q: float) -> float:
    if not vals: return math.nan
    s=sorted(vals); pos=(len(s)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return s[lo] if lo==hi else s[lo]+(s[hi]-s[lo])*(pos-lo)


def build_setups(bars, atr, adx, swings_avail, obs, break_dir, require_trend=True, cohort="trend") -> list[Setup]:
    candidates=[]
    at=[o.atr for o in obs if o.fvg_present]
    lo,hi=quantile(at,1/3),quantile(at,2/3)
    for ob in obs:
        if not ob.fvg_present: continue
        tr=trend_at([s for s in swings_avail[ob.creation_index] if s.confirmed_at <= ob.creation_index], {k:v for k,v in break_dir.items() if k<=ob.creation_index})
        passes = tr == ob.direction
        if require_trend and not passes: continue
        if not require_trend and passes: continue
        touch=None
        for i in range(ob.creation_index+1, min(len(bars), ob.creation_index+RETEST_BARS+1)):
            if bars[i].segment_id != bars[ob.creation_index].segment_id: break
            if ob.direction=="bullish" and bars[i].low <= ob.high: touch=i; break
            if ob.direction=="bearish" and bars[i].high >= ob.low: touch=i; break
        if touch is None or touch+MAX_HOLD_BARS>=len(bars): continue
        ea=atr[touch]
        if ea is None or ea<=0: continue
        sw=[s for s in swings_avail[touch] if s.confirmed_at <= touch]
        if ob.direction=="bullish":
            levels=[s.level for s in sw if s.kind=="high" and s.level>ob.high]
            if not levels: continue
            tp=min(levels)
        else:
            levels=[s.level for s in sw if s.kind=="low" and s.level<ob.low]
            if not levels: continue
            tp=max(levels)
        news=any(bars[j].segment_id==bars[touch].segment_id and (bars[j].high-bars[j].low)>3*ea for j in range(max(0,touch-2),min(len(bars)-1,touch+2)+1))
        state="trending" if (adx[touch] or 0)>=ADX_TREND_THRESHOLD else "ranging"
        bucket="low" if ea<=lo else ("medium" if ea<=hi else "high")
        candidates.append(Setup(len(candidates)+1,cohort,ob,touch,ea,bucket,state,tp,touch+MAX_HOLD_BARS,news))
    return candidates


def spread_r(ts, risk, mode):
    return IUX_SPREAD_USD_OZ[session_for_timestamp(ts)].value(mode) / risk


def run_ticks(path: Path, bars: list[Bar], swings_avail: list[list[Swing]], setups: list[Setup], variant: str) -> list[Trade]:
    trades=[Trade(s,variant,next_bar_update=s.entry_bar) for s in setups if not s.news_proxy]
    trades.sort(key=lambda t: bars[t.setup.entry_bar].start)
    if not trades: return []
    first=min(bars[t.setup.entry_bar].start for t in trades); last=max(bars[t.setup.max_exit_bar].end for t in trades)
    active=[]; done=[]; p=0
    with path.open(newline="") as h:
        r=csv.reader(h); head=next(r); ti,bi,ai=head.index("DateTime"),head.index("Bid"),head.index("Ask")
        for row in r:
            try: ts=parse_ts(row[ti]); bid=float(row[bi]); ask=float(row[ai])
            except Exception: continue
            if ask<=bid or bid<=0 or ask<=0: continue
            if ts<first: continue
            if ts>last: break
            mid=(bid+ask)/2
            while p<len(trades) and bars[trades[p].setup.entry_bar].start<=ts:
                active.append(trades[p]); p+=1
            keep=[]
            for t in active:
                update_bar_events(t,bars,swings_avail,ts,mid)
                if not t.entered:
                    if (t.setup.ob.direction=="bullish" and mid<=t.setup.ob.high) or (t.setup.ob.direction=="bearish" and mid>=t.setup.ob.low):
                        enter(t,ts,mid)
                if t.entered and not t.exited:
                    on_tick(t,ts,mid)
                if t.exited: done.append(t)
                else: keep.append(t)
            active=keep
    for t in active:
        if t.entered and not t.exited:
            exit_weight(t,bars[t.setup.max_exit_bar].end,bars[t.setup.max_exit_bar].close,t.open_weight,price_to_r(t,bars[t.setup.max_exit_bar].close)); t.horizon_exit=True
            done.append(t)
    return done


def run_ticks_many(path: Path, bars: list[Bar], swings_avail: list[list[Swing]], cohorts: list[tuple[str, list[Setup], str]]) -> tuple[dict[str, list[Trade]], dict[str, int]]:
    trades: list[Trade] = []
    labels: dict[int, str] = {}
    for label, setups, variant in cohorts:
        for setup in setups:
            if setup.news_proxy:
                continue
            trade = Trade(setup, variant, next_bar_update=setup.entry_bar)
            labels[id(trade)] = label
            trades.append(trade)
    trades.sort(key=lambda t: bars[t.setup.entry_bar].start)
    out: dict[str, list[Trade]] = {label: [] for label, _, _ in cohorts}
    gap_skips: dict[str, set[int]] = {label: set() for label, _, _ in cohorts}
    if not trades:
        return out, {label: 0 for label in out}

    first = min(bars[t.setup.entry_bar].start for t in trades)
    last = max(bars[t.setup.max_exit_bar].end for t in trades)
    active: list[Trade] = []
    p = 0
    with path.open(newline="") as h:
        r = csv.reader(h)
        head = next(r)
        ti, bi, ai = head.index("DateTime"), head.index("Bid"), head.index("Ask")
        for row in r:
            try:
                ts = parse_ts(row[ti]); bid = float(row[bi]); ask = float(row[ai])
            except Exception:
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            if ts < first:
                continue
            if ts > last:
                break
            mid = (bid + ask) / 2
            while p < len(trades) and bars[trades[p].setup.entry_bar].start <= ts:
                active.append(trades[p]); p += 1
            keep: list[Trade] = []
            for trade in active:
                if not trade.entered:
                    if (
                        (trade.setup.ob.direction == "bullish" and mid <= trade.setup.ob.high)
                        or (trade.setup.ob.direction == "bearish" and mid >= trade.setup.ob.low)
                    ):
                        enter(trade, ts, mid)
                        if trade.gap_skipped:
                            gap_skips[labels[id(trade)]].add(trade.setup.setup_id)
                if trade.entered and not trade.exited:
                    on_tick(trade, ts, mid)
                if trade.entered and not trade.exited:
                    update_bar_events(trade, bars, swings_avail, ts, mid)
                if trade.exited:
                    if trade.entered:
                        out[labels[id(trade)]].append(trade)
                else:
                    keep.append(trade)
            active = keep

    for trade in active:
        if trade.entered and not trade.exited:
            exit_weight(
                trade,
                bars[trade.setup.max_exit_bar].end,
                bars[trade.setup.max_exit_bar].close,
                trade.open_weight,
                price_to_r(trade, bars[trade.setup.max_exit_bar].close),
            )
            trade.horizon_exit = True
            out[labels[id(trade)]].append(trade)
    return out, {label: len(ids) for label, ids in gap_skips.items()}


def enter(t: Trade, ts, mid):
    if t.setup.ob.direction=="bullish":
        if mid <= t.setup.ob.low:
            t.gap_skipped=True; t.exited=True; return
        entry = t.setup.ob.high
        t.stop_price=t.setup.ob.low; t.risk=entry-t.stop_price
    else:
        if mid >= t.setup.ob.high:
            t.gap_skipped=True; t.exited=True; return
        entry = t.setup.ob.low
        t.stop_price=t.setup.ob.high; t.risk=t.stop_price-entry
    t.entered=True; t.entry_time=ts; t.entry_mid=entry
    if t.risk is None or t.risk<=0: t.exited=True; return
    t.net_median_r -= spread_r(ts,t.risk,SpreadMode.MEDIAN)/2
    t.net_p90_r -= spread_r(ts,t.risk,SpreadMode.P90)/2


def price_to_r(t: Trade, price: float) -> float:
    if t.risk is None or t.entry_mid is None: return 0
    return (price-t.entry_mid)/t.risk if t.setup.ob.direction=="bullish" else (t.entry_mid-price)/t.risk


def exit_weight(t: Trade, ts, price, weight, r_value):
    if t.risk is None: return
    t.gross_r += weight*r_value
    t.net_median_r += weight*r_value - weight*spread_r(ts,t.risk,SpreadMode.MEDIAN)/2
    t.net_p90_r += weight*r_value - weight*spread_r(ts,t.risk,SpreadMode.P90)/2
    t.open_weight=max(0,t.open_weight-weight)
    if t.open_weight<=1e-12: t.exited=True


def on_tick(t: Trade, ts, mid):
    assert t.stop_price is not None and t.risk is not None
    if (t.setup.ob.direction=="bullish" and mid<=t.stop_price) or (t.setup.ob.direction=="bearish" and mid>=t.stop_price):
        exit_weight(t,ts,t.stop_price,t.open_weight,price_to_r(t,t.stop_price)); return
    if not t.tp1_hit:
        if (t.setup.ob.direction=="bullish" and mid>=t.setup.tp1_level) or (t.setup.ob.direction=="bearish" and mid<=t.setup.tp1_level):
            exit_weight(t,ts,t.setup.tp1_level,0.25,price_to_r(t,t.setup.tp1_level))
            t.tp1_hit=True; t.stop_price=t.entry_mid


def update_bar_events(t: Trade, bars, swings_avail, ts, mid):
    if not t.entered or t.exited: return
    while t.next_bar_update <= t.setup.max_exit_bar and bars[t.next_bar_update].end <= ts:
        i=t.next_bar_update; b=bars[i]
        if t.variant=="structure":
            sw=[s for s in swings_avail[i] if s.confirmed_at<=i]
            lows=[s.level for s in sw if s.kind=="low"]
            highs=[s.level for s in sw if s.kind=="high"]
            if t.setup.ob.direction=="bullish" and lows and b.close < lows[-1]:
                exit_weight(t,ts,mid,t.open_weight,price_to_r(t,mid)); return
            if t.setup.ob.direction=="bearish" and highs and b.close > highs[-1]:
                exit_weight(t,ts,mid,t.open_weight,price_to_r(t,mid)); return
        if t.variant=="trailing" and t.tp1_hit:
            for s in swings_avail[i]:
                if s.confirmed_at==i:
                    if t.setup.ob.direction=="bullish" and s.kind=="low" and t.stop_price is not None:
                        t.stop_price=max(t.stop_price,s.level)
                    if t.setup.ob.direction=="bearish" and s.kind=="high" and t.stop_price is not None:
                        t.stop_price=min(t.stop_price,s.level)
        t.next_bar_update+=1


def summarize(trades):
    vals=[t.net_median_r for t in trades]; gross=[t.gross_r for t in trades]; p90=[t.net_p90_r for t in trades]
    if not vals: return {}
    m=mean(vals); sd=pstdev(vals) if len(vals)>1 else 0; se=sd/math.sqrt(len(vals))
    costs=[t.gross_r-t.net_median_r for t in trades]
    return dict(n=len(vals),gross=mean(gross),net=m,p90=mean(p90),lo=m-1.96*se,hi=m+1.96*se,win=sum(v>0 for v in vals)/len(vals),std=sd,worst=min(vals),gross_worst=min(gross),gross_below_minus_1=sum(v < -1.000001 for v in gross),maxloss=maxloss(vals),rr=mean([t.gross_r for t in trades]),cost=mean(costs),max_cost=max(costs))


def maxloss(vals):
    best=cur=0
    for v in vals:
        if v<0: cur+=1; best=max(best,cur)
        else: cur=0
    return best


def print_table(title,trades):
    print("\\n"+title); print("group,n,gross,net,p90,ci_low,ci_high,win,std,max_loss,worst_net,worst_gross,gross_below_-1_count,mean_cost_R,max_cost_R")
    for name,rows in trades:
        s=summarize(rows)
        if not s: print(f"{name},0,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a")
        else: print(f"{name},{s['n']},{s['gross']:.4f},{s['net']:.4f},{s['p90']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['win']:.2%},{s['std']:.4f},{s['maxloss']},{s['worst']:.4f},{s['gross_worst']:.4f},{s['gross_below_minus_1']},{s['cost']:.4f},{s['max_cost']:.4f}")


def run(timeframe):
    minutes=15 if timeframe=="M15" else 60
    path=default_tick_path()
    print(f"Loading {timeframe} bars...",flush=True); bars=load_bars(path,minutes)
    atr=compute_atr(bars); adx=compute_adx(bars); events,avail=confirmed_swings(bars)
    obs,break_dir,raw_ob_count=detect_obs(bars,atr,avail)
    trend_setups=build_setups(bars,atr,adx,avail,obs,break_dir,True,"trend")
    base_pool=build_setups(bars,atr,adx,avail,obs,break_dir,False,"baseline")
    # deterministic nearest prior baseline matched direction/session/ATR bucket
    selected=[]; used=set()
    for s in sorted(trend_setups,key=lambda x:x.entry_bar):
        c=[b for b in base_pool if b.setup_id not in used and b.entry_bar<s.entry_bar and b.ob.direction==s.ob.direction and b.ob.session==s.ob.session and b.atr_bucket==s.atr_bucket]
        if c:
            ch=max(c,key=lambda x:x.entry_bar); used.add(ch.setup_id); selected.append(ch)
    print("Running ticks...",flush=True)
    results,gap_skips=run_ticks_many(path,bars,avail,[
        ("trend_structure",trend_setups,"structure"),
        ("trend_trailing",trend_setups,"trailing"),
        ("baseline_structure",selected,"structure"),
        ("baseline_trailing",selected,"trailing"),
    ])
    tr_struct=results["trend_structure"]
    tr_trail=results["trend_trailing"]
    bl_struct=results["baseline_structure"]
    bl_trail=results["baseline_trailing"]
    print(f"\\nTREND_FOLLOWING_OB_CONTEXT,{timeframe}")
    print(f"bars={len(bars)},OB_raw_before_dedup={raw_ob_count},OB_after_dedup={len(obs)},FVG_OB={sum(o.fvg_present for o in obs)},trend_setups={len(trend_setups)},baseline_matched={len(selected)}")
    print(f"news_excluded_trend={sum(s.news_proxy for s in trend_setups)}")
    print(f"gap_skipped_trend={gap_skips['trend_structure']},gap_skipped_baseline={gap_skips['baseline_structure']}")
    print_table("OVERALL", [("trend_structure",tr_struct),("trend_trailing",tr_trail),("baseline_structure",bl_struct),("baseline_trailing",bl_trail)])
    for variant, trs in [("structure",tr_struct),("trailing",tr_trail)]:
        print_table(f"BY_STATE_{variant}", [(st,[t for t in trs if t.setup.state==st]) for st in ("trending","ranging")])
        print_table(f"BY_ATR_{variant}", [(b,[t for t in trs if t.setup.atr_bucket==b]) for b in ("low","medium","high")])
        print_table(f"BY_YEAR_{variant}", [(str(y),[t for t in trs if (t.entry_time.year if t.entry_time else bars[t.setup.entry_bar].end.year)==y]) for y in range(2016,2027)])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--timeframe",choices=("M15","H1"),required=True); args=ap.parse_args(); run(args.timeframe)


if __name__=="__main__": main()
