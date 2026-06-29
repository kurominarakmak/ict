"""
Raw order-flow delta signal audit for XAUUSD.

This is not a strategy engine and uses no ML.

Delta approximation:
- Dukascopy spot ticks are quote updates with bid/ask and tick-count volume, not
  exchange prints with true traded volume.
- Tick rule uses mid-price changes: uptick = buy-initiated tick, downtick =
  sell-initiated tick, unchanged mid = neutral/ignored.
- delta_ratio = (buy_ticks - sell_ticks) / (buy_ticks + sell_ticks).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Optional


TIMEFRAME_MINUTES = 15
GAP_MINUTES = 30.0
ATR_PERIOD = 14
CVD_WINDOW = 20
FORWARD_HORIZONS = (1, 3, 5)
IUX_XAUUSD_ROUNDTRIP_SPREAD = 0.20


@dataclass
class DeltaBar:
    index: int
    segment_id: int
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    ticks: int
    buy_ticks: int
    sell_ticks: int
    neutral_ticks: int
    delta: int
    delta_ratio: float
    cvd20: int = 0
    atr14: Optional[float] = None
    session: str = ""


def default_tick_path() -> Path:
    matches = sorted(Path("data").glob("*XAUUSD*.csv"))
    if not matches:
        raise SystemExit("No XAUUSD tick CSV found under data/")
    return matches[0]


def parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    micro = 0
    if len(raw) > 17 and raw[17] == ".":
        micro = int((raw[18:] + "000000")[:6])
    return datetime(
        int(raw[:4]),
        int(raw[4:6]),
        int(raw[6:8]),
        int(raw[9:11]),
        int(raw[12:14]),
        int(raw[15:17]),
        micro,
        tzinfo=timezone.utc,
    )


def floor_m15(ts: datetime) -> datetime:
    total = ts.hour * 60 + ts.minute
    floored = (total // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
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


def true_range(bar: DeltaBar, prev: Optional[DeltaBar]) -> float:
    if prev is None or prev.segment_id != bar.segment_id:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close))


def load_delta_bars(path: Path) -> list[DeltaBar]:
    bars: list[DeltaBar] = []
    current = None
    previous_ts: Optional[datetime] = None
    previous_mid: Optional[float] = None
    segment_id = 0
    gap_threshold = timedelta(minutes=GAP_MINUTES)

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        (
            start,
            opn,
            high,
            low,
            close,
            ticks,
            buy,
            sell,
            neutral,
            invalid,
            segment,
        ) = current
        current = None
        if invalid:
            return
        denom = buy + sell
        delta = buy - sell
        ratio = delta / denom if denom else 0.0
        bars.append(
            DeltaBar(
                len(bars),
                segment,
                start,
                start + timedelta(minutes=TIMEFRAME_MINUTES),
                opn,
                high,
                low,
                close,
                ticks,
                buy,
                sell,
                neutral,
                delta,
                ratio,
                session=classify_session(start + timedelta(minutes=TIMEFRAME_MINUTES)),
            )
        )

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ti, bi, ai = header.index("DateTime"), header.index("Bid"), header.index("Ask")
        for row in reader:
            try:
                ts = parse_ts(row[ti])
                bid = float(row[bi])
                ask = float(row[ai])
            except Exception:
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            bucket = floor_m15(ts)
            gap = previous_ts is not None and ts - previous_ts > gap_threshold
            if gap:
                if current is not None and current[0] == bucket:
                    current = (*current[:9], True, current[10])
                flush()
                segment_id += 1
                previous_mid = None
            if current is not None and current[0] != bucket:
                flush()
            if previous_mid is None:
                side = 0
            elif mid > previous_mid:
                side = 1
            elif mid < previous_mid:
                side = -1
            else:
                side = 0
            if current is None:
                current = (
                    bucket,
                    mid,
                    mid,
                    mid,
                    mid,
                    1,
                    1 if side > 0 else 0,
                    1 if side < 0 else 0,
                    1 if side == 0 else 0,
                    False,
                    segment_id,
                )
            else:
                start, opn, high, low, _, ticks, buy, sell, neutral, invalid, segment = current
                current = (
                    start,
                    opn,
                    max(high, mid),
                    min(low, mid),
                    mid,
                    ticks + 1,
                    buy + (1 if side > 0 else 0),
                    sell + (1 if side < 0 else 0),
                    neutral + (1 if side == 0 else 0),
                    invalid,
                    segment,
                )
            previous_ts = ts
            previous_mid = mid
    flush()
    add_indicators(bars)
    return bars


def add_indicators(bars: list[DeltaBar]) -> None:
    tr_window: deque[float] = deque(maxlen=ATR_PERIOD)
    cvd_window: deque[int] = deque(maxlen=CVD_WINDOW)
    prev: Optional[DeltaBar] = None
    prev_segment: Optional[int] = None
    for bar in bars:
        if prev_segment is None or bar.segment_id != prev_segment:
            tr_window.clear()
            cvd_window.clear()
            prev = None
        tr_window.append(true_range(bar, prev))
        cvd_window.append(bar.delta)
        if len(tr_window) == ATR_PERIOD:
            bar.atr14 = sum(tr_window) / ATR_PERIOD
        bar.cvd20 = sum(cvd_window)
        prev = bar
        prev_segment = bar.segment_id


def valid_rows(bars: list[DeltaBar], horizon: int) -> list[tuple[DeltaBar, float]]:
    out: list[tuple[DeltaBar, float]] = []
    for i, bar in enumerate(bars):
        j = i + horizon
        if j >= len(bars):
            continue
        if bars[j].segment_id != bar.segment_id:
            continue
        out.append((bar, bars[j].close - bar.close))
    return out


def decile_table(bars: list[DeltaBar]) -> list[dict[str, float]]:
    base = valid_rows(bars, max(FORWARD_HORIZONS))
    ordered = sorted([row[0] for row in base], key=lambda b: b.delta_ratio)
    n = len(ordered)
    decile_by_index: dict[int, int] = {}
    for rank, bar in enumerate(ordered):
        decile_by_index[bar.index] = min(9, int(rank * 10 / n)) + 1
    rows = []
    for decile in range(1, 11):
        members = [bar for bar in ordered if decile_by_index[bar.index] == decile]
        item: dict[str, float] = {
            "decile": decile,
            "n": len(members),
            "mean_delta_ratio": mean([b.delta_ratio for b in members]),
            "median_delta_ratio": median([b.delta_ratio for b in members]),
            "mean_ticks": mean([b.ticks for b in members]),
        }
        for horizon in FORWARD_HORIZONS:
            rets = []
            for bar in members:
                j = bar.index + horizon
                if j < len(bars) and bars[j].segment_id == bar.segment_id:
                    rets.append(bars[j].close - bar.close)
            item[f"ret_{horizon}"] = mean(rets) if rets else math.nan
        rows.append(item)
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def ic_with_ci(rows: list[tuple[DeltaBar, float]]) -> tuple[int, float, float, float]:
    xs = [bar.delta_ratio for bar, _ in rows]
    ys = [ret for _, ret in rows]
    r = pearson(xs, ys)
    n = len(xs)
    if n <= 3 or not math.isfinite(r) or abs(r) >= 1:
        return n, r, math.nan, math.nan
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    lo = math.tanh(z - 1.96 * se)
    hi = math.tanh(z + 1.96 * se)
    return n, r, lo, hi


def atr_bucket(bar: DeltaBar, cuts: tuple[float, float]) -> str:
    if bar.atr14 is None:
        return "missing"
    if bar.atr14 <= cuts[0]:
        return "low"
    if bar.atr14 <= cuts[1]:
        return "medium"
    return "high"


def grouped_signal(rows: list[tuple[DeltaBar, float]]) -> tuple[int, float, float, float, float, float]:
    if len(rows) < 20:
        return len(rows), math.nan, math.nan, math.nan, math.nan, math.nan
    ordered = sorted(rows, key=lambda x: x[0].delta_ratio)
    k = max(1, len(ordered) // 10)
    bottom = [ret for _, ret in ordered[:k]]
    top = [ret for _, ret in ordered[-k:]]
    _, ic, lo, hi = ic_with_ci(rows)
    return len(rows), mean(bottom), mean(top), mean(top) - mean(bottom), ic, lo if math.isfinite(lo) else math.nan


def print_deciles(rows: list[dict[str, float]]) -> None:
    print("\nDELTA_RATIO_DECILES_FORWARD_RETURNS_USD_PER_OZ")
    print("decile,n,mean_delta_ratio,median_delta_ratio,mean_ticks,ret_1bar,ret_3bar,ret_5bar")
    for row in rows:
        print(
            f"{row['decile']},{row['n']},{row['mean_delta_ratio']:.6f},{row['median_delta_ratio']:.6f},"
            f"{row['mean_ticks']:.1f},{row['ret_1']:.6f},{row['ret_3']:.6f},{row['ret_5']:.6f}"
        )
    for horizon in FORWARD_HORIZONS:
        spread = rows[-1][f"ret_{horizon}"] - rows[0][f"ret_{horizon}"]
        print(f"top_minus_bottom_decile_ret_{horizon}bar={spread:.6f}")


def print_ics(bars: list[DeltaBar]) -> None:
    print("\nINFORMATION_COEFFICIENT_DELTA_RATIO")
    print("horizon,n,ic,ci_low,ci_high")
    for horizon in FORWARD_HORIZONS:
        n, ic, lo, hi = ic_with_ci(valid_rows(bars, horizon))
        print(f"{horizon},{n},{ic:.6f},{lo:.6f},{hi:.6f}")


def print_breakdowns(bars: list[DeltaBar]) -> None:
    atrs = [b.atr14 for b in bars if b.atr14 is not None]
    cuts = (quantile(atrs, 1 / 3), quantile(atrs, 2 / 3))
    print("\nBREAKDOWN_TOP_MINUS_BOTTOM_AND_IC")
    print("group_type,group,horizon,n,bottom_decile_ret,top_decile_ret,top_minus_bottom,ic")
    for horizon in FORWARD_HORIZONS:
        rows = valid_rows(bars, horizon)
        for session in ("asian", "london", "ny_overlap", "off_session"):
            group = [(b, r) for b, r in rows if b.session == session]
            n, bottom, top, spread, ic, _ = grouped_signal(group)
            print(f"session,{session},{horizon},{n},{bottom:.6f},{top:.6f},{spread:.6f},{ic:.6f}")
        for bucket in ("low", "medium", "high"):
            group = [(b, r) for b, r in rows if atr_bucket(b, cuts) == bucket]
            n, bottom, top, spread, ic, _ = grouped_signal(group)
            print(f"atr,{bucket},{horizon},{n},{bottom:.6f},{top:.6f},{spread:.6f},{ic:.6f}")


def cost_check(bars: list[DeltaBar]) -> None:
    print("\nTOP_BOTTOM_DECILE_COST_CHECK_USD_PER_OZ")
    print("horizon,n,gross_mean,net_mean,ci_low,ci_high,win_rate")
    table = decile_table(bars)
    # Use global decile cutoffs from full max-horizon eligible sample.
    eligible = sorted([b for b, _ in valid_rows(bars, max(FORWARD_HORIZONS))], key=lambda b: b.delta_ratio)
    n = len(eligible)
    bottom_ids = {b.index for b in eligible[: n // 10]}
    top_ids = {b.index for b in eligible[-(n // 10):]}
    for horizon in FORWARD_HORIZONS:
        vals = []
        for bar, ret in valid_rows(bars, horizon):
            if bar.index in top_ids:
                vals.append(ret - IUX_XAUUSD_ROUNDTRIP_SPREAD)
            elif bar.index in bottom_ids:
                vals.append((-ret) - IUX_XAUUSD_ROUNDTRIP_SPREAD)
        gross = []
        for bar, ret in valid_rows(bars, horizon):
            if bar.index in top_ids:
                gross.append(ret)
            elif bar.index in bottom_ids:
                gross.append(-ret)
        if not vals:
            continue
        m = mean(vals)
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        se = sd / math.sqrt(len(vals))
        print(f"{horizon},{len(vals)},{mean(gross):.6f},{m:.6f},{m - 1.96 * se:.6f},{m + 1.96 * se:.6f},{sum(v > 0 for v in vals) / len(vals):.2%}")


def quantile(vals: list[float], q: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None)
    args = parser.parse_args()
    path = args.ticks or default_tick_path()
    print("Building M15 tick-rule delta bars...", flush=True)
    bars = load_delta_bars(path)
    print("\nDELTA_AUDIT_CONTEXT")
    print(f"tick_file={path}")
    print(f"bars={len(bars)}")
    print(f"date_range={bars[0].start:%Y-%m-%d %H:%M:%S} to {bars[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print("approximation=tick-rule on Dukascopy spot quote updates; buy/sell are uptick/downtick counts, not true traded futures volume")
    print(f"cvd_window_bars={CVD_WINDOW}")
    print_deciles(decile_table(bars))
    print_ics(bars)
    print_breakdowns(bars)
    cost_check(bars)


if __name__ == "__main__":
    main()
