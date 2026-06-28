"""
Order Block detector and locked Phase B evaluator.

Phase A detection:
- Resample Dukascopy bid/ask ticks to completed M15 mid-price bars.
- Keep resampling gap-aware by segmenting on tick gaps; ATR/fractals/BOS never cross
  those segments.
- Detect bare Order Block zones per research/order_block_spec.md.
- Export detected zones and the first 10 zones for visual review.

Phase B evaluation:
- First-touch outcome measurement only.
- Deterministic hard-matched baseline.
- No parameter sweep, no cost/PnL backtest.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Optional


TIMEFRAME_MINUTES = 15
ATR_PERIOD = 14
SWING_LEFT = 10
SWING_RIGHT = 10
DISPLACEMENT_ATR_THRESHOLD = 2.0
SUCCESS_ATR_THRESHOLD = 1.0
REACTION_BARS = 20
GAP_MINUTES = 30.0
BASELINE_K = 5

SESSION_ORDER = ("asian", "london", "ny_overlap", "off_session")
VOLATILITY_BUCKETS = (
    (0.0010, "low"),
    (0.0020, "medium"),
    (math.inf, "high"),
)


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
    tick_count: int


@dataclass(frozen=True)
class Swing:
    index: int
    confirmed_at_index: int
    timestamp: datetime
    level: float


@dataclass(frozen=True)
class Zone:
    zone_id: int
    direction: str
    zone_high: float
    zone_low: float
    zone_creation_time: datetime
    frozen_atr: float
    displacement_atr: float
    bos_swing_level: float
    fvg_present: bool
    session: str
    year: int
    ob_candle_time: datetime
    ob_candle_open: float
    ob_candle_high: float
    ob_candle_low: float
    ob_candle_close: float
    ob_candle_index: int
    impulse_start_time: datetime
    impulse_end_time: datetime
    impulse_start_index: int
    impulse_end_index: int
    impulse_extreme: float
    segment_id: int


@dataclass(frozen=True)
class ZoneOutcome:
    zone: Zone
    touched: bool
    touch_index: Optional[int]
    touch_time: Optional[datetime]
    bars_to_touch: Optional[int]
    success: Optional[bool]
    bars_to_success: Optional[int]
    baseline_count: int
    baseline_successes: int
    under_matched: bool


@dataclass(frozen=True)
class BaselineCandidate:
    candidate_id: int
    source_bar_index: int
    source_bar_time: datetime
    direction: str
    zone_high: float
    zone_low: float
    frozen_atr: float
    displacement_bucket: str
    session: str
    volatility_bucket: str
    trigger_index: int
    trigger_time: datetime
    success: bool
    bars_to_success: Optional[int]


@dataclass(frozen=True)
class HtfTrendPoint:
    end: datetime
    trend: str


@dataclass
class _MutableBar:
    segment_id: int
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    tick_count: int = 1
    invalid: bool = False

    def add(self, price: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1


def parse_timestamp(raw: str) -> datetime:
    raw = raw.strip()
    if len(raw) >= 17 and raw[8] == " " and raw[11] == ":" and raw[14] == ":":
        microsecond = 0
        if len(raw) > 17 and raw[17] == ".":
            microsecond = int((raw[18:] + "000000")[:6])
        return datetime(
            int(raw[0:4]),
            int(raw[4:6]),
            int(raw[6:8]),
            int(raw[9:11]),
            int(raw[12:14]),
            int(raw[15:17]),
            microsecond,
            tzinfo=timezone.utc,
        )
    for fmt in (
        "%Y%m%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp format: {raw!r}")


def floor_timeframe(ts: datetime, minutes: int = TIMEFRAME_MINUTES) -> datetime:
    floored_minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=floored_minute, second=0, microsecond=0)


def classify_session(ts: datetime) -> str:
    t = ts.time()
    if time(0, 0) <= t < time(7, 0):
        return "asian"
    if time(7, 0) <= t < time(12, 0):
        return "london"
    if time(12, 0) <= t < time(17, 0):
        return "ny_overlap"
    return "off_session"


def fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def iter_mid_bars(path: Path, gap_minutes: float = GAP_MINUTES) -> Iterable[Bar]:
    """Stream M15 bars from Dukascopy ticks.

    A tick gap starts a new segment. Indicators and structures are computed inside
    segments only. If a >gap threshold occurs inside the same M15 bucket, that
    bucket is dropped because it would span a data gap.
    """

    gap_threshold = timedelta(minutes=gap_minutes)
    segment_id = 0
    current: Optional[_MutableBar] = None
    previous_ts: Optional[datetime] = None
    next_index = 0

    def flush() -> Optional[Bar]:
        nonlocal current, next_index
        if current is None:
            return None
        item = current
        current = None
        if item.invalid:
            return None
        bar = Bar(
            index=next_index,
            segment_id=item.segment_id,
            start=item.start,
            end=item.end,
            open=item.open,
            high=item.high,
            low=item.low,
            close=item.close,
            tick_count=item.tick_count,
        )
        next_index += 1
        return bar

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            time_idx = header.index("DateTime")
            bid_idx = header.index("Bid")
            ask_idx = header.index("Ask")
        except ValueError as exc:
            raise SystemExit("CSV must contain DateTime, Bid, and Ask columns") from exc

        for row in reader:
            try:
                ts = parse_timestamp(row[time_idx])
                bid = float(row[bid_idx])
                ask = float(row[ask_idx])
            except (IndexError, ValueError):
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue

            gap = previous_ts is not None and ts - previous_ts > gap_threshold
            bucket = floor_timeframe(ts)
            mid = (bid + ask) / 2.0

            if gap:
                if current is not None and bucket == current.start:
                    current.invalid = True
                flushed = flush()
                if flushed is not None:
                    yield flushed
                segment_id += 1

            if current is not None and bucket != current.start:
                flushed = flush()
                if flushed is not None:
                    yield flushed

            if current is None:
                current = _MutableBar(
                    segment_id=segment_id,
                    start=bucket,
                    end=bucket + timedelta(minutes=TIMEFRAME_MINUTES),
                    open=mid,
                    high=mid,
                    low=mid,
                    close=mid,
                )
            else:
                current.add(mid)
            previous_ts = ts

    flushed = flush()
    if flushed is not None:
        yield flushed


def true_range(bar: Bar, previous_bar: Optional[Bar]) -> float:
    if previous_bar is None or previous_bar.segment_id != bar.segment_id:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - previous_bar.close), abs(bar.low - previous_bar.close))


def compute_atr(bars: list[Bar], period: int = ATR_PERIOD) -> list[Optional[float]]:
    atr: list[Optional[float]] = [None] * len(bars)
    tr_window: deque[float] = deque(maxlen=period)
    previous_bar: Optional[Bar] = None
    previous_segment: Optional[int] = None

    for i, bar in enumerate(bars):
        if previous_segment is None or bar.segment_id != previous_segment:
            tr_window.clear()
            previous_bar = None
        tr = true_range(bar, previous_bar)
        tr_window.append(tr)
        if len(tr_window) == period:
            atr[i] = sum(tr_window) / period
        previous_bar = bar
        previous_segment = bar.segment_id
    return atr


def build_h4_bars(m15_bars: list[Bar]) -> list[Bar]:
    h4_bars: list[Bar] = []
    current: list[Bar] = []

    def h4_bucket(ts: datetime) -> datetime:
        return ts.replace(hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0)

    def flush() -> None:
        nonlocal current
        if len(current) == 16:
            start = current[0].start
            expected = [start + timedelta(minutes=TIMEFRAME_MINUTES * i) for i in range(16)]
            if [bar.start for bar in current] == expected:
                h4_bars.append(
                    Bar(
                        index=len(h4_bars),
                        segment_id=current[0].segment_id,
                        start=start,
                        end=start + timedelta(hours=4),
                        open=current[0].open,
                        high=max(bar.high for bar in current),
                        low=min(bar.low for bar in current),
                        close=current[-1].close,
                        tick_count=sum(bar.tick_count for bar in current),
                    )
                )
        current = []

    current_bucket: Optional[datetime] = None
    current_segment: Optional[int] = None
    for bar in m15_bars:
        bucket = h4_bucket(bar.start)
        if current and (bucket != current_bucket or bar.segment_id != current_segment):
            flush()
        if not current:
            current_bucket = bucket
            current_segment = bar.segment_id
        current.append(bar)
        if len(current) == 16:
            flush()
            current_bucket = None
            current_segment = None
    flush()
    return h4_bars


def build_h4_ema50_trend(m15_bars: list[Bar]) -> tuple[list[datetime], list[str]]:
    h4_bars = build_h4_bars(m15_bars)
    alpha = 2.0 / (50 + 1)
    closes: deque[float] = deque(maxlen=50)
    ema: Optional[float] = None
    previous_segment: Optional[int] = None
    points: list[HtfTrendPoint] = []

    for bar in h4_bars:
        if previous_segment is None or bar.segment_id != previous_segment:
            closes.clear()
            ema = None
        closes.append(bar.close)
        if ema is None and len(closes) == 50:
            ema = sum(closes) / 50.0
        elif ema is not None:
            ema = alpha * bar.close + (1 - alpha) * ema

        if ema is None or bar.close == ema:
            trend = "neutral"
        elif bar.close > ema:
            trend = "bullish"
        else:
            trend = "bearish"
        points.append(HtfTrendPoint(end=bar.end, trend=trend))
        previous_segment = bar.segment_id

    return [point.end for point in points], [point.trend for point in points]


def htf_trend(timestamp: datetime, htf_ends: list[datetime], htf_trends: list[str]) -> str:
    index = bisect.bisect_right(htf_ends, timestamp) - 1
    if index < 0:
        return "neutral"
    return htf_trends[index]


def htf_aligned_zones(zones: list[Zone], htf_ends: list[datetime], htf_trends: list[str]) -> list[Zone]:
    aligned: list[Zone] = []
    for zone in zones:
        trend = htf_trend(zone.zone_creation_time, htf_ends, htf_trends)
        if zone.direction == trend:
            aligned.append(zone)
    return aligned


def is_swing_high(bars: list[Bar], pivot: int) -> bool:
    bar = bars[pivot]
    if pivot - SWING_LEFT < 0 or pivot + SWING_RIGHT >= len(bars):
        return False
    window = bars[pivot - SWING_LEFT : pivot + SWING_RIGHT + 1]
    if any(item.segment_id != bar.segment_id for item in window):
        return False
    return all(bar.high > item.high for j, item in enumerate(window) if j != SWING_LEFT)


def is_swing_low(bars: list[Bar], pivot: int) -> bool:
    bar = bars[pivot]
    if pivot - SWING_LEFT < 0 or pivot + SWING_RIGHT >= len(bars):
        return False
    window = bars[pivot - SWING_LEFT : pivot + SWING_RIGHT + 1]
    if any(item.segment_id != bar.segment_id for item in window):
        return False
    return all(bar.low < item.low for j, item in enumerate(window) if j != SWING_LEFT)


def find_last_opposite_candle(bars: list[Bar], end_index: int, direction: str) -> Optional[int]:
    segment_id = bars[end_index].segment_id
    for j in range(end_index - 1, -1, -1):
        if bars[j].segment_id != segment_id:
            return None
        if direction == "bullish" and bars[j].close < bars[j].open:
            return j
        if direction == "bearish" and bars[j].close > bars[j].open:
            return j
    return None


def has_fvg(bars: list[Bar], start_index: int, end_index: int, direction: str) -> bool:
    if end_index - start_index < 2:
        return False
    segment_id = bars[end_index].segment_id
    for j in range(start_index, end_index - 1):
        if any(bars[k].segment_id != segment_id for k in (j, j + 1, j + 2)):
            continue
        if direction == "bullish" and bars[j + 2].low > bars[j].high:
            return True
        if direction == "bearish" and bars[j + 2].high < bars[j].low:
            return True
    return False


def displacement_bucket(value: float) -> str:
    if value < 3.0:
        return "2.0-3.0"
    if value < 4.0:
        return "3.0-4.0"
    return "4.0+"


def volatility_bucket(atr_value: float, close_price: float) -> str:
    ratio = atr_value / close_price if close_price > 0 else math.inf
    for upper, label in VOLATILITY_BUCKETS:
        if ratio < upper:
            return label
    return "high"


def ranges_overlap(high_a: float, low_a: float, high_b: float, low_b: float) -> bool:
    return low_a <= high_b and high_a >= low_b


def find_first_touch(
    bars: list[Bar],
    zone_high: float,
    zone_low: float,
    after_index: int,
) -> Optional[int]:
    for i in range(after_index + 1, len(bars)):
        bar = bars[i]
        if bar.low <= zone_high and bar.high >= zone_low:
            return i
    return None


def measure_success(
    bars: list[Bar],
    touch_index: int,
    direction: str,
    reference_price: float,
    frozen_atr: float,
) -> tuple[bool, Optional[int]]:
    touch_segment = bars[touch_index].segment_id
    target = (
        reference_price + SUCCESS_ATR_THRESHOLD * frozen_atr
        if direction == "bullish"
        else reference_price - SUCCESS_ATR_THRESHOLD * frozen_atr
    )
    for offset in range(1, REACTION_BARS + 1):
        i = touch_index + offset
        if i >= len(bars) or bars[i].segment_id != touch_segment:
            return False, None
        if direction == "bullish" and bars[i].close >= target:
            return True, offset
        if direction == "bearish" and bars[i].close <= target:
            return True, offset
    return False, None


def zone_outcome(bars: list[Bar], zone: Zone) -> ZoneOutcome:
    touch_index = find_first_touch(bars, zone.zone_high, zone.zone_low, zone.impulse_end_index)
    if touch_index is None:
        return ZoneOutcome(
            zone=zone,
            touched=False,
            touch_index=None,
            touch_time=None,
            bars_to_touch=None,
            success=None,
            bars_to_success=None,
            baseline_count=0,
            baseline_successes=0,
            under_matched=True,
        )
    reference = zone.zone_high if zone.direction == "bullish" else zone.zone_low
    success, bars_to_success = measure_success(bars, touch_index, zone.direction, reference, zone.frozen_atr)
    return ZoneOutcome(
        zone=zone,
        touched=True,
        touch_index=touch_index,
        touch_time=bars[touch_index].start,
        bars_to_touch=touch_index - zone.impulse_end_index,
        success=success,
        bars_to_success=bars_to_success,
        baseline_count=0,
        baseline_successes=0,
        under_matched=True,
    )


def detect_order_blocks(bars: list[Bar], atr: list[Optional[float]]) -> list[Zone]:
    zones: list[Zone] = []
    latest_high: Optional[Swing] = None
    latest_low: Optional[Swing] = None
    broken_high_swings: set[int] = set()
    broken_low_swings: set[int] = set()

    for i, bar in enumerate(bars):
        pivot = i - SWING_RIGHT
        if pivot >= SWING_LEFT:
            pivot_bar = bars[pivot]
            if pivot_bar.segment_id == bar.segment_id:
                if is_swing_high(bars, pivot):
                    latest_high = Swing(
                        index=pivot,
                        confirmed_at_index=i,
                        timestamp=pivot_bar.start,
                        level=pivot_bar.high,
                    )
                if is_swing_low(bars, pivot):
                    latest_low = Swing(
                        index=pivot,
                        confirmed_at_index=i,
                        timestamp=pivot_bar.start,
                        level=pivot_bar.low,
                    )

        frozen_atr = atr[i]
        if frozen_atr is None or frozen_atr <= 0:
            continue

        if (
            latest_high is not None
            and latest_high.confirmed_at_index <= i
            and latest_high.index not in broken_high_swings
            and latest_high.level < bar.high
            and bars[latest_high.index].segment_id == bar.segment_id
        ):
            ob_idx = find_last_opposite_candle(bars, i, "bullish")
            if ob_idx is not None:
                ob = bars[ob_idx]
                displacement = bar.high - ob.low
                displacement_atr = displacement / frozen_atr
                if displacement_atr >= DISPLACEMENT_ATR_THRESHOLD:
                    zones.append(
                        make_zone(
                            zone_id=len(zones) + 1,
                            direction="bullish",
                            bars=bars,
                            ob_idx=ob_idx,
                            impulse_idx=i,
                            frozen_atr=frozen_atr,
                            displacement_atr=displacement_atr,
                            bos_swing_level=latest_high.level,
                            impulse_extreme=bar.high,
                        )
                    )
                    broken_high_swings.add(latest_high.index)

        if (
            latest_low is not None
            and latest_low.confirmed_at_index <= i
            and latest_low.index not in broken_low_swings
            and latest_low.level > bar.low
            and bars[latest_low.index].segment_id == bar.segment_id
        ):
            ob_idx = find_last_opposite_candle(bars, i, "bearish")
            if ob_idx is not None:
                ob = bars[ob_idx]
                displacement = ob.high - bar.low
                displacement_atr = displacement / frozen_atr
                if displacement_atr >= DISPLACEMENT_ATR_THRESHOLD:
                    zones.append(
                        make_zone(
                            zone_id=len(zones) + 1,
                            direction="bearish",
                            bars=bars,
                            ob_idx=ob_idx,
                            impulse_idx=i,
                            frozen_atr=frozen_atr,
                            displacement_atr=displacement_atr,
                            bos_swing_level=latest_low.level,
                            impulse_extreme=bar.low,
                        )
                    )
                    broken_low_swings.add(latest_low.index)

    return zones


def build_ob_overlap_index(zones: list[Zone]) -> dict[int, list[Zone]]:
    by_impulse: dict[int, list[Zone]] = {}
    known: list[Zone] = []
    for zone in sorted(zones, key=lambda item: item.impulse_end_index):
        known.append(zone)
        by_impulse[zone.impulse_end_index] = list(known)
    return by_impulse


def is_strict_non_ob_source(bar: Bar, known_zones: list[Zone]) -> bool:
    for zone in known_zones:
        if bar.index == zone.ob_candle_index:
            return False
        if ranges_overlap(bar.high, bar.low, zone.zone_high, zone.zone_low):
            return False
    return True


def build_baseline_candidates(bars: list[Bar], zones: list[Zone]) -> list[BaselineCandidate]:
    candidates: list[BaselineCandidate] = []
    known_by_impulse = build_ob_overlap_index(zones)

    for zone in zones:
        known_zones = known_by_impulse.get(zone.impulse_end_index, [])
        disp_bucket = displacement_bucket(zone.displacement_atr)
        vol_bucket = volatility_bucket(zone.frozen_atr, bars[zone.impulse_end_index].close)
        for source_idx in range(zone.impulse_start_index - 1, -1, -1):
            source = bars[source_idx]
            if source.segment_id != zone.segment_id:
                break
            if source.index >= zone.impulse_start_index:
                continue
            if not is_strict_non_ob_source(source, known_zones):
                continue
            touch_index = find_first_touch(bars, source.high, source.low, zone.impulse_end_index)
            if touch_index is None:
                continue
            reference = source.high if zone.direction == "bullish" else source.low
            success, bars_to_success = measure_success(
                bars=bars,
                touch_index=touch_index,
                direction=zone.direction,
                reference_price=reference,
                frozen_atr=zone.frozen_atr,
            )
            candidates.append(
                BaselineCandidate(
                    candidate_id=len(candidates) + 1,
                    source_bar_index=source.index,
                    source_bar_time=source.start,
                    direction=zone.direction,
                    zone_high=source.high,
                    zone_low=source.low,
                    frozen_atr=zone.frozen_atr,
                    displacement_bucket=disp_bucket,
                    session=zone.session,
                    volatility_bucket=vol_bucket,
                    trigger_index=touch_index,
                    trigger_time=bars[touch_index].start,
                    success=success,
                    bars_to_success=bars_to_success,
                )
            )
    candidates.sort(
        key=lambda item: (
            item.trigger_index,
            item.source_bar_index,
            item.candidate_id,
        )
    )
    return candidates


def select_baselines_for_zone(
    zone_out: ZoneOutcome,
    baseline_candidates: list[BaselineCandidate],
    bars: list[Bar],
) -> list[BaselineCandidate]:
    if not zone_out.touched or zone_out.touch_index is None:
        return []
    zone = zone_out.zone
    disp_bucket = displacement_bucket(zone.displacement_atr)
    vol_bucket = volatility_bucket(zone.frozen_atr, bars[zone.impulse_end_index].close)
    qualified = [
        candidate
        for candidate in baseline_candidates
        if candidate.trigger_index < zone_out.touch_index
        and candidate.direction == zone.direction
        and candidate.displacement_bucket == disp_bucket
        and candidate.session == zone.session
        and candidate.volatility_bucket == vol_bucket
    ]
    qualified.sort(
        key=lambda item: (
            -item.trigger_index,
            item.source_bar_index,
            item.candidate_id,
        )
    )
    return qualified[:BASELINE_K]


def attach_baselines(
    bars: list[Bar],
    outcomes: list[ZoneOutcome],
    baseline_candidates: list[BaselineCandidate],
) -> list[ZoneOutcome]:
    final: list[ZoneOutcome] = []
    for outcome in outcomes:
        selected = select_baselines_for_zone(outcome, baseline_candidates, bars)
        final.append(
            ZoneOutcome(
                zone=outcome.zone,
                touched=outcome.touched,
                touch_index=outcome.touch_index,
                touch_time=outcome.touch_time,
                bars_to_touch=outcome.bars_to_touch,
                success=outcome.success,
                bars_to_success=outcome.bars_to_success,
                baseline_count=len(selected),
                baseline_successes=sum(1 for item in selected if item.success),
                under_matched=len(selected) < BASELINE_K,
            )
        )
    return final


def evaluate_zones(bars: list[Bar], zones: list[Zone]) -> tuple[list[ZoneOutcome], list[BaselineCandidate]]:
    outcomes = [zone_outcome(bars, zone) for zone in zones]
    baseline_candidates = build_baseline_candidates(bars, zones)
    final = attach_baselines(bars, outcomes, baseline_candidates)
    return final, baseline_candidates


def evaluate_zone_subset(
    bars: list[Bar],
    zones: list[Zone],
    baseline_candidates: list[BaselineCandidate],
) -> list[ZoneOutcome]:
    outcomes = [zone_outcome(bars, zone) for zone in zones]
    return attach_baselines(bars, outcomes, baseline_candidates)


def make_zone(
    zone_id: int,
    direction: str,
    bars: list[Bar],
    ob_idx: int,
    impulse_idx: int,
    frozen_atr: float,
    displacement_atr: float,
    bos_swing_level: float,
    impulse_extreme: float,
) -> Zone:
    ob = bars[ob_idx]
    impulse = bars[impulse_idx]
    return Zone(
        zone_id=zone_id,
        direction=direction,
        zone_high=ob.high,
        zone_low=ob.low,
        zone_creation_time=impulse.end,
        frozen_atr=frozen_atr,
        displacement_atr=displacement_atr,
        bos_swing_level=bos_swing_level,
        fvg_present=has_fvg(bars, ob_idx, impulse_idx, direction),
        session=classify_session(impulse.end),
        year=impulse.end.year,
        ob_candle_time=ob.start,
        ob_candle_open=ob.open,
        ob_candle_high=ob.high,
        ob_candle_low=ob.low,
        ob_candle_close=ob.close,
        ob_candle_index=ob.index,
        impulse_start_time=bars[min(ob_idx + 1, impulse_idx)].start,
        impulse_end_time=impulse.end,
        impulse_start_index=min(ob_idx + 1, impulse_idx),
        impulse_end_index=impulse.index,
        impulse_extreme=impulse_extreme,
        segment_id=impulse.segment_id,
    )


ZONE_FIELDS = [
    "zone_id",
    "direction",
    "zone_high",
    "zone_low",
    "zone_creation_time",
    "frozen_atr",
    "displacement_atr",
    "bos_swing_level",
    "fvg_present",
    "session",
    "year",
    "ob_candle_time",
    "ob_candle_open",
    "ob_candle_high",
    "ob_candle_low",
    "ob_candle_close",
    "ob_candle_index",
    "impulse_start_time",
    "impulse_end_time",
    "impulse_start_index",
    "impulse_end_index",
    "impulse_extreme",
    "segment_id",
]


def zone_to_row(zone: Zone) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in ZONE_FIELDS:
        value = getattr(zone, field)
        if isinstance(value, datetime):
            row[field] = fmt_ts(value)
        elif isinstance(value, float):
            row[field] = f"{value:.6f}"
        else:
            row[field] = value
    return row


def write_zones(path: Path, zones: list[Zone]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ZONE_FIELDS)
        writer.writeheader()
        for zone in zones:
            writer.writerow(zone_to_row(zone))


def write_first10(path: Path, zones: list[Zone]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "zone_id",
        "direction",
        "zone_creation_time",
        "zone_high",
        "zone_low",
        "bos_swing_level",
        "displacement_atr",
        "fvg_present",
        "ob_candle_time",
        "ob_candle_open",
        "ob_candle_high",
        "ob_candle_low",
        "ob_candle_close",
        "impulse_start_time",
        "impulse_end_time",
        "impulse_extreme",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for zone in zones[:10]:
            row = zone_to_row(zone)
            writer.writerow({field: row[field] for field in fields})


OUTCOME_FIELDS = [
    *ZONE_FIELDS,
    "touched",
    "touch_time",
    "bars_to_touch",
    "success",
    "bars_to_success",
    "baseline_count",
    "baseline_successes",
    "baseline_success_rate",
    "under_matched",
]


def outcome_to_row(outcome: ZoneOutcome) -> dict[str, object]:
    row = zone_to_row(outcome.zone)
    row.update(
        {
            "touched": outcome.touched,
            "touch_time": fmt_ts(outcome.touch_time) if outcome.touch_time is not None else "",
            "bars_to_touch": outcome.bars_to_touch if outcome.bars_to_touch is not None else "",
            "success": outcome.success if outcome.success is not None else "",
            "bars_to_success": outcome.bars_to_success if outcome.bars_to_success is not None else "",
            "baseline_count": outcome.baseline_count,
            "baseline_successes": outcome.baseline_successes,
            "baseline_success_rate": (
                f"{outcome.baseline_successes / outcome.baseline_count:.6f}"
                if outcome.baseline_count
                else ""
            ),
            "under_matched": outcome.under_matched,
        }
    )
    return row


def write_outcomes(path: Path, outcomes: list[ZoneOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTCOME_FIELDS)
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow(outcome_to_row(outcome))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * pct) - 1)
    return ordered[index]


def rate(successes: int, count: int) -> float:
    return successes / count if count else math.nan


def edge_ci(success_a: int, count_a: int, success_b: int, count_b: int) -> tuple[float, float, float]:
    if count_a == 0 or count_b == 0:
        return math.nan, math.nan, math.nan
    p_a = success_a / count_a
    p_b = success_b / count_b
    edge = p_a - p_b
    se = math.sqrt((p_a * (1 - p_a) / count_a) + (p_b * (1 - p_b) / count_b))
    return edge, edge - 1.96 * se, edge + 1.96 * se


def outcome_stats(outcomes: list[ZoneOutcome]) -> dict[str, float | int]:
    touched = [item for item in outcomes if item.touched]
    successes = sum(1 for item in touched if item.success is True)
    baseline_count = sum(item.baseline_count for item in touched)
    baseline_successes = sum(item.baseline_successes for item in touched)
    under_matched = sum(1 for item in touched if item.under_matched)
    edge, ci_low, ci_high = edge_ci(successes, len(touched), baseline_successes, baseline_count)
    return {
        "raw": len(outcomes),
        "fresh": len(touched),
        "successes": successes,
        "ob_rate": rate(successes, len(touched)),
        "baseline_count": baseline_count,
        "baseline_successes": baseline_successes,
        "baseline_rate": rate(baseline_successes, baseline_count),
        "edge": edge,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "under_matched": under_matched,
    }


def print_report(zones: list[Zone], bars: list[Bar], zones_out: Path, first10_out: Path) -> None:
    by_year = Counter(zone.year for zone in zones)
    by_session = Counter(zone.session for zone in zones)
    by_direction = Counter(zone.direction for zone in zones)
    displacement_values = [zone.displacement_atr for zone in zones]
    fvg_count = sum(1 for zone in zones if zone.fvg_present)

    print("\nOrder Block Phase A Detection Report")
    print("=" * 38)
    print(f"Completed M15 bars: {len(bars):,}")
    print(f"Detected zones:     {len(zones):,}")
    print(f"Zones CSV:          {zones_out}")
    print(f"First 10 CSV:       {first10_out}")

    print("\nDirection")
    for direction in ("bullish", "bearish"):
        print(f"  {direction:<8} {by_direction[direction]:,}")

    print("\nPer Year")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]:,}")

    print("\nPer Session")
    for session in SESSION_ORDER:
        print(f"  {session:<11} {by_session[session]:,}")

    print("\nDisplacement ATR")
    if displacement_values:
        print(f"  min:    {min(displacement_values):.3f}")
        print(f"  median: {median(displacement_values):.3f}")
        print(f"  p90:    {percentile(displacement_values, 0.90):.3f}")
        print(f"  max:    {max(displacement_values):.3f}")
    else:
        print("  n/a")

    rate = fvg_count / len(zones) if zones else 0.0
    print("\nFVG Present")
    print(f"  {fvg_count:,} / {len(zones):,} ({rate:.2%})")

    print("\nPhase A only: no touches, reactions, outcomes, or baselines measured.")


def print_phase_b_report(outcomes: list[ZoneOutcome], outcomes_out: Path) -> None:
    touched = [item for item in outcomes if item.touched]
    successful = [item for item in touched if item.success is True]
    baseline_count = sum(item.baseline_count for item in touched)
    baseline_successes = sum(item.baseline_successes for item in touched)
    under_matched = sum(1 for item in touched if item.under_matched)
    edge, ci_low, ci_high = edge_ci(len(successful), len(touched), baseline_successes, baseline_count)

    print("\nOrder Block Phase B Evaluation Report")
    print("=" * 39)
    print(f"Outcomes CSV:              {outcomes_out}")
    print(f"Raw setup count:           {len(outcomes):,}")
    print(f"Fresh first-touch count:   {len(touched):,}")
    print("Already-touched count:     0 (retests not computed in locked main test)")
    print(f"Untouched zones:           {len(outcomes) - len(touched):,}")
    print(f"Under-matched OBs:         {under_matched:,}")

    print("\nSuccess Rates")
    print(f"  OB fresh first-touch:     {len(successful):,} / {len(touched):,} ({rate(len(successful), len(touched)):.2%})")
    print(
        f"  Matched baseline:         {baseline_successes:,} / {baseline_count:,} "
        f"({rate(baseline_successes, baseline_count):.2%})"
    )
    print(f"  EDGE:                     {edge:.2%}")
    print(f"  95% CI, approx:           [{ci_low:.2%}, {ci_high:.2%}]")

    print("\nSession Breakdown")
    print("  session       OB_rate      baseline_rate   edge       touched  baselines  under_matched")
    for session in SESSION_ORDER:
        rows = [item for item in touched if item.zone.session == session]
        ob_success = sum(1 for item in rows if item.success is True)
        base_n = sum(item.baseline_count for item in rows)
        base_success = sum(item.baseline_successes for item in rows)
        session_edge, _, _ = edge_ci(ob_success, len(rows), base_success, base_n)
        print(
            f"  {session:<11} {rate(ob_success, len(rows)):>9.2%}"
            f"   {rate(base_success, base_n):>12.2%}"
            f"   {session_edge:>7.2%}"
            f"   {len(rows):>7,}"
            f"   {base_n:>9,}"
            f"   {sum(1 for item in rows if item.under_matched):>13,}"
        )

    print("\nFVG Stratification (descriptive only)")
    print("  cohort        OB_rate      baseline_rate   edge       touched  baselines  under_matched")
    for label, expected in (("FVG", True), ("No-FVG", False)):
        rows = [item for item in touched if item.zone.fvg_present is expected]
        ob_success = sum(1 for item in rows if item.success is True)
        base_n = sum(item.baseline_count for item in rows)
        base_success = sum(item.baseline_successes for item in rows)
        cohort_edge, _, _ = edge_ci(ob_success, len(rows), base_success, base_n)
        print(
            f"  {label:<12} {rate(ob_success, len(rows)):>9.2%}"
            f"   {rate(base_success, base_n):>12.2%}"
            f"   {cohort_edge:>7.2%}"
            f"   {len(rows):>7,}"
            f"   {base_n:>9,}"
            f"   {sum(1 for item in rows if item.under_matched):>13,}"
        )

    print("\nLocked parameters only. No parameter sweep, no PnL/cost model, no retest cohort.")


def print_htf_ab_report(
    bare_outcomes: list[ZoneOutcome],
    htf_outcomes: list[ZoneOutcome],
    htf_outcomes_out: Path,
) -> None:
    bare = outcome_stats(bare_outcomes)
    htf = outcome_stats(htf_outcomes)
    shrink = 1.0 - (htf["fresh"] / bare["fresh"] if bare["fresh"] else math.nan)

    print("\nOrder Block HTF A/B Evaluation Report")
    print("=" * 40)
    print("HTF rule locked for this test: H4 close > EMA(50) = bullish; H4 close < EMA(50) = bearish; otherwise neutral.")
    print("HTF uses only completed gap-aware H4 candles with H4 end <= zone_creation_time.")
    print(f"HTF-filtered outcomes CSV: {htf_outcomes_out}")

    print("\nA/B Comparison")
    print("  metric                    Bare OB              HTF-filtered OB")
    print(f"  setup count (fresh)       {bare['fresh']:>7,}              {htf['fresh']:>7,}")
    print(f"  OB success rate           {bare['ob_rate']:>7.2%}              {htf['ob_rate']:>7.2%}")
    print(f"  baseline success rate     {bare['baseline_rate']:>7.2%}              {htf['baseline_rate']:>7.2%}")
    print(f"  EDGE                      {bare['edge']:>7.2%}              {htf['edge']:>7.2%}")
    print(f"  95% CI                    [{bare['ci_low']:.2%}, {bare['ci_high']:.2%}]   [{htf['ci_low']:.2%}, {htf['ci_high']:.2%}]")
    print(f"  under-matched count       {bare['under_matched']:>7,}              {htf['under_matched']:>7,}")

    print("\nSample Shrink")
    print(f"  Fresh setups retained: {htf['fresh']:,} / {bare['fresh']:,} ({1 - shrink:.2%})")
    print(f"  Fresh setups removed:  {bare['fresh'] - htf['fresh']:,} ({shrink:.2%})")
    if htf["fresh"] < 300:
        print("  WARNING: HTF-filtered sample is below a few hundred and is underpowered.")

    print("\nHTF-Filtered Session Breakdown")
    print("  session       OB_rate      baseline_rate   edge       touched  baselines  under_matched")
    touched = [item for item in htf_outcomes if item.touched]
    for session in SESSION_ORDER:
        rows = [item for item in touched if item.zone.session == session]
        ob_success = sum(1 for item in rows if item.success is True)
        base_n = sum(item.baseline_count for item in rows)
        base_success = sum(item.baseline_successes for item in rows)
        session_edge, _, _ = edge_ci(ob_success, len(rows), base_success, base_n)
        print(
            f"  {session:<11} {rate(ob_success, len(rows)):>9.2%}"
            f"   {rate(base_success, base_n):>12.2%}"
            f"   {session_edge:>7.2%}"
            f"   {len(rows):>7,}"
            f"   {base_n:>9,}"
            f"   {sum(1 for item in rows if item.under_matched):>13,}"
        )

    print("\nHonesty Check")
    if htf["edge"] > bare["edge"] and htf["ci_low"] > 0:
        print("  HTF helps by the pre-planned standard: edge increases and CI is clearly above zero.")
    elif htf["edge"] > bare["edge"]:
        print("  HTF raises the point estimate, but the CI still crosses zero. This is not a robust improvement.")
    else:
        print("  HTF does not improve the edge. Bare OB + HTF fails to show a robust standalone edge.")
    print("  No alternate HTF rules tried. No parameter sweep, no PnL/cost model.")


def default_tick_path() -> Path:
    matches = sorted(Path("data").glob("*XAUUSD*.csv"))
    if not matches:
        raise SystemExit("No XAUUSD CSV found under data/")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("detect", "evaluate", "htf-ab"), default="detect")
    parser.add_argument("--ticks", type=Path, default=None, help="Dukascopy XAUUSD tick CSV")
    parser.add_argument("--zones-out", type=Path, default=Path("research/order_block_zones.csv"))
    parser.add_argument("--first10-out", type=Path, default=Path("research/order_block_first10_zones.csv"))
    parser.add_argument("--outcomes-out", type=Path, default=Path("research/order_block_phase_b_outcomes.csv"))
    parser.add_argument("--htf-outcomes-out", type=Path, default=Path("research/order_block_htf_phase_b_outcomes.csv"))
    parser.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)
    parser.add_argument("--max-rows", type=int, default=None, help="development smoke-test row limit")
    return parser.parse_args()


def load_bars(path: Path, gap_minutes: float, max_rows: Optional[int] = None) -> list[Bar]:
    if max_rows is None:
        return list(iter_mid_bars(path, gap_minutes=gap_minutes))

    bars: list[Bar] = []
    gap_threshold = timedelta(minutes=gap_minutes)
    segment_id = 0
    current: Optional[_MutableBar] = None
    previous_ts: Optional[datetime] = None
    next_index = 0

    def flush() -> None:
        nonlocal current, next_index
        if current is None:
            return
        item = current
        current = None
        if item.invalid:
            return
        bars.append(
            Bar(
                index=next_index,
                segment_id=item.segment_id,
                start=item.start,
                end=item.end,
                open=item.open,
                high=item.high,
                low=item.low,
                close=item.close,
                tick_count=item.tick_count,
            )
        )
        next_index += 1

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        time_idx = header.index("DateTime")
        bid_idx = header.index("Bid")
        ask_idx = header.index("Ask")
        for row_number, row in enumerate(reader, start=1):
            if row_number > max_rows:
                break
            try:
                ts = parse_timestamp(row[time_idx])
                bid = float(row[bid_idx])
                ask = float(row[ask_idx])
            except (IndexError, ValueError):
                continue
            if ask <= bid or bid <= 0 or ask <= 0:
                continue
            gap = previous_ts is not None and ts - previous_ts > gap_threshold
            bucket = floor_timeframe(ts)
            mid = (bid + ask) / 2.0
            if gap:
                if current is not None and bucket == current.start:
                    current.invalid = True
                flush()
                segment_id += 1
            if current is not None and bucket != current.start:
                flush()
            if current is None:
                current = _MutableBar(
                    segment_id=segment_id,
                    start=bucket,
                    end=bucket + timedelta(minutes=TIMEFRAME_MINUTES),
                    open=mid,
                    high=mid,
                    low=mid,
                    close=mid,
                )
            else:
                current.add(mid)
            previous_ts = ts
    flush()
    return bars


def main() -> None:
    args = parse_args()
    tick_path = args.ticks or default_tick_path()
    bars = load_bars(tick_path, gap_minutes=args.gap_minutes, max_rows=args.max_rows)
    atr = compute_atr(bars)
    zones = detect_order_blocks(bars, atr)
    write_zones(args.zones_out, zones)
    write_first10(args.first10_out, zones)
    if args.phase == "detect":
        print_report(zones, bars, args.zones_out, args.first10_out)
        return
    if args.phase == "evaluate":
        outcomes, _ = evaluate_zones(bars, zones)
        write_outcomes(args.outcomes_out, outcomes)
        print_phase_b_report(outcomes, args.outcomes_out)
        return
    bare_initial = [zone_outcome(bars, zone) for zone in zones]
    baseline_candidates = build_baseline_candidates(bars, zones)
    outcomes = attach_baselines(bars, bare_initial, baseline_candidates)
    htf_ends, htf_trends = build_h4_ema50_trend(bars)
    htf_zones = htf_aligned_zones(zones, htf_ends, htf_trends)
    htf_outcomes = evaluate_zone_subset(bars, htf_zones, baseline_candidates)
    write_outcomes(args.htf_outcomes_out, htf_outcomes)
    print_htf_ab_report(outcomes, htf_outcomes, args.htf_outcomes_out)


if __name__ == "__main__":
    main()
