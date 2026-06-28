"""
Dukascopy XAUUSD tick-data quality audit.

This script verifies that raw tick files contain both bid and ask prices, then
streams the data to report coverage, gaps, timestamp samples, contract sanity,
and Dukascopy's raw interbank spread distribution by session.

Important: the spread report is a reference measurement of Dukascopy data only.
It is not the trading-cost input for an IUX Standard backtest.

Usage:
    python3 research/data_audit.py
    python3 research/data_audit.py data/*.csv --gap-minutes 30
    python3 research/data_audit.py --max-rows 1000000  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


STD_LOT_OZ = 100.0
MICRO_LOT_OZ = 1.0
PRICE_ABS_MIN = 900.0
PRICE_ABS_MAX = 6500.0
LEGACY_PRICE_ABS_MAX = 5000.0
CLEARED_ATH_CRASH_WINDOW = (
    datetime(2026, 1, 25, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
)

SESSION_ORDER = ("asian", "london", "ny_overlap", "off_session")
SESSION_LABELS = {
    "asian": "Asian",
    "london": "London",
    "ny_overlap": "NY-overlap",
    "off_session": "Off-session",
}

# UTC session approximation for research/audit only. Later strategy code should
# tag sessions with DST-aware London/New York market calendars.
SESSION_WINDOWS_UTC = {
    "asian": (time(0, 0), time(7, 0)),
    "london": (time(7, 0), time(12, 0)),
    "ny_overlap": (time(12, 0), time(17, 0)),
}

KNOWN_EVENT_WINDOWS_UTC = {
    "FOMC decision": (
        datetime(2023, 7, 26, 17, 55, tzinfo=timezone.utc),
        datetime(2023, 7, 26, 18, 5, tzinfo=timezone.utc),
    ),
    "US CPI release": (
        datetime(2022, 7, 13, 12, 25, tzinfo=timezone.utc),
        datetime(2022, 7, 13, 12, 35, tzinfo=timezone.utc),
    ),
    "January FOMC decision": (
        datetime(2024, 1, 31, 18, 55, tzinfo=timezone.utc),
        datetime(2024, 1, 31, 19, 5, tzinfo=timezone.utc),
    ),
}


@dataclass
class Gap:
    start: datetime
    end: datetime
    minutes: float
    classification: str
    reason: str


@dataclass(order=True)
class WorstSpread:
    spread: float
    timestamp: datetime = field(compare=False)
    bid: float = field(compare=False)
    ask: float = field(compare=False)


@dataclass
class TickSample:
    timestamp: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class JumpSample:
    previous: TickSample
    current: TickSample
    pct: float
    before: list[TickSample] = field(default_factory=list)
    after: list[TickSample] = field(default_factory=list)


@dataclass
class AuditState:
    files: list[Path]
    sample_months: Optional[set[int]] = None
    sample_years: Optional[set[int]] = None
    total_ticks: int = 0
    scanned_rows: int = 0
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    previous_ts: Optional[datetime] = None
    previous_month_key: Optional[tuple[int, int]] = None
    previous_mid_by_month: dict[tuple[int, int], tuple[datetime, float]] = field(default_factory=dict)
    ticks_per_year: Counter = field(default_factory=Counter)
    ticks_per_month: Counter = field(default_factory=Counter)
    spread_hist: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    spread_count: Counter = field(default_factory=Counter)
    spread_sum: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    spread_max: dict[str, float] = field(default_factory=dict)
    worst_spreads: dict[str, list[WorstSpread]] = field(default_factory=lambda: defaultdict(list))
    price_min: float = math.inf
    price_max: float = -math.inf
    suspicious_gaps: list[Gap] = field(default_factory=list)
    weekend_gaps: list[Gap] = field(default_factory=list)
    rollover_gaps: list[Gap] = field(default_factory=list)
    out_of_order: int = 0
    bad_rows: int = 0
    bad_tick_counts: Counter = field(default_factory=Counter)
    bad_tick_months: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    legacy_price_bound_false_positives: int = 0
    cleared_jump_candidates: int = 0
    jump_samples: list[JumpSample] = field(default_factory=list)
    _pending_jump_samples: list[JumpSample] = field(default_factory=list, repr=False)
    event_samples: dict[str, list[tuple[datetime, float, float]]] = field(
        default_factory=lambda: {name: [] for name in KNOWN_EVENT_WINDOWS_UTC}
    )


def normalize_column(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def parse_timestamp(raw: str) -> datetime:
    raw = raw.strip()
    if len(raw) >= 17 and raw[8] == " " and raw[11] == ":" and raw[14] == ":":
        try:
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
        except ValueError:
            pass

    formats = (
        "%Y%m%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    )
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp format: {raw!r}")


def classify_session(ts: datetime) -> str:
    t = ts.time()
    for session, (start, end) in SESSION_WINDOWS_UTC.items():
        if start <= t < end:
            return session
    return "off_session"


def holiday_dates(year: int) -> set[date]:
    # Gold spot/CFD liquidity can be sparse around these dates. Treat these as
    # expected-closure candidates, then show them separately instead of hiding.
    jan_first = date(year, 1, 1)
    july_fourth = date(year, 7, 4)
    christmas = date(year, 12, 25)
    easter = easter_sunday(year)
    return {
        jan_first,
        observed_us_holiday(jan_first),
        nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        nth_weekday(year, 2, 0, 3),  # Presidents Day
        easter - timedelta(days=2),  # Good Friday
        easter,
        last_weekday(year, 5, 0),  # Memorial Day
        date(year, 6, 19),  # Juneteenth
        observed_us_holiday(date(year, 6, 19)),
        july_fourth,
        observed_us_holiday(july_fourth),
        nth_weekday(year, 9, 0, 1),  # Labor Day
        nth_weekday(year, 11, 3, 4),  # Thanksgiving
        date(year, 12, 24),  # Christmas Eve / common early close
        christmas,
        observed_us_holiday(christmas),
        date(year, 12, 31),  # New Year's Eve / common early close
    }


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def easter_sunday(year: int) -> date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def observed_us_holiday(actual: date) -> date:
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def is_weekend_or_holiday(ts: datetime) -> bool:
    d = ts.date()
    return ts.weekday() in (5, 6) or d in holiday_dates(ts.year)


def is_daily_maintenance(ts: datetime) -> bool:
    # Common precious-metals maintenance/rollover band. Exact broker feeds vary
    # with DST; this audit uses UTC only and flags the limitation in the report.
    return time(20, 55) <= ts.time() < time(23, 15)


def classify_gap(start: datetime, end: datetime) -> tuple[str, str]:
    duration = end - start
    if duration <= timedelta(0):
        return "suspicious", "non-positive timestamp delta"

    checks = Counter()
    sample = start
    step = timedelta(minutes=30)
    samples = 0
    while sample <= end:
        samples += 1
        if is_weekend_or_holiday(sample):
            checks["weekend_or_holiday"] += 1
        elif is_daily_maintenance(sample):
            checks["daily_maintenance"] += 1
        else:
            checks["market_hours"] += 1
        sample += step

    if samples == 0:
        return "suspicious", "unclassified"
    expected = checks["weekend_or_holiday"] + checks["daily_maintenance"]
    if checks["weekend_or_holiday"] / samples >= 0.80:
        return "weekend", "expected weekend/holiday closure"
    if checks["daily_maintenance"] / samples >= 0.80:
        return "rollover", "expected daily-rollover/metals maintenance pause"
    if expected / samples >= 0.80:
        return "weekend", "expected weekend/holiday + rollover closure"
    return "suspicious", "intra-session market-hours gap"


def percentile_from_hist(hist: Counter, pct: float) -> float:
    total = sum(hist.values())
    if total == 0:
        return math.nan
    target = math.ceil(total * pct)
    seen = 0
    for key in sorted(hist):
        seen += hist[key]
        if seen >= target:
            return key / 1000.0
    return max(hist) / 1000.0


def fmt_ts(ts: Optional[datetime]) -> str:
    if ts is None:
        return "n/a"
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def month_label(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def row_year_month(raw_timestamp: str) -> tuple[int, int]:
    raw = raw_timestamp.strip()
    if len(raw) >= 6 and raw[:6].isdigit():
        return int(raw[0:4]), int(raw[4:6])
    if len(raw) >= 7 and raw[4] == "-" and raw[0:4].isdigit() and raw[5:7].isdigit():
        return int(raw[0:4]), int(raw[5:7])
    ts = parse_timestamp(raw)
    return ts.year, ts.month


def should_sample_month(
    year: int,
    month: int,
    sample_months: Optional[set[int]],
    sample_years: Optional[set[int]],
) -> bool:
    if sample_months is None:
        return True
    if month not in sample_months:
        return False
    if sample_years is not None and year not in sample_years:
        return False
    return True


def add_worst_spread(state: AuditState, session: str, sample: WorstSpread, worst_n: int) -> None:
    if worst_n <= 0:
        return
    heap = state.worst_spreads[session]
    if len(heap) < worst_n:
        heapq.heappush(heap, sample)
    elif sample.spread > heap[0].spread:
        heapq.heapreplace(heap, sample)


def add_tick_to_pending_jump_context(state: AuditState, tick: TickSample) -> None:
    if not state._pending_jump_samples:
        return
    remaining = []
    for sample in state._pending_jump_samples:
        if tick.timestamp <= sample.current.timestamp:
            remaining.append(sample)
            continue
        if len(sample.after) < 3:
            sample.after.append(tick)
        if len(sample.after) < 3:
            remaining.append(sample)
    state._pending_jump_samples = remaining


def is_cleared_ath_crash_jump(timestamp: datetime) -> bool:
    start, end = CLEARED_ATH_CRASH_WINDOW
    return start <= timestamp <= end


def detect_schema(paths: list[Path]) -> tuple[list[str], int, int, int]:
    if not paths:
        raise SystemExit("No input CSV files found.")
    with paths[0].open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise SystemExit(f"{paths[0]} is empty or missing a header row.")

    normalized = {normalize_column(col): index for index, col in enumerate(header)}
    time_col = next(
        (normalized[name] for name in ("datetime", "timestamp", "time", "date") if name in normalized),
        None,
    )
    bid_col = next((normalized[name] for name in ("bid", "bidprice") if name in normalized), None)
    ask_col = next((normalized[name] for name in ("ask", "askprice") if name in normalized), None)

    price_like = [col for col in header if normalize_column(col) in {"price", "last", "mid", "close"}]
    if bid_col is None or ask_col is None:
        found = ", ".join(header)
        if price_like:
            raise SystemExit(
                "SCHEMA FAIL: found only a single/mid/last price column "
                f"({', '.join(price_like)}). Bid/ask spread work is impossible "
                "without BOTH bid and ask columns.\n"
                f"Columns found: {found}"
            )
        raise SystemExit(
            "SCHEMA FAIL: could not find BOTH bid and ask columns. Spread work is "
            f"impossible without both.\nColumns found: {found}"
        )
    if time_col is None:
        raise SystemExit(f"SCHEMA FAIL: could not find a timestamp column. Columns found: {', '.join(header)}")

    return header, time_col, bid_col, ask_col


def validate_file_header(path: Path, expected_header: list[str]) -> None:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header != expected_header:
        raise SystemExit(
            f"SCHEMA FAIL: {path} header does not match the first file.\n"
            f"Expected: {expected_header}\n"
            f"Found:    {header}"
        )


def audit(
    paths: list[Path],
    gap_minutes: float,
    max_rows: Optional[int] = None,
    sample_months: Optional[set[int]] = None,
    sample_years: Optional[set[int]] = None,
    jump_pct: float = 1.0,
    worst_n: int = 10,
) -> AuditState:
    expected_header, time_col, bid_col, ask_col = detect_schema(paths)
    state = AuditState(files=paths, sample_months=sample_months, sample_years=sample_years)
    gap_threshold = timedelta(minutes=gap_minutes)
    jump_fraction = jump_pct / 100.0
    previous_ticks_by_month: dict[tuple[int, int], deque[TickSample]] = defaultdict(lambda: deque(maxlen=3))

    for path in paths:
        validate_file_header(path, expected_header)
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if max_rows is not None and state.total_ticks >= max_rows:
                    return state
                state.scanned_rows += 1
                try:
                    year, month = row_year_month(row[time_col])
                except (IndexError, TypeError, ValueError):
                    state.bad_rows += 1
                    state.bad_tick_counts["malformed"] += 1
                    state.bad_tick_months["malformed"]["unknown"] += 1
                    continue
                if not should_sample_month(year, month, sample_months, sample_years):
                    continue

                try:
                    ts = parse_timestamp(row[time_col])
                    bid = float(row[bid_col])
                    ask = float(row[ask_col])
                except (IndexError, TypeError, ValueError):
                    state.bad_rows += 1
                    state.bad_tick_counts["malformed"] += 1
                    state.bad_tick_months["malformed"][month_label(year, month)] += 1
                    continue

                month_key = (ts.year, ts.month)
                label = month_label(ts.year, ts.month)
                if ask <= bid:
                    state.bad_rows += 1
                    state.bad_tick_counts["ask_lte_bid"] += 1
                    state.bad_tick_months["ask_lte_bid"][label] += 1
                    continue
                if ask <= 0 or bid <= 0:
                    state.bad_rows += 1
                    state.bad_tick_counts["price_lte_zero"] += 1
                    state.bad_tick_months["price_lte_zero"][label] += 1
                    continue

                mid = (bid + ask) / 2.0
                tick = TickSample(timestamp=ts, bid=bid, ask=ask)
                add_tick_to_pending_jump_context(state, tick)

                if min(bid, ask, mid) < PRICE_ABS_MIN or max(bid, ask, mid) > PRICE_ABS_MAX:
                    state.bad_tick_counts["price_out_of_abs_range"] += 1
                    state.bad_tick_months["price_out_of_abs_range"][label] += 1
                elif max(bid, ask, mid) > LEGACY_PRICE_ABS_MAX:
                    state.legacy_price_bound_false_positives += 1

                previous_mid = state.previous_mid_by_month.get(month_key)
                if previous_mid is not None:
                    prev_ts, prev_mid = previous_mid
                    if prev_mid > 0 and abs(mid - prev_mid) / prev_mid > jump_fraction:
                        state.bad_tick_counts["jump_gt_threshold"] += 1
                        state.bad_tick_months["jump_gt_threshold"][label] += 1
                        if is_cleared_ath_crash_jump(ts):
                            state.cleared_jump_candidates += 1
                        else:
                            previous_tick = TickSample(
                                timestamp=prev_ts,
                                bid=previous_ticks_by_month[month_key][-1].bid,
                                ask=previous_ticks_by_month[month_key][-1].ask,
                            )
                            sample = JumpSample(
                                previous=previous_tick,
                                current=tick,
                                pct=abs(mid - prev_mid) / prev_mid * 100.0,
                                before=list(previous_ticks_by_month[month_key]),
                            )
                            state.jump_samples.append(sample)
                            state._pending_jump_samples.append(sample)

                if state.previous_ts is not None and state.previous_month_key == month_key:
                    if ts < state.previous_ts:
                        state.out_of_order += 1
                    else:
                        delta = ts - state.previous_ts
                        if delta > gap_threshold:
                            classification, reason = classify_gap(state.previous_ts, ts)
                            gap = Gap(state.previous_ts, ts, delta.total_seconds() / 60.0, classification, reason)
                            if classification == "weekend":
                                state.weekend_gaps.append(gap)
                            elif classification == "rollover":
                                state.rollover_gaps.append(gap)
                            else:
                                state.suspicious_gaps.append(gap)

                state.total_ticks += 1
                state.start_ts = ts if state.start_ts is None else min(state.start_ts, ts)
                state.end_ts = ts if state.end_ts is None else max(state.end_ts, ts)
                state.previous_ts = ts
                state.previous_month_key = month_key
                state.previous_mid_by_month[month_key] = (ts, mid)
                state.ticks_per_year[ts.year] += 1
                state.ticks_per_month[label] += 1

                state.price_min = min(state.price_min, bid, ask, mid)
                state.price_max = max(state.price_max, bid, ask, mid)
                previous_ticks_by_month[month_key].append(tick)

                spread = ask - bid
                session = classify_session(ts)
                spread_key = int(round(spread * 1000.0))
                state.spread_hist[session][spread_key] += 1
                state.spread_count[session] += 1
                state.spread_sum[session] += spread
                state.spread_max[session] = max(state.spread_max.get(session, 0.0), spread)
                add_worst_spread(state, session, WorstSpread(spread=spread, timestamp=ts, bid=bid, ask=ask), worst_n)

                for event_name, (event_start, event_end) in KNOWN_EVENT_WINDOWS_UTC.items():
                    samples = state.event_samples[event_name]
                    if len(samples) < 8 and event_start <= ts <= event_end:
                        samples.append((ts, bid, ask))

    return state


def abnormal_years(ticks_per_year: Counter) -> list[tuple[int, int, str]]:
    full_year_counts = [count for year, count in ticks_per_year.items() if 2017 <= year <= 2025]
    if not full_year_counts:
        return []
    median = sorted(full_year_counts)[len(full_year_counts) // 2]
    flagged = []
    for year in sorted(ticks_per_year):
        count = ticks_per_year[year]
        # Partial boundary years can be low for legitimate coverage reasons, but
        # still surface them so the user sees where the sample starts/stops.
        if count < median * 0.60:
            flagged.append((year, count, f"below 60% of full-year median ({median:,})"))
    return flagged


def print_gap_table(title: str, gaps: list[Gap], limit: int) -> None:
    print(f"\n{title}")
    if not gaps:
        print("  none")
        return
    for gap in sorted(gaps, key=lambda item: item.minutes, reverse=True)[:limit]:
        print(
            f"  {fmt_ts(gap.start)} -> {fmt_ts(gap.end)}"
            f" | {gap.minutes:,.1f} min | {gap.reason}"
        )
    if len(gaps) > limit:
        print(f"  ... {len(gaps) - limit:,} more not shown; rerun with --gap-list-limit to expand.")


def print_bad_tick_report(state: AuditState, jump_pct: float) -> None:
    print("\nBAD-TICK COUNTS")
    labels = {
        "ask_lte_bid": "ask <= bid",
        "price_lte_zero": "price <= 0",
        "price_out_of_abs_range": f"price outside {PRICE_ABS_MIN:.0f}-{PRICE_ABS_MAX:.0f}",
        "jump_gt_threshold": f"tick-to-tick mid jump > {jump_pct:g}%",
        "malformed": "malformed rows",
    }
    any_bad = False
    for key, label in labels.items():
        count = state.bad_tick_counts[key]
        if count == 0:
            print(f"  {label:<32} 0")
            continue
        any_bad = True
        months = ", ".join(
            f"{month}:{count:,}" for month, count in state.bad_tick_months[key].most_common(12)
        )
        print(f"  {label:<32} {count:,} | months: {months}")
    if not any_bad:
        print("  PASS: no bad ticks found in audited sample.")
    if state.legacy_price_bound_false_positives:
        print(
            f"  Legacy > ${LEGACY_PRICE_ABS_MAX:,.0f}/oz bound false positives: "
            f"{state.legacy_price_bound_false_positives:,} ticks now kept as valid ATH-regime data."
        )
    if state.cleared_jump_candidates:
        start, end = CLEARED_ATH_CRASH_WINDOW
        print(
            "  Cleared ATH/crash jump candidates not reprinted: "
            f"{state.cleared_jump_candidates:,} "
            f"({fmt_ts(start)} -> {fmt_ts(end)})"
        )


def print_jump_samples(state: AuditState) -> None:
    print("\nJUMP SAMPLE CONTEXT")
    if not state.jump_samples:
        print("  none")
        return
    for index, sample in enumerate(state.jump_samples, start=1):
        print(f"  Jump {index}: {sample.pct:.6f}% at {fmt_ts(sample.current.timestamp)}")
        rows = [("before", tick) for tick in sample.before]
        rows.append(("jump", sample.current))
        rows.extend(("after", tick) for tick in sample.after)
        for label, tick in rows:
            print(
                f"    {label:<6} {fmt_ts(tick.timestamp):<27}"
                f" bid={tick.bid:>9.3f} ask={tick.ask:>9.3f}"
                f" mid={tick.mid:>9.3f} spread={tick.spread:>6.3f}"
            )


def print_worst_spreads(state: AuditState) -> None:
    print("\nWORST SPREAD SAMPLES BY SESSION")
    print("  Session        spread   timestamp                    bid        ask")
    for session in SESSION_ORDER:
        samples = sorted(state.worst_spreads[session], key=lambda item: item.spread, reverse=True)
        if not samples:
            print(f"  {SESSION_LABELS[session]:<13} none")
            continue
        for sample in samples:
            print(
                f"  {SESSION_LABELS[session]:<13} {sample.spread:>7.3f}"
                f"   {fmt_ts(sample.timestamp):<27} {sample.bid:>9.3f}  {sample.ask:>9.3f}"
            )


def print_verdict(state: AuditState, flagged_years: list[tuple[int, int, str]]) -> None:
    hard_issues = []
    review_issues = []
    if state.bad_tick_counts["ask_lte_bid"]:
        hard_issues.append("ask <= bid ticks found")
    if state.bad_tick_counts["price_lte_zero"]:
        hard_issues.append("non-positive prices found")
    if state.bad_tick_counts["price_out_of_abs_range"]:
        hard_issues.append("prices outside absolute sanity band found")
    if state.out_of_order:
        hard_issues.append("out-of-order timestamps found")
    if state.suspicious_gaps:
        hard_issues.append("suspicious intra-session gaps found")
    if flagged_years:
        review_issues.append("low-count sampled year(s)")
    uncleared_jumps = state.bad_tick_counts["jump_gt_threshold"] - state.cleared_jump_candidates
    if uncleared_jumps:
        review_issues.append("new large tick-to-tick jumps were flagged for manual inspection")

    print("\nONE-SCREEN VERDICT")
    if not hard_issues and not review_issues:
        print("  PASS: sampled data is clean enough to commit to a full pass.")
        return
    if hard_issues:
        print("  HOLD: fix or explain these issues before the full pass:")
        for issue in hard_issues:
            print(f"    - {issue}")
    if review_issues:
        print("  PASS WITH REVIEW NOTES: sampled data is clean enough for a full pass if the listed jumps are confirmed real:")
        for issue in review_issues:
            print(f"    - {issue}")


def print_report(state: AuditState, gap_minutes: float, gap_list_limit: int, jump_pct: float) -> None:
    print("\nXAUUSD Dukascopy Tick Data Audit")
    print("=" * 34)
    print(f"Files audited: {len(state.files)}")
    for path in state.files:
        print(f"  - {path}")
    if state.sample_months is not None:
        months = ", ".join(str(month) for month in sorted(state.sample_months))
        years = (
            f"{min(state.sample_years)}-{max(state.sample_years)}"
            if state.sample_years
            else "all years"
        )
        print(f"Stratified sample: months {months}; years {years}")
        print(f"Rows scanned for filtering: {state.scanned_rows:,}")

    print("\nSCHEMA")
    print("  PASS: bid and ask columns are present. Spread analysis is possible.")

    print("\nCOVERAGE")
    print(f"  Start:       {fmt_ts(state.start_ts)}")
    print(f"  End:         {fmt_ts(state.end_ts)}")
    print(f"  Total ticks: {state.total_ticks:,}")
    if state.bad_rows:
        print(f"  Bad rows skipped: {state.bad_rows:,}")
    if state.out_of_order:
        print(f"  WARNING: {state.out_of_order:,} out-of-order timestamp transitions found.")
    print("  Ticks per year:")
    for year in sorted(state.ticks_per_year):
        print(f"    {year}: {state.ticks_per_year[year]:,}")
    if state.sample_months is not None:
        print("  Ticks per sampled month:")
        for label in sorted(state.ticks_per_month):
            print(f"    {label}: {state.ticks_per_month[label]:,}")

    flagged_years = abnormal_years(state.ticks_per_year)
    print("\nLOW-COUNT YEAR FLAGS")
    if flagged_years:
        for year, count, reason in flagged_years:
            print(f"  WARNING: {year}: {count:,} ticks, {reason}")
    else:
        print("  none")

    print_bad_tick_report(state, jump_pct)
    print_jump_samples(state)

    print(f"\nGAPS > {gap_minutes:g} MINUTES")
    print("  Gap counts:")
    print(f"    expected weekend/holiday closures: {len(state.weekend_gaps):,}")
    print(f"    expected daily-rollover pauses: {len(state.rollover_gaps):,}")
    print(f"    suspicious intra-session gaps:  {len(state.suspicious_gaps):,}")
    print_gap_table("  Suspicious intra-session/data gaps", state.suspicious_gaps, gap_list_limit)
    print_gap_table("  Expected weekend/holiday closures", state.weekend_gaps, gap_list_limit)
    print_gap_table("  Expected daily-rollover pauses", state.rollover_gaps, gap_list_limit)

    print("\nTIMEZONE")
    print("  Assumption: Dukascopy timestamps are GMT/UTC by default; this script treats naive CSV timestamps as UTC.")
    print("  Later session tagging must be DST-aware because London and New York shift on different dates.")
    print("  Known-event timestamp samples for manual alignment:")
    for event_name, samples in state.event_samples.items():
        print(f"  {event_name}:")
        if not samples:
            print("    no rows found in the configured window")
            continue
        for ts, bid, ask in samples[:8]:
            print(f"    {fmt_ts(ts)} | bid {bid:.3f} | ask {ask:.3f} | spread {ask - bid:.3f}")

    print("\nCONTRACT SANITY")
    print(f"  Assumption: standard gold lot = {STD_LOT_OZ:.0f} oz; 0.01 lot = {MICRO_LOT_OZ:.0f} oz.")
    if math.isfinite(state.price_min) and math.isfinite(state.price_max):
        print(f"  Observed price range: {state.price_min:.3f} to {state.price_max:.3f} USD/oz.")
        if state.price_min < PRICE_ABS_MIN or state.price_max > PRICE_ABS_MAX:
            print(
                "  WARNING: price magnitude is outside the absolute XAUUSD sanity band "
                f"({PRICE_ABS_MIN:.0f}-{PRICE_ABS_MAX:.0f}); inspect symbol/decimal scaling."
            )
        elif state.price_max > 3000:
            print("  PASS: price includes the 2026 ATH regime and remains inside the updated absolute sanity band.")
        else:
            print("  PASS: price magnitude is consistent with spot gold.")

    print("\nDUKASCOPY INTERBANK SPREAD - REFERENCE ONLY, NOT my IUX trading cost")
    print("  Session        ticks             mean    median       p90       max  ($/oz)")
    for session in SESSION_ORDER:
        count = state.spread_count[session]
        if count == 0:
            print(f"  {SESSION_LABELS[session]:<13} {'0':>12}        n/a       n/a       n/a       n/a")
            continue
        mean = state.spread_sum[session] / count
        median = percentile_from_hist(state.spread_hist[session], 0.50)
        p90 = percentile_from_hist(state.spread_hist[session], 0.90)
        max_spread = state.spread_max[session]
        print(
            f"  {SESSION_LABELS[session]:<13} {count:>12,}"
            f"   {mean:>8.3f}  {median:>8.3f}  {p90:>8.3f}  {max_spread:>8.3f}"
        )

    print_worst_spreads(state)

    print("\nBACKTEST-COMPROMISE FLAGS")
    issues = []
    if state.bad_rows:
        issues.append(f"{state.bad_rows:,} malformed or crossed-spread rows were skipped")
    if state.out_of_order:
        issues.append("timestamps are not strictly sorted")
    if state.suspicious_gaps:
        issues.append(f"{len(state.suspicious_gaps):,} suspicious market-hours gaps exceed the threshold")
    if flagged_years:
        issues.append("one or more years have abnormally low tick counts")
    if (
        state.bad_tick_counts["ask_lte_bid"]
        or state.bad_tick_counts["price_lte_zero"]
        or state.bad_tick_counts["price_out_of_abs_range"]
    ):
        issues.append("bad ticks with invalid bid/ask or price values were found")
    uncleared_jumps = state.bad_tick_counts["jump_gt_threshold"] - state.cleared_jump_candidates
    if uncleared_jumps:
        issues.append("new large tick-to-tick jumps need manual inspection")
    if not issues:
        print("  none detected by this audit")
    else:
        for issue in issues:
            print(f"  WARNING: {issue}")
    print_verdict(state, flagged_years)


def default_paths() -> list[Path]:
    return sorted(Path("data").glob("*XAUUSD*.csv"))


def parse_int_set(raw: str, minimum: int, maximum: int, label: str) -> set[int]:
    values = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < minimum or value > maximum:
            raise argparse.ArgumentTypeError(f"{label} must be between {minimum} and {maximum}: {value}")
        values.add(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{label} list is empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="CSV tick files; defaults to data/*XAUUSD*.csv")
    parser.add_argument("--gap-minutes", type=float, default=30.0, help="report timestamp gaps longer than this")
    parser.add_argument("--gap-list-limit", type=int, default=25, help="max gaps to print in each gap section")
    parser.add_argument("--max-rows", type=int, default=None, help="optional smoke-test row limit")
    parser.add_argument(
        "--sample-months",
        nargs="?",
        const="1,7",
        default=None,
        help="audit only selected months, comma-separated; with no value uses January and July",
    )
    parser.add_argument(
        "--sample-years",
        default="2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026",
        help="years used with --sample-months, comma-separated",
    )
    parser.add_argument("--jump-pct", type=float, default=1.0, help="flag tick-to-tick mid jumps above this percent")
    parser.add_argument("--worst-spread-samples", type=int, default=5, help="worst spread samples to print per session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or default_paths()
    sample_months = None
    sample_years = None
    if args.sample_months is not None:
        sample_months = parse_int_set(args.sample_months, 1, 12, "month")
        sample_years = parse_int_set(args.sample_years, 1900, 2200, "year")
    state = audit(
        paths,
        gap_minutes=args.gap_minutes,
        max_rows=args.max_rows,
        sample_months=sample_months,
        sample_years=sample_years,
        jump_pct=args.jump_pct,
        worst_n=args.worst_spread_samples,
    )
    print_report(state, gap_minutes=args.gap_minutes, gap_list_limit=args.gap_list_limit, jump_pct=args.jump_pct)


if __name__ == "__main__":
    main()
