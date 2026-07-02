"""
H-2026-SESS-01: session-conditional quality audit.

Research-only. Does not touch the live bot.

Pre-registered mechanism:
Compression breakouts during high-participation UTC sessions should be more
reliable than thin-liquidity Asian-session breakouts. Asian-session signals are
hypothesized to have lower net R than London/overlap signals.

Pre-registered buckets:
- ASIA:        23:00-06:59 UTC
- LONDON_ONLY: 07:00-11:59 UTC
- OVERLAP:     12:00-15:59 UTC
- NY_LATE:     16:00-21:44 UTC
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
RR = 1.5
HORIZON = 10
SPREAD = 0.20
BOOT_N = 1000
PLACEBO_N = 1000
SEED = 20260702
SESSION_ORDER = ("ASIA", "LONDON_ONLY", "OVERLAP", "NY_LATE")


@dataclass(frozen=True)
class SessionTrade:
    event_id: int
    entry_time: datetime
    session_bucket: str
    atr_regime: str
    duration_bucket: str
    net_r: float
    win: bool
    mfe_r: float
    atr: float


def q(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(vals: list[float], seed: str) -> tuple[float, float]:
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(BOOT_N):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N)]


def session_bucket(ts: datetime) -> str:
    hour = ts.hour
    minute = ts.minute
    if hour >= 23 or hour <= 6:
        return "ASIA"
    if 7 <= hour <= 11:
        return "LONDON_ONLY"
    if 12 <= hour <= 15:
        return "OVERLAP"
    if 16 <= hour < 21 or (hour == 21 and minute <= 44):
        return "NY_LATE"
    return "OUTSIDE_PRE_REGISTERED"


def is_individual_compressed(bars: list[base.DeltaBar], i: int) -> bool:
    cutoff = base.trailing_atr_cutoff(bars, i)
    return cutoff is not None and bars[i].atr14 is not None and bars[i].atr14 <= cutoff


def compression_duration(bars: list[base.DeltaBar], setup_end: int) -> int:
    segment = bars[setup_end].segment_id
    duration = 0
    i = setup_end
    while i >= 0 and bars[i].segment_id == segment and is_individual_compressed(bars, i):
        duration += 1
        i -= 1
    return duration


def duration_bucket_map(events: list[ablate.Event], durations: dict[int, int]) -> dict[int, str]:
    vals = [float(durations[e.event_id]) for e in events]
    lo, hi = q(vals, 1 / 3), q(vals, 2 / 3)
    out = {}
    for e in events:
        d = durations[e.event_id]
        if d <= lo:
            out[e.event_id] = "short"
        elif d <= hi:
            out[e.event_id] = "medium"
        else:
            out[e.event_id] = "long"
    return out


def trailing_atr_env(bars: list[base.DeltaBar], i: int) -> float | None:
    vals = []
    j = i - 1
    while j >= 0 and len(vals) < base.ATR_TRAIL:
        if bars[j].atr14 is not None:
            vals.append(bars[j].atr14)
        j -= 1
    return mean(vals) if len(vals) == base.ATR_TRAIL else None


def atr_regime_map(bars: list[base.DeltaBar], events: list[ablate.Event]) -> dict[int, str]:
    envs = {e.event_id: trailing_atr_env(bars, e.setup_end) for e in events}
    vals = [v for v in envs.values() if v is not None]
    lo, hi = q(vals, 1 / 3), q(vals, 2 / 3)
    out = {}
    for eid, val in envs.items():
        if val is None:
            out[eid] = "unknown"
        elif val <= lo:
            out[eid] = "low"
        elif val <= hi:
            out[eid] = "mid"
        else:
            out[eid] = "high"
    return out


def mfe_for_event(bars: list[base.DeltaBar], event: ablate.Event, risk: float) -> float | None:
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    entry = event.range_high if event.direction == 1 else event.range_low
    end_index = simple.segment_end_index(bars, eval_start, HORIZON)
    mfe = 0.0
    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        if event.direction == 1:
            mfe = max(mfe, bar.high - entry)
        else:
            mfe = max(mfe, entry - bar.low)
    return mfe / risk


def build_trades(bars: list[base.DeltaBar]) -> list[SessionTrade]:
    events = ablate.detect_compression(bars)
    durations = {e.event_id: compression_duration(bars, e.setup_end) for e in events}
    dur_buckets = duration_bucket_map(events, durations)
    regimes = atr_regime_map(bars, events)
    rows: list[SessionTrade] = []
    for event in events:
        trade = ablate.simulate("XAUUSD", bars, event, "session_quality", RR, HORIZON, SPREAD, "range_edge")
        risk = ablate.risk_at_setup_end(bars, event)
        if trade is None or risk is None:
            continue
        bucket = session_bucket(trade.entry_time)
        if bucket == "OUTSIDE_PRE_REGISTERED":
            continue
        mfe = mfe_for_event(bars, event, risk)
        if mfe is None:
            continue
        rows.append(
            SessionTrade(
                event_id=event.event_id,
                entry_time=trade.entry_time,
                session_bucket=bucket,
                atr_regime=regimes[event.event_id],
                duration_bucket=dur_buckets[event.event_id],
                net_r=trade.net_r,
                win=trade.win,
                mfe_r=mfe,
                atr=risk,
            )
        )
    return rows


def period_rows(rows: list[SessionTrade], period: str) -> list[SessionTrade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def summary_line(period: str, bucket: str, rows: list[SessionTrade]) -> str:
    subset = [r for r in period_rows(rows, period) if r.session_bucket == bucket]
    if not subset:
        return f"{period},{bucket},0,n/a,n/a,n/a,n/a,n/a,n/a"
    vals = [r.net_r for r in subset]
    lo, hi = bootstrap_ci(vals, f"{SEED}-{period}-{bucket}")
    return (
        f"{period},{bucket},{len(subset)},{sum(r.win for r in subset)/len(subset):.2%},"
        f"{mean(vals):.4f},{lo:.4f},{hi:.4f},{mean([r.mfe_r for r in subset]):.4f},{mean([r.atr for r in subset]):.4f}"
    )


def placebo_spread_p(rows: list[SessionTrade]) -> tuple[float, float]:
    observed_means = {
        b: mean([r.net_r for r in rows if r.session_bucket == b])
        for b in SESSION_ORDER
        if any(r.session_bucket == b for r in rows)
    }
    observed = max(observed_means.values()) - min(observed_means.values())
    labels = [r.session_bucket for r in rows]
    rng = random.Random(f"{SEED}-session-placebo")
    count = 0
    for _ in range(PLACEBO_N):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        groups = {b: [] for b in SESSION_ORDER}
        for r, b in zip(rows, shuffled):
            groups[b].append(r.net_r)
        means = [mean(v) for v in groups.values() if v]
        stat = max(means) - min(means)
        if stat >= observed:
            count += 1
    return observed, (count + 1) / (PLACEBO_N + 1)


def crosstab(rows: list[SessionTrade], attr: str, values: list[str]) -> list[str]:
    out = [f"session,{attr},n"]
    for session in SESSION_ORDER:
        for val in values:
            n = sum(1 for r in rows if r.session_bucket == session and getattr(r, attr) == val)
            if n:
                out.append(f"{session},{val},{n}")
    return out


def fnum(raw: object) -> float | None:
    try:
        if raw in ("", None):
            return None
        val = float(raw)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def parse_dt(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def live_spread_by_session(path: Path) -> list[str]:
    lines = ["session,n,mean_spread,median_spread,min_spread,max_spread"]
    if not path.exists():
        lines.append(f"NO_LIVE_LOG,0,n/a,n/a,n/a,n/a")
        return lines
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {b: [] for b in SESSION_ORDER}
    for row in rows:
        if str(row.get("event", "")).lower() != "entry":
            continue
        spread = fnum(row.get("real_spread_at_entry"))
        ts = parse_dt(row.get("signal_time") or row.get("timestamp_utc") or "")
        if spread is None or ts is None:
            continue
        bucket = session_bucket(ts)
        if bucket in groups:
            groups[bucket].append(spread)
    for bucket in SESSION_ORDER:
        vals = groups[bucket]
        if vals:
            lines.append(f"{bucket},{len(vals)},{mean(vals):.4f},{median(vals):.4f},{min(vals):.4f},{max(vals):.4f}")
        else:
            lines.append(f"{bucket},0,n/a,n/a,n/a,n/a")
    return lines


def decision_result(rows: list[SessionTrade]) -> str:
    candidates = []
    for bucket in SESSION_ORDER:
        train = [r.net_r for r in period_rows(rows, "train") if r.session_bucket == bucket]
        test = [r.net_r for r in period_rows(rows, "test") if r.session_bucket == bucket]
        if not train or not test:
            continue
        _, train_hi = bootstrap_ci(train, f"{SEED}-decision-train-{bucket}")
        _, test_hi = bootstrap_ci(test, f"{SEED}-decision-test-{bucket}")
        if train_hi < 0 and test_hi < 0:
            candidates.append(bucket)
    if candidates:
        return f"FILTER_CANDIDATE sessions={','.join(candidates)} because CI upper bound < 0 in both train and test; still requires separate post-walk-forward pre-registration."
    return "FAIL_FILTER_RULE no session has significantly negative net R in both train and test; trade all sessions under current evidence."


def update_registry(path: Path, result: str | None = None) -> None:
    registered = (
        "- 2026-07-02: H-2026-SESS-01 registered. Session-conditional compression quality hypothesis: "
        "ASIA 23:00-06:59 UTC should have lower net R than LONDON_ONLY/OVERLAP; pre-registered "
        "four UTC buckets; no boundary tuning; future filter candidate only if a bucket is significantly "
        "negative in both train and test.\n"
    )
    text = path.read_text() if path.exists() else "# Hypothesis Registry\n\n"
    if registered.strip() not in text:
        with path.open("a") as handle:
            if not path.exists() or path.stat().st_size == 0:
                handle.write("# Hypothesis Registry\n\n")
            handle.write(registered)
    if result:
        result_line = f"- 2026-07-02: H-2026-SESS-01 result: {result}\n"
        text = path.read_text()
        if result_line.strip() not in text:
            with path.open("a") as handle:
                handle.write(result_line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--live-log", type=Path, default=Path("research/iux_compression_breakout_live_log.csv"))
    parser.add_argument("--registry", type=Path, default=Path("research/hypothesis_registry.md"))
    args = parser.parse_args()

    update_registry(args.registry)
    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    rows = build_trades(bars)
    print("COMPRESSION_SESSION_QUALITY_AUDIT")
    print(f"rules=validated compression; setup-end ATR; range-edge entry; RR={RR}; horizon={HORIZON}; spread={SPREAD}")
    print("buckets=ASIA 23:00-06:59; LONDON_ONLY 07:00-11:59; OVERLAP 12:00-15:59; NY_LATE 16:00-21:44 UTC")
    print(f"trades_in_registered_buckets={len(rows)}")
    print("\nBUCKET_TABLE")
    print("period,session,n,win_rate,mean_net_r,ci_low,ci_high,mean_mfe_r,mean_atr")
    for period in ("train", "test"):
        for bucket in SESSION_ORDER:
            print(summary_line(period, bucket, rows))
    print("\nPLACEBO")
    observed, p = placebo_spread_p(rows)
    print(f"cross_bucket_net_r_spread={observed:.4f},p={p:.4f},n_shuffle={PLACEBO_N}")
    print("\nCONFOUND_SESSION_X_VOL_TERCILE")
    for line in crosstab(rows, "atr_regime", ["low", "mid", "high", "unknown"]):
        print(line)
    print("\nCONFOUND_SESSION_X_DURATION_TERCILE")
    for line in crosstab(rows, "duration_bucket", ["short", "medium", "long"]):
        print(line)
    print("\nLIVE_SPREAD_BY_SESSION_DESCRIPTIVE")
    for line in live_spread_by_session(args.live_log):
        print(line)
    result = decision_result(rows)
    print("\nVERDICT")
    print(result)
    update_registry(args.registry, result)


if __name__ == "__main__":
    main()
