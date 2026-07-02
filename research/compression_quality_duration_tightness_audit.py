"""
Research Track B: compression duration/tightness quality audit.

Analysis-only. Does not touch the live bot or change entry/exit rules.

Pre-registered hypothesis:
- Longer compression duration and tighter ranges should produce higher MFE and
  higher net R.
- Buckets are terciles only. No threshold search.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev

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


@dataclass(frozen=True)
class QualityTrade:
    event_id: int
    entry_time: datetime
    session: str
    atr_regime: str
    duration: int
    tightness: float
    net_r: float
    win: bool
    mfe_r: float


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


def build_trades(bars: list[base.DeltaBar]) -> list[QualityTrade]:
    events = ablate.detect_compression(bars)
    regimes = atr_regime_map(bars, events)
    rows: list[QualityTrade] = []
    for event in events:
        trade = ablate.simulate("XAUUSD", bars, event, "quality", RR, HORIZON, SPREAD, "range_edge")
        risk = ablate.risk_at_setup_end(bars, event)
        if trade is None or risk is None:
            continue
        mfe = mfe_for_event(bars, event, risk)
        if mfe is None:
            continue
        tightness = (event.range_high - event.range_low) / risk
        rows.append(
            QualityTrade(
                event.event_id,
                trade.entry_time,
                bars[event.breakout_index].session,
                regimes[event.event_id],
                compression_duration(bars, event.setup_end),
                tightness,
                trade.net_r,
                trade.win,
                mfe,
            )
        )
    return rows


def tercile_labels(rows: list[QualityTrade], attr: str, lower_is_better: bool = False) -> dict[int, str]:
    vals = [getattr(r, attr) for r in rows]
    lo, hi = q(vals, 1 / 3), q(vals, 2 / 3)
    out = {}
    for r in rows:
        val = getattr(r, attr)
        if val <= lo:
            bucket = "tight" if lower_is_better else "short"
        elif val <= hi:
            bucket = "medium"
        else:
            bucket = "loose" if lower_is_better else "long"
        out[r.event_id] = bucket
    return out


def summarize_bucket(rows: list[QualityTrade], label: str, bucket: str) -> str:
    vals = [r.net_r for r in rows]
    mfes = [r.mfe_r for r in rows]
    lo, hi = bootstrap_ci(vals, f"{SEED}-{label}-{bucket}-net")
    mfe_lo, mfe_hi = bootstrap_ci(mfes, f"{SEED}-{label}-{bucket}-mfe")
    return (
        f"{label},{bucket},{len(rows)},{sum(r.win for r in rows)/len(rows):.2%},"
        f"{mean(vals):.4f},{lo:.4f},{hi:.4f},{mean(mfes):.4f},{mfe_lo:.4f},{mfe_hi:.4f},"
        f"{min(getattr(r, label) for r in rows):.4f},{median([getattr(r, label) for r in rows]):.4f},{max(getattr(r, label) for r in rows):.4f}"
    )


def bucket_table(rows: list[QualityTrade], attr: str, lower_is_better: bool = False) -> tuple[list[str], dict[str, list[QualityTrade]]]:
    labels = tercile_labels(rows, attr, lower_is_better)
    order = ["short", "medium", "long"] if not lower_is_better else ["tight", "medium", "loose"]
    groups = {name: [r for r in rows if labels[r.event_id] == name] for name in order}
    return [summarize_bucket(groups[name], attr, name) for name in order if groups[name]], groups


def monotonic(values: list[float], increasing: bool = True) -> bool:
    return all(a <= b for a, b in zip(values, values[1:])) if increasing else all(a >= b for a, b in zip(values, values[1:]))


def spread_stat(groups: dict[str, list[QualityTrade]], order: list[str], metric: str) -> float:
    first, last = groups[order[0]], groups[order[-1]]
    if metric == "net":
        return mean([r.net_r for r in last]) - mean([r.net_r for r in first])
    if metric == "mfe":
        return mean([r.mfe_r for r in last]) - mean([r.mfe_r for r in first])
    raise ValueError(metric)


def placebo_p(rows: list[QualityTrade], attr: str, lower_is_better: bool, metric: str) -> tuple[float, float]:
    _, groups = bucket_table(rows, attr, lower_is_better)
    order = ["short", "medium", "long"] if not lower_is_better else ["loose", "medium", "tight"]
    observed = spread_stat(groups, order, metric)
    labels = tercile_labels(rows, attr, lower_is_better)
    bucket_names = [labels[r.event_id] for r in rows]
    rng = random.Random(f"{SEED}-placebo-{attr}-{metric}")
    count = 0
    for _ in range(PLACEBO_N):
        shuffled = bucket_names[:]
        rng.shuffle(shuffled)
        fake = {name: [] for name in set(bucket_names)}
        for r, b in zip(rows, shuffled):
            fake[b].append(r)
        stat = spread_stat(fake, order, metric)
        if stat >= observed:
            count += 1
    return observed, (count + 1) / (PLACEBO_N + 1)


def period_rows(rows: list[QualityTrade], period: str) -> list[QualityTrade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def cross_tab(rows: list[QualityTrade]) -> list[str]:
    dur_labels = tercile_labels(rows, "duration")
    sessions = sorted(set(r.session for r in rows))
    regimes = ["low", "mid", "high", "unknown"]
    out = ["duration_bucket,atr_regime,session,n"]
    for bucket in ("short", "medium", "long"):
        for regime in regimes:
            for session in sessions:
                n = sum(1 for r in rows if dur_labels[r.event_id] == bucket and r.atr_regime == regime and r.session == session)
                if n:
                    out.append(f"{bucket},{regime},{session},{n}")
    return out


def print_analysis(rows: list[QualityTrade], period: str) -> None:
    subset = period_rows(rows, period)
    print(f"\nPERIOD={period},n={len(subset)}")
    print("metric,bucket,n,win_rate,mean_net_r,net_ci_low,net_ci_high,mean_mfe_r,mfe_ci_low,mfe_ci_high,min_metric,median_metric,max_metric")
    dur_lines, dur_groups = bucket_table(subset, "duration")
    tight_lines, tight_groups = bucket_table(subset, "tightness", True)
    for line in dur_lines + tight_lines:
        print(line)
    dur_net = [mean([r.net_r for r in dur_groups[b]]) for b in ("short", "medium", "long")]
    dur_mfe = [mean([r.mfe_r for r in dur_groups[b]]) for b in ("short", "medium", "long")]
    tight_net = [mean([r.net_r for r in tight_groups[b]]) for b in ("tight", "medium", "loose")]
    tight_mfe = [mean([r.mfe_r for r in tight_groups[b]]) for b in ("tight", "medium", "loose")]
    print(
        "MONOTONICITY,"
        f"duration_net_increasing={monotonic(dur_net)},duration_mfe_increasing={monotonic(dur_mfe)},"
        f"tightness_net_decreasing={monotonic(tight_net, increasing=False)},tightness_mfe_decreasing={monotonic(tight_mfe, increasing=False)}"
    )
    if period == "full":
        for attr, lower in (("duration", False), ("tightness", True)):
            for metric in ("net", "mfe"):
                obs, p = placebo_p(subset, attr, lower, metric)
                direction = "high_quality_minus_low_quality"
                print(f"PLACEBO,{attr},{metric},{direction},{obs:.4f},p={p:.4f},n_shuffle={PLACEBO_N}")


def update_registry(path: Path) -> None:
    line = (
        "- 2026-07-02: Track B compression quality audit. Hypothesis: longer compression duration "
        "and tighter range/ATR predict higher MFE and net R. Pre-registered tercile buckets; no "
        "threshold optimization; analysis-only for FDR accounting.\n"
    )
    if path.exists() and line.strip() in path.read_text():
        return
    with path.open("a") as handle:
        if path.stat().st_size == 0:
            handle.write("# Hypothesis Registry\n\n")
        handle.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--registry", type=Path, default=Path("research/hypothesis_registry.md"))
    args = parser.parse_args()

    update_registry(args.registry)
    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    rows = build_trades(bars)
    print("COMPRESSION_QUALITY_DURATION_TIGHTNESS_AUDIT")
    print(f"rules=validated compression; setup-end ATR; range-edge entry; RR={RR}; horizon={HORIZON}; spread={SPREAD}")
    print("duration_definition=consecutive individual bars ending at setup_end where ATR14 <= trailing bottom-tercile ATR cutoff")
    print("tightness_definition=(range_high-range_low)/setup_end_ATR")
    print(f"trades={len(rows)},duration_min={min(r.duration for r in rows)},duration_median={median([r.duration for r in rows])},duration_max={max(r.duration for r in rows)}")
    print_analysis(rows, "full")
    print_analysis(rows, "train")
    print_analysis(rows, "test")
    print("\nCONFOUND_CROSSTAB_DURATION_X_ATR_REGIME_X_SESSION")
    for line in cross_tab(rows):
        print(line)
    print("\nCONCLUSION_GUIDE")
    print("A robust coiled-spring effect requires monotonic high-quality bucket improvement in full, train, and test, plus low placebo p-values. Otherwise treat the binary compression signal as sufficient until a new pre-registered walk-forward test says otherwise.")
    print(f"hypothesis_registry={args.registry}")


if __name__ == "__main__":
    main()
