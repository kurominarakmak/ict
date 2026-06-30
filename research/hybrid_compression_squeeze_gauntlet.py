"""
Research-only gauntlet for Compression ATR, BB/Keltner Squeeze, and hybrids.

This script does not touch any demo/live bot. It reuses the validated research
detectors from squeeze_breakout_gauntlet.py and only composes their event sets.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev

import simple_breakout_atr_exit_audit as simple
import squeeze_breakout_gauntlet as sq
import volatility_compression_breakout_audit as base


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
RRS = (1.5, 2.0)
HORIZON = 10
BOOTSTRAP_N = 1000
RANDOM_SEED = 20260630
DEDUP_WINDOW_BARS = 4
RANGE_NEAR_RISK_FRAC = 0.10
GATES: tuple[float | None, ...] = (None, 0.10, 0.15)

CRISIS_WINDOWS = [
    ("covid_2020", datetime(2020, 2, 20, tzinfo=timezone.utc), datetime(2020, 8, 31, 23, 59, tzinfo=timezone.utc)),
    ("war_2022", datetime(2022, 2, 1, tzinfo=timezone.utc), datetime(2022, 5, 31, 23, 59, tzinfo=timezone.utc)),
    ("gold_blowoff_2025", datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)),
]


@dataclass(frozen=True)
class PortfolioEvent:
    strategy: str
    tag: str
    source_family: str
    event: sq.Event


@dataclass(frozen=True)
class ResultTrade:
    symbol: str
    strategy: str
    tag: str
    source_family: str
    signal: str
    rr: float
    gate: str
    event_id: int
    setup_end: int
    breakout_index: int
    entry_time: datetime
    session: str
    atr_regime: str
    year: int
    net_r: float
    gross_r: float
    exit_reason: str
    bars_held: int
    risk: float
    spread_r: float
    random_net_r: float
    random_gross_r: float


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def bootstrap_ci(vals: list[float], seed: str) -> tuple[float, float]:
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(BOOTSTRAP_N):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N)]


def period_rows(rows: list[ResultTrade], period: str) -> list[ResultTrade]:
    if period == "all":
        return rows
    if period == "train_2016_2021":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test_2022_2026":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def crisis_window(ts: datetime) -> str:
    for name, start, end in CRISIS_WINDOWS:
        if start <= ts <= end:
            return name
    return "outside_crisis"


def segment_years(rows: list[ResultTrade]) -> float:
    if len(rows) < 2:
        return math.nan
    ordered = sorted(rows, key=lambda r: r.entry_time)
    days = (ordered[-1].entry_time - ordered[0].entry_time).days
    return days / 365.25 if days > 0 else math.nan


def equity_metrics(rows: list[ResultTrade]) -> dict[str, float]:
    vals = [r.net_r for r in sorted(rows, key=lambda r: (r.entry_time, r.event_id, r.strategy))]
    if not vals:
        return {"sharpe": math.nan, "t_stat": math.nan, "max_dd": math.nan, "longest_loss": 0, "worst20": math.nan, "worst_month": math.nan}
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    avg = mean(vals)
    sharpe = avg / sd * math.sqrt(len(vals)) if sd > 0 else math.nan
    t_stat = avg / (sd / math.sqrt(len(vals))) if sd > 0 else math.nan
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    longest = 0
    current = 0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        if v <= 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    worst20 = min((sum(vals[i : i + 20]) for i in range(len(vals) - 19)), default=math.nan)
    monthly: dict[tuple[int, int], float] = {}
    for r in rows:
        key = (r.entry_time.year, r.entry_time.month)
        monthly[key] = monthly.get(key, 0.0) + r.net_r
    worst_month = min(monthly.values()) if monthly else math.nan
    return {"sharpe": sharpe, "t_stat": t_stat, "max_dd": max_dd, "longest_loss": longest, "worst20": worst20, "worst_month": worst_month}


def profit_factor(rows: list[ResultTrade]) -> float:
    wins = sum(r.net_r for r in rows if r.net_r > 0)
    losses = -sum(r.net_r for r in rows if r.net_r < 0)
    return wins / losses if losses > 0 else math.inf


def summarize(rows: list[ResultTrade], seed: str, bootstrap: bool = True) -> dict[str, float]:
    vals = [r.net_r for r in rows]
    gross = [r.gross_r for r in rows]
    n, m, normal_lo, normal_hi = ci(vals)
    boot_lo, boot_hi = bootstrap_ci(vals, seed) if bootstrap else (normal_lo, normal_hi)
    years = segment_years(rows)
    em = equity_metrics(rows)
    spread = [r.spread_r for r in rows]
    return {
        "n": n,
        "trades_per_year": n / years if years and years > 0 else math.nan,
        "win_rate": sum(v > 0 for v in vals) / n if n else math.nan,
        "gross": mean(gross) if gross else math.nan,
        "net": m,
        "ci_low": boot_lo,
        "ci_high": boot_hi,
        "normal_ci_low": normal_lo,
        "normal_ci_high": normal_hi,
        "profit_factor": profit_factor(rows),
        "sharpe": em["sharpe"],
        "t_stat": em["t_stat"],
        "max_dd": em["max_dd"],
        "longest_loss": em["longest_loss"],
        "worst20": em["worst20"],
        "worst_month": em["worst_month"],
        "avg_spread_r": mean(spread) if spread else math.nan,
        "median_spread_r": median(spread) if spread else math.nan,
        "spread_gt_010": sum(s > 0.10 for s in spread) / n if n else math.nan,
        "spread_gt_015": sum(s > 0.15 for s in spread) / n if n else math.nan,
        "sl_pct": sum(r.exit_reason == "stop" and r.gross_r <= -0.999 for r in rows) / n if n else math.nan,
        "tp_pct": sum(r.exit_reason == "target" for r in rows) / n if n else math.nan,
        "time_exit_pct": sum(r.exit_reason == "force_close" for r in rows) / n if n else math.nan,
        "total_r": sum(vals),
        "annual_r": sum(vals) / years if years and years > 0 else math.nan,
    }


def trailing_atr_env(bars: list[base.DeltaBar], i: int) -> float | None:
    vals: list[float] = []
    j = i - 1
    while j >= 0 and len(vals) < base.ATR_TRAIL:
        if bars[j].atr14 is not None:
            vals.append(bars[j].atr14)
        j -= 1
    return mean(vals) if len(vals) == base.ATR_TRAIL else None


def percentile(vals: list[float], q: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def atr_regimes(bars: list[base.DeltaBar], events: list[sq.Event]) -> dict[tuple[str, int], str]:
    envs_by_event = {(e.family, e.event_id): trailing_atr_env(bars, e.setup_end) for e in events}
    envs = [v for v in envs_by_event.values() if v is not None]
    low = percentile(envs, 1 / 3)
    high = percentile(envs, 2 / 3)
    out = {}
    for key, env in envs_by_event.items():
        if env is None:
            continue
        out[key] = "low" if env <= low else ("mid" if env <= high else "high")
    return out


def range_near(a: sq.Event, b: sq.Event, bars: list[base.DeltaBar]) -> bool:
    risk_a = bars[a.setup_end].atr14
    risk_b = bars[b.setup_end].atr14
    risk = max(r for r in (risk_a, risk_b) if r is not None) if risk_a or risk_b else 0.0
    tol = max(0.01, RANGE_NEAR_RISK_FRAC * risk)
    return abs(a.range_high - b.range_high) <= tol and abs(a.range_low - b.range_low) <= tol


def active_overlap(a: sq.Event, b: sq.Event) -> bool:
    return max(a.setup_start, b.setup_start) <= min(a.breakout_index, b.breakout_index)


def portfolio_events(strategy: str, comp: list[sq.Event], squeeze: list[sq.Event], bars: list[base.DeltaBar]) -> tuple[list[PortfolioEvent], dict[str, int]]:
    stats = {"exact_both": 0, "dedup_same": 0, "dedup_conflict_skip": 0, "intersection_pairs": 0}
    if strategy == "compression_only":
        return [PortfolioEvent(strategy, "COMPRESSION_ONLY", "compression", e) for e in comp], stats
    if strategy == "squeeze_only":
        return [PortfolioEvent(strategy, "SQUEEZE_ONLY", "squeeze", e) for e in squeeze], stats
    if strategy == "union_raw":
        by_bar: dict[int, list[sq.Event]] = {}
        for e in comp + squeeze:
            by_bar.setdefault(e.breakout_index, []).append(e)
        out = []
        for _, events in sorted(by_bar.items()):
            families = {e.family for e in events}
            if len(families) > 1:
                stats["exact_both"] += 1
                chosen = sorted(events, key=lambda e: (e.breakout_index, e.family))[0]
                tag = "BOTH_EXACT"
            else:
                chosen = events[0]
                tag = "COMPRESSION_ONLY" if chosen.family == "compression" else "SQUEEZE_ONLY"
            out.append(PortfolioEvent(strategy, tag, chosen.family, chosen))
        return out, stats
    if strategy == "intersection":
        out = []
        used: set[int] = set()
        for ce in comp:
            matches = [
                se
                for se in squeeze
                if se.event_id not in used
                and ce.direction == se.direction
                and active_overlap(ce, se)
                and abs(ce.breakout_index - se.breakout_index) <= DEDUP_WINDOW_BARS
            ]
            if not matches:
                continue
            se = min(matches, key=lambda e: abs(e.breakout_index - ce.breakout_index))
            used.add(se.event_id)
            stats["intersection_pairs"] += 1
            chosen = ce if ce.breakout_index <= se.breakout_index else se
            out.append(PortfolioEvent(strategy, "INTERSECTION", chosen.family, chosen))
        return out, stats
    if strategy == "dedup_hybrid":
        candidates = sorted(
            [PortfolioEvent(strategy, "COMPRESSION_ONLY", "compression", e) for e in comp]
            + [PortfolioEvent(strategy, "SQUEEZE_ONLY", "squeeze", e) for e in squeeze],
            key=lambda p: (p.event.breakout_index, p.event.family),
        )
        out: list[PortfolioEvent] = []
        i = 0
        while i < len(candidates):
            current = candidates[i]
            cluster = [current]
            j = i + 1
            while j < len(candidates) and candidates[j].event.breakout_index - current.event.breakout_index <= DEDUP_WINDOW_BARS:
                cluster.append(candidates[j])
                j += 1
            families = {p.source_family for p in cluster}
            if len(families) == 1:
                out.append(current)
            else:
                first = cluster[0]
                compatible = [
                    p
                    for p in cluster[1:]
                    if p.event.direction == first.event.direction and range_near(p.event, first.event, bars)
                ]
                conflicts = [p for p in cluster[1:] if p.event.direction != first.event.direction]
                if conflicts:
                    stats["dedup_conflict_skip"] += 1
                elif compatible:
                    stats["dedup_same"] += 1
                    out.append(PortfolioEvent(strategy, "BOTH_DEDUP_FIRST", first.source_family, first.event))
                else:
                    out.append(first)
            i = j
        return enforce_one_active(strategy, out), stats
    raise ValueError(strategy)


def enforce_one_active(strategy: str, events: list[PortfolioEvent]) -> list[PortfolioEvent]:
    out: list[PortfolioEvent] = []
    next_allowed = -1
    for p in sorted(events, key=lambda e: e.event.breakout_index):
        if p.event.breakout_index < next_allowed:
            continue
        out.append(p)
        next_allowed = p.event.breakout_index + HORIZON + 1
    return out


def spread_bucket(spread_r: float) -> str:
    if spread_r <= 0.10:
        return "le_0p10"
    if spread_r <= 0.15:
        return "le_0p15"
    return "gt_0p15"


def simulate_direction(
    bars: list[base.DeltaBar],
    event: sq.Event,
    direction: int,
    rr: float,
    spread: float,
) -> tuple[float, float, str, int] | None:
    risk = bars[event.setup_end].atr14
    if risk is None or risk <= 0:
        return None
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None
    entry = event.range_high if event.direction == 1 else event.range_low
    stop = entry - direction * risk
    target = entry + direction * rr * risk
    end_index = simple.segment_end_index(bars, eval_start, HORIZON)
    gross_r = direction * (bars[end_index].close - entry) / risk
    exit_reason = "force_close"
    exit_index = end_index
    for i in range(eval_start, end_index + 1):
        bar = bars[i]
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r = direction * (fill - entry) / risk
            exit_reason = "stop"
            exit_index = i
            break
        if target_hit:
            gross_r = rr
            exit_reason = "target"
            exit_index = i
            break
    return gross_r, gross_r - spread / risk, exit_reason, exit_index - entry_index


def build_result_trades(
    symbol: str,
    bars: list[base.DeltaBar],
    events: list[PortfolioEvent],
    regimes: dict[tuple[str, int], str],
    spread: float,
    gate: float | None,
) -> list[ResultTrade]:
    rng = random.Random(f"{RANDOM_SEED}-{symbol}-{gate}")
    rows: list[ResultTrade] = []
    for p in events:
        event = p.event
        risk = bars[event.setup_end].atr14
        if risk is None or risk <= 0:
            continue
        if gate is not None and spread / risk > gate:
            continue
        rand_dir = 1 if rng.random() >= 0.5 else -1
        for rr in RRS:
            strat = simulate_direction(bars, event, event.direction, rr, spread)
            rnd = simulate_direction(bars, event, rand_dir, rr, spread)
            if strat is None or rnd is None:
                continue
            gross, net, exit_reason, bars_held = strat
            random_gross, random_net, _, _ = rnd
            bar = bars[event.breakout_index]
            rows.append(
                ResultTrade(
                    symbol=symbol,
                    strategy=p.strategy,
                    tag=p.tag,
                    source_family=p.source_family,
                    signal="breakout",
                    rr=rr,
                    gate="none" if gate is None else f"le_{gate:.2f}",
                    event_id=event.event_id,
                    setup_end=event.setup_end,
                    breakout_index=event.breakout_index,
                    entry_time=bar.start,
                    session=bar.session,
                    atr_regime=regimes.get((event.family, event.event_id), "unknown"),
                    year=bar.start.year,
                    net_r=net,
                    gross_r=gross,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    risk=risk,
                    spread_r=spread / risk,
                    random_net_r=random_net,
                    random_gross_r=random_gross,
                )
            )
    return rows


def monthly_rows(rows: list[ResultTrade]) -> list[dict[str, str]]:
    out = []
    groups: dict[tuple[str, str, float, str, int, int], list[ResultTrade]] = {}
    for r in rows:
        groups.setdefault((r.symbol, r.strategy, r.rr, r.gate, r.entry_time.year, r.entry_time.month), []).append(r)
    for key, subset in sorted(groups.items()):
        symbol, strategy, rr, gate, year, month = key
        out.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "rr": f"{rr:.1f}",
                "gate": gate,
                "year": str(year),
                "month": str(month),
                "n": str(len(subset)),
                "net_r": f"{sum(r.net_r for r in subset):.6f}",
                "random_net_r": f"{sum(r.random_net_r for r in subset):.6f}",
            }
        )
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def monthly_net_map(rows: list[ResultTrade]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for r in rows:
        key = (r.entry_time.year, r.entry_time.month)
        out[key] = out.get(key, 0.0) + r.net_r
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trade_csv_rows(rows: list[ResultTrade]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        out.append({k: str(v) for k, v in r.__dict__.items()})
    return out


def make_report(
    contexts: list[str],
    rows: list[ResultTrade],
    comp_events_by_symbol: dict[str, list[sq.Event]],
    sq_events_by_symbol: dict[str, list[sq.Event]],
    event_stats: dict[tuple[str, str], dict[str, int]],
) -> str:
    lines: list[str] = []
    lines.append("HYBRID_COMPRESSION_SQUEEZE_GAUNTLET_CONTEXT")
    lines.append("research_only=true")
    lines.append("logic=existing compression detector + existing squeeze detector; no live/demo bot touched")
    lines.append("hybrid_conflict_rule=within 4 bars opposite direction skips cluster; same direction and range within 0.10R de-dupes to first signal")
    lines.append("portfolio_rule=max one active trade per symbol for union/intersection/hybrid via 10-bar non-overlap")
    lines.append("random_benchmark=same event timing/count/SL/TP/hold/cost; direction randomized with deterministic seed; stratification covariates are preserved by same timing")
    lines.append(f"bootstrap_ci={BOOTSTRAP_N} deterministic resamples of net R/trade")
    lines.extend(contexts)

    lines.append("\nSUMMARY")
    header = "period,symbol,strategy,rr,gate,n,trades_per_year,win_rate,gross_r,net_r,boot_ci_low,boot_ci_high,train_net,train_ci_low,train_ci_high,test_net,test_ci_low,test_ci_high,profit_factor,sharpe,t_stat,max_dd,longest_loss,worst20,worst_month,avg_spread_r,median_spread_r,spread_gt_010,spread_gt_015,sl_pct,tp_pct,time_exit_pct,total_r"
    lines.append(header)
    for period in ("all",):
        base_rows = period_rows(rows, period)
        keys = sorted({(r.symbol, r.strategy, r.rr, r.gate) for r in base_rows})
        for symbol, strategy, rr, gate in keys:
            subset = [r for r in base_rows if (r.symbol, r.strategy, r.rr, r.gate) == (symbol, strategy, rr, gate)]
            s = summarize(subset, f"{symbol}-{strategy}-{rr}-{gate}-all")
            train = summarize(period_rows(subset, "train_2016_2021"), f"{symbol}-{strategy}-{rr}-{gate}-train")
            test = summarize(period_rows(subset, "test_2022_2026"), f"{symbol}-{strategy}-{rr}-{gate}-test")
            lines.append(
                f"{period},{symbol},{strategy},{rr:.1f},{gate},{int(s['n'])},{s['trades_per_year']:.2f},{s['win_rate']:.2%},"
                f"{s['gross']:.6f},{s['net']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},"
                f"{train['net']:.6f},{train['ci_low']:.6f},{train['ci_high']:.6f},"
                f"{test['net']:.6f},{test['ci_low']:.6f},{test['ci_high']:.6f},"
                f"{s['profit_factor']:.6f},{s['sharpe']:.6f},{s['t_stat']:.6f},{s['max_dd']:.6f},{int(s['longest_loss'])},"
                f"{s['worst20']:.6f},{s['worst_month']:.6f},{s['avg_spread_r']:.6f},{s['median_spread_r']:.6f},"
                f"{s['spread_gt_010']:.2%},{s['spread_gt_015']:.2%},{s['sl_pct']:.2%},{s['tp_pct']:.2%},{s['time_exit_pct']:.2%},{s['total_r']:.6f}"
            )

    lines.append("\nRANDOM_BENCHMARK")
    lines.append("symbol,strategy,rr,gate,n,strategy_net,random_net,delta_net,strategy_gross,random_gross,delta_gross")
    for symbol, strategy, rr, gate in sorted({(r.symbol, r.strategy, r.rr, r.gate) for r in rows}):
        subset = [r for r in rows if (r.symbol, r.strategy, r.rr, r.gate) == (symbol, strategy, rr, gate)]
        lines.append(
            f"{symbol},{strategy},{rr:.1f},{gate},{len(subset)},{mean([r.net_r for r in subset]):.6f},"
            f"{mean([r.random_net_r for r in subset]):.6f},{mean([r.net_r - r.random_net_r for r in subset]):.6f},"
            f"{mean([r.gross_r for r in subset]):.6f},{mean([r.random_gross_r for r in subset]):.6f},{mean([r.gross_r - r.random_gross_r for r in subset]):.6f}"
        )

    lines.append("\nYEARLY_BREAKDOWN")
    lines.append("symbol,strategy,rr,gate,year,n,net_r,ci_low,ci_high,total_r")
    for symbol, strategy, rr, gate, year in sorted({(r.symbol, r.strategy, r.rr, r.gate, r.year) for r in rows}):
        subset = [r for r in rows if (r.symbol, r.strategy, r.rr, r.gate, r.year) == (symbol, strategy, rr, gate, year)]
        s = summarize(subset, f"year-{symbol}-{strategy}-{rr}-{gate}-{year}", bootstrap=False)
        lines.append(f"{symbol},{strategy},{rr:.1f},{gate},{year},{int(s['n'])},{s['net']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{s['total_r']:.6f}")

    lines.append("\nATR_REGIME_BREAKDOWN")
    lines.append("symbol,strategy,rr,gate,atr_regime,n,net_r,ci_low,ci_high,total_r")
    for symbol, strategy, rr, gate, regime in sorted({(r.symbol, r.strategy, r.rr, r.gate, r.atr_regime) for r in rows}):
        subset = [r for r in rows if (r.symbol, r.strategy, r.rr, r.gate, r.atr_regime) == (symbol, strategy, rr, gate, regime)]
        s = summarize(subset, f"regime-{symbol}-{strategy}-{rr}-{gate}-{regime}", bootstrap=False)
        lines.append(f"{symbol},{strategy},{rr:.1f},{gate},{regime},{int(s['n'])},{s['net']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{s['total_r']:.6f}")

    lines.append("\nOUTSIDE_CRISIS")
    lines.append("symbol,strategy,rr,gate,crisis_bucket,n,net_r,ci_low,ci_high,total_r")
    for symbol, strategy, rr, gate, bucket in sorted({(r.symbol, r.strategy, r.rr, r.gate, crisis_window(r.entry_time)) for r in rows}):
        subset = [r for r in rows if (r.symbol, r.strategy, r.rr, r.gate) == (symbol, strategy, rr, gate) and crisis_window(r.entry_time) == bucket]
        s = summarize(subset, f"crisis-{symbol}-{strategy}-{rr}-{gate}-{bucket}", bootstrap=False)
        lines.append(f"{symbol},{strategy},{rr:.1f},{gate},{bucket},{int(s['n'])},{s['net']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{s['total_r']:.6f}")

    lines.append("\nOVERLAP_ANALYSIS")
    lines.append("symbol,exact_overlap,exact_pct_compression,exact_pct_squeeze,within4_overlap,within4_pct_compression,within4_pct_squeeze,monthly_corr_rr_1_5,monthly_corr_rr_2")
    for symbol in sorted(comp_events_by_symbol):
        comp = comp_events_by_symbol[symbol]
        sqz = sq_events_by_symbol[symbol]
        comp_idx = {e.breakout_index for e in comp}
        sq_idx = {e.breakout_index for e in sqz}
        exact = len(comp_idx & sq_idx)
        within = sum(1 for idx in comp_idx if any(abs(idx - s) <= DEDUP_WINDOW_BARS for s in sq_idx))
        corrs = []
        for rr in RRS:
            comp_rows = [r for r in rows if r.symbol == symbol and r.strategy == "compression_only" and r.rr == rr and r.gate == "none"]
            sq_rows = [r for r in rows if r.symbol == symbol and r.strategy == "squeeze_only" and r.rr == rr and r.gate == "none"]
            cm, sm = monthly_net_map(comp_rows), monthly_net_map(sq_rows)
            months = sorted(set(cm) | set(sm))
            corrs.append(pearson([cm.get(m, 0.0) for m in months], [sm.get(m, 0.0) for m in months]))
        lines.append(f"{symbol},{exact},{exact / len(comp):.2%},{exact / len(sqz):.2%},{within},{within / len(comp):.2%},{within / len(sqz):.2%},{corrs[0]:.6f},{corrs[1]:.6f}")

    lines.append("\nOVERLAP_BY_YEAR")
    lines.append("symbol,year,compression_events,squeeze_events,exact_overlap,within4_overlap")
    for symbol in sorted(comp_events_by_symbol):
        comp_year = event_year_map(rows, symbol, "compression_only")
        sq_year = event_year_map(rows, symbol, "squeeze_only")
        for year in range(2016, 2027):
            comp = [e for e in comp_events_by_symbol[symbol] if comp_year.get(e.event_id) == year]
            sqz = [e for e in sq_events_by_symbol[symbol] if sq_year.get(e.event_id) == year]
            comp_idx = {e.breakout_index for e in comp}
            sq_idx = {e.breakout_index for e in sqz}
            exact = len(comp_idx & sq_idx)
            within = sum(1 for idx in comp_idx if any(abs(idx - s) <= DEDUP_WINDOW_BARS for s in sq_idx))
            lines.append(f"{symbol},{year},{len(comp)},{len(sqz)},{exact},{within}")

    lines.append("\nOVERLAP_BY_ATR_REGIME")
    lines.append("symbol,atr_regime,compression_events,squeeze_events,exact_overlap,within4_overlap")
    for symbol in sorted(comp_events_by_symbol):
        for regime in ("low", "mid", "high", "unknown"):
            comp_ids = {r.event_id for r in rows if r.symbol == symbol and r.strategy == "compression_only" and r.rr == 1.5 and r.gate == "none" and r.atr_regime == regime}
            sq_ids = {r.event_id for r in rows if r.symbol == symbol and r.strategy == "squeeze_only" and r.rr == 1.5 and r.gate == "none" and r.atr_regime == regime}
            comp = [e for e in comp_events_by_symbol[symbol] if e.event_id in comp_ids]
            sqz = [e for e in sq_events_by_symbol[symbol] if e.event_id in sq_ids]
            comp_idx = {e.breakout_index for e in comp}
            sq_idx = {e.breakout_index for e in sqz}
            exact = len(comp_idx & sq_idx)
            within = sum(1 for idx in comp_idx if any(abs(idx - s) <= DEDUP_WINDOW_BARS for s in sq_idx))
            lines.append(f"{symbol},{regime},{len(comp)},{len(sqz)},{exact},{within}")

    lines.append("\nEVENT_CONSTRUCTION_STATS")
    lines.append("symbol,strategy,exact_both,dedup_same,dedup_conflict_skip,intersection_pairs")
    for (symbol, strategy), stats in sorted(event_stats.items()):
        lines.append(f"{symbol},{strategy},{stats['exact_both']},{stats['dedup_same']},{stats['dedup_conflict_skip']},{stats['intersection_pairs']}")

    lines.extend(decision_table(rows))
    return "\n".join(lines) + "\n"


def event_year_map(rows: list[ResultTrade], symbol: str, strategy: str) -> dict[int, int]:
    out = {}
    for r in rows:
        if r.symbol == symbol and r.strategy == strategy and r.rr == 1.5 and r.gate == "none":
            out[r.event_id] = r.year
    return out


def decision_table(rows: list[ResultTrade]) -> list[str]:
    out = ["\nFINAL_VERDICT_TABLE", "symbol,strategy,status,reason"]
    for symbol in sorted({r.symbol for r in rows}):
        comp = [r for r in rows if r.symbol == symbol and r.strategy == "compression_only" and r.rr == 1.5 and r.gate == "none"]
        comp_s = summarize(comp, f"decision-{symbol}-comp", bootstrap=False)
        for strategy in ("compression_only", "squeeze_only", "union_raw", "intersection", "dedup_hybrid"):
            subset = [r for r in rows if r.symbol == symbol and r.strategy == strategy and r.rr == 1.5 and r.gate == "none"]
            s = summarize(subset, f"decision-{symbol}-{strategy}", bootstrap=False)
            train = summarize(period_rows(subset, "train_2016_2021"), f"decision-{symbol}-{strategy}-train", bootstrap=False)
            test = summarize(period_rows(subset, "test_2022_2026"), f"decision-{symbol}-{strategy}-test", bootstrap=False)
            if strategy == "compression_only":
                status = "baseline_v1"
                reason = "reference strategy"
            elif strategy == "squeeze_only":
                status = "v2_candidate" if train["ci_low"] > 0 and test["ci_low"] > 0 and s["max_dd"] >= comp_s["max_dd"] else "not_v2"
                reason = "requires train/test CI > 0 and DD <= compression"
            elif strategy == "union_raw":
                comp_tpy = comp_s["trades_per_year"]
                status = "pass" if s["annual_r"] > comp_s["annual_r"] and s["max_dd"] >= comp_s["max_dd"] else "fail"
                reason = "must improve R/year without worse max DD"
            elif strategy == "intersection":
                status = "diagnostic_only"
                if s["n"] >= 300 and s["ci_low"] > 0 and s["net"] > comp_s["net"] + 0.05:
                    status = "pass"
                reason = "sample-size sensitive; pass requires n>=300, CI>0, materially better net"
            else:
                years = yearly_totals_positive_enough(subset)
                status = "main_candidate" if s["ci_low"] > 0 and s["total_r"] > comp_s["total_r"] and s["max_dd"] >= comp_s["max_dd"] and years else "fail"
                reason = "must beat compression total R, DD, CI, and yearly breadth"
            if s["n"] < 100:
                reason += "; sample_size_fragile"
            out.append(f"{symbol},{strategy},{status},{reason}")
    out.append("recommendation,A_keep_compression_live_baseline=true")
    out.append("recommendation,B_promote_squeeze_to_shadow_demo=depends_on_symbol_gold_yes_silver_gated_only")
    out.append("recommendation,C_test_dedup_hybrid_in_shadow_demo=see_symbol_verdict")
    out.append("recommendation,D_reject_intersection_if_sample_starved=true")
    return out


def yearly_totals_positive_enough(rows: list[ResultTrade]) -> bool:
    totals: dict[int, float] = {}
    for r in rows:
        totals[r.year] = totals.get(r.year, 0.0) + r.net_r
    if not totals:
        return False
    positive = sum(v > 0 for v in totals.values())
    return positive / len(totals) >= 0.70


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xag-ticks", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--xag-cache", type=Path, default=Path("data/xagusd_m15_delta_bars.csv"))
    parser.add_argument("--xau-spread", type=float, default=0.20)
    parser.add_argument("--xag-spread", type=float, default=0.02)
    parser.add_argument("--results", type=Path, default=Path("research/hybrid_compression_squeeze_gauntlet_results.txt"))
    parser.add_argument("--trades-csv", type=Path, default=Path("research/hybrid_compression_squeeze_trades.csv"))
    parser.add_argument("--monthly-csv", type=Path, default=Path("research/hybrid_compression_squeeze_monthly.csv"))
    args = parser.parse_args()

    all_rows: list[ResultTrade] = []
    contexts: list[str] = []
    comp_events_by_symbol: dict[str, list[sq.Event]] = {}
    sq_events_by_symbol: dict[str, list[sq.Event]] = {}
    event_stats: dict[tuple[str, str], dict[str, int]] = {}
    for symbol, ticks, cache, spread in (
        ("XAUUSD", args.xau_ticks, args.xau_cache, args.xau_spread),
        ("XAGUSD", args.xag_ticks, args.xag_cache, args.xag_spread),
    ):
        bars = simple.load_symbol_bars(symbol, ticks, cache)
        comp_events = sq.compression_events(bars)
        squeeze_events = sq.detect_squeeze_events(bars)
        comp_events_by_symbol[symbol] = comp_events
        sq_events_by_symbol[symbol] = squeeze_events
        regimes = atr_regimes(bars, comp_events + squeeze_events)
        contexts.append(
            f"symbol_context={symbol},tick_file={ticks},bars={len(bars)},date_range={bars[0].start:%Y-%m-%d %H:%M:%S} to {bars[-1].end:%Y-%m-%d %H:%M:%S} UTC,spread={spread:.4f},compression_events={len(comp_events)},squeeze_events={len(squeeze_events)}"
        )
        for strategy in ("compression_only", "squeeze_only", "union_raw", "intersection", "dedup_hybrid"):
            events, stats = portfolio_events(strategy, comp_events, squeeze_events, bars)
            event_stats[(symbol, strategy)] = stats
            gated_variants = GATES if symbol == "XAGUSD" else (None,)
            for gate in gated_variants:
                all_rows.extend(build_result_trades(symbol, bars, events, regimes, spread, gate))

    report = make_report(contexts, all_rows, comp_events_by_symbol, sq_events_by_symbol, event_stats)
    args.results.write_text(report)
    write_csv(args.trades_csv, trade_csv_rows(all_rows))
    write_csv(args.monthly_csv, monthly_rows(all_rows))
    print(f"results={args.results}")
    print(f"trades_csv={args.trades_csv}")
    print(f"monthly_csv={args.monthly_csv}")


if __name__ == "__main__":
    main()
