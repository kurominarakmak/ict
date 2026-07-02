"""
H-2026-MR-01: XAUUSD mean-reversion core audit.

Research-only. No regime detection, no ML, no threshold tuning.

Primary judged model is realistic limit execution: signal on a closed M15 bar,
then fill only if a later bar actually retraces to the limit price.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import simple_breakout_atr_exit_audit as simple
from delta_signal_audit import DeltaBar, IUX_XAUUSD_ROUNDTRIP_SPREAD


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
RESULTS_PATH = Path("research/gold_mean_reversion_core_results.txt")
REGISTRY_PATH = Path("research/hypothesis_registry.md")

EMA_PERIOD = 20
EXTREME_ATR = 2.5
ENTRY_ATR = 1.5
STOP_EXTRA_ATR = 1.0
TIME_STOP_BARS = 10
BOOT_N = 1000
SEED = 20260703


@dataclass(frozen=True)
class Setup:
    event_id: int
    signal_index: int
    signal_time: datetime
    direction: int
    mean_price: float
    atr: float
    extreme: float
    entry: float
    stop: float
    target: float


@dataclass(frozen=True)
class Trade:
    model: str
    event_id: int
    signal_time: datetime
    entry_time: datetime | None
    direction: int
    net_r: float
    gross_r: float
    win: bool
    exit_reason: str
    skipped: bool
    bars_waited: int
    bars_held: int
    risk: float
    spread_r: float


def q(vals: list[float], frac: float) -> float:
    if not vals:
        return math.nan
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * frac
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(vals: list[float], seed: str) -> tuple[float, float]:
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    means = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT_N)]
    return q(means, 0.025), q(means, 0.975)


def safe_mean(vals: list[float]) -> float:
    return mean(vals) if vals else math.nan


def same_segment(bars: list[DeltaBar], start: int, end: int) -> bool:
    if start < 0 or end >= len(bars):
        return False
    segment = bars[start].segment_id
    return all(bars[i].segment_id == segment for i in range(start, end + 1))


def typical_price(bar: DeltaBar) -> float:
    return (bar.high + bar.low + bar.close) / 3.0


def session_vwap(bars: list[DeltaBar]) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    current_key = None
    pv = 0.0
    vol = 0.0
    for i, bar in enumerate(bars):
        key = (bar.segment_id, bar.start.date())
        if key != current_key:
            current_key = key
            pv = 0.0
            vol = 0.0
        weight = max(float(bar.ticks), 1.0)
        pv += typical_price(bar) * weight
        vol += weight
        out[i] = pv / vol if vol > 0 else None
    return out


def ema20(bars: list[DeltaBar]) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    alpha = 2.0 / (EMA_PERIOD + 1)
    prev_segment = None
    ema = None
    count = 0
    for i, bar in enumerate(bars):
        if prev_segment is None or bar.segment_id != prev_segment:
            ema = None
            count = 0
        price = bar.close
        count += 1
        ema = price if ema is None else alpha * price + (1 - alpha) * ema
        if count >= EMA_PERIOD:
            out[i] = ema
        prev_segment = bar.segment_id
    return out


def build_setups(bars: list[DeltaBar], means: list[float | None]) -> list[Setup]:
    setups: list[Setup] = []
    for i, bar in enumerate(bars):
        atr = bar.atr14
        mean_price = means[i]
        if atr is None or atr <= 0 or mean_price is None:
            continue
        if bar.close >= mean_price + EXTREME_ATR * atr:
            entry = mean_price + ENTRY_ATR * atr
            extreme = bar.high
            stop = extreme + STOP_EXTRA_ATR * atr
            target = mean_price
            if stop > entry > target:
                setups.append(Setup(len(setups) + 1, i, bar.start, -1, mean_price, atr, extreme, entry, stop, target))
        elif bar.close <= mean_price - EXTREME_ATR * atr:
            entry = mean_price - ENTRY_ATR * atr
            extreme = bar.low
            stop = extreme - STOP_EXTRA_ATR * atr
            target = mean_price
            if stop < entry < target:
                setups.append(Setup(len(setups) + 1, i, bar.start, 1, mean_price, atr, extreme, entry, stop, target))
    return setups


def segment_last_index(bars: list[DeltaBar], start: int) -> int:
    segment = bars[start].segment_id
    i = start
    while i + 1 < len(bars) and bars[i + 1].segment_id == segment:
        i += 1
    return i


def find_limit_fill(bars: list[DeltaBar], setup: Setup) -> int | None:
    j = setup.signal_index + 1
    while j < len(bars) and bars[j].segment_id == bars[setup.signal_index].segment_id:
        hit = bars[j].low <= setup.entry if setup.direction == 1 else bars[j].high >= setup.entry
        if hit:
            return j
        j += 1
    return None


def simulate_from_entry(
    bars: list[DeltaBar],
    setup: Setup,
    model: str,
    entry_index: int,
    spread: float,
    *,
    forced_entry: bool = False,
) -> Trade:
    risk = abs(setup.entry - setup.stop)
    if risk <= 0:
        return Trade(model, setup.event_id, setup.signal_time, None, setup.direction, math.nan, math.nan, False, "invalid_risk", True, 0, 0, risk, math.nan)
    end = min(segment_last_index(bars, entry_index), entry_index + TIME_STOP_BARS)
    gross_r = setup.direction * (bars[end].close - setup.entry) / risk
    exit_reason = "time_stop"
    exit_index = end
    for i in range(entry_index + 1, end + 1):
        bar = bars[i]
        stop_hit = bar.low <= setup.stop if setup.direction == 1 else bar.high >= setup.stop
        target_hit = bar.high >= setup.target if setup.direction == 1 else bar.low <= setup.target
        if stop_hit:
            fill = min(setup.stop, bar.low) if setup.direction == 1 else max(setup.stop, bar.high)
            gross_r = setup.direction * (fill - setup.entry) / risk
            exit_reason = "stop"
            exit_index = i
            break
        if target_hit:
            gross_r = setup.direction * (setup.target - setup.entry) / risk
            exit_reason = "target"
            exit_index = i
            break
    spread_r = spread / risk
    return Trade(
        model=model,
        event_id=setup.event_id,
        signal_time=setup.signal_time,
        entry_time=bars[entry_index].start,
        direction=setup.direction,
        net_r=gross_r - spread_r,
        gross_r=gross_r,
        win=gross_r - spread_r > 0,
        exit_reason=exit_reason,
        skipped=False,
        bars_waited=0 if forced_entry else entry_index - setup.signal_index,
        bars_held=exit_index - entry_index,
        risk=risk,
        spread_r=spread_r,
    )


def skipped_trade(model: str, setup: Setup) -> Trade:
    return Trade(model, setup.event_id, setup.signal_time, None, setup.direction, math.nan, math.nan, False, "skipped_no_retrace", True, 0, 0, math.nan, math.nan)


def simulate_realistic_limit(bars: list[DeltaBar], setups: list[Setup], spread: float) -> list[Trade]:
    rows: list[Trade] = []
    blocked_until = -1
    for setup in setups:
        if setup.signal_index <= blocked_until:
            continue
        fill_index = find_limit_fill(bars, setup)
        if fill_index is None:
            rows.append(skipped_trade("realistic_limit", setup))
            blocked_until = segment_last_index(bars, setup.signal_index)
            continue
        trade = simulate_from_entry(bars, setup, "realistic_limit", fill_index, spread)
        rows.append(trade)
        blocked_until = fill_index + trade.bars_held
    return rows


def simulate_idealized(bars: list[DeltaBar], setups: list[Setup], spread: float) -> list[Trade]:
    return [simulate_from_entry(bars, setup, "idealized_signal_fill", setup.signal_index, spread, forced_entry=True) for setup in setups]


def random_direction_control(bars: list[DeltaBar], setups: list[Setup], spread: float) -> list[Trade]:
    rng = random.Random(SEED)
    random_setups: list[Setup] = []
    for setup in setups:
        direction = rng.choice([-1, 1])
        entry = setup.mean_price - ENTRY_ATR * setup.atr if direction == 1 else setup.mean_price + ENTRY_ATR * setup.atr
        if direction == 1:
            extreme = bars[setup.signal_index].low
            stop = extreme - STOP_EXTRA_ATR * setup.atr
            target = setup.mean_price
            if not (stop < entry < target):
                continue
        else:
            extreme = bars[setup.signal_index].high
            stop = extreme + STOP_EXTRA_ATR * setup.atr
            target = setup.mean_price
            if not (stop > entry > target):
                continue
        random_setups.append(Setup(setup.event_id, setup.signal_index, setup.signal_time, direction, setup.mean_price, setup.atr, extreme, entry, stop, target))
    return simulate_realistic_limit(bars, random_setups, spread)


def period_rows(rows: list[Trade], period: str) -> list[Trade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.signal_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.signal_time >= TEST_START]
    raise ValueError(period)


def summarize(rows: list[Trade], seed: str) -> dict[str, float]:
    filled = [r for r in rows if not r.skipped and math.isfinite(r.net_r)]
    vals = [r.net_r for r in filled]
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    lo, hi = bootstrap_ci(vals, seed)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    top_wins = sorted(wins, reverse=True)[: max(1, math.ceil(len(wins) * 0.05))] if wins else []
    tail_losses = sorted(losses)[: max(1, math.ceil(len(losses) * 0.05))] if losses else []
    sd = pstdev(vals) if len(vals) > 1 else math.nan
    return {
        "total_signals": len(rows),
        "n": len(filled),
        "skipped": sum(1 for r in rows if r.skipped),
        "skip_rate": sum(1 for r in rows if r.skipped) / len(rows) if rows else math.nan,
        "win_rate": len(wins) / len(vals) if vals else math.nan,
        "net": safe_mean(vals),
        "ci_low": lo,
        "ci_high": hi,
        "worst_loss": min(vals) if vals else math.nan,
        "p95_loss": q(vals, 0.05),
        "avg_win": safe_mean(wins),
        "avg_loss_abs": abs(safe_mean(losses)) if losses else math.nan,
        "avg_win_loss_ratio": safe_mean(wins) / abs(safe_mean(losses)) if wins and losses else math.nan,
        "top5_win_profit_frac": sum(top_wins) / gross_profit if gross_profit > 0 else math.nan,
        "tail5_loss_frac": abs(sum(tail_losses)) / gross_loss if gross_loss > 0 else math.nan,
        "sharpe_trade": safe_mean(vals) / sd * math.sqrt(len(vals)) if vals and sd and math.isfinite(sd) and sd > 0 else math.nan,
    }


def yearly(rows: list[Trade]) -> list[tuple[int, int, float]]:
    out = []
    years = sorted({r.signal_time.year for r in rows})
    for year in years:
        subset = [r for r in rows if r.signal_time.year == year and not r.skipped and math.isfinite(r.net_r)]
        out.append((year, len(subset), safe_mean([r.net_r for r in subset])))
    return out


def passive_vol_target_baseline(bars: list[DeltaBar]) -> dict[str, float]:
    vals: list[float] = []
    train_vals: list[float] = []
    test_vals: list[float] = []
    for i in range(len(bars) - 1):
        bar = bars[i]
        nxt = bars[i + 1]
        if bar.atr14 is None or bar.atr14 <= 0 or nxt.segment_id != bar.segment_id:
            continue
        r = (nxt.close - bar.close) / bar.atr14
        vals.append(r)
        if bar.start <= TRAIN_END:
            train_vals.append(r)
        elif bar.start >= TEST_START:
            test_vals.append(r)
    def s(items: list[float]) -> tuple[float, float]:
        sd = pstdev(items) if len(items) > 1 else math.nan
        sharpe = mean(items) / sd * math.sqrt(252 * 24 * 4) if items and sd and math.isfinite(sd) and sd > 0 else math.nan
        return safe_mean(items), sharpe
    full_mean, full_sharpe = s(vals)
    train_mean, train_sharpe = s(train_vals)
    test_mean, test_sharpe = s(test_vals)
    return {
        "full_mean_r_per_bar": full_mean,
        "full_sharpe": full_sharpe,
        "train_mean_r_per_bar": train_mean,
        "train_sharpe": train_sharpe,
        "test_mean_r_per_bar": test_mean,
        "test_sharpe": test_sharpe,
    }


def append_registry(verdict: str, train: dict[str, float], test: dict[str, float]) -> None:
    existing = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else "# Hypothesis Registry\n"
    lines = [line for line in existing.rstrip().splitlines() if "H-2026-MR-01" not in line]
    lines.append("- 2026-07-03: H-2026-MR-01 registered. Gold mean-reversion core audit: UTC-session VWAP mean, closed-bar +/-2.5ATR signal, realistic limit at VWAP +/-1.5ATR, TP at VWAP, SL one ATR beyond signal extreme, 10-bar time stop, no regime detection or ML.")
    lines.append(
        "- 2026-07-03: H-2026-MR-01 result: "
        f"{verdict}; realistic_limit_train={train['net']:.4f} [{train['ci_low']:.4f},{train['ci_high']:.4f}], "
        f"test={test['net']:.4f} [{test['ci_low']:.4f},{test['ci_high']:.4f}]."
    )
    REGISTRY_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--spread", type=float, default=IUX_XAUUSD_ROUNDTRIP_SPREAD)
    parser.add_argument("--mean", choices=("vwap", "ema20"), default="vwap")
    args = parser.parse_args()

    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    means = session_vwap(bars) if args.mean == "vwap" else ema20(bars)
    setups = build_setups(bars, means)
    realistic = simulate_realistic_limit(bars, setups, args.spread)
    idealized = simulate_idealized(bars, setups, args.spread)
    random_control = random_direction_control(bars, setups, args.spread)
    passive = passive_vol_target_baseline(bars)

    summaries = {
        ("realistic_limit", p): summarize(period_rows(realistic, p), f"{SEED}-realistic-{p}") for p in ("full", "train", "test")
    }
    summaries.update({("idealized_signal_fill", p): summarize(period_rows(idealized, p), f"{SEED}-ideal-{p}") for p in ("full", "train", "test")})
    summaries.update({("random_direction_limit", p): summarize(period_rows(random_control, p), f"{SEED}-random-{p}") for p in ("full", "train", "test")})
    train = summaries[("realistic_limit", "train")]
    test = summaries[("realistic_limit", "test")]
    pass_core = train["ci_low"] > 0 and test["ci_low"] > 0 and train["avg_win_loss_ratio"] > 0 and test["avg_win_loss_ratio"] > 0
    verdict = "PASS_CORE_EDGE_WORTH_REGIME_RESEARCH" if pass_core else "FAIL_CORE_EDGE_DOES_NOT_CLEAR_REALISTIC_EV_GATE"

    lines: list[str] = []
    lines.append("H_2026_MR_01_GOLD_MEAN_REVERSION_CORE_AUDIT")
    lines.append("symbol,XAUUSD")
    lines.append(f"mean,{args.mean}")
    lines.append("mean_definition,UTC-session VWAP using M15 typical price weighted by tick count")
    lines.append("signal,closed_bar_close >= mean+2.5ATR short or <= mean-2.5ATR long")
    lines.append("primary_fill,realistic_limit_after_signal")
    lines.append(f"spread,{args.spread}")
    lines.append(f"raw_setups,{len(setups)}")
    lines.append("")
    lines.append("MODEL_SUMMARY")
    lines.append("model,period,total_signals,filled,skipped,skip_rate,win_rate,net_r,ci_low,ci_high,worst_loss,p95_loss,avg_win,avg_loss_abs,avg_win_loss_ratio,top5_win_profit_frac,tail5_loss_frac,trade_sharpe")
    for model in ("realistic_limit", "idealized_signal_fill", "random_direction_limit"):
        for p in ("full", "train", "test"):
            s = summaries[(model, p)]
            lines.append(
                f"{model},{p},{s['total_signals']},{s['n']},{s['skipped']},{s['skip_rate']:.4f},"
                f"{s['win_rate']:.4f},{s['net']:.4f},{s['ci_low']:.4f},{s['ci_high']:.4f},"
                f"{s['worst_loss']:.4f},{s['p95_loss']:.4f},{s['avg_win']:.4f},{s['avg_loss_abs']:.4f},"
                f"{s['avg_win_loss_ratio']:.4f},{s['top5_win_profit_frac']:.4f},{s['tail5_loss_frac']:.4f},{s['sharpe_trade']:.4f}"
            )
    lines.append("")
    lines.append("PASSIVE_VOL_TARGET_BUY_AND_HOLD")
    lines.append("period,mean_r_per_bar,annualized_sharpe")
    lines.append(f"full,{passive['full_mean_r_per_bar']:.6f},{passive['full_sharpe']:.4f}")
    lines.append(f"train,{passive['train_mean_r_per_bar']:.6f},{passive['train_sharpe']:.4f}")
    lines.append(f"test,{passive['test_mean_r_per_bar']:.6f},{passive['test_sharpe']:.4f}")
    lines.append("")
    lines.append("YEARLY_REALISTIC_LIMIT")
    lines.append("year,n,net_r")
    for year, n, net in yearly(realistic):
        lines.append(f"{year},{n},{net:.4f}")
    lines.append("")
    lines.append("VERDICT")
    if pass_core:
        lines.append("Realistic-limit mean reversion clears the core train/test EV gate. This is worth a separate future regime-detection audit.")
    else:
        lines.append("Realistic-limit mean reversion does not clear the core train/test EV gate. Do not add regime detection or ML to rescue it yet.")
    report = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(report)
    print(report, end="")
    append_registry(verdict, train, test)
    print(f"results_file={RESULTS_PATH}")


if __name__ == "__main__":
    main()
