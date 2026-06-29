"""
Pre-registered volatility-compression breakout audit for XAUUSD.

Theory under test:
- Volatility clustering: compression should predict magnitude.
- Wyckoff/auction ideas: accumulation/distribution may predict direction.

This script reports all declared direction methods and baselines. It does not
select the best-looking method after the fact.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev

from delta_signal_audit import (
    FORWARD_HORIZONS,
    IUX_XAUUSD_ROUNDTRIP_SPREAD,
    DeltaBar,
    add_indicators,
    classify_session,
    default_tick_path,
    pearson,
    quantile,
)
from orderflow_absorption_divergence_audit import absorption_score


COMPRESSION_WINDOW = 16
ATR_TRAIL = 100
ATR_TERCILE_Q = 1 / 3
COMPRESSION_MIN_FRACTION = 0.75
EXIT_HORIZON = 5
FALSE_BREAKOUT_BARS = 3
RSI_PERIOD = 14
COIN_FLIP_SEED = 20260629
DEFAULT_BAR_CACHE = Path("data/xauusd_m15_delta_bars.csv")


@dataclass(frozen=True)
class CompressionEvent:
    event_id: int
    setup_start: int
    setup_end: int
    breakout_index: int
    range_high: float
    range_low: float
    breakout_direction: int
    simple_return: float
    false_breakout: bool
    predictions: dict[str, int | None]


def ci(vals: list[float]) -> tuple[int, float, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se, sd


def corr_ci(xs: list[float], ys: list[float]) -> tuple[int, float, float, float]:
    r = pearson(xs, ys)
    n = len(xs)
    if n <= 3 or not math.isfinite(r) or abs(r) >= 1:
        return n, r, math.nan, math.nan
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return n, r, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def forward_return(bars: list[DeltaBar], i: int, horizon: int) -> float | None:
    j = i + horizon
    if j >= len(bars) or bars[j].segment_id != bars[i].segment_id:
        return None
    return bars[j].close - bars[i].close


def rank_residuals(values: list[float], controls: list[float]) -> list[float]:
    mx = mean(controls)
    my = mean(values)
    denom = sum((x - mx) ** 2 for x in controls)
    if denom == 0:
        return [v - my for v in values]
    beta = sum((x - mx) * (y - my) for x, y in zip(controls, values)) / denom
    alpha = my - beta * mx
    return [y - (alpha + beta * x) for x, y in zip(controls, values)]


def print_atr_absorption_check(bars: list[DeltaBar]) -> None:
    print("\nCHECK_A_ATR_VS_ABSORPTION")
    print("horizon,n,ic_atr_absret,atr_ci_low,atr_ci_high,ic_absorption_absret,abs_ci_low,abs_ci_high,partial_corr_absorption_after_atr,partial_ci_low,partial_ci_high")
    for h in FORWARD_HORIZONS:
        atrs: list[float] = []
        absorptions: list[float] = []
        absrets: list[float] = []
        for i, bar in enumerate(bars):
            ret = forward_return(bars, i, h)
            score = absorption_score(bar)
            if ret is None or bar.atr14 is None or not math.isfinite(score):
                continue
            atrs.append(bar.atr14)
            absorptions.append(score)
            absrets.append(abs(ret))
        n1, atr_ic, atr_lo, atr_hi = corr_ci(atrs, absrets)
        _, abs_ic, abs_lo, abs_hi = corr_ci(absorptions, absrets)
        # Partial correlation of absorption with abs return after removing ATR
        # from both variables. With one control, this is equivalent to OLS
        # incremental signal beyond ATR.
        abs_resid = rank_residuals(absorptions, atrs)
        ret_resid = rank_residuals(absrets, atrs)
        _, partial, plo, phi = corr_ci(abs_resid, ret_resid)
        print(f"{h},{n1},{atr_ic:.6f},{atr_lo:.6f},{atr_hi:.6f},{abs_ic:.6f},{abs_lo:.6f},{abs_hi:.6f},{partial:.6f},{plo:.6f},{phi:.6f}")


def add_rsi14(bars: list[DeltaBar]) -> list[float | None]:
    rsis: list[float | None] = [None] * len(bars)
    gains: list[float] = []
    losses: list[float] = []
    prev_segment: int | None = None
    avg_gain: float | None = None
    avg_loss: float | None = None
    for i, bar in enumerate(bars):
        if prev_segment is None or bar.segment_id != prev_segment:
            gains.clear()
            losses.clear()
            avg_gain = None
            avg_loss = None
            prev_segment = bar.segment_id
            continue
        change = bar.close - bars[i - 1].close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if avg_gain is None or avg_loss is None:
            gains.append(gain)
            losses.append(loss)
            if len(gains) == RSI_PERIOD:
                avg_gain = sum(gains) / RSI_PERIOD
                avg_loss = sum(losses) / RSI_PERIOD
            else:
                prev_segment = bar.segment_id
                continue
        else:
            avg_gain = (avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
            avg_loss = (avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100.0 - (100.0 / (1.0 + rs))
        prev_segment = bar.segment_id
    return rsis


def same_segment(bars: list[DeltaBar], start: int, end: int) -> bool:
    if start < 0 or end >= len(bars):
        return False
    segment = bars[start].segment_id
    return all(bars[i].segment_id == segment for i in range(start, end + 1))


def trailing_atr_cutoff(bars: list[DeltaBar], i: int) -> float | None:
    vals: list[float] = []
    j = i - 1
    while j >= 0 and len(vals) < ATR_TRAIL:
        if bars[j].atr14 is not None:
            vals.append(bars[j].atr14)
        j -= 1
    if len(vals) < ATR_TRAIL:
        return None
    return quantile(vals, ATR_TERCILE_Q)


def is_compression_end(bars: list[DeltaBar], i: int) -> bool:
    start = i - COMPRESSION_WINDOW + 1
    if not same_segment(bars, start, i):
        return False
    compressed = 0
    checked = 0
    for j in range(start, i + 1):
        cutoff = trailing_atr_cutoff(bars, j)
        if cutoff is None or bars[j].atr14 is None:
            return False
        checked += 1
        if bars[j].atr14 <= cutoff:
            compressed += 1
    return checked == COMPRESSION_WINDOW and compressed / checked >= COMPRESSION_MIN_FRACTION


def method_cvd(bars: list[DeltaBar], start: int, end: int) -> int | None:
    total = sum(bars[i].delta for i in range(start, end + 1))
    if total > 0:
        return 1
    if total < 0:
        return -1
    return None


def method_price_position(bars: list[DeltaBar], start: int, end: int, range_high: float, range_low: float) -> int | None:
    width = range_high - range_low
    if width <= 0:
        return None
    mid = start + (end - start + 1) // 2
    first_width = max(b.high for b in bars[start:mid]) - min(b.low for b in bars[start:mid])
    second_width = max(b.high for b in bars[mid : end + 1]) - min(b.low for b in bars[mid : end + 1])
    if second_width >= first_width:
        return None
    avg_pos = mean([(bars[i].close - range_low) / width for i in range(start, end + 1)])
    if avg_pos <= 1 / 3:
        return 1
    if avg_pos >= 2 / 3:
        return -1
    return None


def method_wyckoff_rsi(rsis: list[float | None], start: int, end: int) -> int | None:
    window = rsis[start : end + 1]
    if any(v is None for v in window):
        return None
    vals = [float(v) for v in window if v is not None]
    mid = len(vals) // 2
    first = vals[:mid]
    second = vals[mid:]
    last = vals[-1]
    if last > 50 and min(second) > min(first):
        return 1
    if last < 50 and max(second) < max(first):
        return -1
    return None


def detect_events(bars: list[DeltaBar]) -> list[CompressionEvent]:
    rsis = add_rsi14(bars)
    rng = random.Random(COIN_FLIP_SEED)
    events: list[CompressionEvent] = []
    i = ATR_TRAIL + COMPRESSION_WINDOW
    while i < len(bars) - EXIT_HORIZON:
        if not is_compression_end(bars, i):
            i += 1
            continue
        start = i - COMPRESSION_WINDOW + 1
        range_high = max(b.high for b in bars[start : i + 1])
        range_low = min(b.low for b in bars[start : i + 1])
        breakout = None
        for j in range(i + 1, len(bars) - EXIT_HORIZON):
            if bars[j].segment_id != bars[i].segment_id:
                break
            if bars[j].close > range_high:
                breakout = (j, 1)
                break
            if bars[j].close < range_low:
                breakout = (j, -1)
                break
        if breakout is None:
            i += 1
            continue
        bidx, bdir = breakout
        exit_idx = bidx + EXIT_HORIZON
        if bars[exit_idx].segment_id != bars[bidx].segment_id:
            i = bidx + 1
            continue
        gross = bdir * (bars[exit_idx].close - bars[bidx].close)
        simple_return = gross - IUX_XAUUSD_ROUNDTRIP_SPREAD
        false_breakout = False
        for k in range(bidx + 1, min(bidx + FALSE_BREAKOUT_BARS, len(bars) - 1) + 1):
            if bars[k].segment_id != bars[bidx].segment_id:
                break
            if bdir == 1 and bars[k].close < range_low:
                false_breakout = True
            if bdir == -1 and bars[k].close > range_high:
                false_breakout = True
        prior_momentum = None
        if bars[i].close > bars[i - 1].close:
            prior_momentum = 1
        elif bars[i].close < bars[i - 1].close:
            prior_momentum = -1
        predictions = {
            "method_1_cvd": method_cvd(bars, start, i),
            "method_2_price_position": method_price_position(bars, start, i, range_high, range_low),
            "method_3_wyckoff_rsi": method_wyckoff_rsi(rsis, start, i),
            "baseline_coin_flip": 1 if rng.random() >= 0.5 else -1,
            "baseline_prior_momentum": prior_momentum,
        }
        events.append(
            CompressionEvent(
                len(events) + 1,
                start,
                i,
                bidx,
                range_high,
                range_low,
                bdir,
                simple_return,
                false_breakout,
                predictions,
            )
        )
        i = bidx + EXIT_HORIZON + 1
    return events


def summarize_method(events: list[CompressionEvent], name: str) -> dict[str, float]:
    vals = []
    correct = 0
    false_count = 0
    for event in events:
        pred = event.predictions[name]
        if pred is None:
            continue
        if pred == event.breakout_direction:
            correct += 1
        if event.false_breakout:
            false_count += 1
        exit_idx = event.breakout_index + EXIT_HORIZON
        direction = int(pred)
        gross = direction * (bars_global[exit_idx].close - bars_global[event.breakout_index].close)
        vals.append(gross - IUX_XAUUSD_ROUNDTRIP_SPREAD)
    n, m, lo, hi, _ = ci(vals)
    return {
        "n": n,
        "coverage": n / len(events) if events else math.nan,
        "pct_correct": correct / n if n else math.nan,
        "net_mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "win_rate": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "false_breakouts": false_count,
    }


def print_event_summary(events: list[CompressionEvent]) -> None:
    print("\nCHECK_B_COMPRESSION_BREAKOUT_CONTEXT")
    print(f"compression_window_bars={COMPRESSION_WINDOW}")
    print(f"atr_rule=ATR14 <= bottom tercile of prior {ATR_TRAIL} valid ATR14 values")
    print(f"compression_min_fraction={COMPRESSION_MIN_FRACTION:.2f}")
    print("range=high/low of compression window")
    print("breakout=first close beyond range high/low")
    print(f"exit=close after {EXIT_HORIZON} bars, net of ${IUX_XAUUSD_ROUNDTRIP_SPREAD:.2f}/oz round-trip spread")
    print(f"false_breakout=reclose through opposite side of compression range within {FALSE_BREAKOUT_BARS} bars")
    print(f"coin_flip_seed={COIN_FLIP_SEED}")
    print(f"events={len(events)}")
    if events:
        print(f"event_range={bars_global[events[0].setup_start].start:%Y-%m-%d %H:%M} to {bars_global[events[-1].breakout_index].start:%Y-%m-%d %H:%M} UTC")


def print_methods(events: list[CompressionEvent]) -> None:
    print("\nDIRECTION_METHODS_AND_BASELINES")
    print("method,total_events,n_predictions,coverage,pct_correct_breakout_direction,net_return_mean,ci_low,ci_high,win_rate,false_breakouts,beats_coinflip_accuracy,beats_momentum_accuracy,ci_clears_zero")
    names = [
        "method_1_cvd",
        "method_2_price_position",
        "method_3_wyckoff_rsi",
        "baseline_coin_flip",
        "baseline_prior_momentum",
    ]
    summaries = {name: summarize_method(events, name) for name in names}
    coin_acc = summaries["baseline_coin_flip"]["pct_correct"]
    mom_acc = summaries["baseline_prior_momentum"]["pct_correct"]
    for name in names:
        s = summaries[name]
        beats_coin = s["pct_correct"] > coin_acc if math.isfinite(s["pct_correct"]) and math.isfinite(coin_acc) else False
        beats_mom = s["pct_correct"] > mom_acc if math.isfinite(s["pct_correct"]) and math.isfinite(mom_acc) else False
        clears = s["ci_low"] > 0 if math.isfinite(s["ci_low"]) else False
        print(
            f"{name},{len(events)},{int(s['n'])},{s['coverage']:.2%},{s['pct_correct']:.2%},"
            f"{s['net_mean']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},{s['win_rate']:.2%},"
            f"{int(s['false_breakouts'])},{beats_coin},{beats_mom},{clears}"
        )


def print_simple_follow(events: list[CompressionEvent]) -> None:
    vals = [e.simple_return for e in events]
    n, m, lo, hi, _ = ci(vals)
    print("\nSIMPLE_BREAKOUT_FOLLOW")
    print("method,total_events,n,net_return_mean,ci_low,ci_high,win_rate,false_breakouts,ci_clears_zero")
    print(
        f"simple_follow_breakout_direction,{len(events)},{n},{m:.6f},{lo:.6f},{hi:.6f},"
        f"{(sum(v > 0 for v in vals) / len(vals)) if vals else math.nan:.2%},"
        f"{sum(e.false_breakout for e in events)},{lo > 0 if math.isfinite(lo) else False}"
    )


def print_verdict(events: list[CompressionEvent]) -> None:
    names = ["method_1_cvd", "method_2_price_position", "method_3_wyckoff_rsi"]
    summaries = {name: summarize_method(events, name) for name in names + ["baseline_coin_flip", "baseline_prior_momentum"]}
    coin = summaries["baseline_coin_flip"]
    mom = summaries["baseline_prior_momentum"]
    winners = []
    for name in names:
        s = summaries[name]
        if s["pct_correct"] > coin["pct_correct"] and s["pct_correct"] > mom["pct_correct"] and s["ci_low"] > 0:
            winners.append(name)
    simple_vals = [e.simple_return for e in events]
    _, simple_mean, simple_lo, simple_hi, _ = ci(simple_vals)
    print("\nVERDICT")
    if winners:
        print(f"direction_methods_passing_accuracy_baselines_and_positive_net_ci={';'.join(winners)}")
        print("interpretation=Treat cautiously: three pre-registered direction shots raise false-positive risk; needs out-of-sample confirmation.")
    else:
        print("direction_methods_passing_accuracy_baselines_and_positive_net_ci=none")
        print("interpretation=No declared direction method cleared chance/momentum and positive net-return CI on this spot-tick sample.")
    print(f"simple_follow_net_mean={simple_mean:.6f},ci_low={simple_lo:.6f},ci_high={simple_hi:.6f},ci_clears_zero={simple_lo > 0 if math.isfinite(simple_lo) else False}")


bars_global: list[DeltaBar] = []


def load_cached_bars(path: Path) -> list[DeltaBar]:
    bars: list[DeltaBar] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = datetime.fromtimestamp(int(row["start_epoch"]), tz=timezone.utc)
            end = start + timedelta(minutes=15)
            bars.append(
                DeltaBar(
                    len(bars),
                    int(row["segment_id"]),
                    start,
                    end,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["ticks"]),
                    int(row["buy_ticks"]),
                    int(row["sell_ticks"]),
                    int(row["neutral_ticks"]),
                    int(row["delta"]),
                    float(row["delta_ratio"]),
                    session=classify_session(end),
                )
            )
    add_indicators(bars)
    return bars


def ensure_bar_cache(tick_path: Path, cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_mtime >= tick_path.stat().st_mtime:
        return
    source = Path("research/fast_m15_delta_bars.cpp")
    binary = Path("/tmp/fast_m15_delta_bars")
    print(f"Building M15 bar cache with compiled converter: {cache_path}", flush=True)
    subprocess.run(["c++", "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(tick_path), str(cache_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None)
    parser.add_argument("--bar-cache", type=Path, default=DEFAULT_BAR_CACHE)
    args = parser.parse_args()
    path = args.ticks or default_tick_path()
    ensure_bar_cache(path, args.bar_cache)
    print("Loading M15 tick-rule delta bars...", flush=True)
    global bars_global
    bars_global = load_cached_bars(args.bar_cache)
    print("\nVOLATILITY_COMPRESSION_BREAKOUT_AUDIT_CONTEXT")
    print(f"tick_file={path}")
    print(f"bars={len(bars_global)}")
    print(f"date_range={bars_global[0].start:%Y-%m-%d %H:%M:%S} to {bars_global[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print("data_caveat=Dukascopy spot quote ticks; tick-count volume and tick-rule CVD, not futures traded volume")
    print_atr_absorption_check(bars_global)
    events = detect_events(bars_global)
    print_event_summary(events)
    print_methods(events)
    print_simple_follow(events)
    print_verdict(events)


if __name__ == "__main__":
    main()
