"""
Raw order-flow absorption/divergence audit for XAUUSD.

This extends the tick-rule delta audit. It is not a strategy engine and uses no ML.

Data caveat:
- Dukascopy spot ticks are quote updates, not exchange prints.
- "Volume" is tick count, not traded contracts.
- Delta/CVD are inferred from mid-price uptick/downtick tick rule.

Features:
- Absorption: high tick count with small ATR-normalized range.
- Directional absorption: high absorption near a 20-bar local low/high.
- Divergence: price pivot makes new extreme while rolling CVD(20) fails to confirm.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

from delta_signal_audit import (
    FORWARD_HORIZONS,
    IUX_XAUUSD_ROUNDTRIP_SPREAD,
    DeltaBar,
    default_tick_path,
    load_delta_bars,
    pearson,
    quantile,
)


LOCAL_EXTREME_LOOKBACK = 20
PIVOT_LEFT = 5
PIVOT_RIGHT = 1


@dataclass(frozen=True)
class Divergence:
    kind: str
    pivot_index: int
    signal_index: int
    price_level: float
    cvd20: int
    prior_pivot_index: int
    prior_price_level: float
    prior_cvd20: int


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


def absorption_score(bar: DeltaBar) -> float:
    if bar.atr14 is None or bar.atr14 <= 0:
        return math.nan
    rng = max(bar.high - bar.low, 1e-9)
    return bar.ticks / (rng / bar.atr14)


def local_position(bars: list[DeltaBar], i: int) -> str:
    start = max(0, i - LOCAL_EXTREME_LOOKBACK + 1)
    window = bars[start : i + 1]
    if len(window) < LOCAL_EXTREME_LOOKBACK:
        return "none"
    is_low = bars[i].low <= min(b.low for b in window)
    is_high = bars[i].high >= max(b.high for b in window)
    if is_low and not is_high:
        return "low"
    if is_high and not is_low:
        return "high"
    return "none"


def forward_return(bars: list[DeltaBar], i: int, horizon: int) -> float | None:
    j = i + horizon
    if j >= len(bars) or bars[j].segment_id != bars[i].segment_id:
        return None
    return bars[j].close - bars[i].close


def decile_absorption(bars: list[DeltaBar]) -> None:
    eligible = [
        (i, absorption_score(b))
        for i, b in enumerate(bars)
        if b.atr14 is not None and all(forward_return(bars, i, h) is not None for h in FORWARD_HORIZONS)
    ]
    ordered = sorted(eligible, key=lambda x: x[1])
    n = len(ordered)
    print("\nABSORPTION_SCORE_DECILES")
    print("decile,n,mean_absorption,median_absorption,mean_ticks,ret_1bar,ret_3bar,ret_5bar,absret_1bar,absret_3bar,absret_5bar")
    decile_rows = []
    for decile in range(1, 11):
        lo = (decile - 1) * n // 10
        hi = decile * n // 10
        members = ordered[lo:hi]
        idxs = [i for i, _ in members]
        scores = [score for _, score in members]
        row = {
            "decile": decile,
            "n": len(idxs),
            "mean_absorption": mean(scores),
            "median_absorption": median(scores),
            "mean_ticks": mean([bars[i].ticks for i in idxs]),
        }
        for h in FORWARD_HORIZONS:
            rets = [forward_return(bars, i, h) for i in idxs]
            clean = [r for r in rets if r is not None]
            row[f"ret_{h}"] = mean(clean)
            row[f"absret_{h}"] = mean([abs(r) for r in clean])
        decile_rows.append(row)
        print(
            f"{decile},{row['n']},{row['mean_absorption']:.3f},{row['median_absorption']:.3f},"
            f"{row['mean_ticks']:.1f},{row['ret_1']:.6f},{row['ret_3']:.6f},{row['ret_5']:.6f},"
            f"{row['absret_1']:.6f},{row['absret_3']:.6f},{row['absret_5']:.6f}"
        )
    for h in FORWARD_HORIZONS:
        print(f"top_minus_bottom_absret_{h}bar={decile_rows[-1][f'absret_{h}'] - decile_rows[0][f'absret_{h}']:.6f}")
        print(f"top_minus_bottom_signed_ret_{h}bar={decile_rows[-1][f'ret_{h}'] - decile_rows[0][f'ret_{h}']:.6f}")


def absorption_ics(bars: list[DeltaBar]) -> None:
    print("\nABSORPTION_IC")
    print("horizon,n,ic_absorption_vs_abs_return,ci_low,ci_high,ic_directional_absorption_vs_signed_return,ci_low,ci_high")
    for h in FORWARD_HORIZONS:
        scores = []
        absrets = []
        dir_scores = []
        signed = []
        for i, bar in enumerate(bars):
            score = absorption_score(bar)
            ret = forward_return(bars, i, h)
            if not math.isfinite(score) or ret is None:
                continue
            scores.append(score)
            absrets.append(abs(ret))
            pos = local_position(bars, i)
            if pos == "low":
                dir_score = score
            elif pos == "high":
                dir_score = -score
            else:
                dir_score = 0.0
            dir_scores.append(dir_score)
            signed.append(ret)
        n1, ic1, lo1, hi1 = corr_ci(scores, absrets)
        _, ic2, lo2, hi2 = corr_ci(dir_scores, signed)
        print(f"{h},{n1},{ic1:.6f},{lo1:.6f},{hi1:.6f},{ic2:.6f},{lo2:.6f},{hi2:.6f}")


def absorption_directional_test(bars: list[DeltaBar]) -> list[tuple[str, int, int]]:
    eligible = [
        (i, absorption_score(b))
        for i, b in enumerate(bars)
        if b.atr14 is not None and all(forward_return(bars, i, h) is not None for h in FORWARD_HORIZONS)
    ]
    cutoff = sorted(score for _, score in eligible)[int(len(eligible) * 0.9)]
    signals = []
    for i, score in eligible:
        if score < cutoff:
            continue
        pos = local_position(bars, i)
        if pos == "low":
            signals.append(("bullish_absorption", i, 1))
        elif pos == "high":
            signals.append(("bearish_absorption", i, -1))
    print("\nHIGH_ABSORPTION_DIRECTIONAL_REVERSAL")
    print("type,horizon,n,mean_directional_return,ci_low,ci_high,win_rate")
    for sig_type in ("bullish_absorption", "bearish_absorption"):
        rows = [s for s in signals if s[0] == sig_type]
        for h in FORWARD_HORIZONS:
            vals = []
            for _, i, direction in rows:
                ret = forward_return(bars, i, h)
                if ret is not None:
                    vals.append(direction * ret)
            n, m, lo, hi, _ = ci(vals)
            win = sum(v > 0 for v in vals) / len(vals) if vals else math.nan
            print(f"{sig_type},{h},{n},{m:.6f},{lo:.6f},{hi:.6f},{win:.2%}")
    return signals


def is_pivot_low(bars: list[DeltaBar], i: int) -> bool:
    if i - PIVOT_LEFT < 0 or i + PIVOT_RIGHT >= len(bars):
        return False
    window = bars[i - PIVOT_LEFT : i + PIVOT_RIGHT + 1]
    return all(b.segment_id == bars[i].segment_id for b in window) and bars[i].low < min(b.low for j, b in enumerate(window) if j != PIVOT_LEFT)


def is_pivot_high(bars: list[DeltaBar], i: int) -> bool:
    if i - PIVOT_LEFT < 0 or i + PIVOT_RIGHT >= len(bars):
        return False
    window = bars[i - PIVOT_LEFT : i + PIVOT_RIGHT + 1]
    return all(b.segment_id == bars[i].segment_id for b in window) and bars[i].high > max(b.high for j, b in enumerate(window) if j != PIVOT_LEFT)


def detect_divergences(bars: list[DeltaBar]) -> list[Divergence]:
    out = []
    last_low: int | None = None
    last_high: int | None = None
    for i in range(len(bars) - PIVOT_RIGHT):
        signal_index = i + PIVOT_RIGHT
        if is_pivot_low(bars, i):
            if last_low is not None and bars[last_low].segment_id == bars[i].segment_id:
                if bars[i].low < bars[last_low].low and bars[i].cvd20 > bars[last_low].cvd20:
                    out.append(Divergence("bullish", i, signal_index, bars[i].low, bars[i].cvd20, last_low, bars[last_low].low, bars[last_low].cvd20))
            last_low = i
        if is_pivot_high(bars, i):
            if last_high is not None and bars[last_high].segment_id == bars[i].segment_id:
                if bars[i].high > bars[last_high].high and bars[i].cvd20 < bars[last_high].cvd20:
                    out.append(Divergence("bearish", i, signal_index, bars[i].high, bars[i].cvd20, last_high, bars[last_high].high, bars[last_high].cvd20))
            last_high = i
    return out


def matched_baseline_indices(bars: list[DeltaBar], divs: list[Divergence]) -> dict[int, int]:
    used = set()
    div_signal_idxs = {d.signal_index for d in divs}
    out = {}
    atrs = [b.atr14 for b in bars if b.atr14 is not None]
    cuts = (quantile(atrs, 1 / 3), quantile(atrs, 2 / 3))

    def bucket(bar: DeltaBar) -> str:
        if bar.atr14 is None:
            return "missing"
        if bar.atr14 <= cuts[0]:
            return "low"
        if bar.atr14 <= cuts[1]:
            return "medium"
        return "high"

    for d in sorted(divs, key=lambda x: x.signal_index):
        ref = bars[d.signal_index]
        candidates = [
            i for i in range(max(0, d.signal_index - 5000), d.signal_index)
            if i not in used
            and i not in div_signal_idxs
            and bars[i].segment_id == ref.segment_id
            and bars[i].session == ref.session
            and bucket(bars[i]) == bucket(ref)
            and all(forward_return(bars, i, h) is not None for h in FORWARD_HORIZONS)
        ]
        if candidates:
            choice = candidates[-1]
            used.add(choice)
            out[d.signal_index] = choice
    return out


def divergence_test(bars: list[DeltaBar], divs: list[Divergence]) -> None:
    baseline = matched_baseline_indices(bars, divs)
    print("\nDELTA_PRICE_DIVERGENCE_REVERSAL")
    print(f"pivot_left={PIVOT_LEFT},pivot_right={PIVOT_RIGHT},signal_at=pivot+{PIVOT_RIGHT}")
    print("type,horizon,n,mean_directional_return,ci_low,ci_high,win_rate,baseline_n,baseline_mean,baseline_ci_low,baseline_ci_high")
    for kind, direction in (("bullish", 1), ("bearish", -1)):
        kind_divs = [d for d in divs if d.kind == kind]
        for h in FORWARD_HORIZONS:
            vals = []
            base_vals = []
            for d in kind_divs:
                ret = forward_return(bars, d.signal_index, h)
                if ret is not None:
                    vals.append(direction * ret)
                bidx = baseline.get(d.signal_index)
                if bidx is not None:
                    bret = forward_return(bars, bidx, h)
                    if bret is not None:
                        base_vals.append(direction * bret)
            n, m, lo, hi, _ = ci(vals)
            bn, bm, blo, bhi, _ = ci(base_vals)
            win = sum(v > 0 for v in vals) / len(vals) if vals else math.nan
            print(f"{kind},{h},{n},{m:.6f},{lo:.6f},{hi:.6f},{win:.2%},{bn},{bm:.6f},{blo:.6f},{bhi:.6f}")


def combo_test(bars: list[DeltaBar], absorption_signals: list[tuple[str, int, int]], divs: list[Divergence]) -> None:
    abs_by_index = {i: direction for _, i, direction in absorption_signals}
    div_by_index = {d.signal_index: 1 if d.kind == "bullish" else -1 for d in divs}
    combo = [(i, abs_by_index[i]) for i in abs_by_index if i in div_by_index and abs_by_index[i] == div_by_index[i]]
    print("\nCOMBO_HIGH_ABSORPTION_PLUS_DIVERGENCE")
    print("horizon,n,mean_directional_return,ci_low,ci_high,win_rate")
    for h in FORWARD_HORIZONS:
        vals = []
        for i, direction in combo:
            ret = forward_return(bars, i, h)
            if ret is not None:
                vals.append(direction * ret)
        n, m, lo, hi, _ = ci(vals)
        win = sum(v > 0 for v in vals) / len(vals) if vals else math.nan
        print(f"{h},{n},{m:.6f},{lo:.6f},{hi:.6f},{win:.2%}")


def cost_check(bars: list[DeltaBar], absorption_signals: list[tuple[str, int, int]], divs: list[Divergence]) -> None:
    tests = []
    tests.append(("high_absorption_directional", [(i, direction) for _, i, direction in absorption_signals]))
    tests.append(("divergence", [(d.signal_index, 1 if d.kind == "bullish" else -1) for d in divs]))
    print("\nCOST_CHECK_DIRECTIONAL_REVERSAL_SIGNALS")
    print("feature,horizon,n,gross_mean,net_mean,ci_low,ci_high,win_rate")
    for name, signals in tests:
        for h in FORWARD_HORIZONS:
            gross = []
            net = []
            for i, direction in signals:
                j = i + 1
                k = i + h + 1
                if k >= len(bars) or j >= len(bars) or bars[k].segment_id != bars[i].segment_id or bars[j].segment_id != bars[i].segment_id:
                    continue
                ret = direction * (bars[k].close - bars[j].open)
                gross.append(ret)
                net.append(ret - IUX_XAUUSD_ROUNDTRIP_SPREAD)
            n, m, lo, hi, _ = ci(net)
            win = sum(v > 0 for v in net) / len(net) if net else math.nan
            print(f"{name},{h},{n},{mean(gross) if gross else math.nan:.6f},{m:.6f},{lo:.6f},{hi:.6f},{win:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=Path, default=None)
    args = parser.parse_args()
    path = args.ticks or default_tick_path()
    print("Building M15 tick-rule delta bars...", flush=True)
    bars = load_delta_bars(path)
    print("\nORDERFLOW_ABSORPTION_DIVERGENCE_CONTEXT")
    print(f"tick_file={path}")
    print(f"bars={len(bars)}")
    print(f"date_range={bars[0].start:%Y-%m-%d %H:%M:%S} to {bars[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print("caveat=spot quote ticks; tick-count volume and tick-rule delta, not true futures traded volume")
    print(f"absorption_score=ticks/((high-low)/ATR14), local_extreme_lookback={LOCAL_EXTREME_LOOKBACK}")
    decile_absorption(bars)
    absorption_ics(bars)
    absorption_signals = absorption_directional_test(bars)
    divs = detect_divergences(bars)
    print(f"\ndivergence_count_total={len(divs)},bullish={sum(d.kind == 'bullish' for d in divs)},bearish={sum(d.kind == 'bearish' for d in divs)}")
    divergence_test(bars, divs)
    combo_test(bars, absorption_signals, divs)
    cost_check(bars, absorption_signals, divs)


if __name__ == "__main__":
    main()
