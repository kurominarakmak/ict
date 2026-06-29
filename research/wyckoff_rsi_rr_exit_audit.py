"""
RR-exit retest for Method 3 (Wyckoff RSI) from the compression breakout audit.

Only the exit scheme changes:
- Same compression definition and breakout trigger as volatility_compression_breakout_audit.py.
- Same Wyckoff RSI direction call as the original audit: RSI > 50 with higher
  lows predicts up; RSI < 50 with lower highs predicts down. This is equivalent
  to sensitivity=0 around the 50 line in the prior implementation.
- Entry is at breakout confirmation close.
- Fixed 1R stop is $10/oz.
- TP ladder: close 50% at +1R and move stop to breakeven, close 25% at +2R,
  close final 25% at +3R.
- Force-close unresolved remainder at 20 M15 bars, or the last available bar
  before a segment gap if sooner.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import volatility_compression_breakout_audit as base
from delta_signal_audit import IUX_XAUUSD_ROUNDTRIP_SPREAD, default_tick_path


RISK_USD_OZ = 10.0
FORCE_CLOSE_BARS = 20
COIN_FLIP_SEED = 20260629
TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=base.timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=base.timezone.utc)


@dataclass(frozen=True)
class RRTrade:
    method: str
    event_id: int
    entry_index: int
    entry_time: datetime
    direction: int
    breakout_direction: int
    direction_correct: bool
    r_net: float
    gross_r: float
    exit_reason: str
    horizon_bars: int
    unresolved_forced: bool


def ci(vals: list[float]) -> tuple[int, float, float, float]:
    if not vals:
        return 0, math.nan, math.nan, math.nan
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals))
    return len(vals), m, m - 1.96 * se, m + 1.96 * se


def segment_end_index(bars: list[base.DeltaBar], start_index: int, max_horizon: int) -> int:
    end = min(len(bars) - 1, start_index + max_horizon)
    segment = bars[start_index].segment_id
    j = start_index
    while j + 1 <= end and bars[j + 1].segment_id == segment:
        j += 1
    return j


def simulate_rr_trade(bars: list[base.DeltaBar], event: base.CompressionEvent, direction: int, method: str) -> RRTrade:
    entry = bars[event.breakout_index].close
    stop = entry - direction * RISK_USD_OZ
    targets = [entry + direction * RISK_USD_OZ, entry + direction * 2 * RISK_USD_OZ, entry + direction * 3 * RISK_USD_OZ]
    target_weights = [0.50, 0.25, 0.25]
    remaining = 1.0
    gross_r = 0.0
    next_target = 0
    stop_moved_be = False
    exit_reason = "force_close"
    end_index = segment_end_index(bars, event.breakout_index, FORCE_CLOSE_BARS)

    for i in range(event.breakout_index + 1, end_index + 1):
        bar = bars[i]

        # Conservative OHLC ordering: if a bar can both stop and target, the
        # stop is assumed to fill first. Stop slippage uses the adverse bar
        # extreme because M15 bars do not preserve tick order.
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        if stop_hit and remaining > 0:
            fill = min(stop, bar.low) if direction == 1 else max(stop, bar.high)
            gross_r += remaining * direction * (fill - entry) / RISK_USD_OZ
            remaining = 0.0
            exit_reason = "stop"
            break

        while next_target < len(targets) and remaining > 0:
            target = targets[next_target]
            target_hit = bar.high >= target if direction == 1 else bar.low <= target
            if not target_hit:
                break
            weight = min(target_weights[next_target], remaining)
            gross_r += weight * direction * (target - entry) / RISK_USD_OZ
            remaining -= weight
            next_target += 1
            if next_target == 1 and not stop_moved_be:
                stop = entry
                stop_moved_be = True
            if remaining <= 1e-12:
                remaining = 0.0
                exit_reason = "tp3"
                break

    if remaining > 0:
        close = bars[end_index].close
        gross_r += remaining * direction * (close - entry) / RISK_USD_OZ
        exit_reason = "force_close"

    net_r = gross_r - (IUX_XAUUSD_ROUNDTRIP_SPREAD / RISK_USD_OZ)
    return RRTrade(
        method=method,
        event_id=event.event_id,
        entry_index=event.breakout_index,
        entry_time=bars[event.breakout_index].start,
        direction=direction,
        breakout_direction=event.breakout_direction,
        direction_correct=direction == event.breakout_direction,
        r_net=net_r,
        gross_r=gross_r,
        exit_reason=exit_reason,
        horizon_bars=end_index - event.breakout_index,
        unresolved_forced=exit_reason == "force_close",
    )


def build_trade_sets(bars: list[base.DeltaBar], events: list[base.CompressionEvent]) -> dict[str, list[RRTrade]]:
    wyckoff_events = [e for e in events if e.predictions["method_3_wyckoff_rsi"] is not None]
    rng = random.Random(COIN_FLIP_SEED)
    out = {
        "wyckoff_rsi": [],
        "coin_flip": [],
        "prior_momentum": [],
        "simple_follow": [],
    }
    for event in wyckoff_events:
        wyckoff_dir = int(event.predictions["method_3_wyckoff_rsi"])
        coin_dir = 1 if rng.random() >= 0.5 else -1
        momentum_dir = event.predictions["baseline_prior_momentum"]
        simple_dir = event.breakout_direction
        directions = {
            "wyckoff_rsi": wyckoff_dir,
            "coin_flip": coin_dir,
            "simple_follow": simple_dir,
        }
        if momentum_dir is not None:
            directions["prior_momentum"] = int(momentum_dir)
        for method, direction in directions.items():
            out[method].append(simulate_rr_trade(bars, event, direction, method))
    return out


def summarize(trades: list[RRTrade]) -> dict[str, float]:
    vals = [t.r_net for t in trades]
    n, m, lo, hi = ci(vals)
    return {
        "n": n,
        "direction_correct": sum(t.direction_correct for t in trades) / n if n else math.nan,
        "win_rate": sum(t.r_net > 0 for t in trades) / n if n else math.nan,
        "expectancy_r": m,
        "ci_low": lo,
        "ci_high": hi,
        "unresolved": sum(t.unresolved_forced for t in trades) / n if n else math.nan,
        "stop": sum(t.exit_reason == "stop" for t in trades),
        "tp3": sum(t.exit_reason == "tp3" for t in trades),
        "force": sum(t.exit_reason == "force_close" for t in trades),
        "avg_horizon": mean([t.horizon_bars for t in trades]) if trades else math.nan,
    }


def print_summary_table(label: str, trade_sets: dict[str, list[RRTrade]]) -> None:
    print(f"\n{label}")
    print("method,n,direction_correct,win_rate,expectancy_r,ci_low,ci_high,unresolved_pct,stop_count,tp3_count,force_close_count,avg_horizon_bars,ci_clears_zero")
    for method in ["wyckoff_rsi", "coin_flip", "prior_momentum", "simple_follow"]:
        s = summarize(trade_sets.get(method, []))
        print(
            f"{method},{int(s['n'])},{s['direction_correct']:.2%},{s['win_rate']:.2%},"
            f"{s['expectancy_r']:.6f},{s['ci_low']:.6f},{s['ci_high']:.6f},"
            f"{s['unresolved']:.2%},{int(s['stop'])},{int(s['tp3'])},{int(s['force'])},"
            f"{s['avg_horizon']:.2f},{s['ci_low'] > 0 if math.isfinite(s['ci_low']) else False}"
        )


def filter_trade_sets(trade_sets: dict[str, list[RRTrade]], start: datetime | None, end: datetime | None) -> dict[str, list[RRTrade]]:
    out: dict[str, list[RRTrade]] = {}
    for method, trades in trade_sets.items():
        out[method] = [
            t for t in trades
            if (start is None or t.entry_time >= start) and (end is None or t.entry_time <= end)
        ]
    return out


def print_verdict(trade_sets: dict[str, list[RRTrade]]) -> None:
    summaries = {method: summarize(trades) for method, trades in trade_sets.items()}
    wy = summaries["wyckoff_rsi"]
    beats = (
        wy["expectancy_r"] > summaries["coin_flip"]["expectancy_r"]
        and wy["expectancy_r"] > summaries["prior_momentum"]["expectancy_r"]
        and wy["expectancy_r"] > summaries["simple_follow"]["expectancy_r"]
    )
    print("\nVERDICT")
    print(f"wyckoff_expectancy_r={wy['expectancy_r']:.6f},ci_low={wy['ci_low']:.6f},ci_high={wy['ci_high']:.6f},ci_clears_zero={wy['ci_low'] > 0 if math.isfinite(wy['ci_low']) else False}")
    print(f"wyckoff_beats_all_rr_baselines_by_expectancy={beats}")
    if wy["ci_low"] > 0 and beats:
        print("interpretation=RR ladder converts the Wyckoff direction signal into positive net R expectancy on this sample; train/test stability still matters.")
    else:
        print("interpretation=The RR ladder does not establish a robust Wyckoff edge here; direction accuracy alone is not enough if expectancy CI does not clear zero and/or baselines are competitive.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None)
    parser.add_argument("--bar-cache", type=Path, default=base.DEFAULT_BAR_CACHE)
    args = parser.parse_args()

    tick_path = args.ticks or default_tick_path()
    base.ensure_bar_cache(tick_path, args.bar_cache)
    bars = base.load_cached_bars(args.bar_cache)
    base.bars_global = bars
    events = base.detect_events(bars)
    trade_sets = build_trade_sets(bars, events)
    wyckoff_n = len(trade_sets["wyckoff_rsi"])

    print("WYCKOFF_RSI_RR_EXIT_AUDIT_CONTEXT")
    print(f"tick_file={tick_path}")
    print(f"bars={len(bars)}")
    print(f"date_range={bars[0].start:%Y-%m-%d %H:%M:%S} to {bars[-1].end:%Y-%m-%d %H:%M:%S} UTC")
    print(f"compression_events_total={len(events)}")
    print(f"comparison_universe=Wyckoff-covered compression events only; n={wyckoff_n}")
    print(f"entry=breakout confirmation close")
    print(f"risk_1r_usd_oz={RISK_USD_OZ:.2f}")
    print(f"spread_usd_oz={IUX_XAUUSD_ROUNDTRIP_SPREAD:.2f},spread_r={IUX_XAUUSD_ROUNDTRIP_SPREAD / RISK_USD_OZ:.4f}")
    print(f"tp_ladder=50% at +1R, 25% at +2R, 25% at +3R; stop moves to breakeven after TP1")
    print(f"force_close_bars={FORCE_CLOSE_BARS}; earlier at segment end if a market gap occurs")
    print("ohlc_fill_rule=conservative M15 OHLC: same-bar stop before target; TP at limit; SL at adverse bar extreme")
    print("wyckoff_rsi_rule=same as prior audit, sensitivity=0 around RSI 50")

    print_summary_table("ALL_SAMPLE_RR_RESULTS", trade_sets)
    print_summary_table("TRAIN_2016_2021_RR_RESULTS", filter_trade_sets(trade_sets, None, TRAIN_END))
    print_summary_table("TEST_2022_2026_RR_RESULTS", filter_trade_sets(trade_sets, TEST_START, None))
    print_verdict(trade_sets)


if __name__ == "__main__":
    main()
