"""
H-2026-TF-01: cross-timeframe replication of the validated compression spec.

Research-only. Does not touch the live bot.

This is a cost-rescue / replication audit, not a threshold search. The M15
bars are resampled to H1/H4 by exact OHLC aggregation, then the unchanged
validated compression breakout spec is applied at each timeframe:

- compression window 16 bars
- ATR14, trailing ATR tercile over 100 bars, bottom tercile
- at least 75% of the 16-bar window compressed
- range-edge stop entry on close-confirmed breakout
- SL 1.0 ATR, TP 1.5R, 10-bar force close
- segment-gap close / session-flatten exclusions inherited from the M15 cache
"""

from __future__ import annotations

import argparse
import io
import math
import random
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

import compression_breakout_ablation_study as ablate
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as base
from delta_signal_audit import DeltaBar, IUX_XAUUSD_ROUNDTRIP_SPREAD, add_indicators, classify_session, pearson


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
RR = 1.5
HORIZON = 10
BOOT_N = 1000
BOOT_SEED = 20260702
MIN_JUDGED_N = 150
RESULTS_PATH = Path("research/compression_timeframe_replication_results.txt")
REGISTRY_PATH = Path("research/hypothesis_registry.md")


@dataclass(frozen=True)
class TFTrade:
    symbol: str
    timeframe: str
    event_id: int
    entry_time: datetime
    gross_r: float
    net_r: float
    win: bool
    exit_reason: str
    bars_held: int
    risk: float
    spread_r: float
    truncated_by_segment: bool


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
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
    return quantile(means, 0.025), quantile(means, 0.975)


def period_rows(rows: list[TFTrade], period: str) -> list[TFTrade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def tf_seconds(tf_minutes: int) -> int:
    return tf_minutes * 60


def bucket_start(ts: datetime, tf_minutes: int) -> datetime:
    epoch = int(ts.timestamp())
    bucket = (epoch // tf_seconds(tf_minutes)) * tf_seconds(tf_minutes)
    return datetime.fromtimestamp(bucket, tz=timezone.utc)


def aggregate_bars(m15: list[DeltaBar], tf_minutes: int) -> list[DeltaBar]:
    if tf_minutes == 15:
        return m15
    expected = tf_minutes // 15
    if tf_minutes % 15 != 0:
        raise ValueError(f"timeframe must be a multiple of M15: {tf_minutes}")
    grouped: dict[tuple[int, datetime], list[DeltaBar]] = {}
    for bar in m15:
        grouped.setdefault((bar.segment_id, bucket_start(bar.start, tf_minutes)), []).append(bar)

    out: list[DeltaBar] = []
    prev_key: tuple[int, datetime] | None = None
    out_segment = -1
    for key, members in sorted(grouped.items(), key=lambda item: item[1][0].start):
        members = sorted(members, key=lambda b: b.start)
        if len(members) != expected:
            continue
        if any(members[i].end != members[i + 1].start for i in range(len(members) - 1)):
            continue
        segment_id, start = key
        if (
            prev_key is None
            or segment_id != prev_key[0]
            or start - prev_key[1] != timedelta(minutes=tf_minutes)
        ):
            out_segment += 1
        prev_key = key
        ticks = sum(b.ticks for b in members)
        buy_ticks = sum(b.buy_ticks for b in members)
        sell_ticks = sum(b.sell_ticks for b in members)
        neutral_ticks = sum(b.neutral_ticks for b in members)
        delta = sum(b.delta for b in members)
        denom = buy_ticks + sell_ticks
        out.append(
            DeltaBar(
                index=len(out),
                segment_id=out_segment,
                start=start,
                end=start + timedelta(minutes=tf_minutes),
                open=members[0].open,
                high=max(b.high for b in members),
                low=min(b.low for b in members),
                close=members[-1].close,
                ticks=ticks,
                buy_ticks=buy_ticks,
                sell_ticks=sell_ticks,
                neutral_ticks=neutral_ticks,
                delta=delta,
                delta_ratio=delta / denom if denom else 0.0,
                session=classify_session(start + timedelta(minutes=tf_minutes)),
            )
        )
    add_indicators(out)
    return out


def end_index_with_truncation(
    bars: list[DeltaBar], start_index: int, horizon: int
) -> tuple[int, bool]:
    desired = min(len(bars) - 1, start_index + horizon)
    end = simple.segment_end_index(bars, start_index, horizon)
    return end, end < desired


def simulate_tf_trade(
    symbol: str,
    timeframe: str,
    bars: list[DeltaBar],
    event: ablate.Event,
    spread: float,
) -> TFTrade | None:
    risk = ablate.risk_at_setup_end(bars, event)
    if risk is None:
        return None
    entry_index = event.breakout_index
    eval_start = entry_index + 1
    if eval_start >= len(bars) or bars[eval_start].segment_id != bars[entry_index].segment_id:
        return None

    direction = event.direction
    entry = event.range_high if direction == 1 else event.range_low
    stop = entry - direction * risk
    target = entry + direction * RR * risk
    end_index, truncated = end_index_with_truncation(bars, eval_start, HORIZON)

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
            gross_r = RR
            exit_reason = "target"
            exit_index = i
            break

    net_r = gross_r - spread / risk
    return TFTrade(
        symbol=symbol,
        timeframe=timeframe,
        event_id=event.event_id,
        entry_time=bars[entry_index].start,
        gross_r=gross_r,
        net_r=net_r,
        win=net_r > 0,
        exit_reason=exit_reason,
        bars_held=exit_index - entry_index,
        risk=risk,
        spread_r=spread / risk,
        truncated_by_segment=truncated and exit_reason == "force_close",
    )


def build_trades(symbol: str, timeframe: str, bars: list[DeltaBar], spread: float) -> list[TFTrade]:
    rows = []
    for event in ablate.detect_compression(bars):
        trade = simulate_tf_trade(symbol, timeframe, bars, event, spread)
        if trade is not None:
            rows.append(trade)
    return rows


def summarize(rows: list[TFTrade], seed: str) -> dict[str, float | str]:
    n = len(rows)
    gross_vals = [r.gross_r for r in rows]
    net_vals = [r.net_r for r in rows]
    gross_lo, gross_hi = bootstrap_ci(gross_vals, f"{seed}-gross")
    net_lo, net_hi = bootstrap_ci(net_vals, f"{seed}-net")
    status = "JUDGED" if n >= MIN_JUDGED_N else "DESCRIPTIVE_N_LT_150"
    return {
        "n": n,
        "status": status,
        "win_rate": sum(r.win for r in rows) / n if n else math.nan,
        "gross": mean(gross_vals) if gross_vals else math.nan,
        "gross_lo": gross_lo,
        "gross_hi": gross_hi,
        "net": mean(net_vals) if net_vals else math.nan,
        "net_lo": net_lo,
        "net_hi": net_hi,
        "spread_atr": mean([r.spread_r for r in rows]) if rows else math.nan,
        "flatten_trunc": sum(r.truncated_by_segment for r in rows) / n if n else math.nan,
        "avg_atr": mean([r.risk for r in rows]) if rows else math.nan,
        "target_rate": sum(r.exit_reason == "target" for r in rows) / n if n else math.nan,
        "stop_rate": sum(r.exit_reason == "stop" for r in rows) / n if n else math.nan,
        "force_rate": sum(r.exit_reason == "force_close" for r in rows) / n if n else math.nan,
    }


def daily_returns(rows: list[TFTrade]) -> dict[datetime.date, float]:
    out: dict[datetime.date, float] = {}
    for row in rows:
        out[row.entry_time.date()] = out.get(row.entry_time.date(), 0.0) + row.net_r
    return out


def xau_h1_overlap_and_corr(m15_rows: list[TFTrade], h1_rows: list[TFTrade]) -> tuple[float, float, int, int]:
    m15_times = sorted(r.entry_time for r in m15_rows)
    window = timedelta(hours=3)
    overlap = 0
    cursor = 0
    for row in sorted(h1_rows, key=lambda r: r.entry_time):
        while cursor < len(m15_times) and m15_times[cursor] < row.entry_time - window:
            cursor += 1
        if cursor < len(m15_times) and abs(m15_times[cursor] - row.entry_time) <= window:
            overlap += 1
    h1_overlap_pct = overlap / len(h1_rows) if h1_rows else math.nan

    m15_daily = daily_returns(m15_rows)
    h1_daily = daily_returns(h1_rows)
    dates = sorted(set(m15_daily) | set(h1_daily))
    corr = pearson([m15_daily.get(d, 0.0) for d in dates], [h1_daily.get(d, 0.0) for d in dates])
    return h1_overlap_pct, corr, overlap, len(h1_rows)


def load_symbol_bars(args: argparse.Namespace, symbol: str) -> list[DeltaBar]:
    if symbol == "XAUUSD":
        return simple.load_symbol_bars(symbol, args.xau_ticks, args.xau_cache)
    if symbol == "XAGUSD":
        return simple.load_symbol_bars(symbol, args.xag_ticks, args.xag_cache)
    raise ValueError(symbol)


def format_float(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def compression_diagnostics(bars: list[DeltaBar]) -> dict[str, float]:
    flags = []
    for i, bar in enumerate(bars):
        cutoff = base.trailing_atr_cutoff(bars, i)
        flags.append(cutoff is not None and bar.atr14 is not None and bar.atr14 <= cutoff)
    max_streak = 0
    streak = 0
    for flag in flags:
        if flag:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    max_fraction = 0.0
    eligible_windows = 0
    for i in range(base.ATR_TRAIL + base.COMPRESSION_WINDOW, len(bars)):
        start = i - base.COMPRESSION_WINDOW + 1
        if ablate.same_segment(bars, start, i):
            eligible_windows += 1
            max_fraction = max(max_fraction, sum(flags[start : i + 1]) / base.COMPRESSION_WINDOW)
    seg_lengths = []
    current_segment = None
    current_len = 0
    for bar in bars:
        if current_segment is None or bar.segment_id != current_segment:
            if current_len:
                seg_lengths.append(current_len)
            current_segment = bar.segment_id
            current_len = 1
        else:
            current_len += 1
    if current_len:
        seg_lengths.append(current_len)
    return {
        "bars": len(bars),
        "atr_bars": sum(1 for b in bars if b.atr14 is not None),
        "segments": len(seg_lengths),
        "max_segment_len": max(seg_lengths) if seg_lengths else 0,
        "eligible_16_bar_windows": eligible_windows,
        "compressed_flag_count": sum(flags),
        "max_compressed_streak": max_streak,
        "max_16_bar_compression_fraction": max_fraction,
        "compression_end_count": sum(
            1
            for i in range(base.ATR_TRAIL + base.COMPRESSION_WINDOW, len(bars) - base.EXIT_HORIZON)
            if base.is_compression_end(bars, i)
        ),
    }


def emit_report(
    all_rows: dict[tuple[str, str], list[TFTrade]],
    bar_contexts: dict[tuple[str, str], list[DeltaBar]],
) -> dict[str, object]:
    print("H_2026_TF_01_COMPRESSION_TIMEFRAME_REPLICATION")
    print("spec=unchanged_validated_compression; rr=1.5; max_hold=10_bars_of_each_timeframe; no_threshold_tuning")
    print("costs=XAUUSD_0.20_roundtrip,XAGUSD_0.02_roundtrip")
    print("judgement_rule=split_n_lt_150_marked_descriptive_not_judged")

    print("\nRESAMPLE_AND_SIGNAL_DIAGNOSTICS")
    print(
        "symbol,timeframe,bars,atr_bars,segments,max_segment_len,eligible_16_bar_windows,"
        "compressed_flag_count,max_compressed_streak,max_16_bar_compression_fraction,compression_end_count"
    )
    for symbol in ("XAUUSD", "XAGUSD"):
        for timeframe in ("M15_REF", "H1", "H4"):
            d = compression_diagnostics(bar_contexts[(symbol, timeframe)])
            print(
                f"{symbol},{timeframe},{int(d['bars'])},{int(d['atr_bars'])},{int(d['segments'])},"
                f"{int(d['max_segment_len'])},{int(d['eligible_16_bar_windows'])},"
                f"{int(d['compressed_flag_count'])},{int(d['max_compressed_streak'])},"
                f"{format_float(float(d['max_16_bar_compression_fraction']), 4)},"
                f"{int(d['compression_end_count'])}"
            )

    print("\nM15_REFERENCE_AND_FIXED_TF_GRID")
    print(
        "period,symbol,timeframe,n,status,win_rate,spread_atr,gross_r,gross_ci_low,gross_ci_high,"
        "net_r,net_ci_low,net_ci_high,flatten_trunc_pct,avg_atr,target_pct,stop_pct,force_pct,ci_clears_zero"
    )
    for period in ("train", "test"):
        for symbol in ("XAUUSD", "XAGUSD"):
            for timeframe in ("M15_REF", "H1", "H4"):
                rows = period_rows(all_rows[(symbol, timeframe)], period)
                s = summarize(rows, f"{BOOT_SEED}-{period}-{symbol}-{timeframe}")
                ci_clears = bool(math.isfinite(float(s["net_lo"])) and float(s["net_lo"]) > 0)
                print(
                    f"{period},{symbol},{timeframe},{int(s['n'])},{s['status']},"
                    f"{format_float(float(s['win_rate']), 2)},{format_float(float(s['spread_atr']), 6)},"
                    f"{format_float(float(s['gross']))},{format_float(float(s['gross_lo']))},{format_float(float(s['gross_hi']))},"
                    f"{format_float(float(s['net']))},{format_float(float(s['net_lo']))},{format_float(float(s['net_hi']))},"
                    f"{format_float(float(s['flatten_trunc']), 2)},{format_float(float(s['avg_atr']))},"
                    f"{format_float(float(s['target_rate']), 2)},{format_float(float(s['stop_rate']), 2)},"
                    f"{format_float(float(s['force_rate']), 2)},{ci_clears}"
                )

    print("\nFULL_PERIOD_CONTEXT")
    print("symbol,timeframe,n,win_rate,spread_atr,gross_r,net_r,net_ci_low,net_ci_high,flatten_trunc_pct,avg_atr")
    for symbol in ("XAUUSD", "XAGUSD"):
        for timeframe in ("M15_REF", "H1", "H4"):
            rows = all_rows[(symbol, timeframe)]
            s = summarize(rows, f"{BOOT_SEED}-full-{symbol}-{timeframe}")
            print(
                f"{symbol},{timeframe},{int(s['n'])},{format_float(float(s['win_rate']), 2)},"
                f"{format_float(float(s['spread_atr']), 6)},{format_float(float(s['gross']))},"
                f"{format_float(float(s['net']))},{format_float(float(s['net_lo']))},"
                f"{format_float(float(s['net_hi']))},{format_float(float(s['flatten_trunc']), 2)},"
                f"{format_float(float(s['avg_atr']))}"
            )

    h1_overlap_pct, daily_corr, overlap_n, h1_n = xau_h1_overlap_and_corr(
        all_rows[("XAUUSD", "M15_REF")], all_rows[("XAUUSD", "H1")]
    )
    print("\nXAU_H1_VS_M15_INDEPENDENCE")
    print("metric,value")
    print(f"h1_trades,{h1_n}")
    print(f"h1_with_m15_trade_within_plus_minus_3_h1_bars,{overlap_n}")
    print(f"h1_overlap_pct,{format_float(h1_overlap_pct, 4)}")
    print(f"daily_net_r_correlation,{format_float(daily_corr, 4)}")

    def clears(symbol: str, timeframe: str, period: str) -> bool:
        rows = period_rows(all_rows[(symbol, timeframe)], period)
        s = summarize(rows, f"{BOOT_SEED}-gate-{symbol}-{timeframe}-{period}")
        return int(s["n"]) >= MIN_JUDGED_N and math.isfinite(float(s["net_lo"])) and float(s["net_lo"]) > 0

    xag_h1_pass = clears("XAGUSD", "H1", "train") and clears("XAGUSD", "H1", "test")
    xag_h4_pass = clears("XAGUSD", "H4", "train") and clears("XAGUSD", "H4", "test")
    xau_h1_pass = clears("XAUUSD", "H1", "train") and clears("XAUUSD", "H1", "test") and h1_overlap_pct < 0.40

    print("\nPRE_REGISTERED_GATE_VERDICTS")
    print("gate,result,detail")
    print(
        "SILVER_RESCUE,"
        f"{'PASS' if (xag_h1_pass or xag_h4_pass) else 'FAIL'},"
        f"XAG_H1={'PASS' if xag_h1_pass else 'FAIL'}; XAG_H4={'PASS' if xag_h4_pass else 'FAIL_OR_DESCRIPTIVE'}"
    )
    print(
        "GOLD_H1_SECOND_STREAM,"
        f"{'PASS' if xau_h1_pass else 'FAIL'},"
        f"net_CI_gate_and_overlap_lt_40pct; overlap_pct={format_float(h1_overlap_pct, 4)}; daily_corr={format_float(daily_corr, 4)}"
    )
    print(
        "H4_LIVE_CAUTION,INFO,"
        "10 H4 bars is roughly two days exposure; any live use needs a separate gap/financing audit."
    )

    return {
        "xag_h1_pass": xag_h1_pass,
        "xag_h4_pass": xag_h4_pass,
        "xau_h1_pass": xau_h1_pass,
        "h1_overlap_pct": h1_overlap_pct,
        "daily_corr": daily_corr,
    }


def append_registry(verdicts: dict[str, object]) -> None:
    overlap = float(verdicts["h1_overlap_pct"])
    corr = float(verdicts["daily_corr"])
    overlap_text = f"{overlap:.2%}" if math.isfinite(overlap) else "no_h1_trades"
    corr_text = f"{corr:.3f}" if math.isfinite(corr) else "no_h1_trades"
    registered = (
        "- 2026-07-02: H-2026-TF-01 registered. Cross-timeframe replication/cost-rescue audit: "
        "unchanged validated compression spec on resampled H1/H4 XAU/XAG; fixed costs; no threshold tuning."
    )
    result = (
        "- 2026-07-02: H-2026-TF-01 result: "
        f"SILVER_RESCUE={'PASS' if (verdicts['xag_h1_pass'] or verdicts['xag_h4_pass']) else 'FAIL'} "
        f"(XAG_H1={'PASS' if verdicts['xag_h1_pass'] else 'FAIL'}, "
        f"XAG_H4={'PASS' if verdicts['xag_h4_pass'] else 'FAIL_OR_DESCRIPTIVE'}); "
        f"GOLD_H1_SECOND_STREAM={'PASS' if verdicts['xau_h1_pass'] else 'FAIL'} "
        f"(overlap={overlap_text}, daily_corr={corr_text})."
    )
    existing = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else "# Hypothesis Registry\n"
    lines = [line for line in existing.rstrip().splitlines() if "H-2026-TF-01" not in line]
    for entry in (registered, result):
        if entry not in lines:
            lines.append(entry)
    REGISTRY_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xag-ticks", type=Path, default=Path("data/2026.6.28XAGUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--xag-cache", type=Path, default=Path("data/xagusd_m15_delta_bars.csv"))
    parser.add_argument("--xau-spread", type=float, default=IUX_XAUUSD_ROUNDTRIP_SPREAD)
    parser.add_argument("--xag-spread", type=float, default=0.02)
    args = parser.parse_args()

    bars_by_symbol = {
        "XAUUSD": load_symbol_bars(args, "XAUUSD"),
        "XAGUSD": load_symbol_bars(args, "XAGUSD"),
    }
    spreads = {"XAUUSD": args.xau_spread, "XAGUSD": args.xag_spread}
    all_rows: dict[tuple[str, str], list[TFTrade]] = {}
    bar_contexts: dict[tuple[str, str], list[DeltaBar]] = {}
    for symbol, m15 in bars_by_symbol.items():
        for timeframe, minutes in (("M15_REF", 15), ("H1", 60), ("H4", 240)):
            bars = aggregate_bars(m15, minutes)
            bar_contexts[(symbol, timeframe)] = bars
            all_rows[(symbol, timeframe)] = build_trades(symbol, timeframe, bars, spreads[symbol])

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        verdicts = emit_report(all_rows, bar_contexts)
    report = buffer.getvalue()
    print(report, end="")
    RESULTS_PATH.write_text(report)
    append_registry(verdicts)
    print(f"\nresults_file={RESULTS_PATH}")


if __name__ == "__main__":
    main()
