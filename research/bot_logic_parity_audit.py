"""
V-2026-PARITY-01: engineering parity audit for the live compression bot.

This is not a new trading hypothesis. It verifies whether the live bot decision
logic, replayed over historical M15 XAUUSD bars, reproduces the validated
research compression edge.

Hard rule: this harness imports research/iux_mt5_compression_breakout_bot.py
read-only after installing a minimal MetaTrader5 stub.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
import types
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))


def install_mt5_stub() -> None:
    mt5 = types.ModuleType("MetaTrader5")
    constants = {
        "TIMEFRAME_M15": 15,
        "ORDER_TYPE_BUY_STOP": 4,
        "ORDER_TYPE_SELL_STOP": 5,
        "ORDER_TYPE_BUY": 0,
        "ORDER_TYPE_SELL": 1,
        "POSITION_TYPE_BUY": 0,
        "POSITION_TYPE_SELL": 1,
        "TRADE_ACTION_PENDING": 5,
        "TRADE_ACTION_REMOVE": 8,
        "TRADE_ACTION_DEAL": 1,
        "ORDER_TIME_GTC": 0,
        "ORDER_FILLING_RETURN": 2,
        "ORDER_FILLING_FOK": 0,
        "ORDER_FILLING_IOC": 1,
        "TRADE_RETCODE_DONE": 10009,
        "TRADE_RETCODE_DONE_PARTIAL": 10010,
        "TRADE_RETCODE_PLACED": 10008,
        "TRADE_RETCODE_MARKET_CLOSED": 10018,
        "SYMBOL_TRADE_MODE_FULL": 4,
        "DEAL_ENTRY_OUT": 1,
        "DEAL_ENTRY_INOUT": 2,
    }
    for key, value in constants.items():
        setattr(mt5, key, value)
    mt5.copy_rates_from_pos = lambda *args, **kwargs: None
    mt5.last_error = lambda: (0, "stub")
    mt5.symbol_info = lambda symbol: None
    mt5.symbol_info_tick = lambda symbol: None
    mt5.order_send = lambda request: None
    mt5.orders_get = lambda *args, **kwargs: []
    mt5.positions_get = lambda *args, **kwargs: []
    mt5.history_deals_get = lambda *args, **kwargs: []
    mt5.history_orders_get = lambda *args, **kwargs: []
    sys.modules["MetaTrader5"] = mt5


install_mt5_stub()

import compression_breakout_ablation_study as ablate
import iux_mt5_compression_breakout_bot as bot
import simple_breakout_atr_exit_audit as simple
import volatility_compression_breakout_audit as research
from delta_signal_audit import DeltaBar, IUX_XAUUSD_ROUNDTRIP_SPREAD


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
BOOT_N = 1000
SEED = 20260702
SPREAD = IUX_XAUUSD_ROUNDTRIP_SPREAD
RESULTS_PATH = Path("research/bot_logic_parity_results.txt")
REGISTRY_PATH = Path("research/hypothesis_registry.md")
LIVE_LOG_PATH = Path("research/iux_compression_breakout_live_log.csv")


@dataclass(frozen=True)
class Signal:
    index: int
    time: datetime
    range_high: float
    range_low: float
    atr: float


@dataclass(frozen=True)
class ReplayTrade:
    signal_index: int
    signal_time: datetime
    direction: int
    entry_index: int
    entry_time: datetime
    entry: float
    sl: float
    tp: float
    exit_index: int
    exit_time: datetime
    exit_price: float
    exit_reason: str
    gross_r: float
    net_r: float
    risk: float
    skipped_sides: int
    stopped_by_validity: bool
    session_flattened: bool


def q(vals: list[float], pct: float) -> float:
    if not vals:
        return math.nan
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
    return q(means, 0.025), q(means, 0.975)


def to_live_bar(bar: DeltaBar, idx: int) -> bot.LiveBar:
    return bot.LiveBar(
        index=idx,
        time=bar.start,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
    )


def build_bot_live_bars(delta_bars: list[DeltaBar]) -> list[bot.LiveBar]:
    live = [to_live_bar(bar, i) for i, bar in enumerate(delta_bars)]
    bot.add_atr14(live)
    return live


def bot_signal_at(delta_bars: list[DeltaBar], live_bars: list[bot.LiveBar], index: int) -> Signal | None:
    if index < bot.HISTORY_BARS - 1:
        return None
    if not bot.is_compression_end(live_bars, index):
        return None
    window = live_bars[index - bot.COMPRESSION_WINDOW + 1 : index + 1]
    atr = live_bars[index].atr14
    if atr is None or atr <= 0:
        return None
    return Signal(
        index=index,
        time=delta_bars[index].start,
        range_high=max(b.high for b in window),
        range_low=min(b.low for b in window),
        atr=atr,
    )


def research_signal_indexes(bars: list[DeltaBar]) -> set[int]:
    return {
        i
        for i in range(research.ATR_TRAIL + research.COMPRESSION_WINDOW, len(bars) - research.EXIT_HORIZON)
        if research.is_compression_end(bars, i)
    }


def bot_signal_indexes(bars: list[DeltaBar], live_bars: list[bot.LiveBar]) -> dict[int, Signal]:
    out: dict[int, Signal] = {}
    for i in range(len(bars)):
        signal = bot_signal_at(bars, live_bars, i)
        if signal is not None:
            out[i] = signal
    return out


def is_flatten_time(ts: datetime) -> bool:
    if not bot.SESSION_FLATTEN_ENABLED:
        return False
    if ts.weekday() > 4:
        return False
    return (ts.hour, ts.minute) >= (bot.SESSION_FLATTEN_HOUR, bot.SESSION_FLATTEN_MINUTE)


def pending_valid(direction: int, entry: float, setup_bar: DeltaBar, stops_level_usd: float) -> bool:
    bid = setup_bar.close
    ask = setup_bar.close + SPREAD
    if direction == 1:
        return entry > ask + stops_level_usd
    return entry < bid - stops_level_usd


def session_gap_next(bars: list[DeltaBar], index: int) -> bool:
    return index + 1 >= len(bars) or bars[index + 1].segment_id != bars[index].segment_id


def replay_bot_logic(
    bars: list[DeltaBar],
    signals: dict[int, Signal],
    stops_level_usd: float,
) -> tuple[list[ReplayTrade], dict[str, int]]:
    trades: list[ReplayTrade] = []
    stats = {
        "signals": len(signals),
        "signals_blocked_by_flatten": 0,
        "signals_no_valid_pending": 0,
        "side_skips": 0,
        "replaced_pending_sets": 0,
    }
    pending: Signal | None = None
    pending_buy_valid = False
    pending_sell_valid = False
    active: dict[str, object] | None = None

    for i, bar in enumerate(bars):
        if pending is not None and active is None and i > pending.index:
            buy_hit = pending_buy_valid and bar.high >= pending.range_high
            sell_hit = pending_sell_valid and bar.low <= pending.range_low
            direction = 0
            entry = math.nan
            if buy_hit and sell_hit:
                direction = -1  # conservative: sell side first in same bar.
                entry = pending.range_low
            elif buy_hit:
                direction = 1
                entry = pending.range_high
            elif sell_hit:
                direction = -1
                entry = pending.range_low
            if direction:
                active = {
                    "signal": pending,
                    "direction": direction,
                    "entry_index": i,
                    "entry_time": bar.start,
                    "entry": entry,
                    "sl": entry - direction * pending.atr,
                    "tp": entry + direction * bot.RR_TARGET * pending.atr,
                    "skipped_sides": int(not pending_buy_valid) + int(not pending_sell_valid),
                    "stopped_by_validity": (not pending_buy_valid) or (not pending_sell_valid),
                }
                pending = None

        if active is not None:
            direction = int(active["direction"])
            entry = float(active["entry"])
            sl = float(active["sl"])
            tp = float(active["tp"])
            signal = active["signal"]
            assert isinstance(signal, Signal)
            exit_reason = ""
            exit_price = math.nan
            session_flattened = False
            stop_hit = bar.low <= sl if direction == 1 else bar.high >= sl
            target_hit = bar.high >= tp if direction == 1 else bar.low <= tp
            if stop_hit:
                exit_reason = "stop"
                exit_price = min(sl, bar.low) if direction == 1 else max(sl, bar.high)
            elif target_hit:
                exit_reason = "target"
                exit_price = tp
            elif i - int(active["entry_index"]) >= bot.FORCE_CLOSE_BARS:
                exit_reason = "force_close"
                exit_price = bar.close
            elif is_flatten_time(bar.start) or session_gap_next(bars, i):
                exit_reason = "session_flatten"
                exit_price = bar.close
                session_flattened = True
            if exit_reason:
                gross = direction * (exit_price - entry) / signal.atr
                trades.append(
                    ReplayTrade(
                        signal_index=signal.index,
                        signal_time=signal.time,
                        direction=direction,
                        entry_index=int(active["entry_index"]),
                        entry_time=active["entry_time"],
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        exit_index=i,
                        exit_time=bar.start,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        gross_r=gross,
                        net_r=gross - SPREAD / signal.atr,
                        risk=signal.atr,
                        skipped_sides=int(active["skipped_sides"]),
                        stopped_by_validity=bool(active["stopped_by_validity"]),
                        session_flattened=session_flattened,
                    )
                )
                active = None
                pending = None
                continue

        signal = signals.get(i)
        if active is not None:
            continue
        if is_flatten_time(bar.start):
            if signal is not None:
                stats["signals_blocked_by_flatten"] += 1
            pending = None
            pending_buy_valid = False
            pending_sell_valid = False
            continue
        if signal is not None:
            if pending is not None:
                stats["replaced_pending_sets"] += 1
            buy_valid = pending_valid(1, signal.range_high, bar, stops_level_usd)
            sell_valid = pending_valid(-1, signal.range_low, bar, stops_level_usd)
            stats["side_skips"] += int(not buy_valid) + int(not sell_valid)
            if not buy_valid and not sell_valid:
                stats["signals_no_valid_pending"] += 1
                pending = None
                continue
            pending = signal
            pending_buy_valid = buy_valid
            pending_sell_valid = sell_valid
    return trades, stats


def research_trades(bars: list[DeltaBar]) -> list[ablate.Trade]:
    out = []
    for event in ablate.detect_compression(bars):
        trade = ablate.simulate("XAUUSD", bars, event, "research_A", 1.5, 10, SPREAD, "range_edge")
        if trade is not None:
            out.append(trade)
    return out


def summarize_replay(rows: list[ReplayTrade], period: str) -> dict[str, float]:
    if period == "train":
        subset = [r for r in rows if r.signal_time <= TRAIN_END]
    elif period == "test":
        subset = [r for r in rows if r.signal_time >= TEST_START]
    else:
        subset = rows
    vals = [r.net_r for r in subset]
    lo, hi = bootstrap_ci(vals, f"{SEED}-bot-{period}")
    years = ((max(r.signal_time for r in subset) - min(r.signal_time for r in subset)).days / 365.25) if len(subset) > 1 else math.nan
    return {
        "n": len(subset),
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "net": mean(vals) if vals else math.nan,
        "lo": lo,
        "hi": hi,
        "trades_per_year": len(subset) / years if years and math.isfinite(years) and years > 0 else math.nan,
    }


def summarize_research(rows: list[ablate.Trade], period: str) -> dict[str, float]:
    if period == "train":
        subset = [r for r in rows if r.entry_time <= TRAIN_END]
    elif period == "test":
        subset = [r for r in rows if r.entry_time >= TEST_START]
    else:
        subset = rows
    vals = [r.net_r for r in subset]
    lo, hi = bootstrap_ci(vals, f"{SEED}-research-{period}")
    years = ((max(r.entry_time for r in subset) - min(r.entry_time for r in subset)).days / 365.25) if len(subset) > 1 else math.nan
    return {
        "n": len(subset),
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "net": mean(vals) if vals else math.nan,
        "lo": lo,
        "hi": hi,
        "trades_per_year": len(subset) / years if years and math.isfinite(years) and years > 0 else math.nan,
    }


def signal_mismatch_causes(bot_only: set[int], research_only: set[int], bars: list[DeltaBar]) -> dict[str, int]:
    causes: dict[str, int] = {}
    for i in bot_only:
        cause = "bot_rolling_atr_no_segment_reset_or_window_indexing"
        if i - 1 in research_only or i + 1 in research_only:
            cause = "one_bar_boundary_shift"
        causes[cause] = causes.get(cause, 0) + 1
    for i in research_only:
        cause = "research_segment_reset_or_full_history_atr_diff"
        if i - 1 in bot_only or i + 1 in bot_only:
            cause = "one_bar_boundary_shift"
        elif i > 0 and bars[i].segment_id != bars[i - 1].segment_id:
            cause = "near_segment_boundary"
        causes[cause] = causes.get(cause, 0) + 1
    return causes


def golden_live_check(bars: list[DeltaBar], live_bars: list[bot.LiveBar]) -> list[str]:
    lines = []
    start = datetime(2026, 6, 29, tzinfo=timezone.utc)
    end = datetime(2026, 7, 2, 23, 59, tzinfo=timezone.utc)
    window = [b for b in bars if start <= b.start <= end]
    idxs = [i for i, b in enumerate(bars) if start <= b.start <= end]
    if not idxs:
        return ["golden_live_replay_status,NO_HISTORICAL_BARS_IN_WINDOW"]
    bot_sigs = [bot_signal_at(bars, live_bars, i) for i in idxs]
    bot_sigs = [s for s in bot_sigs if s is not None]
    lines.append("golden_live_replay_status,LIVE_CSV_MISSING" if not LIVE_LOG_PATH.exists() else "golden_live_replay_status,LIVE_CSV_FOUND")
    lines.append("harness_window_bars," + str(len(window)))
    lines.append("harness_signal_count," + str(len(bot_sigs)))
    for sig in bot_sigs[:25]:
        lines.append(f"harness_signal,{sig.time.isoformat()},{sig.range_high:.2f},{sig.range_low:.2f},{sig.atr:.6f}")
    if not LIVE_LOG_PATH.exists():
        lines.append("note,Actual live CSV not present at research/iux_compression_breakout_live_log.csv; cannot compare tickets/order_skip rows.")
        return lines
    with LIVE_LOG_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    live_rows = [r for r in rows if start.isoformat()[:10] <= (r.get("signal_time") or r.get("timestamp_utc") or "")[:10] <= end.isoformat()[:10]]
    lines.append("live_csv_rows_in_window," + str(len(live_rows)))
    return lines


def append_registry(pass_gate: bool, bot_train: dict[str, float], bot_test: dict[str, float]) -> None:
    registered = (
        "- 2026-07-02: V-2026-PARITY-01 registered. Engineering verification, not a new hypothesis: "
        "replay live bot compression decision logic over historical XAUUSD M15 and compare to validated research pipeline."
    )
    result = (
        "- 2026-07-02: V-2026-PARITY-01 result: "
        f"{'PASS' if pass_gate else 'FAIL'}; "
        f"bot_logic_train={bot_train['net']:.4f} [{bot_train['lo']:.4f},{bot_train['hi']:.4f}], "
        f"test={bot_test['net']:.4f} [{bot_test['lo']:.4f},{bot_test['hi']:.4f}]."
    )
    existing = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else "# Hypothesis Registry\n"
    lines = [line for line in existing.rstrip().splitlines() if "V-2026-PARITY-01" not in line]
    lines.extend([registered, result])
    REGISTRY_PATH.write_text("\n".join(lines) + "\n")


def fmt(v: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(v) else f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    parser.add_argument("--stops-level-usd", type=float, default=0.0)
    parser.add_argument("--realistic-stops-level-usd", type=float, default=0.50)
    args = parser.parse_args()

    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    live_bars = build_bot_live_bars(bars)
    research_signal_set = research_signal_indexes(bars)
    bot_signal_map = bot_signal_indexes(bars, live_bars)
    bot_signal_set = set(bot_signal_map)
    matched_signals = research_signal_set & bot_signal_set
    bot_only = bot_signal_set - research_signal_set
    research_only = research_signal_set - bot_signal_set

    research_rows = research_trades(bars)
    bot_rows_0, stats_0 = replay_bot_logic(bars, bot_signal_map, args.stops_level_usd)
    bot_rows_real, stats_real = replay_bot_logic(bars, bot_signal_map, args.realistic_stops_level_usd)

    bot_by_signal = {r.signal_index: r for r in bot_rows_0}
    research_events = ablate.detect_compression(bars)
    research_event_by_signal = {e.setup_end: e for e in research_events}
    research_trade_by_signal = {
        e.setup_end: t
        for e in research_events
        for t in research_rows
        if t.event_id == e.event_id
    }
    matched_trade_signals = set(bot_by_signal) & set(research_trade_by_signal)
    entry_diffs = []
    sl_diffs = []
    tp_diffs = []
    net_diffs = []
    for i in matched_trade_signals:
        event = research_event_by_signal[i]
        bot_trade = bot_by_signal[i]
        research_trade = research_trade_by_signal[i]
        research_entry = event.range_high if bot_trade.direction == 1 else event.range_low
        research_sl = research_entry - bot_trade.direction * research_trade.risk
        research_tp = research_entry + bot_trade.direction * 1.5 * research_trade.risk
        entry_diffs.append(abs(bot_trade.entry - research_entry))
        sl_diffs.append(abs(bot_trade.sl - research_sl))
        tp_diffs.append(abs(bot_trade.tp - research_tp))
        net_diffs.append(abs(bot_trade.net_r - research_trade.net_r))
    max_entry_diff = max(entry_diffs, default=math.nan)
    max_sl_diff = max(sl_diffs, default=math.nan)
    max_tp_diff = max(tp_diffs, default=math.nan)
    max_net_diff = max(net_diffs, default=math.nan)

    bot_train = summarize_replay(bot_rows_0, "train")
    bot_test = summarize_replay(bot_rows_0, "test")
    research_train = summarize_research(research_rows, "train")
    research_test = summarize_research(research_rows, "test")
    pass_gate = (
        bot_train["lo"] > 0
        and bot_test["lo"] > 0
        and 0.1147 <= bot_train["net"] <= 0.2733
        and 0.1731 <= bot_test["net"] <= 0.3518
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("V_2026_PARITY_01_BOT_LOGIC_PARITY_AUDIT")
        print("live_bot_file_modified,false")
        print("mt5_import_stubbed,true")
        print("xau_cache," + str(args.xau_cache))
        print("\nSIGNAL_PARITY")
        match_pct = len(matched_signals) / len(research_signal_set | bot_signal_set) if (research_signal_set | bot_signal_set) else math.nan
        print("research_signal_ends,bot_signal_ends,matched,bot_only,research_only,match_pct")
        print(f"{len(research_signal_set)},{len(bot_signal_set)},{len(matched_signals)},{len(bot_only)},{len(research_only)},{fmt(match_pct, 6)}")
        print("mismatch_cause,count")
        for cause, count in sorted(signal_mismatch_causes(bot_only, research_only, bars).items()):
            print(f"{cause},{count}")
        print("first_50_mismatches,type,index,time")
        for i in sorted(list(bot_only))[:25]:
            print(f"bot_only,{i},{bars[i].start.isoformat()}")
        for i in sorted(list(research_only))[:25]:
            print(f"research_only,{i},{bars[i].start.isoformat()}")

        print("\nTRADE_PARITY_AND_BOT_MECHANICS")
        print("stops_level_usd,bot_trades,bot_signals,signals_blocked_by_flatten,signals_no_valid_pending,side_skips,replaced_pending_sets")
        for stops, rows, stats in (
            (args.stops_level_usd, bot_rows_0, stats_0),
            (args.realistic_stops_level_usd, bot_rows_real, stats_real),
        ):
            print(
                f"{stops:.4f},{len(rows)},{stats['signals']},{stats['signals_blocked_by_flatten']},"
                f"{stats['signals_no_valid_pending']},{stats['side_skips']},{stats['replaced_pending_sets']}"
            )
        bot_net_by_signal = {r.signal_index: r.net_r for r in bot_rows_0}
        research_net_by_signal = {i: t.net_r for i, t in research_trade_by_signal.items()}
        common = set(bot_net_by_signal) & set(research_net_by_signal)
        delta_common = sum(bot_net_by_signal[i] - research_net_by_signal[i] for i in common)
        skipped_research = set(research_net_by_signal) - set(bot_net_by_signal)
        bot_only_trade_signals = set(bot_net_by_signal) - set(research_net_by_signal)
        print("matched_trade_signals,bot_only_trade_signals,research_only_trade_signals,common_net_r_delta,skipped_research_net_r,max_entry_diff,max_sl_diff,max_tp_diff,max_net_r_diff")
        print(
            f"{len(common)},{len(bot_only_trade_signals)},{len(skipped_research)},"
            f"{delta_common:.4f},{sum(research_net_by_signal[i] for i in skipped_research):.4f},"
            f"{fmt(max_entry_diff, 6)},{fmt(max_sl_diff, 6)},{fmt(max_tp_diff, 6)},{fmt(max_net_diff, 6)}"
        )

        print("\nMETRIC_PARITY")
        print("logic,period,n,win_rate,net_r,ci_low,ci_high,trades_per_year,reference_ci_gate")
        for logic, rows in (("research_A", research_rows),):
            for period in ("train", "test"):
                s = summarize_research(rows, period)
                print(f"{logic},{period},{s['n']},{fmt(s['win'], 4)},{fmt(s['net'])},{fmt(s['lo'])},{fmt(s['hi'])},{fmt(s['trades_per_year'], 2)},n/a")
        for period in ("train", "test"):
            s = summarize_replay(bot_rows_0, period)
            ref_gate = (0.1147 <= s["net"] <= 0.2733) if period == "train" else (0.1731 <= s["net"] <= 0.3518)
            print(f"bot_logic_stops0,{period},{s['n']},{fmt(s['win'], 4)},{fmt(s['net'])},{fmt(s['lo'])},{fmt(s['hi'])},{fmt(s['trades_per_year'], 2)},{ref_gate}")
        for period in ("train", "test"):
            s = summarize_replay(bot_rows_real, period)
            print(f"bot_logic_realistic_stops,{period},{s['n']},{fmt(s['win'], 4)},{fmt(s['net'])},{fmt(s['lo'])},{fmt(s['hi'])},{fmt(s['trades_per_year'], 2)},diagnostic")

        print("\nGOLDEN_LIVE_REPLAY_CHECK")
        for line in golden_live_check(bars, live_bars):
            print(line)

        print("\nDIVERGENCE_CLASSIFICATION")
        print("class,item,assessment")
        print("BUG_OR_DESIGN_DIVERGENCE,entry timing,bot arms OCO immediately after compression end; research pipeline waits for close-confirmed breakout then assigns range-edge fill")
        print("BUG_OR_DESIGN_DIVERGENCE,ATR/compression segmentation,bot uses rolling MT5 window with no segment reset while research resets ATR/compression at gaps")
        print("REALISTIC_CONSTRAINT,pending_stop_is_valid,bot may skip one/both pending sides when price is already too close to range edge")
        print("REALISTIC_CONSTRAINT,re-arming,bot replaces unfilled pendings on each new compression signal while flat")
        print("REALISTIC_CONSTRAINT,session_flatten,bot blocks/cancels during flatten window; research closes at segment gaps")
        if not LIVE_LOG_PATH.exists():
            print("RESEARCH_ARTIFACT,golden_live_csv_missing,cannot verify ticket-level live decisions from repo because live CSV is absent")

        print("\nVERDICT")
        if pass_gate:
            print("PASS: bot-logic replay clears zero in train/test and point estimates sit inside the pre-stated research reference CIs.")
        else:
            print("FAIL: bot-logic replay does not satisfy the parity gate. The live walk-forward is testing a deployable bot variant, not a bit-for-bit replay of the validated research pipeline.")
    report = buffer.getvalue()
    print(report, end="")
    RESULTS_PATH.write_text(report)
    append_registry(pass_gate, bot_train, bot_test)
    print(f"\nresults_file={RESULTS_PATH}")


if __name__ == "__main__":
    main()
