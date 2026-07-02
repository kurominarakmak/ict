"""
V-2026-TICK-01: tick-level fill simulation for compression breakout.

Research-only verification. This replays the fixed live-bot entry model on
Dukascopy XAUUSD bid/ask ticks: OCO stop orders at compression range edges,
buy stops fill on ask, sell stops fill on bid, exits fill on bid/ask ticks.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))


def install_mt5_stub() -> None:
    mt5 = types.ModuleType("MetaTrader5")
    mt5.TIMEFRAME_M15 = 15
    sys.modules["MetaTrader5"] = mt5


install_mt5_stub()

import iux_mt5_compression_breakout_bot as bot
import simple_breakout_atr_exit_audit as simple
from delta_signal_audit import DeltaBar


TRAIN_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
RESULTS_PATH = Path("research/tick_fill_simulation_results.txt")
REGISTRY_PATH = Path("research/hypothesis_registry.md")
BOOT_N = 1000
SEED = 20260702
M15 = timedelta(minutes=15)
M15_STOP_ORDER_REFERENCE = {
    "full": -0.4760,
    "train": -0.5246,
    "test": -0.4010,
}


CPP_SOURCE = r'''
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr long long M15_SECONDS = 15LL * 60LL;
constexpr int FORCE_CLOSE_BARS = 10;
constexpr int SESSION_FLATTEN_HOUR = 21;
constexpr int SESSION_FLATTEN_MINUTE = 45;
constexpr double RR_TARGET = 1.5;

struct Signal {
    long long setup_epoch = 0;
    long long arm_epoch = 0;
    double range_high = 0.0;
    double range_low = 0.0;
    double atr = 0.0;
};

struct Active {
    Signal signal;
    int direction = 0;
    double intended_entry = 0.0;
    double actual_entry = 0.0;
    long long entry_epoch = 0;
    long long entry_bar_start = 0;
    double entry_spread = 0.0;
    double sl = 0.0;
    double tp = 0.0;
};

long long days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097LL + static_cast<long long>(doe) - 719468LL;
}

int parse_int(const std::string& s, size_t pos, size_t len) {
    int out = 0;
    for (size_t i = pos; i < pos + len; ++i) out = out * 10 + (s[i] - '0');
    return out;
}

long long parse_epoch_seconds(const std::string& ts) {
    const int y = parse_int(ts, 0, 4);
    const int mo = parse_int(ts, 4, 2);
    const int d = parse_int(ts, 6, 2);
    const int h = parse_int(ts, 9, 2);
    const int mi = parse_int(ts, 12, 2);
    const int se = parse_int(ts, 15, 2);
    return days_from_civil(y, static_cast<unsigned>(mo), static_cast<unsigned>(d)) * 86400LL
        + h * 3600LL + mi * 60LL + se;
}

long long floor_m15(long long epoch) {
    return epoch - (epoch % M15_SECONDS);
}

bool flatten_window(long long epoch) {
    long long sod = epoch % 86400LL;
    if (sod < 0) sod += 86400LL;
    const int hour = static_cast<int>(sod / 3600LL);
    const int minute = static_cast<int>((sod % 3600LL) / 60LL);
    return hour > SESSION_FLATTEN_HOUR || (hour == SESSION_FLATTEN_HOUR && minute >= SESSION_FLATTEN_MINUTE);
}

void write_trade(std::ofstream& out, const Active& a, long long exit_epoch, double exit_price, double exit_spread, const std::string& reason) {
    const double gross_r = a.direction * (exit_price - a.actual_entry) / a.signal.atr;
    const double fill_slip_r = a.direction * (a.actual_entry - a.intended_entry) / a.signal.atr;
    out << a.signal.setup_epoch << ','
        << a.entry_epoch << ','
        << exit_epoch << ','
        << a.direction << ','
        << std::fixed << std::setprecision(8)
        << a.intended_entry << ','
        << a.actual_entry << ','
        << exit_price << ','
        << a.signal.atr << ','
        << gross_r << ','
        << gross_r << ','
        << reason << ','
        << a.entry_spread << ','
        << exit_spread << ','
        << fill_slip_r << '\n';
}

std::vector<Signal> load_signals(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open signals");
    std::vector<Signal> signals;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        Signal s;
        std::getline(ss, cell, ','); s.setup_epoch = std::stoll(cell);
        std::getline(ss, cell, ','); s.arm_epoch = std::stoll(cell);
        std::getline(ss, cell, ','); s.range_high = std::stod(cell);
        std::getline(ss, cell, ','); s.range_low = std::stod(cell);
        std::getline(ss, cell, ','); s.atr = std::stod(cell);
        signals.push_back(s);
    }
    return signals;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: tick_sim TICKS SIGNALS TRADES_OUT STATS_OUT\n";
        return 2;
    }
    const std::string tick_path = argv[1];
    const std::string signal_path = argv[2];
    const std::string trades_path = argv[3];
    const std::string stats_path = argv[4];
    const auto signals = load_signals(signal_path);

    std::ifstream in(tick_path);
    if (!in) {
        std::cerr << "failed to open ticks: " << tick_path << "\n";
        return 1;
    }
    std::ofstream trades(trades_path);
    std::ofstream stats(stats_path);
    trades << "signal_epoch,entry_epoch,exit_epoch,direction,intended_entry,actual_entry,exit_price,atr,gross_r,net_r,exit_reason,entry_spread,exit_spread,fill_slip_r\n";

    size_t sig_i = 0;
    bool have_pending = false;
    Signal pending;
    bool have_active = false;
    Active active;
    long long oco_armed = 0;
    long long signals_skipped_pending = 0;
    long long signals_skipped_active = 0;
    long long signals_skipped_flatten = 0;
    long long pending_cancelled_flatten = 0;
    long long rows = 0;

    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const size_t c1 = line.find(',');
        if (c1 == std::string::npos || c1 < 17) continue;
        const size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos) continue;
        const size_t c3 = line.find(',', c2 + 1);
        const char* bid_start = line.c_str() + c1 + 1;
        char* bid_end = nullptr;
        const double bid = std::strtod(bid_start, &bid_end);
        const char* ask_start = line.c_str() + c2 + 1;
        char* ask_end = nullptr;
        const double ask = std::strtod(ask_start, &ask_end);
        if (!std::isfinite(bid) || !std::isfinite(ask) || ask <= bid || bid <= 0.0) continue;
        const long long epoch = parse_epoch_seconds(line);
        const double spread = ask - bid;

        while (sig_i < signals.size() && signals[sig_i].arm_epoch <= epoch) {
            const Signal& s = signals[sig_i++];
            if (have_active) {
                ++signals_skipped_active;
            } else if (have_pending) {
                ++signals_skipped_pending;
            } else if (flatten_window(s.arm_epoch)) {
                ++signals_skipped_flatten;
            } else {
                pending = s;
                have_pending = true;
                ++oco_armed;
            }
        }

        if (have_pending && !have_active && flatten_window(epoch)) {
            have_pending = false;
            ++pending_cancelled_flatten;
        }

        if (have_active) {
            bool closed = false;
            if (active.direction == 1) {
                if (bid <= active.sl) {
                    write_trade(trades, active, epoch, bid, spread, "stop");
                    closed = true;
                } else if (bid >= active.tp) {
                    write_trade(trades, active, epoch, bid, spread, "target");
                    closed = true;
                } else if (epoch >= active.entry_bar_start + M15_SECONDS * (FORCE_CLOSE_BARS + 1)) {
                    write_trade(trades, active, epoch, bid, spread, "force_close");
                    closed = true;
                } else if (flatten_window(epoch)) {
                    write_trade(trades, active, epoch, bid, spread, "session_flatten");
                    closed = true;
                }
            } else {
                if (ask >= active.sl) {
                    write_trade(trades, active, epoch, ask, spread, "stop");
                    closed = true;
                } else if (ask <= active.tp) {
                    write_trade(trades, active, epoch, ask, spread, "target");
                    closed = true;
                } else if (epoch >= active.entry_bar_start + M15_SECONDS * (FORCE_CLOSE_BARS + 1)) {
                    write_trade(trades, active, epoch, ask, spread, "force_close");
                    closed = true;
                } else if (flatten_window(epoch)) {
                    write_trade(trades, active, epoch, ask, spread, "session_flatten");
                    closed = true;
                }
            }
            if (closed) {
                have_active = false;
                continue;
            }
        }

        if (have_pending && !have_active && !flatten_window(epoch)) {
            const bool buy_hit = ask >= pending.range_high;
            const bool sell_hit = bid <= pending.range_low;
            if (buy_hit || sell_hit) {
                int direction = 0;
                double intended = 0.0;
                double actual = 0.0;
                if (buy_hit && sell_hit) {
                    direction = -1;
                    intended = pending.range_low;
                    actual = bid;
                } else if (buy_hit) {
                    direction = 1;
                    intended = pending.range_high;
                    actual = ask;
                } else {
                    direction = -1;
                    intended = pending.range_low;
                    actual = bid;
                }
                active = Active{};
                active.signal = pending;
                active.direction = direction;
                active.intended_entry = intended;
                active.actual_entry = actual;
                active.entry_epoch = epoch;
                active.entry_bar_start = floor_m15(epoch);
                active.entry_spread = spread;
                active.sl = actual - direction * pending.atr;
                active.tp = actual + direction * RR_TARGET * pending.atr;
                have_active = true;
                have_pending = false;
            }
        }

        ++rows;
        if (rows % 50000000LL == 0) std::cerr << "processed_rows=" << rows << "\n";
        (void)c3;
    }

    stats << "signals," << signals.size() << "\n";
    stats << "oco_armed," << oco_armed << "\n";
    stats << "signals_skipped_pending," << signals_skipped_pending << "\n";
    stats << "signals_skipped_active," << signals_skipped_active << "\n";
    stats << "signals_skipped_flatten," << signals_skipped_flatten << "\n";
    stats << "pending_cancelled_flatten," << pending_cancelled_flatten << "\n";
    stats << "processed_rows," << rows << "\n";
    return 0;
}
'''


@dataclass(frozen=True)
class Signal:
    index: int
    setup_time: datetime
    arm_time: datetime
    range_high: float
    range_low: float
    atr: float


@dataclass
class Pending:
    signal: Signal


@dataclass
class Active:
    signal: Signal
    direction: int
    intended_entry: float
    actual_entry: float
    entry_time: datetime
    entry_bar_start: datetime
    entry_spread: float
    sl: float
    tp: float


@dataclass(frozen=True)
class TickTrade:
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    direction: int
    intended_entry: float
    actual_entry: float
    exit_price: float
    atr: float
    gross_r: float
    net_r: float
    exit_reason: str
    entry_spread: float
    exit_spread: float
    fill_slip_r: float


def parse_ts(raw: str) -> datetime:
    micro = 0
    if len(raw) > 17 and raw[17] == ".":
        micro = int((raw[18:] + "000000")[:6])
    return datetime(
        int(raw[:4]),
        int(raw[4:6]),
        int(raw[6:8]),
        int(raw[9:11]),
        int(raw[12:14]),
        int(raw[15:17]),
        micro,
        tzinfo=timezone.utc,
    )


def floor_m15(ts: datetime) -> datetime:
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


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
    means: list[float] = []
    for _ in range(BOOT_N):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    return q(means, 0.025), q(means, 0.975)


def to_live_bar(bar: DeltaBar, idx: int) -> bot.LiveBar:
    return bot.LiveBar(
        index=idx,
        segment_id=bar.segment_id,
        time=bar.start,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        atr14=bar.atr14,
    )


def build_signals(bars: list[DeltaBar]) -> list[Signal]:
    live_bars = [to_live_bar(bar, i) for i, bar in enumerate(bars)]
    bot.add_atr14(live_bars)
    signals: list[Signal] = []
    for i in range(bot.HISTORY_BARS - 1, len(live_bars)):
        if not bot.is_compression_end(live_bars, i):
            continue
        atr = live_bars[i].atr14
        if atr is None or atr <= 0:
            continue
        start = i - bot.COMPRESSION_WINDOW + 1
        window = live_bars[start : i + 1]
        signals.append(
            Signal(
                index=i,
                setup_time=live_bars[i].time,
                arm_time=live_bars[i].time + M15,
                range_high=max(b.high for b in window),
                range_low=min(b.low for b in window),
                atr=atr,
            )
        )
    return signals


def epoch(ts: datetime) -> int:
    return int(ts.timestamp())


def period_rows(rows: list[TickTrade], period: str) -> list[TickTrade]:
    if period == "full":
        return rows
    if period == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if period == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(period)


def summarize(rows: list[TickTrade], period: str) -> dict[str, float]:
    vals = [r.net_r for r in rows]
    lo, hi = bootstrap_ci(vals, f"{SEED}-{period}")
    return {
        "n": len(rows),
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "net": mean(vals) if vals else math.nan,
        "lo": lo,
        "hi": hi,
        "avg_entry_spread": mean([r.entry_spread for r in rows]) if rows else math.nan,
        "avg_exit_spread": mean([r.exit_spread for r in rows]) if rows else math.nan,
        "avg_roundtrip_spread": mean([r.entry_spread + r.exit_spread for r in rows]) if rows else math.nan,
        "false_breakout_rate": sum(r.exit_reason == "stop" for r in rows) / len(rows) if rows else math.nan,
    }


def is_flatten_window(ts: datetime) -> bool:
    return (ts.hour, ts.minute) >= (bot.SESSION_FLATTEN_HOUR, bot.SESSION_FLATTEN_MINUTE)


def close_active(active: Active, ts: datetime, exit_price: float, spread: float, reason: str) -> TickTrade:
    gross_r = active.direction * (exit_price - active.actual_entry) / active.signal.atr
    fill_slip_r = active.direction * (active.actual_entry - active.intended_entry) / active.signal.atr
    return TickTrade(
        signal_time=active.signal.setup_time,
        entry_time=active.entry_time,
        exit_time=ts,
        direction=active.direction,
        intended_entry=active.intended_entry,
        actual_entry=active.actual_entry,
        exit_price=exit_price,
        atr=active.signal.atr,
        gross_r=gross_r,
        net_r=gross_r,
        exit_reason=reason,
        entry_spread=active.entry_spread,
        exit_spread=spread,
        fill_slip_r=fill_slip_r,
    )


def compile_cpp(binary: Path) -> None:
    source = binary.with_suffix(".cpp")
    source.write_text(CPP_SOURCE)
    subprocess.run(["c++", "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)


def run_cpp_tick_sim(tick_path: Path, signals: list[Signal]) -> tuple[list[TickTrade], dict[str, int]]:
    tmp = Path(tempfile.mkdtemp(prefix="tick_fill_sim_"))
    signal_path = tmp / "signals.csv"
    trades_path = tmp / "trades.csv"
    stats_path = tmp / "stats.csv"
    binary = tmp / "tick_sim"
    with signal_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setup_epoch", "arm_epoch", "range_high", "range_low", "atr"])
        for signal in signals:
            writer.writerow([epoch(signal.setup_time), epoch(signal.arm_time), signal.range_high, signal.range_low, signal.atr])
    compile_cpp(binary)
    subprocess.run([str(binary), str(tick_path), str(signal_path), str(trades_path), str(stats_path)], check=True)

    stats: dict[str, int] = {}
    with stats_path.open() as handle:
        for raw in handle:
            key, value = raw.strip().split(",", 1)
            stats[key] = int(value)

    trades: list[TickTrade] = []
    with trades_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trades.append(
                TickTrade(
                    signal_time=datetime.fromtimestamp(int(row["signal_epoch"]), tz=timezone.utc),
                    entry_time=datetime.fromtimestamp(int(row["entry_epoch"]), tz=timezone.utc),
                    exit_time=datetime.fromtimestamp(int(row["exit_epoch"]), tz=timezone.utc),
                    direction=int(row["direction"]),
                    intended_entry=float(row["intended_entry"]),
                    actual_entry=float(row["actual_entry"]),
                    exit_price=float(row["exit_price"]),
                    atr=float(row["atr"]),
                    gross_r=float(row["gross_r"]),
                    net_r=float(row["net_r"]),
                    exit_reason=row["exit_reason"],
                    entry_spread=float(row["entry_spread"]),
                    exit_spread=float(row["exit_spread"]),
                    fill_slip_r=float(row["fill_slip_r"]),
                )
            )
    return trades, stats


def append_registry(verdict: str, train: dict[str, float], test: dict[str, float]) -> None:
    existing = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else "# Hypothesis Registry\n"
    lines = [line for line in existing.rstrip().splitlines() if "V-2026-TICK-01" not in line]
    lines.append("- 2026-07-02: V-2026-TICK-01 registered. Tick-level verification of compression fill realism: fixed live-bot OCO stop model on Dukascopy XAUUSD bid/ask ticks, one realistic fill model, no live bot changes.")
    lines.append(
        "- 2026-07-02: V-2026-TICK-01 result: "
        f"{verdict}; train={train['net']:.4f} [{train['lo']:.4f},{train['hi']:.4f}], "
        f"test={test['net']:.4f} [{test['lo']:.4f},{test['hi']:.4f}]."
    )
    REGISTRY_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-ticks", type=Path, default=Path("data/2026.6.15XAUUSD-TICK-No Session.csv"))
    parser.add_argument("--xau-cache", type=Path, default=Path("data/xauusd_m15_delta_bars.csv"))
    args = parser.parse_args()

    bars = simple.load_symbol_bars("XAUUSD", args.xau_ticks, args.xau_cache)
    signals = build_signals(bars)
    trades, stats = run_cpp_tick_sim(args.xau_ticks, signals)
    full = summarize(period_rows(trades, "full"), "full")
    train = summarize(period_rows(trades, "train"), "train")
    test = summarize(period_rows(trades, "test"), "test")
    fill_slips = [r.fill_slip_r for r in trades]
    verdict = "FAIL_CONFIRMS_NOT_TRADABLE" if train["hi"] < 0 and test["hi"] < 0 else "REVIEW_TICK_RESULT_DIFFERS_FROM_M15"

    lines: list[str] = []
    lines.append("V_2026_TICK_01_TICK_FILL_SIMULATION_AUDIT")
    lines.append(f"tick_file,{args.xau_ticks}")
    lines.append(f"bar_cache,{args.xau_cache}")
    lines.append("fill_model,buy_stop_ask_sell_stop_bid_tick_exits_real_spread")
    lines.append("live_bot_modified,false")
    lines.append("")
    lines.append("ORDER_MANAGEMENT_COUNTS")
    lines.append("metric,value")
    for key, value in stats.items():
        lines.append(f"{key},{value}")
    lines.append("")
    lines.append("NET_R_SUMMARY")
    lines.append("period,n,win_rate,net_r,ci_low,ci_high,avg_entry_spread,avg_exit_spread,avg_roundtrip_spread,false_breakout_rate,m15_stop_order_ref")
    for name, summary in (("full", full), ("train", train), ("test", test)):
        lines.append(
            f"{name},{summary['n']},{summary['win']:.4f},{summary['net']:.4f},"
            f"{summary['lo']:.4f},{summary['hi']:.4f},{summary['avg_entry_spread']:.4f},"
            f"{summary['avg_exit_spread']:.4f},{summary['avg_roundtrip_spread']:.4f},"
            f"{summary['false_breakout_rate']:.4f},{M15_STOP_ORDER_REFERENCE[name]:.4f}"
        )
    lines.append("")
    lines.append("ACTUAL_FILL_VS_EDGE_DISTRIBUTION_R")
    lines.append("n,median,p75,p90,mean")
    lines.append(
        f"{len(fill_slips)},{q(fill_slips,0.50):.4f},{q(fill_slips,0.75):.4f},"
        f"{q(fill_slips,0.90):.4f},{mean(fill_slips):.4f}"
    )
    lines.append("")
    lines.append("EXIT_REASON_COUNTS")
    lines.append("reason,count")
    for reason in sorted({r.exit_reason for r in trades}):
        lines.append(f"{reason},{sum(r.exit_reason == reason for r in trades)}")
    lines.append("")
    lines.append("VERDICT")
    if verdict == "FAIL_CONFIRMS_NOT_TRADABLE":
        lines.append("Tick-level bid/ask simulation confirms the M15 fill-realism finding: the compression edge is not tradable as designed.")
    else:
        lines.append("Tick-level result differs materially from the M15 finding; inspect fill logic before drawing a positive conclusion.")
    report = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(report)
    print(report, end="")
    append_registry(verdict, train, test)
    print(f"results_file={RESULTS_PATH}")


if __name__ == "__main__":
    main()
