"""
Tick-level M1 confirmation audit for the compression breakout setup.

Research-only. M15 compression defines the range and ATR. Breakout confirmation
uses a CLOSED M1 mid-price bar beyond the range, then enters on the first later
tick at executable bid/ask. This avoids using an M15 close while pretending to
fill back at the range edge.
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
RESULTS_PATH = Path("research/tick_m1_confirmation_results.txt")
REGISTRY_PATH = Path("research/hypothesis_registry.md")
BOOT_N = 1000
SEED = 20260702
M15 = timedelta(minutes=15)


CPP_SOURCE = r'''
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr long long M1_SECONDS = 60LL;
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
    double actual_entry = 0.0;
    long long entry_epoch = 0;
    long long entry_bar_start = 0;
    double entry_spread = 0.0;
    double sl = 0.0;
    double tp = 0.0;
    long long confirm_epoch = 0;
    double confirm_close = 0.0;
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

long long floor_tf(long long epoch, long long seconds) {
    return epoch - (epoch % seconds);
}

bool flatten_window(long long epoch) {
    long long sod = epoch % 86400LL;
    if (sod < 0) sod += 86400LL;
    const int hour = static_cast<int>(sod / 3600LL);
    const int minute = static_cast<int>((sod % 3600LL) / 60LL);
    return hour > SESSION_FLATTEN_HOUR || (hour == SESSION_FLATTEN_HOUR && minute >= SESSION_FLATTEN_MINUTE);
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

void write_trade(std::ofstream& out, const Active& a, long long exit_epoch, double exit_price, double exit_spread, const std::string& reason) {
    const double gross_r = a.direction * (exit_price - a.actual_entry) / a.signal.atr;
    const double edge = a.direction == 1 ? a.signal.range_high : a.signal.range_low;
    const double confirm_beyond_r = a.direction * (a.confirm_close - edge) / a.signal.atr;
    const double fill_vs_edge_r = a.direction * (a.actual_entry - edge) / a.signal.atr;
    out << a.signal.setup_epoch << ','
        << a.confirm_epoch << ','
        << a.entry_epoch << ','
        << exit_epoch << ','
        << a.direction << ','
        << std::fixed << std::setprecision(8)
        << edge << ','
        << a.confirm_close << ','
        << a.actual_entry << ','
        << exit_price << ','
        << a.signal.atr << ','
        << gross_r << ','
        << gross_r << ','
        << reason << ','
        << a.entry_spread << ','
        << exit_spread << ','
        << confirm_beyond_r << ','
        << fill_vs_edge_r << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: tick_m1_sim TICKS SIGNALS TRADES_OUT STATS_OUT\n";
        return 2;
    }
    const auto signals = load_signals(argv[2]);
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "failed to open ticks\n";
        return 1;
    }
    std::ofstream trades(argv[3]);
    std::ofstream stats(argv[4]);
    trades << "signal_epoch,confirm_epoch,entry_epoch,exit_epoch,direction,edge,confirm_close,actual_entry,exit_price,atr,gross_r,net_r,exit_reason,entry_spread,exit_spread,confirm_beyond_r,fill_vs_edge_r\n";

    size_t sig_i = 0;
    bool have_pending = false;
    Signal pending;
    bool have_active = false;
    Active active;
    bool have_confirm = false;
    int confirm_direction = 0;
    long long confirm_epoch = 0;
    double confirm_close = 0.0;

    bool have_m1 = false;
    long long m1_start = 0;
    double m1_close = 0.0;

    long long oco_armed = 0, signals_skipped_pending = 0, signals_skipped_active = 0;
    long long signals_skipped_flatten = 0, pending_cancelled_flatten = 0, m1_confirmations = 0;
    long long rows = 0;

    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const size_t c1 = line.find(',');
        if (c1 == std::string::npos || c1 < 17) continue;
        const size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos) continue;
        const char* bid_start = line.c_str() + c1 + 1;
        char* bid_end = nullptr;
        const double bid = std::strtod(bid_start, &bid_end);
        const char* ask_start = line.c_str() + c2 + 1;
        char* ask_end = nullptr;
        const double ask = std::strtod(ask_start, &ask_end);
        if (!std::isfinite(bid) || !std::isfinite(ask) || ask <= bid || bid <= 0.0) continue;
        const long long epoch = parse_epoch_seconds(line);
        const double mid = (bid + ask) / 2.0;
        const double spread = ask - bid;
        const long long bucket1 = floor_tf(epoch, M1_SECONDS);

        if (!have_m1) {
            have_m1 = true;
            m1_start = bucket1;
            m1_close = mid;
        } else if (bucket1 != m1_start) {
            const long long closed_epoch = m1_start + M1_SECONDS;
            if (have_pending && !have_active && !have_confirm && closed_epoch >= pending.arm_epoch) {
                if (m1_close > pending.range_high) {
                    have_confirm = true;
                    confirm_direction = 1;
                    confirm_epoch = closed_epoch;
                    confirm_close = m1_close;
                    ++m1_confirmations;
                } else if (m1_close < pending.range_low) {
                    have_confirm = true;
                    confirm_direction = -1;
                    confirm_epoch = closed_epoch;
                    confirm_close = m1_close;
                    ++m1_confirmations;
                }
            }
            m1_start = bucket1;
            m1_close = mid;
        } else {
            m1_close = mid;
        }

        while (sig_i < signals.size() && signals[sig_i].arm_epoch <= epoch) {
            const Signal& s = signals[sig_i++];
            if (have_active) ++signals_skipped_active;
            else if (have_pending || have_confirm) ++signals_skipped_pending;
            else if (flatten_window(s.arm_epoch)) ++signals_skipped_flatten;
            else {
                pending = s;
                have_pending = true;
                ++oco_armed;
            }
        }

        if ((have_pending || have_confirm) && !have_active && flatten_window(epoch)) {
            have_pending = false;
            have_confirm = false;
            ++pending_cancelled_flatten;
        }

        if (have_active) {
            bool closed = false;
            if (active.direction == 1) {
                if (bid <= active.sl) { write_trade(trades, active, epoch, bid, spread, "stop"); closed = true; }
                else if (bid >= active.tp) { write_trade(trades, active, epoch, bid, spread, "target"); closed = true; }
                else if (epoch >= active.entry_bar_start + M15_SECONDS * (FORCE_CLOSE_BARS + 1)) { write_trade(trades, active, epoch, bid, spread, "force_close"); closed = true; }
                else if (flatten_window(epoch)) { write_trade(trades, active, epoch, bid, spread, "session_flatten"); closed = true; }
            } else {
                if (ask >= active.sl) { write_trade(trades, active, epoch, ask, spread, "stop"); closed = true; }
                else if (ask <= active.tp) { write_trade(trades, active, epoch, ask, spread, "target"); closed = true; }
                else if (epoch >= active.entry_bar_start + M15_SECONDS * (FORCE_CLOSE_BARS + 1)) { write_trade(trades, active, epoch, ask, spread, "force_close"); closed = true; }
                else if (flatten_window(epoch)) { write_trade(trades, active, epoch, ask, spread, "session_flatten"); closed = true; }
            }
            if (closed) {
                have_active = false;
                continue;
            }
        }

        if (have_confirm && have_pending && !have_active && epoch >= confirm_epoch && !flatten_window(epoch)) {
            const int direction = confirm_direction;
            const double actual = direction == 1 ? ask : bid;
            active = Active{};
            active.signal = pending;
            active.direction = direction;
            active.actual_entry = actual;
            active.entry_epoch = epoch;
            active.entry_bar_start = floor_tf(epoch, M15_SECONDS);
            active.entry_spread = spread;
            active.sl = actual - direction * pending.atr;
            active.tp = actual + direction * RR_TARGET * pending.atr;
            active.confirm_epoch = confirm_epoch;
            active.confirm_close = confirm_close;
            have_active = true;
            have_pending = false;
            have_confirm = false;
        }

        ++rows;
        if (rows % 50000000LL == 0) std::cerr << "processed_rows=" << rows << "\n";
    }

    stats << "signals," << signals.size() << "\n";
    stats << "oco_armed," << oco_armed << "\n";
    stats << "m1_confirmations," << m1_confirmations << "\n";
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
    setup_time: datetime
    arm_time: datetime
    range_high: float
    range_low: float
    atr: float


@dataclass(frozen=True)
class Trade:
    signal_time: datetime
    confirm_time: datetime
    entry_time: datetime
    net_r: float
    exit_reason: str
    entry_spread: float
    exit_spread: float
    confirm_beyond_r: float
    fill_vs_edge_r: float


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
    rng = random.Random(seed)
    n = len(vals)
    means = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT_N)]
    return q(means, 0.025), q(means, 0.975)


def to_live_bar(bar: DeltaBar, idx: int) -> bot.LiveBar:
    return bot.LiveBar(idx, bar.segment_id, bar.start, bar.open, bar.high, bar.low, bar.close, bar.atr14)


def build_signals(bars: list[DeltaBar]) -> list[Signal]:
    live = [to_live_bar(bar, i) for i, bar in enumerate(bars)]
    bot.add_atr14(live)
    signals: list[Signal] = []
    for i in range(bot.HISTORY_BARS - 1, len(live)):
        if not bot.is_compression_end(live, i):
            continue
        atr = live[i].atr14
        if atr is None or atr <= 0:
            continue
        start = i - bot.COMPRESSION_WINDOW + 1
        window = live[start : i + 1]
        signals.append(
            Signal(
                setup_time=live[i].time,
                arm_time=live[i].time + M15,
                range_high=max(b.high for b in window),
                range_low=min(b.low for b in window),
                atr=atr,
            )
        )
    return signals


def compile_cpp(binary: Path) -> None:
    source = binary.with_suffix(".cpp")
    source.write_text(CPP_SOURCE)
    subprocess.run(["c++", "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)


def run_cpp(tick_path: Path, signals: list[Signal]) -> tuple[list[Trade], dict[str, int]]:
    tmp = Path(tempfile.mkdtemp(prefix="tick_m1_confirm_"))
    signal_path = tmp / "signals.csv"
    trades_path = tmp / "trades.csv"
    stats_path = tmp / "stats.csv"
    binary = tmp / "tick_m1_sim"
    with signal_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setup_epoch", "arm_epoch", "range_high", "range_low", "atr"])
        for signal in signals:
            writer.writerow([int(signal.setup_time.timestamp()), int(signal.arm_time.timestamp()), signal.range_high, signal.range_low, signal.atr])
    compile_cpp(binary)
    subprocess.run([str(binary), str(tick_path), str(signal_path), str(trades_path), str(stats_path)], check=True)
    stats: dict[str, int] = {}
    with stats_path.open() as handle:
        for raw in handle:
            key, value = raw.strip().split(",", 1)
            stats[key] = int(value)
    trades: list[Trade] = []
    with trades_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trades.append(
                Trade(
                    signal_time=datetime.fromtimestamp(int(row["signal_epoch"]), tz=timezone.utc),
                    confirm_time=datetime.fromtimestamp(int(row["confirm_epoch"]), tz=timezone.utc),
                    entry_time=datetime.fromtimestamp(int(row["entry_epoch"]), tz=timezone.utc),
                    net_r=float(row["net_r"]),
                    exit_reason=row["exit_reason"],
                    entry_spread=float(row["entry_spread"]),
                    exit_spread=float(row["exit_spread"]),
                    confirm_beyond_r=float(row["confirm_beyond_r"]),
                    fill_vs_edge_r=float(row["fill_vs_edge_r"]),
                )
            )
    return trades, stats


def period(rows: list[Trade], name: str) -> list[Trade]:
    if name == "full":
        return rows
    if name == "train":
        return [r for r in rows if r.entry_time <= TRAIN_END]
    if name == "test":
        return [r for r in rows if r.entry_time >= TEST_START]
    raise ValueError(name)


def summarize(rows: list[Trade], seed: str) -> dict[str, float]:
    vals = [r.net_r for r in rows]
    lo, hi = bootstrap_ci(vals, seed)
    return {
        "n": len(rows),
        "win": sum(v > 0 for v in vals) / len(vals) if vals else math.nan,
        "net": mean(vals) if vals else math.nan,
        "lo": lo,
        "hi": hi,
        "avg_entry_spread": mean([r.entry_spread for r in rows]) if rows else math.nan,
        "avg_exit_spread": mean([r.exit_spread for r in rows]) if rows else math.nan,
        "stop_rate": sum(r.exit_reason == "stop" for r in rows) / len(rows) if rows else math.nan,
    }


def safe_mean(vals: list[float]) -> float:
    return mean(vals) if vals else math.nan


def append_registry(verdict: str, train: dict[str, float], test: dict[str, float]) -> None:
    existing = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else "# Hypothesis Registry\n"
    lines = [line for line in existing.rstrip().splitlines() if "V-2026-M1FILL-01" not in line]
    lines.append("- 2026-07-02: V-2026-M1FILL-01 registered. Tick-level M1 close-confirmation audit: M15 compression setup, closed M1 breakout confirmation, first later tick market fill, no lookahead, no live bot changes.")
    lines.append(
        "- 2026-07-02: V-2026-M1FILL-01 result: "
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
    trades, stats = run_cpp(args.xau_ticks, signals)
    summaries = {name: summarize(period(trades, name), f"{SEED}-{name}") for name in ("full", "train", "test")}
    verdict = "FAIL_M1_CONFIRMATION_NOT_TRADABLE" if summaries["train"]["hi"] < 0 and summaries["test"]["hi"] < 0 else "REVIEW_M1_CONFIRMATION_RESULT"
    confirm_beyond = [r.confirm_beyond_r for r in trades]
    fill_vs_edge = [r.fill_vs_edge_r for r in trades]

    lines: list[str] = []
    lines.append("V_2026_M1FILL_01_TICK_M1_CONFIRMATION_AUDIT")
    lines.append(f"tick_file,{args.xau_ticks}")
    lines.append(f"bar_cache,{args.xau_cache}")
    lines.append("model,M15 compression setup + closed M1 breakout confirmation + first later tick market fill")
    lines.append("lookahead,false")
    lines.append("live_bot_modified,false")
    lines.append("")
    lines.append("ORDER_MANAGEMENT_COUNTS")
    lines.append("metric,value")
    for key, value in stats.items():
        lines.append(f"{key},{value}")
    lines.append("")
    lines.append("NET_R_SUMMARY")
    lines.append("period,n,win_rate,net_r,ci_low,ci_high,avg_entry_spread,avg_exit_spread,stop_rate")
    for name in ("full", "train", "test"):
        s = summaries[name]
        lines.append(f"{name},{s['n']},{s['win']:.4f},{s['net']:.4f},{s['lo']:.4f},{s['hi']:.4f},{s['avg_entry_spread']:.4f},{s['avg_exit_spread']:.4f},{s['stop_rate']:.4f}")
    lines.append("")
    lines.append("CONFIRM_CLOSE_BEYOND_EDGE_R")
    lines.append("n,median,p75,p90,mean")
    lines.append(f"{len(confirm_beyond)},{q(confirm_beyond,0.5):.4f},{q(confirm_beyond,0.75):.4f},{q(confirm_beyond,0.9):.4f},{safe_mean(confirm_beyond):.4f}")
    lines.append("")
    lines.append("ACTUAL_FILL_VS_EDGE_R")
    lines.append("n,median,p75,p90,mean")
    lines.append(f"{len(fill_vs_edge)},{q(fill_vs_edge,0.5):.4f},{q(fill_vs_edge,0.75):.4f},{q(fill_vs_edge,0.9):.4f},{safe_mean(fill_vs_edge):.4f}")
    lines.append("")
    lines.append("EXIT_REASON_COUNTS")
    lines.append("reason,count")
    for reason in sorted({r.exit_reason for r in trades}):
        lines.append(f"{reason},{sum(r.exit_reason == reason for r in trades)}")
    lines.append("")
    lines.append("VERDICT")
    if verdict.startswith("FAIL"):
        lines.append("M1 close confirmation still has negative train/test CIs. Faster confirmation reduces lag but does not rescue the compression edge.")
    else:
        lines.append("M1 confirmation differs from prior negative audits; inspect execution logic before interpreting as tradable.")
    report = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(report)
    print(report, end="")
    append_registry(verdict, summaries["train"], summaries["test"])
    print(f"results_file={RESULTS_PATH}")


if __name__ == "__main__":
    main()
