"""
Standalone live shadow tracker for validated trailing-exit config C.

This script does not import or modify the live bot. It reads completed live
compression trades and computes what trailing config C would have done on the
same entry, ATR, direction, and live M15 bars.

Config C source note:
research/compression_exit_trailing_audit.py was requested as the canonical
source, but it is not present in this checkout. The simulator below implements
the specified H-2026-EXIT-01 config C semantics:
- initial stop 1.0 ATR from actual entry
- no fixed TP
- arm when price reaches +1.0R
- after arming, trail 1.0 ATR behind the most favorable CLOSED-bar extreme
- 10-bar force close
- conservative intrabar convention: active stop is checked before new
  close-bar trail updates.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean


TIMEFRAME_MINUTES = 15
HORIZON_BARS = 10
BACKTEST_ROUNDTRIP_SPREAD = 0.20


@dataclass(frozen=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class ClosedTrade:
    ticket: str
    entry_time: datetime
    direction: int
    actual_entry: float
    atr: float
    realized_exit_reason: str
    realized_exit_price: float | None
    realized_net_r: float
    realized_net_r_realized: float | None


@dataclass(frozen=True)
class ShadowResult:
    exit_price: float
    exit_time: datetime
    exit_reason: str
    gross_r: float
    net_r: float
    net_r_realized: float
    bars_used: int


def parse_dt(raw: str) -> datetime:
    value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def fnum(raw: object) -> float | None:
    try:
        if raw in ("", None):
            return None
        val = float(raw)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def direction_from_row(row: dict[str, str]) -> int | None:
    raw = str(row.get("breakout_direction", "")).lower()
    if raw == "long":
        return 1
    if raw == "short":
        return -1
    return None


def load_completed_trades(path: Path) -> tuple[list[ClosedTrade], list[str]]:
    if not path.exists():
        return [], [f"log not found: {path}"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = {str(r.get("ticket", "")).strip(): r for r in rows if str(r.get("event", "")).lower() == "entry" and r.get("ticket")}
    completed: list[ClosedTrade] = []
    skips: list[str] = []
    for row in rows:
        if str(row.get("event", "")).lower() != "exit":
            continue
        ticket = str(row.get("ticket", "")).strip()
        realized_net = fnum(row.get("net_r_vs_020_spread"))
        reason = str(row.get("exit_reason", "")).lower()
        if realized_net is None or reason == "unknown":
            skips.append(f"{ticket or 'missing_ticket'}: exit row missing valid net_r/reason")
            continue
        merged = dict(entries.get(ticket, {}))
        merged.update({k: v for k, v in row.items() if v not in ("", None)})
        direction = direction_from_row(merged)
        actual_entry = fnum(merged.get("actual_fill_price"))
        atr = fnum(merged.get("atr_at_entry"))
        signal_time = merged.get("signal_time") or merged.get("entry_time") or merged.get("timestamp_utc")
        if direction is None or actual_entry is None or atr is None or atr <= 0 or not signal_time:
            skips.append(f"{ticket}: missing direction/actual_entry/ATR/entry_time")
            continue
        completed.append(
            ClosedTrade(
                ticket=ticket,
                entry_time=parse_dt(str(signal_time)),
                direction=direction,
                actual_entry=actual_entry,
                atr=atr,
                realized_exit_reason=reason,
                realized_exit_price=fnum(merged.get("exit_price")),
                realized_net_r=realized_net,
                realized_net_r_realized=fnum(merged.get("net_r_realized")),
            )
        )
    return completed, skips


def fetch_mt5_bars(symbol: str, start: datetime, horizon_bars: int) -> tuple[list[Bar], str | None]:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError:
        return [], "MetaTrader5 package is not installed"
    if mt5.terminal_info() is None and not mt5.initialize():
        return [], f"MT5 initialize failed: {mt5.last_error()}"
    end = start + timedelta(minutes=TIMEFRAME_MINUTES * (horizon_bars + 2))
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start, end)
    if rates is None:
        return [], f"copy_rates_range failed: {mt5.last_error()}"
    bars: list[Bar] = []
    for row in rates:
        ts = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc)
        if ts <= start:
            continue
        bars.append(Bar(ts, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])))
    return bars[:horizon_bars], None


def simulate_config_c(trade: ClosedTrade, bars: list[Bar]) -> ShadowResult | None:
    if not bars:
        return None
    d = trade.direction
    entry = trade.actual_entry
    atr = trade.atr
    stop = entry - d * atr
    arm_level = entry + d * atr
    armed = False
    best = entry
    exit_price = bars[-1].close
    exit_time = bars[-1].time
    exit_reason = "force_close"

    for i, bar in enumerate(bars, start=1):
        stop_hit = bar.low <= stop if d == 1 else bar.high >= stop
        if stop_hit:
            exit_price = min(stop, bar.low) if d == 1 else max(stop, bar.high)
            exit_time = bar.time
            exit_reason = "trail_hit" if armed else "initial_stop"
            break

        arm_hit = bar.high >= arm_level if d == 1 else bar.low <= arm_level
        if arm_hit:
            armed = True

        if armed:
            if d == 1:
                best = max(best, bar.high)
                stop = max(stop, best - atr)
            else:
                best = min(best, bar.low)
                stop = min(stop, best + atr)

        if i == len(bars):
            exit_price = bar.close
            exit_time = bar.time
            exit_reason = "force_close"

    gross = d * (exit_price - entry) / atr
    return ShadowResult(
        exit_price=exit_price,
        exit_time=exit_time,
        exit_reason=exit_reason,
        gross_r=gross,
        net_r=gross - BACKTEST_ROUNDTRIP_SPREAD / atr,
        net_r_realized=gross,
        bars_used=len(bars),
    )


def write_shadow_log(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticket",
        "entry_time",
        "direction",
        "atr",
        "realized_exit_reason",
        "realized_exit_price",
        "realized_net_r",
        "realized_net_r_realized",
        "shadow_exit_reason",
        "shadow_exit_price",
        "shadow_exit_time",
        "shadow_gross_r",
        "shadow_net_r",
        "shadow_net_r_realized",
        "shadow_minus_realized",
        "bars_used",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def update_registry(path: Path, today: str) -> None:
    line = (
        f"- {today}: H-2026-EXIT-01 live shadow validation phase started; "
        "decision on exit switch deferred until live gate (50-100 trades) with "
        "realized-A vs shadow-C comparison.\n"
    )
    text = path.read_text() if path.exists() else "# Hypothesis Registry\n\n"
    if line.strip() in text:
        return
    with path.open("a") as handle:
        if not path.exists() or path.stat().st_size == 0:
            handle.write("# Hypothesis Registry\n\n")
        handle.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("research/iux_compression_breakout_live_log.csv"))
    parser.add_argument("--out", type=Path, default=Path("research/live_shadow_trail_log.csv"))
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--registry", type=Path, default=Path("research/hypothesis_registry.md"))
    args = parser.parse_args()

    update_registry(args.registry, "2026-07-02")
    trades, skips = load_completed_trades(args.log)
    output_rows: list[dict[str, object]] = []
    for trade in trades:
        bars, err = fetch_mt5_bars(args.symbol, trade.entry_time, HORIZON_BARS)
        if err:
            skips.append(f"{trade.ticket}: {err}")
            continue
        shadow = simulate_config_c(trade, bars)
        if shadow is None:
            skips.append(f"{trade.ticket}: no M15 bars available for shadow window")
            continue
        output_rows.append(
            {
                "ticket": trade.ticket,
                "entry_time": trade.entry_time.isoformat(),
                "direction": "long" if trade.direction == 1 else "short",
                "atr": trade.atr,
                "realized_exit_reason": trade.realized_exit_reason,
                "realized_exit_price": trade.realized_exit_price if trade.realized_exit_price is not None else "",
                "realized_net_r": trade.realized_net_r,
                "realized_net_r_realized": trade.realized_net_r_realized if trade.realized_net_r_realized is not None else "",
                "shadow_exit_reason": shadow.exit_reason,
                "shadow_exit_price": shadow.exit_price,
                "shadow_exit_time": shadow.exit_time.isoformat(),
                "shadow_gross_r": shadow.gross_r,
                "shadow_net_r": shadow.net_r,
                "shadow_net_r_realized": shadow.net_r_realized,
                "shadow_minus_realized": shadow.net_r - trade.realized_net_r,
                "bars_used": shadow.bars_used,
                "notes": "",
            }
        )
    write_shadow_log(args.out, output_rows)

    print("LIVE_SHADOW_TRAIL_TRACKER")
    print(f"source_log={args.log}")
    print(f"shadow_log={args.out}")
    print("ticket,entry_time,direction,realized_exit,realized_net_r,shadow_exit,shadow_net_r,shadow_minus_realized,shadow_exit_time")
    for row in output_rows:
        print(
            f"{row['ticket']},{row['entry_time']},{row['direction']},"
            f"{row['realized_exit_reason']},{float(row['realized_net_r']):.4f},"
            f"{row['shadow_exit_reason']},{float(row['shadow_net_r']):.4f},"
            f"{float(row['shadow_minus_realized']):.4f},{row['shadow_exit_time']}"
        )
    if output_rows:
        realized = [float(r["realized_net_r"]) for r in output_rows]
        shadow = [float(r["shadow_net_r"]) for r in output_rows]
        diffs = [float(r["shadow_minus_realized"]) for r in output_rows]
        print("\nSUMMARY")
        print(f"n={len(output_rows)}")
        print(f"cum_realized_A_R={sum(realized):.4f}")
        print(f"cum_shadow_C_R={sum(shadow):.4f}")
        print(f"realized_win_rate={sum(v > 0 for v in realized) / len(realized):.2%}")
        print(f"shadow_win_rate={sum(v > 0 for v in shadow) / len(shadow):.2%}")
        print(f"mean_shadow_minus_realized={mean(diffs):.4f}")
    else:
        print("\nSUMMARY")
        print("n=0")
    print("\nSKIPS")
    for skip in skips:
        print(skip)
    print("\nREMINDER: n is far too small for conclusions until roughly 50-100 completed live trades.")


if __name__ == "__main__":
    main()
