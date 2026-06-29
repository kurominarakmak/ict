"""
Local Streamlit dashboard for the IUX MT5 compression breakout demo bot.

Run:
    streamlit run dashboard.py

The data layer is intentionally small (`load_log_source`) so it can later be
replaced with a web/API source without changing metric code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


DEFAULT_LOG_PATH = Path("research/iux_compression_breakout_live_log.csv")

BACKTEST_EXPECTANCY_BAND = (0.22, 0.27)
BACKTEST_WIN_RATE_BAND = (0.57, 0.64)
BACKTEST_SPREAD_USD = 0.20
BACKTEST_TRADES_PER_MONTH = 22
BACKTEST_WORST_LOSS_STREAK = 6
BACKTEST_MAX_DD_R = -40.12
STALE_MINUTES = 20


TRADE_COLUMNS = [
    "timestamp_utc",
    "ticket",
    "breakout_direction",
    "actual_fill_price",
    "exit_price",
    "exit_reason",
    "gross_r",
    "net_r",
    "net_r_vs_020_spread",
    "realized_spread_or_slippage",
    "entry_slippage",
    "trade_number",
    "cum_r",
    "equity_high",
    "drawdown_r",
]


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: float
    ci_low: float
    ci_high: float


def load_log_source(path: Path) -> pd.DataFrame:
    """Local CSV data layer. Swap this function later for S3/API/web source."""

    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    for col in ("timestamp_utc", "signal_time", "setup_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    for col in (
        "range_high",
        "range_low",
        "intended_entry",
        "actual_fill_price",
        "sl",
        "tp",
        "atr_at_entry",
        "exit_price",
        "gross_r",
        "net_r_vs_020_spread",
        "realized_spread_or_slippage",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def completed_trades(df: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=TRADE_COLUMNS)
    if df.empty or "event" not in df.columns:
        return empty
    exits = df[df["event"].astype(str).str.lower() == "exit"].copy()
    if exits.empty:
        return empty
    entries = df[df["event"].astype(str).str.lower() == "entry"].copy()
    if not entries.empty and "ticket" in entries.columns and "ticket" in exits.columns:
        entry_cols = [
            "ticket",
            "actual_fill_price",
            "intended_entry",
            "realized_spread_or_slippage",
            "timestamp_utc",
        ]
        entry_cols = [c for c in entry_cols if c in entries.columns]
        entries = entries[entry_cols].rename(
            columns={
                "actual_fill_price": "entry_fill_price",
                "intended_entry": "entry_intended_price",
                "realized_spread_or_slippage": "entry_slippage",
                "timestamp_utc": "entry_logged_at",
            }
        )
        exits = exits.merge(entries, on="ticket", how="left")
    exits = exits.sort_values("timestamp_utc").reset_index(drop=True)
    exits["trade_number"] = range(1, len(exits) + 1)
    exits["net_r"] = exits.get("net_r_vs_020_spread", pd.Series(dtype=float))
    exits["gross_r"] = exits.get("gross_r", pd.Series(dtype=float))
    exits["cum_r"] = exits["net_r"].fillna(0).cumsum()
    exits["equity_high"] = exits["cum_r"].cummax()
    exits["drawdown_r"] = exits["cum_r"] - exits["equity_high"]
    return exits


def summarize(vals: pd.Series) -> MetricSummary:
    clean = pd.to_numeric(vals, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return MetricSummary(0, math.nan, math.nan, math.nan)
    mean = float(clean.mean())
    sd = float(clean.std(ddof=0)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else math.nan
    return MetricSummary(n, mean, mean - 1.96 * se, mean + 1.96 * se)


def readiness(n: int) -> tuple[str, str]:
    if n < 30:
        return "Too early", "n < 30: wide CI, do not interpret."
    if n < 50:
        return "Preliminary", "30-49 trades: useful but still noisy."
    if n < 100:
        return "Meaningful", "50-99 trades: live estimate becoming useful."
    return "Solid", "100+ trades: live estimate is statistically more useful."


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty or "net_r" not in trades.columns:
        return math.nan
    wins = trades.loc[trades["net_r"] > 0, "net_r"].sum()
    losses = -trades.loc[trades["net_r"] < 0, "net_r"].sum()
    return float(wins / losses) if losses > 0 else math.inf


def streaks(trades: pd.DataFrame) -> tuple[int, int]:
    if trades.empty or "net_r" not in trades.columns:
        return 0, 0
    current = 0
    worst = 0
    for val in trades["net_r"].fillna(0):
        if val < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    current_loss_streak = 0
    for val in reversed(trades["net_r"].fillna(0).tolist()):
        if val < 0:
            current_loss_streak += 1
        else:
            break
    return current_loss_streak, worst


def month_count(trades: pd.DataFrame) -> int:
    if trades.empty or "timestamp_utc" not in trades.columns:
        return 0
    now = pd.Timestamp.now(tz="UTC")
    return int((trades["timestamp_utc"] >= now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)).sum())


def daily_drawdown(trades: pd.DataFrame) -> float:
    if trades.empty or "net_r" not in trades.columns or "timestamp_utc" not in trades.columns:
        return math.nan
    tmp = trades.dropna(subset=["timestamp_utc"]).copy()
    if tmp.empty:
        return math.nan
    tmp["day"] = tmp["timestamp_utc"].dt.date
    daily = tmp.groupby("day")["net_r"].sum()
    return float(daily.min()) if not daily.empty else math.nan


def time_in_drawdown(trades: pd.DataFrame) -> str:
    if trades.empty or "cum_r" not in trades.columns or "equity_high" not in trades.columns:
        return "n/a"
    at_high = trades["cum_r"] >= trades["equity_high"]
    if bool(at_high.iloc[-1]):
        return "0 trades"
    last_high_idx = at_high[at_high].index.max()
    if pd.isna(last_high_idx):
        return f"{len(trades)} trades"
    return f"{len(trades) - int(last_high_idx) - 1} trades"


def infer_health(df: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    now = pd.Timestamp.now(tz="UTC")
    if df.empty or "timestamp_utc" not in df.columns:
        return {
            "last_log": None,
            "stale": True,
            "minutes_since_log": math.nan,
            "time_since_trade": "n/a",
            "open_positions": 0,
            "pending_orders": 0,
        }
    last_log = df["timestamp_utc"].dropna().max()
    minutes = (now - last_log).total_seconds() / 60 if pd.notna(last_log) else math.nan
    entries = df[df["event"].astype(str).str.lower() == "entry"] if "event" in df else pd.DataFrame()
    exits = df[df["event"].astype(str).str.lower() == "exit"] if "event" in df else pd.DataFrame()
    entry_tickets = set(entries.get("ticket", pd.Series(dtype=object)).dropna().astype(str))
    exit_tickets = set(exits.get("ticket", pd.Series(dtype=object)).dropna().astype(str))
    open_positions = len(entry_tickets - exit_tickets)
    signal_count = int((df["event"].astype(str).str.lower() == "signal").sum()) if "event" in df else 0
    pending_orders = max(0, signal_count - len(entry_tickets))
    if trades.empty or trades["timestamp_utc"].dropna().empty:
        since_trade = "n/a"
    else:
        last_trade = trades["timestamp_utc"].dropna().max()
        hours = (now - last_trade).total_seconds() / 3600
        since_trade = f"{hours:.1f}h"
    return {
        "last_log": last_log,
        "stale": bool(minutes > STALE_MINUTES) if math.isfinite(minutes) else True,
        "minutes_since_log": minutes,
        "time_since_trade": since_trade,
        "open_positions": open_positions,
        "pending_orders": pending_orders,
    }


def ci_badge(summary: MetricSummary) -> str:
    if summary.n == 0:
        return "No live trades yet"
    if summary.ci_low > 0:
        return "CI clears zero"
    return "CI does not clear zero"


def alert_messages(trades: pd.DataFrame, expectancy: MetricSummary, win_rate: float, avg_spread_proxy: float, health: dict[str, object]) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []
    n = expectancy.n
    if health["stale"]:
        alerts.append(("error", f"Bot/log appears stale: last log {health['minutes_since_log']:.1f} minutes ago."))
    if n >= 30:
        if expectancy.ci_high < BACKTEST_EXPECTANCY_BAND[0] or expectancy.ci_low > BACKTEST_EXPECTANCY_BAND[1]:
            alerts.append(("error", "Live expectancy CI is outside the backtest reference band. Check logic/fills/regime."))
        if win_rate < BACKTEST_WIN_RATE_BAND[0] - 0.10 or win_rate > BACKTEST_WIN_RATE_BAND[1] + 0.10:
            alerts.append(("warning", "Live win rate materially diverges from backtest band after n >= 30."))
    if math.isfinite(avg_spread_proxy) and avg_spread_proxy > BACKTEST_SPREAD_USD * 1.5:
        alerts.append(("warning", "Average logged slippage/spread proxy is materially above the $0.20 backtest assumption."))
    if trades.empty:
        alerts.append(("info", "No completed trades yet. Dashboard will become useful after exits are logged."))
    return alerts


def main() -> None:
    st.set_page_config(page_title="Compression Breakout Bot", layout="wide")
    st.title("Compression Breakout Demo Bot Monitor")

    with st.sidebar:
        log_path = Path(st.text_input("CSV log path", str(DEFAULT_LOG_PATH)))
        refresh_seconds = st.slider("Auto-refresh seconds", 30, 120, 60, 15)
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        st.caption("Data layer currently reads a local CSV. It can later be swapped for a remote/API source.")

    df = load_log_source(log_path)
    trades = completed_trades(df)
    health = infer_health(df, trades)
    expectancy = summarize(trades.get("net_r", pd.Series(dtype=float)))
    win_rate = float((trades["net_r"] > 0).mean()) if not trades.empty and "net_r" in trades.columns else math.nan
    pf = profit_factor(trades)
    avg_win = float(trades.loc[trades["net_r"] > 0, "net_r"].mean()) if not trades.empty and "net_r" in trades.columns else math.nan
    avg_loss = float(trades.loc[trades["net_r"] < 0, "net_r"].mean()) if not trades.empty and "net_r" in trades.columns else math.nan
    current_streak, worst_streak = streaks(trades)
    ready_label, ready_text = readiness(expectancy.n)

    st.info(
        "Demo is slow (~22 trades/mo). Need 50-100+ trades for meaningful conclusions. "
        "Early n<30 has very wide CIs. The CI-clears-zero indicator is the real live test. "
        "Live spread/slippage is the execution gate; backtest assumed $0.20 round trip."
    )

    st.subheader("1. Bot Health")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", "STALE" if health["stale"] else "Running")
    c2.metric("Last log", str(health["last_log"]) if health["last_log"] is not None else "none")
    c3.metric("Open positions", int(health["open_positions"]))
    c4.metric("Pending orders", int(health["pending_orders"]))
    c5.metric("Since last trade", str(health["time_since_trade"]))
    st.caption("MT5 connection status is only visible if the bot writes log/heartbeat rows; current dashboard infers health from CSV freshness.")

    st.subheader("Divergence Alerts")
    avg_slip_proxy = float(trades["realized_spread_or_slippage"].abs().mean()) if "realized_spread_or_slippage" in trades and not trades.empty else math.nan
    for kind, msg in alert_messages(trades, expectancy, win_rate, avg_slip_proxy, health):
        getattr(st, kind)(msg)

    st.subheader("2. Performance vs Backtest")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Completed trades", expectancy.n, help=ready_text)
    p2.metric("Readiness", ready_label)
    p3.metric("Live expectancy", f"{expectancy.mean:.3f}R" if expectancy.n else "n/a", f"CI [{expectancy.ci_low:.3f}, {expectancy.ci_high:.3f}]")
    p4.metric("CI test", ci_badge(expectancy))
    p5.metric("Backtest band", f"{BACKTEST_EXPECTANCY_BAND[0]:.2f}-{BACKTEST_EXPECTANCY_BAND[1]:.2f}R")

    p6, p7, p8, p9 = st.columns(4)
    p6.metric("Win rate", f"{win_rate:.1%}" if math.isfinite(win_rate) else "n/a", f"BT {BACKTEST_WIN_RATE_BAND[0]:.0%}-{BACKTEST_WIN_RATE_BAND[1]:.0%}")
    p7.metric("Profit factor", f"{pf:.2f}" if math.isfinite(pf) else "n/a")
    p8.metric("Avg win / loss", f"{avg_win:.2f}R / {avg_loss:.2f}R" if math.isfinite(avg_win) and math.isfinite(avg_loss) else "n/a")
    p9.metric("Trades this month", month_count(trades), f"BT ~{BACKTEST_TRADES_PER_MONTH}/mo")

    if not trades.empty:
        chart_df = trades[["timestamp_utc", "cum_r"]].dropna().set_index("timestamp_utc")
        if not chart_df.empty:
            st.line_chart(chart_df, height=280)
        expected = pd.DataFrame(
            {
                "trade_number": trades["trade_number"],
                "expected_low": trades["trade_number"] * BACKTEST_EXPECTANCY_BAND[0],
                "expected_high": trades["trade_number"] * BACKTEST_EXPECTANCY_BAND[1],
                "live": trades["cum_r"],
            }
        ).set_index("trade_number")
        st.caption("Equity curve over trade count with backtest expected slope band.")
        st.line_chart(expected, height=240)

    st.subheader("3. Execution Quality")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Backtest spread", f"${BACKTEST_SPREAD_USD:.2f}/oz")
    e2.metric("Avg entry slippage", f"{trades.get('entry_slippage', pd.Series(dtype=float)).mean():.3f}" if not trades.empty and "entry_slippage" in trades else "n/a")
    e3.metric("Largest loss", f"{trades['net_r'].min():.2f}R" if not trades.empty else "n/a")
    slippage_loss_count = int((trades["net_r"] < -1.05).sum()) if not trades.empty and "net_r" in trades.columns else 0
    e4.metric("Losses worse than -1.05R", slippage_loss_count)
    gap = float(trades["cum_r"].iloc[-1] - trades["trade_number"].iloc[-1] * BACKTEST_EXPECTANCY_BAND[0]) if not trades.empty else math.nan
    e5.metric("Live vs BT low-band gap", f"{gap:.2f}R" if math.isfinite(gap) else "n/a")

    if not trades.empty:
        st.write("Exit reason counts")
        st.dataframe(trades.get("exit_reason", pd.Series(dtype=object)).fillna("unknown").value_counts().rename("count"))
        if "entry_slippage" in trades:
            st.write("Entry slippage distribution")
            st.bar_chart(trades["entry_slippage"].dropna())

    st.subheader("4. Risk")
    r1, r2, r3, r4, r5 = st.columns(5)
    current_dd = float(trades["drawdown_r"].iloc[-1]) if not trades.empty and "drawdown_r" in trades.columns else math.nan
    max_dd = float(trades["drawdown_r"].min()) if not trades.empty and "drawdown_r" in trades.columns else math.nan
    r1.metric("Current DD", f"{current_dd:.2f}R" if math.isfinite(current_dd) else "n/a")
    r2.metric("Max DD", f"{max_dd:.2f}R" if math.isfinite(max_dd) else "n/a", f"BT {BACKTEST_MAX_DD_R:.1f}R")
    r3.metric("Daily DD worst", f"{daily_drawdown(trades):.2f}R" if not trades.empty else "n/a")
    r4.metric("Loss streak", f"{current_streak} now / {worst_streak} worst", f"BT worst {BACKTEST_WORST_LOSS_STREAK}")
    r5.metric("Time in DD", time_in_drawdown(trades))

    st.subheader("5. Trade Log")
    if trades.empty:
        st.warning(f"No completed exit rows found in {log_path}")
    else:
        cols = [
            "timestamp_utc",
            "ticket",
            "breakout_direction",
            "actual_fill_price",
            "exit_price",
            "exit_reason",
            "gross_r",
            "net_r",
            "cum_r",
            "drawdown_r",
        ]
        st.dataframe(trades[[c for c in cols if c in trades.columns]].tail(100), use_container_width=True)

    st.caption(f"Last dashboard update: {datetime.now(timezone.utc).isoformat()}")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
