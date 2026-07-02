"""
Decompose live compression fill differences into spread and pure slippage.

Usage:
    python3 research/live_slippage_decomposition.py --log path/to/live_log.csv

MT5 M15 bars are bid-side bars. Therefore:
- long buy-stop intended range high is bid-bar based; actual fill is ask, so
  total fill diff contains the full entry spread.
- short sell-stop intended range low is bid-bar based; actual fill is bid, so
  total fill diff does not contain entry spread.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


BACKTEST_SPREAD = 0.20


def fnum(raw: object) -> float | None:
    try:
        if raw in ("", None):
            return None
        val = float(raw)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def direction(row: dict[str, str]) -> int:
    raw = (row.get("breakout_direction") or "").lower()
    return 1 if raw == "long" else -1


def decompose(row: dict[str, str]) -> dict[str, float | str]:
    d = direction(row)
    intended = fnum(row.get("intended_entry"))
    actual = fnum(row.get("actual_fill_price"))
    atr = fnum(row.get("atr_at_entry"))
    exit_price = fnum(row.get("exit_price"))
    entry_spread = fnum(row.get("real_spread_at_entry"))
    exit_spread = fnum(row.get("real_spread_at_exit"))
    spread_lag = fnum(row.get("spread_sample_lag_seconds"))
    if intended is None or actual is None or atr is None:
        raise ValueError("missing intended/actual/ATR")
    total_fill_diff = d * (actual - intended)
    spread_component = entry_spread if d == 1 and entry_spread is not None else 0.0
    pure_slippage = total_fill_diff - spread_component
    gross_r = fnum(row.get("gross_r"))
    if gross_r is None and exit_price is not None:
        gross_r = d * (exit_price - actual) / atr
    net_flat_020 = gross_r - BACKTEST_SPREAD / atr if gross_r is not None else None
    # gross_r is from actual broker fill to actual broker exit, so it already
    # includes live bid/ask execution cost. Do not subtract spread again.
    net_realized = gross_r
    roundtrip_spread = ""
    if entry_spread is not None:
        roundtrip_spread = entry_spread + (exit_spread or 0.0)
    return {
        "ticket": row.get("ticket", ""),
        "side": "long" if d == 1 else "short",
        "intended": intended,
        "actual": actual,
        "exit": exit_price if exit_price is not None else "",
        "atr": atr,
        "total_fill_diff": total_fill_diff,
        "spread_at_entry": entry_spread if entry_spread is not None else "",
        "spread_sample_lag_seconds": spread_lag if spread_lag is not None else "",
        "spread_component": spread_component,
        "pure_slippage": pure_slippage,
        "gross_r_actual_fills": gross_r if gross_r is not None else "",
        "net_flat_020": net_flat_020 if net_flat_020 is not None else "",
        "net_realized_no_extra_spread": net_realized if net_realized is not None else "",
        "estimated_roundtrip_spread": roundtrip_spread,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("research/iux_compression_breakout_live_log.csv"))
    args = parser.parse_args()
    if not args.log.exists():
        raise SystemExit(f"log not found: {args.log}")
    with args.log.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = {row.get("ticket", ""): row for row in rows if row.get("event") == "entry" and row.get("ticket")}
    exits = [row for row in rows if row.get("event") == "exit" and row.get("ticket")]
    merged = []
    for exit_row in exits:
        entry = entries.get(exit_row.get("ticket", ""), {})
        merged_row = dict(entry)
        merged_row.update({k: v for k, v in exit_row.items() if v not in ("", None)})
        try:
            merged.append(decompose(merged_row))
        except ValueError:
            continue
    columns = [
        "ticket",
        "side",
        "intended",
        "actual",
        "exit",
        "atr",
        "total_fill_diff",
        "spread_at_entry",
        "spread_sample_lag_seconds",
        "spread_component",
        "pure_slippage",
        "gross_r_actual_fills",
        "net_flat_020",
        "net_realized_no_extra_spread",
        "estimated_roundtrip_spread",
    ]
    print(",".join(columns))
    for row in merged:
        print(",".join(str(row[k]) for k in columns))
    pure = [float(row["pure_slippage"]) for row in merged if row["pure_slippage"] != ""]
    flat = [float(row["net_flat_020"]) for row in merged if row["net_flat_020"] != ""]
    realized = [float(row["net_realized_no_extra_spread"]) for row in merged if row["net_realized_no_extra_spread"] != ""]
    if merged:
        print("\nSUMMARY")
        print(f"trades={len(merged)}")
        print(f"avg_pure_slippage={mean(pure):.4f}" if pure else "avg_pure_slippage=n/a")
        print(f"avg_net_flat_020={mean(flat):.4f}" if flat else "avg_net_flat_020=n/a")
        print(f"avg_net_realized_no_extra_spread={mean(realized):.4f}" if realized else "avg_net_realized_no_extra_spread=n/a")
        print("double_count_check=gross_r_actual_fills already uses actual bid/ask fills; net_flat_020 subtracts another backtest spread for comparability, not true realized live PnL")


if __name__ == "__main__":
    main()
