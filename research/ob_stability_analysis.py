"""
OB Edge Stability Analysis — Phase B outcomes decomposition.

No parameter changes. Reads the locked Phase B CSV and reports:
  1. Edge by year (2016-2026)
  2. Edge by ATR volatility regime (frozen_atr terciles, dataset-wide)
  3. Edge by trending vs ranging context (consecutive-OB-direction rule)

Trending/ranging rule (pre-registered, uses only pre-zone info from CSV):
  For each OB, find the most recent prior OB in the same segment (same segment_id,
  strictly smaller ob_candle_index). If that prior OB exists and has the SAME
  direction as the current OB, classify "trending". Otherwise (no prior OB in
  segment, or prior OB is opposite direction) classify "ranging/counter".

This approximates consecutive higher-highs/lower-lows using the OB event stream
as a proxy for price structure, without requiring raw OHLC bars.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Optional


CSV_PATH = Path(__file__).parent / "order_block_phase_b_outcomes.csv"


def edge_ci(
    ob_success: int, ob_n: int, base_success: int, base_n: int
) -> tuple[float, float, float]:
    if ob_n == 0 or base_n == 0:
        return math.nan, math.nan, math.nan
    p_ob = ob_success / ob_n
    p_base = base_success / base_n
    edge = p_ob - p_base
    se = math.sqrt(p_ob * (1 - p_ob) / ob_n + p_base * (1 - p_base) / base_n)
    return edge, edge - 1.96 * se, edge + 1.96 * se


def pct(n: int, d: int) -> str:
    return f"{n / d:.2%}" if d else "n/a"


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_bool(val: str) -> bool:
    return val.strip().lower() == "true"


def parse_int_or_none(val: str) -> Optional[int]:
    val = val.strip()
    return int(val) if val else None


def parse_float_or_none(val: str) -> Optional[float]:
    val = val.strip()
    return float(val) if val else None


def tercile_boundaries(values: list[float]) -> tuple[float, float]:
    """Return the 33rd and 67th percentile boundaries."""
    ordered = sorted(values)
    n = len(ordered)
    lo = ordered[int(n * 0.333)]
    hi = ordered[int(n * 0.667)]
    return lo, hi


def assign_trend_context(rows: list[dict]) -> list[str]:
    """
    Assign trending / ranging label per row.

    Pre-registered rule:
    - For each OB (identified by segment_id + ob_candle_index), look up the most
      recent prior OB in the same segment (smallest ob_candle_index strictly less
      than the current one).
    - If prior OB direction == current OB direction → "trending"
    - Otherwise (no prior, or opposite direction)      → "ranging"
    """
    # Build lookup: segment_id -> sorted list of (ob_candle_index, direction, row_i)
    from collections import defaultdict
    seg_obs: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for i, row in enumerate(rows):
        seg_id = row["segment_id"]
        ob_idx = parse_int_or_none(row["ob_candle_index"])
        if ob_idx is None:
            continue
        seg_obs[seg_id].append((ob_idx, row["direction"]))

    # Sort each segment by ob_candle_index
    for seg_id in seg_obs:
        seg_obs[seg_id].sort(key=lambda x: x[0])

    labels = []
    for row in rows:
        seg_id = row["segment_id"]
        ob_idx = parse_int_or_none(row["ob_candle_index"])
        direction = row["direction"]
        if ob_idx is None:
            labels.append("ranging")
            continue
        prior_list = seg_obs[seg_id]
        # Find prior OBs with strictly smaller ob_candle_index
        prior_same_seg = [(idx, d) for idx, d in prior_list if idx < ob_idx]
        if not prior_same_seg:
            labels.append("ranging")
        else:
            # Most recent prior OB (largest index strictly less than current)
            prior_dir = prior_same_seg[-1][1]
            if prior_dir == direction:
                labels.append("trending")
            else:
                labels.append("ranging")
    return labels


def print_bucket_table(
    header: str,
    groups: list[tuple[str, list[tuple[int, int, int, int]]]],
) -> None:
    """
    groups: list of (label, [(ob_success, ob_n, base_success, base_n), ...])
    Each group aggregates its list of observation tuples.
    """
    print(f"\n{header}")
    print(f"  {'bucket':<20} {'OB_n':>6} {'OB_rate':>8} {'base_rate':>9} {'edge':>8} {'95% CI':>22} {'base_n':>6}")
    print("  " + "-" * 90)
    for label, obs_list in groups:
        ob_success = sum(o[0] for o in obs_list)
        ob_n = sum(o[1] for o in obs_list)
        base_success = sum(o[2] for o in obs_list)
        base_n = sum(o[3] for o in obs_list)
        edge, ci_lo, ci_hi = edge_ci(ob_success, ob_n, base_success, base_n)
        ob_rate_str = pct(ob_success, ob_n)
        base_rate_str = pct(base_success, base_n)
        edge_str = f"{edge:.2%}" if not math.isnan(edge) else "n/a"
        ci_str = (
            f"[{ci_lo:.2%}, {ci_hi:.2%}]"
            if not math.isnan(ci_lo)
            else "n/a"
        )
        # Sign flag
        if not math.isnan(ci_lo):
            if ci_lo > 0:
                flag = "(+)"
            elif ci_hi < 0:
                flag = "(-)"
            else:
                flag = "(?)"
        else:
            flag = ""
        print(
            f"  {label:<20} {ob_n:>6,} {ob_rate_str:>8} {base_rate_str:>9} "
            f"{edge_str:>8} {ci_str:>22}  {flag:<4} {base_n:>6,}"
        )


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")

    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows):,} rows from {CSV_PATH.name}")

    # Attach trend context labels to all rows
    trend_labels = assign_trend_context(rows)
    for row, label in zip(rows, trend_labels):
        row["_trend_context"] = label

    # --- filter to first-touch only ---
    touched = [row for row in rows if parse_bool(row["touched"])]
    print(f"Fresh first-touch rows: {len(touched):,}")

    # Collect frozen_atr values from touched rows for tercile boundaries
    atr_values = [float(row["frozen_atr"]) for row in touched]
    t33, t67 = tercile_boundaries(atr_values)
    print(f"ATR tercile boundaries (across touched setups): low<{t33:.4f}, med<{t67:.4f}, high≥{t67:.4f}")

    # Helper: extract obs tuple for a single row
    def row_obs(row: dict) -> tuple[int, int, int, int]:
        success = parse_bool(row["success"]) if row["success"].strip() else False
        ob_success = 1 if success else 0
        ob_n = 1
        bc = parse_int_or_none(row["baseline_count"]) or 0
        bs = parse_int_or_none(row["baseline_successes"]) or 0
        return ob_success, ob_n, bs, bc

    # =========================================================================
    # BREAKDOWN 1: Per Year
    # =========================================================================
    print("\n" + "=" * 100)
    print("BREAKDOWN 1 — Edge by Year")
    print("Rule: group touched setups by year of zone_creation_time.")
    print("Edge = OB_success_rate - matched_baseline_success_rate")
    print("CI = 95% normal approximation (Wilson would require per-baseline SE; this is the aggregate approximation).")
    print("(+) = CI lower bound > 0   (-) = CI upper bound < 0   (?) = CI crosses zero")

    years = sorted(set(row["year"] for row in touched))
    year_groups = []
    for yr in years:
        yr_rows = [row for row in touched if row["year"] == yr]
        year_groups.append((str(yr), [row_obs(r) for r in yr_rows]))

    print_bucket_table("  year          OB_n  OB_rate  base_rate    edge              95% CI       base_n", year_groups)

    # Also print raw success counts for transparency
    print("\n  Per-year raw counts (for small-N caution):")
    print(f"  {'year':<6} {'touched':>8} {'OB_succ':>9} {'base_n':>8} {'base_succ':>10} {'under_matched':>14}")
    for yr in years:
        yr_rows = [row for row in touched if row["year"] == yr]
        ob_succ = sum(1 for r in yr_rows if parse_bool(r["success"]) if r["success"].strip())
        base_n = sum((parse_int_or_none(r["baseline_count"]) or 0) for r in yr_rows)
        base_succ = sum((parse_int_or_none(r["baseline_successes"]) or 0) for r in yr_rows)
        under = sum(1 for r in yr_rows if parse_bool(r["under_matched"]))
        print(f"  {yr:<6} {len(yr_rows):>8,} {ob_succ:>9,} {base_n:>8,} {base_succ:>10,} {under:>14,}")

    # =========================================================================
    # BREAKDOWN 2: Per ATR Volatility Regime (frozen_atr terciles)
    # =========================================================================
    print("\n" + "=" * 100)
    print("BREAKDOWN 2 — Edge by ATR Volatility Regime (frozen_atr terciles, dataset-wide)")
    print(f"Tercile cuts: low = frozen_atr < {t33:.4f}, medium = {t33:.4f}–{t67:.4f}, high = ≥ {t67:.4f}")
    print("ATR is the frozen ATR(14) at zone creation time — NOT a fixed dollar value.")

    def atr_regime(atr: float) -> str:
        if atr < t33:
            return "low_atr"
        if atr < t67:
            return "medium_atr"
        return "high_atr"

    regime_order = ["low_atr", "medium_atr", "high_atr"]
    regime_groups = []
    for regime in regime_order:
        regime_rows = [row for row in touched if atr_regime(float(row["frozen_atr"])) == regime]
        regime_groups.append((regime, [row_obs(r) for r in regime_rows]))

    print_bucket_table("  ATR regime", regime_groups)

    # Year × Regime cross-tab to show whether high-ATR bias is year-driven
    print("\n  Year × ATR regime cross-tab (counts of touched setups):")
    print(f"  {'year':<6} {'low_atr':>9} {'medium_atr':>11} {'high_atr':>9}")
    for yr in years:
        yr_rows = [row for row in touched if row["year"] == yr]
        lo = sum(1 for r in yr_rows if atr_regime(float(r["frozen_atr"])) == "low_atr")
        med = sum(1 for r in yr_rows if atr_regime(float(r["frozen_atr"])) == "medium_atr")
        hi = sum(1 for r in yr_rows if atr_regime(float(r["frozen_atr"])) == "high_atr")
        print(f"  {yr:<6} {lo:>9,} {med:>11,} {hi:>9,}")

    # =========================================================================
    # BREAKDOWN 3: Trending vs Ranging
    # =========================================================================
    print("\n" + "=" * 100)
    print("BREAKDOWN 3 — Edge by Market Context: Trending vs Ranging")
    print("Pre-registered rule: for each OB, find the most recent prior OB in the same segment.")
    print("  If prior OB direction == current OB direction → 'trending' (consecutive same-dir BOS)")
    print("  Otherwise (no prior OB in segment, or prior OB is opposite direction) → 'ranging'")
    print("Rationale: same-direction consecutive BOS approximates a series of HH/LH or LL/HL")
    print("without requiring raw OHLC bars beyond what is already in the Phase B CSV.")

    context_order = ["trending", "ranging"]
    context_groups = []
    for ctx in context_order:
        ctx_rows = [row for row in touched if row["_trend_context"] == ctx]
        context_groups.append((ctx, [row_obs(r) for r in ctx_rows]))

    print_bucket_table("  context", context_groups)

    # Also show by year to see if trending/ranging distribution shifts over time
    print("\n  Context distribution by year (trending | ranging counts):")
    print(f"  {'year':<6} {'trending':>10} {'ranging':>9}  {'trend_edge':>11} {'range_edge':>11}")
    for yr in years:
        yr_rows = [row for row in touched if row["year"] == yr]
        t_rows = [r for r in yr_rows if r["_trend_context"] == "trending"]
        r_rows = [r for r in yr_rows if r["_trend_context"] == "ranging"]
        t_succ = sum(1 for r in t_rows if parse_bool(r["success"]) if r["success"].strip())
        r_succ = sum(1 for r in r_rows if parse_bool(r["success"]) if r["success"].strip())
        t_base_n = sum((parse_int_or_none(r["baseline_count"]) or 0) for r in t_rows)
        t_base_s = sum((parse_int_or_none(r["baseline_successes"]) or 0) for r in t_rows)
        r_base_n = sum((parse_int_or_none(r["baseline_count"]) or 0) for r in r_rows)
        r_base_s = sum((parse_int_or_none(r["baseline_successes"]) or 0) for r in r_rows)
        t_edge, _, _ = edge_ci(t_succ, len(t_rows), t_base_s, t_base_n)
        r_edge, _, _ = edge_ci(r_succ, len(r_rows), r_base_s, r_base_n)
        t_edge_str = f"{t_edge:.2%}" if not math.isnan(t_edge) else "n/a"
        r_edge_str = f"{r_edge:.2%}" if not math.isnan(r_edge) else "n/a"
        print(f"  {yr:<6} {len(t_rows):>10,} {len(r_rows):>9,}  {t_edge_str:>11} {r_edge_str:>11}")

    # =========================================================================
    # AGGREGATE REMINDER
    # =========================================================================
    print("\n" + "=" * 100)
    print("AGGREGATE (Phase B locked result, for reference):")
    all_obs = [row_obs(r) for r in touched]
    ob_succ = sum(o[0] for o in all_obs)
    ob_n = sum(o[1] for o in all_obs)
    base_s = sum(o[2] for o in all_obs)
    base_n = sum(o[3] for o in all_obs)
    agg_edge, agg_ci_lo, agg_ci_hi = edge_ci(ob_succ, ob_n, base_s, base_n)
    print(f"  OB rate: {pct(ob_succ, ob_n)} ({ob_succ}/{ob_n})  |  Baseline: {pct(base_s, base_n)} ({base_s}/{base_n})")
    print(f"  EDGE: {agg_edge:.2%}  |  95% CI: [{agg_ci_lo:.2%}, {agg_ci_hi:.2%}]")
    print()
    print("NOTE: This script performs DIAGNOSTIC decomposition only.")
    print("Do NOT use the best-performing regime as a new filter — that would be p-hacking.")
    print("No parameter changes, no new timeframes, no tuning. Locked Phase B parameters.")


if __name__ == "__main__":
    main()
