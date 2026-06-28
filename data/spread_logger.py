"""
Spread Logger — XAUUSD bid/ask sampler for IUX Standard account

PURPOSE
-------
The IUX Standard account has no commission — the spread is the ENTIRE visible
round-trip cost. The PLACEHOLDER values in research/breakeven.py must be replaced
with measured numbers before Gate 0 is real and before Gate 1 can start.

This script polls the MT5 terminal for XAUUSD bid/ask, tags each sample with its
trading session (UTC-based), stores everything in SQLite, and prints a per-session
summary when done. The summary section tells you exactly what to put back into
breakeven.py.

It also doubles as the beginning of a realized-slippage log: once you have actual
trade fills (requested vs filled price), you can join them against the spread_samples
table on timestamp to measure whether your backtest cost assumptions held in practice.

HOW TO RUN
----------
  Prerequisites (Windows, MT5 running and logged in to IUX):
    pip install MetaTrader5 pandas numpy

  Collect 1 hour of data at 5-second intervals:
    python data/spread_logger.py --interval 5 --duration 3600

  Run indefinitely until Ctrl+C:
    python data/spread_logger.py --interval 5

  macOS / Linux — simulate mode (tests logger logic; data is synthetic):
    python3 data/spread_logger.py --simulate --duration 300

  Print summary of already-collected data (no new collection):
    python data/spread_logger.py --summary-only

  Full options:
    --symbol        MT5 symbol name                    (default: XAUUSD)
    --interval      Polling interval in seconds        (default: 5)
    --duration      Seconds to run; 0 = until Ctrl+C  (default: 0)
    --db            SQLite database path               (default: data/spread_log.sqlite)
    --simulate      Use synthetic data (macOS/offline testing)
    --summary-only  Skip collection; just report
    --verbose       Print each sample while collecting

SESSION BOUNDARIES (all times UTC — approximate; gold market has no hard session wall)
---------------------------------------------------------------------------------------
  ASIAN:      00:00–07:00 UTC  — thinner volume, spread typically widest here
  LONDON:     07:00–12:00 UTC  — Europe open, gold usually tightens
  NY_OVERLAP: 12:00–17:00 UTC  — peak volume, tightest spreads
  NY:         17:00–21:00 UTC  — New York afternoon, moderate volume
  OFF:        21:00–24:00 UTC  — very thin; spreads can spike around 22:00 rollover

HOW LONG TO COLLECT
-------------------
  Minimum: 5 full trading days to cover all sessions and intra-week variation.
  Recommended: 10+ days to catch spread behaviour around economic events.
  At 5-second intervals, 5 days = ~86,400 samples (~3 MB SQLite).

AFTER COLLECTION — updating breakeven.py
------------------------------------------
  The summary section prints specific values:
    SPREAD_QUIET_USD_OZ  ← use London or NY_OVERLAP median
    SPREAD_WIDE_USD_OZ   ← use Asian or OFF median

  Replace those two constants at the top of research/breakeven.py and re-run it.
  Only then is Gate 0 complete and Gate 1 can start.

REALIZED SLIPPAGE (future extension)
--------------------------------------
  When you have real trade fills, record (symbol, requested_price, filled_price,
  side, utc_time) in a separate fills table. Join it against spread_samples on
  timestamp to compute actual slippage vs quoted spread. That tells you whether
  the slippage=$0 assumption in the Standard account model is holding.
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — gracefully absent on macOS/Linux
# ---------------------------------------------------------------------------

try:
    import MetaTrader5 as mt5_module  # type: ignore
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5_module = None  # type: ignore


# ---------------------------------------------------------------------------
# Session classification (UTC)
# ---------------------------------------------------------------------------

# (start_hour_inclusive, end_hour_exclusive, label)
SESSION_MAP: list[tuple[int, int, str]] = [
    (0,  7,  "ASIAN"),
    (7,  12, "LONDON"),
    (12, 17, "NY_OVERLAP"),
    (17, 21, "NY"),
    (21, 24, "OFF"),
]


def classify_session(utc_hour: int) -> str:
    for start, end, label in SESSION_MAP:
        if start <= utc_hour < end:
            return label
    return "OFF"


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS spread_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    utc_time      TEXT    NOT NULL,   -- ISO 8601 UTC timestamp of sample
    broker_epoch  INTEGER NOT NULL,   -- Unix epoch from MT5 tick (UTC); 0 in simulate mode
    bid           REAL    NOT NULL,   -- bid price USD/oz (raw from tick)
    ask           REAL    NOT NULL,   -- ask price USD/oz (raw from tick)
    spread_oz     REAL    NOT NULL,   -- ask - bid in USD/oz; computed from prices, never from pip fields
    session       TEXT    NOT NULL    -- ASIAN / LONDON / NY_OVERLAP / NY / OFF
);
CREATE INDEX IF NOT EXISTS idx_session  ON spread_samples (session);
CREATE INDEX IF NOT EXISTS idx_utc_time ON spread_samples (utc_time);

-- One row per database, written once at startup.
-- Lets us verify the unit chain: spread_oz × trade_contract_size = spread per lot.
CREATE TABLE IF NOT EXISTS symbol_meta (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),  -- enforces single row
    symbol              TEXT    NOT NULL,
    digits              INTEGER NOT NULL,  -- decimal places in quoted price (2 or 3 for gold)
    point_size          REAL    NOT NULL,  -- smallest price increment in USD/oz (0.01 or 0.001)
    trade_contract_size REAL    NOT NULL,  -- oz per standard lot (must be 100 for XAUUSD)
    recorded_at         TEXT    NOT NULL,  -- ISO 8601 UTC, when this row was written
    simulated           INTEGER NOT NULL DEFAULT 0  -- 1 if synthetic data, 0 if live MT5
);
"""


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit
    conn.executescript(_DDL)
    return conn


def insert_sample(
    conn: sqlite3.Connection,
    utc_time: datetime,
    broker_epoch: int,
    bid: float,
    ask: float,
) -> None:
    # Spread is always computed as ask - bid from raw prices.
    # We never use symbol_info.spread (pip count) or multiply by point — that
    # introduces a unit dependency on digits and is unnecessary when the prices
    # themselves are already in USD/oz.
    spread_oz = round(ask - bid, 5)
    session   = classify_session(utc_time.hour)
    conn.execute(
        "INSERT INTO spread_samples"
        " (utc_time, broker_epoch, bid, ask, spread_oz, session)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (utc_time.strftime("%Y-%m-%dT%H:%M:%SZ"), broker_epoch, bid, ask, spread_oz, session),
    )


def write_symbol_meta(
    conn: sqlite3.Connection,
    symbol: str,
    digits: int,
    point_size: float,
    trade_contract_size: float,
    simulated: bool = False,
) -> None:
    """Write (or overwrite) the one-row symbol metadata record."""
    recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT OR REPLACE INTO symbol_meta"
        " (id, symbol, digits, point_size, trade_contract_size, recorded_at, simulated)"
        " VALUES (1, ?, ?, ?, ?, ?, ?)",
        (symbol, digits, point_size, trade_contract_size, recorded_at, int(simulated)),
    )


def read_symbol_meta(conn: sqlite3.Connection) -> Optional[dict]:
    """Return the stored symbol metadata, or None if the table/row is absent."""
    try:
        row = conn.execute(
            "SELECT symbol, digits, point_size, trade_contract_size, recorded_at, simulated"
            " FROM symbol_meta WHERE id = 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None  # old database without the symbol_meta table
    if row is None:
        return None
    return {
        "symbol":               row[0],
        "digits":               row[1],
        "point_size":           row[2],
        "trade_contract_size":  row[3],
        "recorded_at":          row[4],
        "simulated":            bool(row[5]),
    }


# ---------------------------------------------------------------------------
# Simulation (macOS / Linux / offline)
# Synthetic parameters mirror realistic IUX Standard gold spreads.
# ⚠  NEVER use simulated data to update research/breakeven.py — measure the real thing.
# ---------------------------------------------------------------------------

# (mean $/oz, std $/oz) per session — purely for plausible simulation
_SIM_PARAMS: dict[str, tuple[float, float]] = {
    "ASIAN":      (0.48, 0.10),
    "LONDON":     (0.28, 0.06),
    "NY_OVERLAP": (0.25, 0.05),
    "NY":         (0.32, 0.08),
    "OFF":        (0.65, 0.18),
}


def simulate_tick(utc_now: datetime, rng: random.Random) -> tuple[float, float, int]:
    session = classify_session(utc_now.hour)
    mean, std = _SIM_PARAMS[session]
    spread = max(mean + rng.gauss(0, std), 0.05)
    if rng.random() < 0.02:          # 2% spike probability
        spread *= rng.uniform(2.0, 5.0)
    mid = 2350.0 + rng.gauss(0, 3.0)
    bid = round(mid - spread / 2, 2)
    ask = round(bid + spread, 2)
    return bid, ask, 0


# ---------------------------------------------------------------------------
# MT5 connector
# ---------------------------------------------------------------------------

def mt5_connect(symbol: str, retries: int = 3) -> bool:
    if not MT5_AVAILABLE:
        return False
    for attempt in range(1, retries + 1):
        if mt5_module.initialize():
            info = mt5_module.symbol_info(symbol)
            if info is None:
                log.error("Symbol %r not found. Add it to Market Watch in MT5.", symbol)
                mt5_module.shutdown()
                return False
            if not info.visible:
                mt5_module.symbol_select(symbol, True)
            log.info("MT5 connected | symbol=%s | spread_digits=%s", symbol, getattr(info, "spread_digits", "?"))
            return True
        log.warning("MT5 init attempt %d/%d failed: %s", attempt, retries, mt5_module.last_error())
        time.sleep(2.0)
    return False


def fetch_symbol_meta(symbol: str) -> dict:
    """
    Read quoting precision from MT5 once at startup.

    Fields we care about:
      digits              — decimal places in the quoted price (2 → $0.01/oz resolution;
                            3 → $0.001/oz). Regardless of digits, ask-bid gives $/oz
                            directly because the prices are already in USD/oz.
      point               — smallest price increment (0.01 if digits=2; 0.001 if digits=3).
                            We store this for sanity but do NOT use it to compute spread.
      trade_contract_size — oz per standard lot; must be 100 for XAUUSD. If this is
                            wrong, every $/lot figure in breakeven.py is wrong.
    """
    info = mt5_module.symbol_info(symbol)
    return {
        "symbol":               symbol,
        "digits":               info.digits,
        "point_size":           info.point,
        "trade_contract_size":  info.trade_contract_size,
    }


def get_live_tick(symbol: str) -> Optional[tuple[float, float, int]]:
    tick = mt5_module.symbol_info_tick(symbol)
    if tick is None:
        return None
    # tick.time is UTC Unix epoch (integer seconds).
    # Spread = tick.ask - tick.bid; computed in insert_sample, not here, to keep
    # this function a pure data-fetch with no unit assumptions.
    return tick.bid, tick.ask, tick.time


# ---------------------------------------------------------------------------
# Statistics (pure Python — no pandas required)
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    n   = len(sorted_vals)
    idx = (n - 1) * p / 100.0
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _stats(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    n = len(s)
    return {
        "n":      n,
        "mean":   sum(s) / n,
        "median": _percentile(s, 50),
        "p90":    _percentile(s, 90),
        "max":    s[-1],
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

SEP82 = "─" * 82


def _print_meta_block(meta: Optional[dict]) -> None:
    """
    Print the symbol metadata header and unit-chain verification.
    This is the first thing shown in every summary so the user can confirm
    that spread_oz values are in the right ballpark before reading the table.
    """
    print()
    print(SEP82)
    if meta is None:
        print("  SYMBOL METADATA  — not recorded (old database; re-collect to populate)")
        print(SEP82)
        return

    source = "SYNTHETIC — simulate mode" if meta["simulated"] else "LIVE — from MT5 symbol_info"
    print(f"  SYMBOL METADATA  [{source}]")
    print(SEP82)

    digits        = meta["digits"]
    point_size    = meta["point_size"]
    contract_size = meta["trade_contract_size"]

    # Explain what digits means in plain terms
    resolution = point_size  # USD/oz per point
    digits_note = (
        f"cent-level quoting (${resolution:.2f}/oz resolution)"   if digits == 2 else
        f"sub-cent quoting   (${resolution:.3f}/oz resolution)"   if digits == 3 else
        f"{digits} decimal places (${resolution}/oz resolution)"
    )

    contract_ok = "✓ correct" if abs(contract_size - 100.0) < 0.01 else f"⚠ UNEXPECTED — expected 100.0"

    print(f"  Symbol              : {meta['symbol']}")
    print(f"  digits              : {digits}    ← {digits_note}")
    print(f"  point               : ${point_size:.4f}/oz  ← smallest quoted increment")
    print(f"  trade_contract_size : {contract_size:.1f} oz/lot  ← {contract_ok}")
    print(f"  Recorded at         : {meta['recorded_at']}")
    print()
    print("  Spread computation:")
    print("    spread_usd_per_oz  = tick.ask − tick.bid   ← raw prices, no pip conversion")
    print(f"    spread_usd_per_lot = spread_usd_per_oz × {contract_size:.0f}  ← × contract_size")
    print()

    # Sanity thresholds — if digits=3, ask-bid is still $/oz, just more precise;
    # no adjustment needed.  The thresholds below catch 10× off-by-one errors.
    print("  Sanity check (spread_usd_per_oz):")
    print("    Expected range for XAUUSD during active sessions: $0.20 – $0.80/oz")
    print("    ✓  $0.35  → correct unit")
    print("    ✗  $3.50  → off by 10× (would mean digits confusion — report immediately)")
    print("    ✗  $0.035 → off by ÷10 (same issue)")
    if digits == 3:
        print()
        print("    digits=3 note: sub-cent quoting; ask-bid still gives $/oz directly.")
        print("    You may see values like $0.352 instead of $0.35 — that is precision,")
        print("    not a unit error.")


def print_summary(conn: sqlite3.Connection, is_simulated: bool = False) -> None:
    meta = read_symbol_meta(conn)
    _print_meta_block(meta)

    rows = conn.execute(
        "SELECT session, spread_oz, utc_time FROM spread_samples ORDER BY utc_time"
    ).fetchall()

    if not rows:
        print("\n  No spread samples collected yet.")
        return

    by_session: dict[str, list[float]] = {}
    all_spreads: list[float] = []
    for session, spread, _ in rows:
        by_session.setdefault(session, []).append(spread)
        all_spreads.append(spread)

    first_ts = rows[0][2]
    last_ts  = rows[-1][2]
    order    = ["ASIAN", "LONDON", "NY_OVERLAP", "NY", "OFF"]

    print()
    print(SEP82)
    print("  SPREAD SUMMARY — XAUUSD / IUX Standard (USD/oz)")
    if is_simulated:
        print("  ⚠  SYNTHETIC DATA — do NOT use to update breakeven.py")
    print(SEP82)
    print(f"  Period : {first_ts}  →  {last_ts}")
    print(f"  Samples: {len(all_spreads):,}  |  "
          f"Sessions covered: {', '.join(s for s in order if s in by_session)}")
    print()

    hdr = (f"  {'Session':<12} {'N':>7} {'Mean':>8} {'Median':>8} "
           f"{'P90':>8} {'Max':>8}  Use for")
    print(hdr)
    print("  " + "─" * 73)

    quiet_candidates: list[tuple[str, float]] = []
    wide_candidates:  list[tuple[str, float]] = []

    for sess in order:
        if sess not in by_session:
            continue
        st = _stats(by_session[sess])
        use_for = ""
        if sess in ("LONDON", "NY_OVERLAP"):
            use_for = "→ SPREAD_QUIET_USD_OZ"
            quiet_candidates.append((sess, st["median"]))
        elif sess in ("ASIAN", "OFF"):
            use_for = "→ SPREAD_WIDE_USD_OZ"
            wide_candidates.append((sess, st["median"]))
        print(
            f"  {sess:<12} {st['n']:>7,d} "
            f"${st['mean']:>6.3f}  "
            f"${st['median']:>6.3f}  "
            f"${st['p90']:>6.3f}  "
            f"${st['max']:>6.3f}  "
            f"{use_for}"
        )

    all_st = _stats(all_spreads)
    print("  " + "─" * 73)
    print(
        f"  {'ALL':<12} {all_st['n']:>7,d} "
        f"${all_st['mean']:>6.3f}  "
        f"${all_st['median']:>6.3f}  "
        f"${all_st['p90']:>6.3f}  "
        f"${all_st['max']:>6.3f}"
    )
    print()

    if is_simulated:
        print("  ⚠  Simulated — collect real MT5 data before updating breakeven.py.")
        return

    # Unit chain example — uses the best quiet-session median and the stored
    # contract_size so the user can verify: spread_oz × 100 == RTC/lot in breakeven.py
    contract_size = (meta["trade_contract_size"] if meta else 100.0)
    if quiet_candidates:
        best_quiet = min(quiet_candidates, key=lambda x: x[1])
        lot_cost   = best_quiet[1] * contract_size
        print("  ─── UNIT CHAIN CHECK ────────────────────────────────────────────────")
        print(f"  {best_quiet[0]} median  :  ${best_quiet[1]:.3f}/oz  ×  "
              f"{contract_size:.0f} oz/lot  =  ${lot_cost:.2f}/lot")
        print(f"  → This should match the 'RTC/lot' column in research/breakeven.py")
        print(f"    (currently PLACEHOLDER = $35.00/lot; yours = ${lot_cost:.2f}/lot)")
        print()

    # Recommended replacement values for breakeven.py
    if quiet_candidates or wide_candidates:
        print("  ─── RECOMMENDED UPDATES for research/breakeven.py ──────────────────")
        if quiet_candidates:
            best_quiet = min(quiet_candidates, key=lambda x: x[1])
            print(f"  SPREAD_QUIET_USD_OZ = {best_quiet[1]:.2f}"
                  f"  # {best_quiet[0]} median  ← replace PLACEHOLDER")
        if wide_candidates:
            best_wide = max(wide_candidates, key=lambda x: x[1])
            print(f"  SPREAD_WIDE_USD_OZ  = {best_wide[1]:.2f}"
                  f"  # {best_wide[0]} median  ← replace PLACEHOLDER")
        print()
        print("  After updating, re-run: python3 research/breakeven.py")
        print("  Compare new breakeven lines against the PLACEHOLDER run.")
        print("  If any scenario flips tradeable/untradeable, re-read Gate 0 before Gate 1.")
        print("  ─────────────────────────────────────────────────────────────────────")
    print()


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect(args: argparse.Namespace) -> None:
    db_path    = Path(args.db)
    conn       = init_db(db_path)
    rng        = random.Random()

    # Decide mode
    is_simulate = args.simulate
    if not is_simulate and not MT5_AVAILABLE:
        print(
            "\n  ⚠  MetaTrader5 package not available (macOS/Linux).\n"
            "  Switching to --simulate mode. Install on Windows to collect real data.\n"
            "  Or pass --simulate explicitly to suppress this warning.\n"
        )
        is_simulate = True

    if not is_simulate:
        if not mt5_connect(args.symbol):
            print(
                "\n  ERROR: Could not connect to MT5.\n"
                "  Ensure MT5 is running, logged in to IUX, and the terminal\n"
                "  allows external connections (Tools → Options → Expert Advisors →\n"
                "  'Allow automated trading' and 'Allow DLL imports').\n"
            )
            sys.exit(1)

    # Record symbol quoting metadata once at startup.
    # Live: pulled from MT5 symbol_info (digits, point, trade_contract_size).
    # Simulate: hardcoded to gold's standard values so the summary looks identical.
    if is_simulate:
        write_symbol_meta(
            conn, args.symbol,
            digits=2, point_size=0.01, trade_contract_size=100.0,
            simulated=True,
        )
    else:
        meta = fetch_symbol_meta(args.symbol)
        write_symbol_meta(conn, simulated=False, **meta)
        log.info(
            "Symbol meta: digits=%d  point=%.4f  contract_size=%.1f oz/lot",
            meta["digits"], meta["point_size"], meta["trade_contract_size"],
        )

    # Graceful shutdown handler
    _running = [True]
    def _stop(sig, frame):
        _running[0] = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    mode_label = "SIMULATE ⚠ (synthetic)" if is_simulate else f"LIVE — {args.symbol} via MT5"
    print()
    print(f"  Mode    : {mode_label}")
    print(f"  Interval: {args.interval}s  |  "
          f"Duration: {'∞ (Ctrl+C to stop)' if args.duration == 0 else f'{args.duration}s'}  |  "
          f"DB: {db_path}")
    print()

    start      = time.monotonic()
    n_samples  = 0
    n_errors   = 0
    MT5_RETRY_EVERY = 5   # attempt reconnect after this many consecutive errors

    while _running[0]:
        elapsed = time.monotonic() - start
        if args.duration > 0 and elapsed >= args.duration:
            break

        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            if is_simulate:
                bid, ask, epoch = simulate_tick(utc_now, rng)
            else:
                result = get_live_tick(args.symbol)
                if result is None:
                    n_errors += 1
                    log.warning("No tick (error #%d): %s", n_errors, mt5_module.last_error())
                    if n_errors % MT5_RETRY_EVERY == 0:
                        log.info("Attempting MT5 reconnect...")
                        mt5_module.shutdown()
                        if not mt5_connect(args.symbol):
                            log.error("Reconnect failed — stopping.")
                            break
                    time.sleep(args.interval)
                    continue
                bid, ask, epoch = result

            insert_sample(conn, utc_now, epoch, bid, ask)
            n_samples += 1

            if args.verbose:
                spread  = ask - bid
                session = classify_session(utc_now.hour)
                print(f"  {utc_now.strftime('%H:%M:%S')} UTC  [{session:<10}]  "
                      f"bid={bid:.2f}  ask={ask:.2f}  spread=${spread:.3f}/oz")

        except Exception as exc:
            n_errors += 1
            log.error("Sample error: %s", exc)

        time.sleep(args.interval)

    # Wrap up
    elapsed = time.monotonic() - start
    print()
    print(f"  Stopped. Samples: {n_samples:,} | Errors: {n_errors} | "
          f"Elapsed: {elapsed:.0f}s")

    if not is_simulate and MT5_AVAILABLE:
        mt5_module.shutdown()

    print_summary(conn, is_simulated=is_simulate)
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XAUUSD spread logger — IUX Standard account (no commission, spread-only cost)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",       default="XAUUSD",                   help="MT5 symbol name")
    p.add_argument("--interval",     type=float, default=5.0,             help="Polling interval in seconds")
    p.add_argument("--duration",     type=int,   default=0,               help="Collection seconds; 0=until Ctrl+C")
    p.add_argument("--db",           default="data/spread_log.sqlite",    help="SQLite database path")
    p.add_argument("--simulate",     action="store_true",                  help="Synthetic data (macOS/offline)")
    p.add_argument("--summary-only", action="store_true", dest="summary_only",
                   help="Print summary of existing db without collecting new data")
    p.add_argument("--verbose",      action="store_true",                  help="Print each sample while running")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.summary_only:
        db_path = Path(args.db)
        if not db_path.exists():
            print(f"  Database not found: {db_path}")
            sys.exit(1)
        conn = sqlite3.connect(str(db_path))
        # Detect simulate-mode data: broker_epoch = 0 on all rows means simulated
        sim_check = conn.execute(
            "SELECT COUNT(*) FROM spread_samples WHERE broker_epoch != 0"
        ).fetchone()[0]
        is_sim = (sim_check == 0)
        print_summary(conn, is_simulated=is_sim)
        conn.close()
        return

    collect(args)


if __name__ == "__main__":
    main()
