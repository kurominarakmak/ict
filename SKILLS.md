# SKILL.md — Building Systematic Trading Bots

> **Read this before starting any new trading-bot project.** This is a general playbook distilled from building and live-testing real MT5 bots. It is strategy-agnostic — it encodes the discipline, the traps, and the process that apply to _any_ systematic trading system, whatever the edge. The user is a rigorous quant who catches overstatements; match that standard. When a lesson below has a concrete example, it's from a real bug that actually cost time — don't relearn it the hard way.

---

## 1. Research discipline (the part that decides success)

Most systematic strategies fail not because the idea is bad but because the research fooled itself. Guard against this above all.

- **Mechanism before code.** State _why_ the edge should exist before writing a backtest. If you can't articulate the mechanism, you're fishing, and fishing finds noise.
- **Pre-register, don't tune.** Decide parameters and thresholds up front. NO sweeping/optimizing to make results look good. Every knob you turn post-hoc inflates your false-discovery rate.
- **Make all thresholds relative** (ATR-relative for price, percentile-relative for rankings). Hardcoded absolute levels overfit to the sample's price regime.
- **Train/test split, always.** Fit intuition on train, judge on a held-out test period. The edge must clear zero **net of cost** in BOTH. Report confidence intervals, not point estimates.
- **Compare to matched baseline + random.** An edge that doesn't beat a random-entry / buy-and-hold baseline of the same exposure isn't an edge.
- **Cross-asset / cross-regime replication is a validity gate.** If the mechanism is real, a related instrument should show the same GROSS edge. If it only works on one symbol in one period, suspect overfitting.
- **Apply multiple-testing corrections** (Deflated Sharpe, Benjamini-Hochberg FDR) when you've tested many hypotheses. The more you tried, the higher your bar for significance.

### Statistical traps that repeatedly bite (internalize these)

- **CI crossing zero ≠ edge absent.** Absence of evidence is not evidence of absence. Small-sample slices (single years, thin regimes) will have wide CIs from noise; judge the aggregate.
- **Net-negative on one instrument ≠ edge absent.** Separate GROSS (does the edge exist?) from COST (is it tradable _here_?). An edge can be real but untradable on an instrument whose spread/ATR ratio is too high.
- **"Crisis dependence" is usually a red herring.** Markets always have crises. Test whether the edge survives with crisis windows removed — if it keeps most of its profit, it's _crisis-enhanced_, not crisis-dependent.
- **In-sample cherry-picks are diagnostic only.** A filtered subset showing spectacular stats (huge win rate, huge R) is a hypothesis to test out-of-sample, NEVER a reason to risk more capital now.
- **The live forward test is the real gate, not more backtest variants.** When you feel the urge to run yet another variant, stop — you're entering multiple-testing territory. Collect live/out-of-sample data instead.
- **Beware the fantasy return.** If your backtest implies returns above the best funds in history (Medallion ~66%/yr), your cost model is too optimistic or you're overfit. Realistic retail systematic edges are single-to-low-double-digit %/yr after honest costs and prop-safe sizing.

---

## 2. Backtest fidelity (make the sim match reality)

The backtest is only useful if it can't do things the live bot can't.

- **No look-ahead.** Only use information available at decision time. Classic leak: computing an indicator (e.g. ATR, a level) using the breakout bar while entering intrabar. Recompute using the value the bot would actually have (e.g. compression-end ATR), and re-verify the edge survives.
- **Entry/exit realism.** Model the actual order type. A stop entry fills at or _worse_ than the trigger; a limit entry fills at or _better_. These have opposite slippage signs — don't model one as the other.
- **Honest costs.** Use real measured spread, not a hopeful constant. Model the round-trip. Add slippage, especially for strategies that trigger into volatility.
- **Gap and session handling.** Decide explicitly what happens over weekends/gaps. Don't let the backtest assume it can exit at a price that wasn't tradable.
- **The load-bearing component.** Ablate to find what actually carries the edge (remove each piece, see what breaks). Protect that piece; treat the rest as replaceable. Example finding: for a breakout edge, entering at the range EDGE was load-bearing — waiting for close-confirmation destroyed it.

---

## 3. MT5 / broker execution gotchas (bot-agnostic, all learned live)

These recur across brokers and strategies. Handle them from day one.

- **Filling mode matters and varies by broker.** A market order with the wrong `type_filling` fails with **retcode 10030 "Unsupported filling mode"**. Prefer `ORDER_FILLING_RETURN`; fall back to the symbol's `symbol_info(symbol).filling_mode`, then FOK/IOC. IOC in particular is often rejected. Query, don't assume.
- **Broker-side SL/TP closes carry `magic=0`.** When you reconcile a closed position from history, match by **`position_id`**, not by magic — or you'll never find TP/SL exits. Pull the true open time from `history_deals_get` and widen the search window.
- **Bounded retries + verify + alert.** If a close/send fails, retry a small bounded number of times (e.g. 5), then log ONE clear error and raise a visible alert. NEVER retry every poll forever — a persistent failure otherwise spams hundreds of identical log rows and hides the real problem. After sending a close, **verify the position is actually gone** by polling; don't trust the return code alone.
- **Restarts are dangerous while a position is open.** Reconstructing an in-flight trade after a restart tends to lose setup state (ranges default to 0, entry_time is wrong), which breaks downstream logic. Build orphan-recovery that backfills missing exits from broker history on startup — but still prefer to restart only when `open positions = 0` (check `Equity == Balance`; pending orders are safe to restart on).
- **Broker-side SL/TP is your safety net.** Even if your in-code close logic fails, a live SL/TP on the broker keeps the position bounded. Always set them; treat them as the last line of defense.
- **MT5 bars are bid-side.** This matters for measuring fills (see §4).

---

## 4. Live vs backtest measurement traps (subtle, easy to get wrong)

Getting the _measurement_ wrong makes a good edge look bad (or vice versa). These are the ones that fooled us:

- **Entry slippage with adaptive SL/TP is RR-neutral.** If you compute SL and TP from the ACTUAL fill, then where the entry filled doesn't change the risk/reward — pure entry slippage does NOT degrade net R. Don't panic over entry-slippage numbers.
- **Don't conflate slippage with spread.** With bid-side bars and STOP entries: a BUY_STOP at the bar's range-high fills on the ASK, so `actual - intended` = pure_slippage **+ spread**. A SELL_STOP fills on the bid, so its fill diff contains **no** entry spread. Decompose: `pure_slippage = total_fill_diff − spread_component` (spread_component = entry spread for buys, 0 for sells).
- **Don't double-count spread in net R.** If `gross_r` is computed from actual broker fills, it _already_ includes bid/ask spread. Subtracting another flat spread on top double-counts it. Keep two clearly-labeled numbers: `net_r_realized = gross_r` (true live PnL) and a separate conservative `net_r_vs_flat_spread` for backtest comparability only.
- **Slippage correlates with volatility.** Breakout/momentum strategies fire into volatility expansion — exactly when slippage is worst. Model exit slippage as volatility-correlated (stochastic, scaled by ATR), not a fixed constant. If the edge survives that, it's robust. EXIT slippage (TP/SL filling worse) is what actually degrades net R — entry slippage doesn't.
- **Log the raw components, compute derived fields downstream.** Store `intended`, `actual`, `real_spread_at_entry`, `real_spread_at_exit`, `atr`, exit price. Then any ratio (R, slippage, net) can be recomputed correctly later. If you only log a pre-combined "slippage" field, you can't un-mix it.

---

## 5. Sizing, drawdown, and prop-firm reality

- **1% risk is usually too aggressive for prop DD limits.** Run Monte Carlo on your own trade R-series: for many edges, 1% risk gives P(drawdown > 10%) ≈ 90%. For a 10% max-DD rule, realistic sizing is often **0.25-0.30%**.
- **Report drawdown as a distribution.** "Max DD" from one backtest path is meaningless. Bootstrap/Monte-Carlo the trade sequence and report the 95th-percentile worst DD.
- **Calmar (return/DD) > 1 is the target; DD > ~30% is the ruin zone.** Size so the worst-case DD stays inside the challenge's limit with margin.
- **Prop firms let you trade their capital** — proving the edge on demo then passing a challenge beats risking a large personal account. Don't skip demo to rush to live.

---

## 6. Instrumentation & the dashboard

- **A dashboard is only as honest as its row filter.** Performance metrics (win rate, expectancy, R) must be computed from **exit rows** (which carry R), matched to entry rows by ticket. If win rate reads 0% while real trades closed, it's reading entry rows.
- **Surface a CI-clears-zero indicator, not just an average.** With small n the average is noise; show the confidence interval and a "too early / need N trades" readiness flag.
- **Alert on divergence**: live vs backtest expectancy gap, stale log, failed closes, execution-quality drift. Make failures loud.
- **Separate execution quality from edge.** Track spread paid, pure slippage, and exit slippage as their own panels — they answer "is the edge tradable _here_," which is different from "does the edge exist."

---

## 7. Process (how to actually run a build)

1. **Audit before trusting a reported result.** Clone the repo fresh, read the actual code, reproduce the number. Reported results drift from code reality constantly.
2. **Diagnose from real data.** When something looks wrong, read the actual log/CSV, not what the code "should" produce. The truth is in the fills.
3. **Fix safety before features.** A broken close/flatten makes ALL live data unreliable — fix it before any new research. Order of operations: correctness of execution → correctness of logging → correctness of metrics → new edges.
4. **Brake illusory numbers out loud.** If a metric looks too good (fantasy returns, in-sample cherry-picks, small-sample win rates), say so plainly and explain the mechanism of the illusion. Enthusiasm is not evidence.
5. **Small, self-contained changes.** Each fix should be verifiable in isolation. State what changed, why, and how to validate it.
6. **Let it run.** The honest gate for any edge is 50-100 live/out-of-sample trades (often 2-4 months). Resist concluding from 5. Collect data; the market decides.

### Environment gotchas

- Windows PowerShell: `python`, not `python3`; commands on one line (no `\` continuation).
- Package installs in a restricted container may need `--break-system-packages`.
- Keep the live log gitignored (it's data, not code); read it directly when debugging.

---

## 8. The mindset to hold

- **Edges are real but modest.** Expect single-to-low-double-digit %/yr after honest costs and safe sizing, not the backtest's headline. Under-promise.
- **Fragility lives in execution.** A statistically real edge dies to exit slippage, wide spreads, and failed exits. Most of the work after finding an edge is making execution not kill it.
- **Small sample = no conclusion.** However good 5 trades look, the CI crosses zero. Discipline is refusing to believe your own good luck.
- **The skeptic's eye is the most valuable tool.** Verify, decompose, and question every number — especially the ones you want to be true. Good measurement beats a good story.
