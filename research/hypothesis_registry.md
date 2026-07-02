# Hypothesis Registry

- 2026-07-02: Track B compression quality audit. Hypothesis: longer compression duration and tighter range/ATR predict higher MFE and net R. Pre-registered tercile buckets; no threshold optimization; analysis-only for FDR accounting.
- 2026-07-02: H-2026-EXIT-01 live shadow validation phase started; decision on exit switch deferred until live gate (50-100 trades) with realized-A vs shadow-C comparison.
- 2026-07-02: H-2026-REV-01 registered. Failed-breakout reversal hypothesis: validated compression trades that hit 1R stop become opposite-direction next-bar-open signals with 1R SL, 1.5R TP, 10-bar force close, $0.20 spread; no parameter tuning.
- 2026-07-02: H-2026-REV-01 result: FAIL under pre-registered gates; reversal does not clear train/test net-of-cost CI and/or does not beat required controls. Hypothesis closed unless re-registered with new data.
- 2026-07-02: H-2026-SESS-01 registered. Session-conditional compression quality hypothesis: ASIA 23:00-06:59 UTC should have lower net R than LONDON_ONLY/OVERLAP; pre-registered four UTC buckets; no boundary tuning; future filter candidate only if a bucket is significantly negative in both train and test.
- 2026-07-02: H-2026-SESS-01 result: FAIL_FILTER_RULE no session has significantly negative net R in both train and test; trade all sessions under current evidence.
- 2026-07-02: H-2026-EXIT-01 registered. Compression trailing exit hypothesis: Config C initial 1ATR stop, arm at closed-bar +1R, trail 1ATR behind best closed-bar favorable extreme, no fixed TP, 10-bar force close; pass only if C beats A in train/test, C CI clears zero in train/test, and maxDD is not >25% worse than A.
- 2026-07-02: H-2026-EXIT-01 result: FAIL: C does not satisfy all pre-registered gates. C_train=-0.1337 vs A_train=0.1968; C_test=-0.1044 vs A_test=0.2633; C_CI_train=[-0.2213,-0.0422], C_CI_test=[-0.2002,-0.0011]; DD_train C/A=280.27/40.12, DD_test C/A=117.93/23.98
- 2026-07-02: H-2026-TF-01 registered. Cross-timeframe replication/cost-rescue audit: unchanged validated compression spec on resampled H1/H4 XAU/XAG; fixed costs; no threshold tuning.
- 2026-07-02: H-2026-TF-01 result: SILVER_RESCUE=FAIL (XAG_H1=FAIL, XAG_H4=FAIL_OR_DESCRIPTIVE); GOLD_H1_SECOND_STREAM=FAIL (overlap=no_h1_trades, daily_corr=no_h1_trades).
- 2026-07-02: V-2026-PARITY-01 registered. Engineering verification, not a new hypothesis: replay live bot compression decision logic over historical XAUUSD M15 and compare to validated research pipeline.
- 2026-07-02: V-2026-PARITY-01 result: FAIL; bot_logic_train=-1.0241 [-1.0612,-0.9876], test=-0.8774 [-0.9213,-0.8369].
