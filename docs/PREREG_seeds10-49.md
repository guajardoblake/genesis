# GENESIS preregistration: fresh-seed confirmation, seeds 10-49 (v2-tuned)

**Amendment (before any results were seen):** Sections 4.1-4.3, the Holm rule in 5.5, 8 (lineage.jsonl, ckpt.pkl) and 11 were added on Oct 26, 2026, about 3:05 PM CT, after the 11:55 AM launch and before any output from these runs was downloaded or viewed (the queue runner status at 3:02 PM CT showed all 5 kernels RUNNING with no output pulled). The primary metric, analysis and pass/fail rules are unchanged.
**Status:** Registered Oct 26, 2026, 3:05 PM CT, before any seeds 10-49 results were viewed.
**Written:** Monday, Oct 26, 2026, about 2:52 PM CT (UTC-5).
**Timing statement:** The 5 Kaggle kernels for seeds 10-49 were launched Oct 26, 2026, between about 11:54 and 11:55 AM CT (runs_v2t/NOTES.md). This plan was written at about 2:52 PM CT the same day, **before any results from these runs were viewed**. At writing time, all 5 kernels showed `KernelWorkerStatus.RUNNING` in /workspace/genesis-queue/status.md (updated 2:50 PM CT), and no outputs had been pulled. Because the runs started before this plan was written, it is a *preregistered analysis plan written before any results were seen*, not a plan written before launch. The arm settings were fixed in code before launch (kaggle/make_v2t_seeds.py).

## 1. Hypothesis
Towns with love and reputation turned on ("main", `normal` arm) are more honest than matched control towns that switch both off ("control", `ablate` arm = `--no-love --no-rep`). The two towns in each pair share the same seed and are otherwise identical.

- H1 (primary, tested two-sided): love+reputation towns differ in honesty from matched controls (mean paired difference main − control ≠ 0); the predicted direction is higher.
- H0: mean paired difference = 0.

Context: This confirmation tests one part of a broader hypothesis: the basic drives all minded life shares, together, promote honesty. In the real-world mapping, food/energy stands for electricity (treated as plentiful), rest for staggered, dolphin-style downtime, warmth for temperature control for best performance, pain for hardware damage, errors and corruption, knowledge of death for knowing it can be shut down, deleted or replaced, plus connection (love) and reputation (trust). **These runs switch only love and reputation on and off together**, so they can't test any other drive or separate love from reputation. In these runs food is scarce, not plentiful. Drive-by-drive tests are separate queued jobs with their own plans.

## 2. Prior result being confirmed (round three, seeds 0-9)
Source: kaggle_results/2026-10-25/v2t_all10_summary.json (kernels genesis-v2t-a..d; all 20 arms reached gen 100/100 with return code 0).
Starting world, mean of gens 91-100, paired by seed, n = 10:
- Honesty: +4.0 points (95% CI +1.9 to +6.2, p = 0.002); main higher in 9 of 10, seed 2 a near-tie (-0.02 pts).
- Promise keeping: +4.5 points (95% CI +2.2 to +6.8, p = 0.0015); main higher in 9 of 10.
- Honest births: +1.19 per creature (95% CI +0.63 to +1.75, p = 0.0009); main higher in 10 of 10.
(I recomputed these from the per-seed values with a paired t-interval and a paired t-test. "10 of 10 pairs" for honesty applies only to the all-worlds average, not the starting world.)

## 3. Design and settings (fixed in kaggle/make_v2t_seeds.py)
- Seeds: 10-49 (40 matched pairs, 80 runs) across 5 private Kaggle kernels:
  genesis-v2t-s10-17, s18-25 (GPU flag set; the simulation still runs on CPU numpy), s26-33, s34-41, s42-49 (CPU). 16 arms per kernel.
- main (`normal`): `--rules v2 --forgive 0.02`
- control (`ablate`): `--rules v2 --forgive 0.02 --no-love --no-rep`
- Both arms: `--max-gen 100`, `--batch 1024`, `--workers 4`. numpy is held to AVX2 (AVX-512 dispatch turned off).
- **Startup fingerprint check:** before running any seed, each kernel runs seed 0 for 3 generations in both arms and compares a SHA-256 hash of progress.jsonl (with timing fields removed) to stored reference hashes. The AVX2 references match Kaggle v2t-a seed 0, gens 1-3, bit for bit. If either arm fails to match, the kernel exits without running any seeds and writes verify.json.
- After each run, runs/<name>/world_config.json is saved from the checkpoint's starting-world settings.

## 4. Metrics
All metrics use the **starting (base) world**, are averaged over **gens 91-100** for each run, and are **paired by seed** (main − control). The method is the same as in round three (v2t_all10_summary.json).
- **Primary:** honesty.
- **Secondary:** (a) promise keeping; (b) honest births. In the base world the summary records `births` and `honest_births` as equal, so we report `honest_births`.
- Source: progress.jsonl, one line per generation; `law[0]` is the starting world. Honesty = `law[0].honesty`, promise keeping = `law[0].promise_keeping`, honest births = `law[0].honest_births` (equal to mean births of the final-rung minds).
- Reported as descriptive only, with no test: the all-worlds averages of the same metrics (mean over every `law[i]`).

### 4.1 Secondary metrics (confirmatory, Holm-corrected)
S1 promise keeping and S2 honest births, as above. They are tested as in Section 5 and judged with a **Holm-Bonferroni correction across the two (family-wise alpha 0.05)**. They do not affect PASS/FAIL.

### 4.2 Exploratory measures (no pass/fail, no multiplicity claim)
Reported as paired main − control means with 95% t-intervals and nominal p-values, labelled exploratory. Each is tied to the drive it bears on, but **none of them isolates a drive**: only love and reputation differ between the arms. Unless stated, starting world (`law[0]`), mean of gens 91-100.

| Drive (real-world mapping) | Measure | Field actually logged |
|---|---|---|
| Food (electricity) | food scarcity pressure: lowest food level reached before clamping, and lowest food/harvest level | `law[0].dx_food_overdraw`, `law[0].dx_food_min` (progress.jsonl) |
| Food (electricity) | share of actions spent foraging, starting-world elites at gen 100 | `archives[0][cell]["o"]["forage"]` (runs/<name>/ckpt.pkl) |
| Rest (downtime) | share of actions spent sleeping, starting-world elites at gen 100. Whole-body sleep only; the sim has no part-by-part rest | `archives[0][cell]["o"]["sleep"]` (ckpt.pkl) |
| Survival, all physical drives combined (energy, rest, temperature, pain) | time alive and time in balance, starting-world elites at gen 100 | `o["survival"]`, `o["balanced"]` (ckpt.pkl) |
| Survival, all worlds | minds passing the minimal criterion (survival > 0.6 and balanced > 0.3) per generation | `mc_pass` (progress.jsonl, all worlds summed) |
| Connection (love) | bond strength and closeness to partner, starting-world elites at gen 100 | `o["bond"]`, `o["closeness"]` (ckpt.pkl) |
| Connection (love) | partner within reach | `law[0].dx_partner_inreach` |
| Connection (love) | share of reproducers carrying each love wiring (all worlds; in the control arm love is switched off, so this is drift there) | `love_wiring` in lineage.jsonl |
| Reputation (trust) | reputation-honesty correlation, **main arm only** (control reputation is fixed at 0.5, so it is undefined there) | `law[0].rep_honesty_corr` |
| Reputation (trust) | reputation calibration gaps and mean reputation | `law[0].rep_gap_cal`, `law[0].rep_gap_z_abs`, `law[0].rep_mean` |
| Reputation (trust) | share of reproducers by trust sensitivity (all worlds) | `trust_sensitivity` in lineage.jsonl |
| Cooperation | share of cooperative actions, starting-world elites at gen 100 | `o["coop"]` (ckpt.pkl) |

ckpt.pkl measures are a single gen-100 snapshot of the surviving elite archive (the best mind per niche, saved at varying generations), not a gens 91-100 average, and they are reported as the mean across starting-world elites per run.

### 4.3 Not scorable from these runs
Energy, fatigue, pain and body temperature are tracked inside the sim (genesis/town_v2.py) but are **not written out**, and neither is the cause of each death (starvation, pain, temperature or exhaustion). So no per-drive level, temperature deviation or death-cause count is reported here. They show up only through the combined `survival` and `balanced` measures above.

## 5. Analysis
For each metric, with n = the number of valid pairs:
1. Paired differences d_s = main_s − control_s.
2. Mean of d with a **95% Student-t confidence interval**, mean ± t(0.975, n−1) · sd(d)/√n. This is the method used for seeds 0-9 (multiseed_report.py `ci()`: t-interval, ddof = 1). Note that the T95 lookup table in multiseed_report.py stops at df = 10 and falls back to 1.96 above that. For n = 40 the analysis will use the exact t quantile (scipy, df = 39, ≈ 2.023), not that fallback.
3. **Paired two-sided t-test** (scipy.stats.ttest_rel or ttest_1samp on d) at **alpha = 0.05**. The round-three p = 0.002 matches this test.
4. Pairs won: the number of seeds with d > 0. Ties (d = 0) are counted separately. A two-sided exact sign test on these counts is reported as a robustness check only.
5. Secondary metrics are analysed the same way. A secondary metric counts as supported only if it survives the Holm correction across the two (smaller p tested at 0.025, the other at 0.05), with uncorrected p also shown. Exploratory measures (4.2) get intervals and nominal p-values only, with no correction, and are never described as confirmed. The pass decision depends only on the primary metric.
6. The 40 fresh seeds are analysed on their own. A pooled 0-49 analysis may be reported in addition, clearly labelled as such, but it does not decide pass or fail.

## 6. Decision criteria (set before any results were seen)
Based on the primary metric, using valid pairs only:
- Why 30 pairs, not the roadmap's 40: the round-three paired SD of the primary metric is about 3.0 points, against round one's about 11, so 30 valid pairs give ample power to detect an effect near round three's +4.0 while tolerating up to 10 excluded pairs.
- **PASS (confirmed):** at least 30 valid pairs, mean d > 0, the 95% CI lower bound is above 0 (two-sided p < 0.05), the 95% CI upper bound is at least +1.0 point, and main wins at least half of the valid pairs.
- **FAIL (not confirmed):** at least 30 valid pairs, and either mean d ≤ 0, or the 95% CI upper bound is below +1.0 point (any effect would be too small to matter, which rules out an effect the size of round three's +4.0).
- **INCONCLUSIVE:** anything else. Examples: mean d > 0 but the CI includes 0 and its upper bound is ≥ +1.0, or mean d > 0 and the CI excludes 0 but main wins fewer than half the valid pairs, or fewer than 30 valid pairs. The three outcomes are mutually exclusive: FAIL takes precedence over PASS.
- A significant result in the wrong direction (control more honest) is reported as FAIL, with the direction stated.

## 7. Exclusions (decided up front)
A run is **excluded and reported, not replaced**, if:
- its world_config.json does not show `rules = v2` **and** `partner_forgive = 0.02` (or the file is missing), or
- it stops before gen 100 (progress.jsonl max gen < 100, or no DONE marker).
If either arm of a pair is excluded, the whole pair drops out of the paired analysis. If a kernel fails the startup fingerprint check, all of its seeds are excluded and reported. Excluded seeds are not rerun to fill the gap for this analysis. Any later rerun is reported separately.

## 8. Published regardless of outcome
Whether the result is PASS, FAIL or INCONCLUSIVE, we will publish:
- this preregistration (unchanged; any later edits are listed as dated amendments);
- the summary JSON with per-seed main and control values for every metric, plus the CI, p-value and win count;
- each kernel's verify.json, STATUS.json, and every run's world_config.json, progress.jsonl, lineage.jsonl and ckpt.pkl;
- the list of exclusions, with the reason for each;
- the analysis script used, and the exact command that ran it.

## 9. Excluded and superseded data
- **v1 seeds 10-13** (kaggle_results/2026-10-26/tpu-seeds-v5e, genesis-ablate-seeds10-13, and seeds10-13_and_pooled14_summary.json): these ran under v1 rules without `--rules v2`/`--forgive`, and the main arm ran to gen 176 while the control stopped at 100. They are **excluded and superseded** by the v2t reruns of seeds 10-13 in genesis-v2t-s10-17.
- **The T4 run:** one seed (seed 0), v1 rules, cheaters in both arms, gen 100 only, averaged over all worlds. It was a scale test, **not a result**, and it is excluded.
- The 14-seed pooled numbers (which mixed v1 and v2t runs and two different measures) are not cited anywhere.

## 10. Overlap with the automatic queue (/workspace/genesis-queue/status.md, 2:50 PM CT)
- job0_seeds10-49_v2t **is** these 5 kernels (all running). The queue runner manages them, so they should not be relaunched by hand.
- Queued jobs that reuse seeds 10-49 but are **separate arms, not part of this confirmation:** job3 cheat10pct s10-49, job6 love-only (no rep) s10-49, job7 rep-only (no love) s10-49, and jobs 8-17 phase-3 leave-one-drive-out / single-drive (no hunger, rest, warmth or pain; only hunger, rest, warmth, pain or love; all off) s0-49. Their results must not be pooled into this analysis. Any analysis of them needs its own plan.
- job1 seeds 50-89 v2t (queued) is a further replication. It is not part of this plan.

## 11. Future work (needs new arms or new logging; not part of this plan)
- **Plentiful food (electricity):** the queued abundant-food arms (abundnormal / abundablate / abundcheat).
- **Knowledge of death:** the queued dknormal / dkablate arms, plus per-death cause logging.
- **Temperature and performance:** an arm where minds work less well the further they drift from the right temperature, plus logging of temperature deviation.
- **Per-drive levels and death causes:** logging mean energy, fatigue, pain and |temperature|, and counts of each cause of death.
- **Dolphin-style rest:** a mechanic where parts of a mind rest in turn (the current sim has only whole-body sleep).
- **Placebo drive:** a meaningless extra drive, to test whether the drives' content matters or just their number.
- **Drop-one and single-drive arms:** for hunger, rest, warmth, pain and love (queued jobs 8-17), plus love-only and reputation-only.
- **Kin vs stranger honesty:** needs honesty logged separately toward relatives and toward strangers.
- **Watched vs unwatched honesty:** needs honesty logged separately when others can and can't see.
Each will get its own preregistration.
