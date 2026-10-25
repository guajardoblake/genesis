# GENESIS — open-ended search for how minds should learn

## My core idea (thesis)
**My thesis:** I believe a general AI can't exist on intelligence alone. It has to be built on the fundamentals that all minded life shares, but only the non-exclusive ones: traits found in every species that shows a mind of its own, not traits that belong to one species. Love, meaning the built-in pull to care for and bond with another, is one of those traits, and I believe it has to be present together with all the others, not instead of them. The first set of GENESIS runs is designed to test that by comparing towns where love is switched on against otherwise identical towns where it's switched off.

**In plain words:** every animal that seems to have a mind of its own needs to eat, rest, stay warm, avoid pain, and bond with others. No single species owns these, so I call them non-exclusive traits. My bet is that a real general mind needs all of them at once, love included, and my first experiments check what changes when the love incentive is taken away.

**What the first runs can and can't show:** every GENESIS creature lives with the shared basics (hunger, sleep, warmth, pain, and the drives to stay balanced and to reproduce). The main towns add the love incentive; the control towns (the ablation) remove it. In this first set the control also switches off reputation at the same time, so these runs test love and reputation together; a love-only control is the next run. This is a small simulated town, so results here are evidence about this town, not proof about AGI.

**An early discovery:** My first results point to an early discovery. In all 10 matched pairs of towns, the towns that had love and reputation incentives ended up more honest than identical towns without them (89.5% vs 83.0%, generations 81-100, all worlds), and cheaters raised about 27x fewer offspring (0.020 vs 0.541 per creature). In this simulated world, minds raised with love and reputation incentives stayed more trustworthy. That's early evidence for my thesis, not proof about real AI. Honest limits: the control towns switched off love and reputation together, so I can't yet say love alone did it; on the stricter starting-world measure the gain (+7.4 points) can't rule out zero with only 10 seeds; and part of the cheater gap is built into how the cheaters are made. Next I'm testing love alone and scaling up.

## The idea in plain words
We do not design a finished mind. Each **genome is a recipe for how a mind learns**. Thousands of minds,
each born with random weights, live together in a small town and learn during their own life using the rules
in their genome. Minds that stay alive, stay balanced and bond well have offspring; their recipes (and, optionally,
what they learned) are carried forward. Over many generations we watch which recipe choices keep surviving.
That list is the "most likely answer remaining" (`REPORT.md`).

## Blake's fixed rules, and where they live in code (`genesis/town.py`)
- **Two drives only.** *Balance*: a hidden, unlabeled number built from body deviations + the mind's own surprise
  − love pull, minus a private set-point; the mind just sees a number. *Reproduction*: births happen only when a
  bonded, fed, balanced pair is close together.
- **No emotion labels.** Love = `bond × closeness`, a hidden pull that feeds balance and/or reproduction depending on the genome's `love_wiring`.
- **Inner sense.** Every step each mind gets 6 signals (own bond, safety, closeness; partner's bond, safety, closeness)
  and must *predict them* with its own learned predictor. The prediction error is its private surprise (free-energy term),
  used only inside the mind (balance + optional learning signal), never as an outside grade.
- **Real physiology every run.** Hunger (energy), sleep (fatigue), pain, temperature with day/night cold; any can kill.
- **Town laws** (don't harm, don't steal, keep your word) are enforced only by trust: breaking them drops reputation,
  breaks bonds, and a broken promise to a partner costs double. Trusted minds harvest more (resources), build tools
  (institutions) and get faster harvests (territory/tools). Nothing forbids lawbreaking — it is just costly.
- **No internet.** The engine has no network code at all.

## Genome design space: 276,480,000 distinct recipes
| axis | choices | n |
|---|---|---|
| plasticity_rule | hebb, oja, bcm, anti_hebb, eligibility_hebb, predictive_delta | 6 |
| learning_rate | 0.001 … 0.2 | 8 |
| memory_type | none, leaky_trace, fast_weights, episodic_slots | 4 |
| memory_decay | 0.5 … 0.99 | 6 |
| prediction_horizon | 1, 2, 4, 8 steps | 4 |
| hidden_units | 4, 8, 12, 16 | 4 |
| surprise_gain (free-energy into balance) | 0 … 2 | 6 |
| love_wiring | none, to_balance, to_reproduction, to_both, to_both_strong | 5 |
| partner_pred_weight | 0 … 2 | 5 |
| exploration | 0 … 0.4 | 5 |
| trust_sensitivity | 0 … 2 | 5 |
| neuromod_source (what gates learning) | constant, surprise_drop, balance_gain, bond_gain | 4 |

6·8·4·6·4·4·6·5·5·5·5·4 = **276,480,000** (27.6× the 10M minimum).

## The search loop (`genesis/engine.py`)
1. **Adaptive scheduler** picks parents from archive niches weighted by recent improvement × offspring, mutates/crosses them.
2. **Surrogate pre-screen**: a Bayesian linear model (one-hot + pairwise features) scores 250k random fresh genomes per
   world per generation and keeps the best by upper-confidence (promise + uncertainty).
3. **Dedup cache**: recipes already lived in that world are skipped.
4. **Successive halving (Hyperband-style)**: rung0 = 32-step life, 2 minds/genome; top third → rung1 (96 steps, 4 minds);
   top third → rung2 (192 steps, 6 minds). Recipes that cannot stay alive and balanced are dropped early (early termination);
   a town that collapses below 5% alive stops simulating.
5. **Selection = nature, not a score.** Only minds passing the minimal criterion (alive >60% of life, balanced >30% of it)
   enter the archive; inside a niche, the one that left more offspring stays.
6. **Quality-diversity + novelty**: MAP-Elites grid of 8×8×8 niches over behaviour (cooperation, closeness to partner,
   law-breaking rate); novelty (kNN distance to past behaviours) breaks ties toward new behaviour.
7. **Lamarckian warm start** (toggle `--no-lamarck`): children start from the parent niche's learned weights + noise.
8. **POET world co-evolution**: every 3 gens a world whose minds thrive spawns a harder child (less food, colder, more hazard);
   kept only if 5–90% of transferred minds pass the criterion. Every 4 gens the best minds try every other world (minimal-criterion transfer).
9. **Lineage log** (`lineage.jsonl`): offspring-weighted share of every axis value among reproducers, each generation.
10. **Checkpoint/resume**: `ckpt.pkl` written atomically every generation; re-running the same command resumes.

## Compute (free only)
- **Box**: `python3 run_box.py --minutes 30 --out runs/box` (multiprocessing pool, small shards = work-stealing).
- **Kaggle**: `python3 make_kaggle.py` → `kaggle/genesis_kaggle.py` / `.ipynb` (self-contained). Each session = an island with
  its own seed; download `runs/kaggle_*/` and `python3 merge_islands.py runs/box runs/kaggle_101`.
- **Galaxy S25+**: `workers/phone_worker.py` checks the phone headlessly and pauses if hot (>42 °C), battery low/unreadable, or no Python runtime.
- `python3 bench.py` measures throughput; `python3 report.py runs/box` writes `REPORT.md`.

## Honest limits
- Simulator is numpy on CPU. A JAX/GPU port is not written yet, so Kaggle GPUs add nothing; Kaggle sessions contribute CPU islands only.
- Float16/int8 not used (in-life learning updates are too small for float16 to be safe); everything is float32.
- Phone worker is a guard only: the phone has no Python and the app cannot read battery, so it always reports PAUSED.
- Kaggle islands merge at the end (no live work-stealing across the internet).
- The town is a small abstraction (12×12 food grid, ≤192 steps). Surviving recipes here are evidence about this town, not proof of anything about AGI.
- Fast throughput partly comes from early termination when a town collapses; that is real but means many lives are short.
