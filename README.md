# GENESIS

## My core idea (thesis)
**My thesis:** I believe a general AI can't exist on intelligence alone. It has to be built on the fundamentals that all minded life shares, but only the non-exclusive ones: traits found in every species that shows a mind of its own, not traits that belong to one species. Love, meaning the built-in pull to care for and bond with another, is one of those traits, and I believe it has to be present together with all the others, not instead of them. The first set of GENESIS runs is designed to test that by comparing towns where love is switched on against otherwise identical towns where it's switched off.

**In plain words:** every animal that seems to have a mind of its own needs to eat, rest, stay warm, avoid pain, and bond with others. No single species owns these, so I call them non-exclusive traits. My bet is that a real general mind needs all of them at once, love included, and my first experiments check what changes when the love incentive is taken away.

**What the first runs can and can't show:** every GENESIS creature lives with the shared basics (hunger, sleep, warmth, pain, and the drives to stay balanced and to reproduce). The main towns add the love incentive; the control towns (the ablation) remove it. In this first set the control also switches off reputation at the same time, so these runs test love and reputation together; a love-only control is the next run. This is a small simulated town, so results here are evidence about this town, not proof about AGI.

Open-ended evolutionary search for **how minds should learn**, by DTOM AI Labs.

## In plain words
GENESIS does not design a finished mind. Each genome is a *recipe for how a mind learns*. Thousands of small minds,
each born with random weights, live together in a simulated town and learn during their own lives using the rules in
their genome. They have only two hidden drives — **balance** (keeping body and inner state steady, including their own
prediction surprise) and **reproduction**. Hunger, sleep, pain and temperature are simulated in every run. There is no
outside score: minds that stay alive, stay balanced and bond well have offspring, and their recipes carry forward.
Over many generations GENESIS records which recipe choices keep surviving.

Methods combined: meta-learned plasticity rules, MAP-Elites quality-diversity, novelty search, POET-style world
co-evolution, a Bayesian surrogate pre-screen, and successive halving. Full design: [DESIGN.md](DESIGN.md).

## Honest status (October 2026)
- Early research code. The simulator is **numpy on CPU**. A **JAX port (for TPU/GPU) is planned, not written**.
- The phone worker (`workers/phone_worker.py`) is a safety guard only and currently contributes no compute.
- The town is a small abstraction (12×12 food grid, ≤192 steps per life). Results are evidence about this town, not proof about general intelligence.

## Measured numbers (one 8-core CPU box, from `bench_result.json`)
| measure | value |
|---|---|
| Genome design space | 276,480,000 distinct recipes |
| Short trial lives (rung 0), 7 workers | ~47.0 million genomes/hour |
| Full lives, 1 core (baseline) | ~0.96 million genomes/hour |
| Triage speed-up vs. baseline | ~14× |
| Surrogate screens | ~1.16 billion/hour |

First box run (in progress at publication): generation 76 after ~13 minutes, 532,517 genomes simulated,
98.75 million screened by the surrogate, 6 co-evolving worlds.

## How to run
Requires Python 3.10+ and numpy.
```bash
pip install numpy
python3 run_box.py --minutes 30 --out runs/box   # resumes from runs/box/ckpt.pkl if present
python3 report.py runs/box                       # writes REPORT.md (surviving recipe features)
python3 bench.py                                 # measures throughput
python3 make_kaggle.py                           # builds a self-contained Kaggle notebook (CPU island)
python3 merge_islands.py runs/box runs/kaggle_101
```
Flags: `--no-lamarck` (no inherited learned weights), `--no-speed` (baseline without surrogate/halving/cache).

## License
Apache-2.0 — Copyright 2026 Blake Guajardo / DTOM AI Labs. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
