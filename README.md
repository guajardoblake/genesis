# GENESIS

## Why I built GENESIS
As a father of two boys, I'm doing this to help secure a safe future for our children. I think AGI is in our near future, and I want the next generation to grow up with AI they can trust. I don't believe a general AI can be built on intelligence alone. The many species that show a mind of their own share a handful of basics: they need food, rest and warmth, they avoid pain, and they bond with others. For an AI, I think these have natural counterparts: food becomes energy and compute, rest becomes sleep that works like a dolphin's, where one part recovers and sorts its memory while the rest keeps working, warmth becomes a safe operating range, pain becomes damage and error signals, and bonding stays bonding. No single species owns these traits, so I call them non-exclusive, and I believe a real general mind needs all of them together. Love, the built-in pull to care for and bond with another, is the one I think keeps a mind trustworthy.

GENESIS is my test of that idea. It's a simulated town where evolving minds live with all of these shared basics in simple form: they eat food on a small grid to keep their energy up, rest to recover from fatigue, stay warm through cold nights, and avoid hazards that cause pain. Some towns get love and reputation incentives, and matched twin towns are identical except without them. Then I watch which minds become honest.

The first results lean my way. In all 10 matched pairs, the town with love and reputation ended up more honest than its twin without them (89.5% vs 83.0% over generations 81-100, averaged across all worlds). Planted cheaters raised about 27 times fewer offspring, though part of that gap comes from how the cheaters are built. The towns without love and reputation got more honest too, so what I'm measuring is the extra gain love and reputation add on top.

This is a first step in a small simulated world, not proof about real AI. Three things come next. First, separate love from reputation, since this round switched them off together. Second, run many more seeds: on the strictest measure the gain is +7.4 points, and 10 seeds can't yet rule out zero. Third, scale the towns up.

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
