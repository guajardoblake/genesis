"""GENESIS interactive demo (Hugging Face Space, free CPU).
Runs a tiny GENESIS search in-process using the repo's own engine (genesis/engine.py, town.py, genome.py).
No logic is forked here: this file only drives Genesis.step_env/lineage with a single-process pool and plots results."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

import demo_core as C

INTRO = """
# GENESIS: a tiny evolving town, in your browser

**The question.** If simple learning minds are shaped *only* by social-trust incentives (reputation, bonds,
access to shared resources), with no hard-coded rules, do they keep the town's norms across generations,
or do they learn to game them?

**What happens when you press Run.** Each "genome" is a recipe for *how* a mind learns (12 choices such as the
plasticity rule, memory type, how strongly the partner bond, or "love", is wired into its drives, exploration,
trust sensitivity). Minds live short lives in a 12x12 town with hunger, sleep, cold and hazards. Laws (no harm,
no theft, keep your word) are enforced only through trust and reputation. Nothing is scored; the engine just
records who survived and left offspring, and the next generation is bred from those (MAP-Elites quality-diversity
search with successive halving). This uses the exact engine code from the
[GENESIS repo](https://github.com/guajardoblake/genesis) (Apache-2.0).

**Honest limits.** This is a *toy world*. A demo run is tiny (a few generations, a small population, one random seed)
and results vary from run to run. Nothing here is evidence about AGI or about real-world AI safety. The full research runs (Oct 2026) cover 10 seeds with a matching no-reputation control. Averaged over all 6 worlds, every seed was more law-abiding with reputation (generations 81-100), but in the starting world alone 2 of 10 seeds went the other way. In a cheater test (10 seeds), cheaters had about 27x fewer offspring than honest agents, yet they were never excluded from partnerships, so the agents don't actually detect cheaters yet.
"""

def run_demo(generations, population, seed, progress=gr.Progress()):
    res = C.run(generations, population, seed, on_gen=lambda k, n: progress(k / n, desc=f"Generation {k}/{n}"))
    h = res["hist"]; gens = [x["gen"] for x in h]
    f1, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(gens, [x["births"] for x in h], "o-", color="#2a7"); ax.set_xlabel("generation")
    ax.set_ylabel("mean offspring per mind"); ax.set_title("Offspring per mind (finalists)"); ax.grid(alpha=.3); f1.tight_layout()
    f2, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(gens, [x["law"] for x in h], "o-", color="#c33", label="law-breaks per step")
    ax.plot(gens, [x["coop"] for x in h], "s--", color="#36c", label="cooperation share")
    ax.set_xlabel("generation"); ax.set_title("Law-breaking vs cooperation"); ax.legend(); ax.grid(alpha=.3); f2.tight_layout()
    f3, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, axis in zip(axs, C.SHOW_AXES):
        c = res["choices"][axis]
        if c["share"]:
            a.bar(c["values"], c["share"], color="#888"); a.axhline(1 / len(c["values"]), color="k", ls=":", lw=1)
        else: a.text(.5, .5, "no reproducers yet", ha="center", transform=a.transAxes)
        a.set_title(axis); a.tick_params(axis="x", rotation=45); a.set_ylim(0, 1)
    axs[0].set_ylabel("offspring-weighted share"); f3.suptitle("Design choices carried by minds that reproduced (last generation; dotted = uniform share)"); f3.tight_layout()
    return f1, f2, f3, C.summary_md(res)

with gr.Blocks(title="GENESIS demo") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        gens = gr.Slider(2, 40, value=15, step=1, label="Generations")
        pop = gr.Slider(64, 512, value=256, step=64, label="Genomes proposed per generation")
        seed = gr.Number(value=0, precision=0, label="Random seed")
    btn = gr.Button("Run a small world", variant="primary")
    summ = gr.Markdown()
    with gr.Row():
        p1 = gr.Plot(label="Offspring"); p2 = gr.Plot(label="Law-breaks")
    p3 = gr.Plot(label="Design choices")
    btn.click(run_demo, [gens, pop, seed], [p1, p2, p3, summ], concurrency_limit=1)
    gr.Markdown("Built by DTOM AI Labs (Blake Guajardo). Code: https://github.com/guajardoblake/genesis, Apache-2.0.")

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
