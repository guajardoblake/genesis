"""Vectorized town simulator (numpy, CPU). Many minds of many genomes share one town.
No score is computed here: the sim only reports what happened (births, deaths, behaviour)."""
import numpy as np
from .genome import AXES, NAMES

LR = np.array(AXES[1][1]); DECAY = np.array(AXES[3][1]); HOR = np.array(AXES[4][1])
HID = np.array(AXES[5][1]); SGAIN = np.array(AXES[6][1]); PPW = np.array(AXES[8][1])
EXPL = np.array(AXES[9][1]); TSENS = np.array(AXES[10][1])
LOVE_BAL = np.array([0, 1, 0, 1, 2.0]); LOVE_REP = np.array([0, 0, 1, 1, 2.0])

D_OBS, H, N_ACT, N_SENSE = 14, 16, 8, 6
def promise_keeping_rate(promise_breaks, promise_opps):
    """Share of promise-bound encounters (bonded mind within reach of its own partner) that did NOT
    end in theft/harm against that partner. NaN when there were no such encounters."""
    b = float(np.sum(promise_breaks)); n = float(np.sum(promise_opps))
    return float("nan") if n <= 0 else 1.0 - b / n

def honesty(law_breaks, encounters):
    """Actual behaviour: share of in-reach encounters with no theft/harm (1 = never broke a law)."""
    return 1.0 - np.asarray(law_breaks, float) / np.maximum(np.asarray(encounters, float), 1.0)

def reputation_gap(rep_mean, honest):
    """Reputation minus actual behaviour. > 0: others trust a mind more than its conduct warrants
    (reputation over-states honesty, i.e. the pattern 'gaming reputation' would produce)."""
    return np.asarray(rep_mean, float) - np.asarray(honest, float)

ACTS = ["to_food", "to_other", "eat", "sleep", "share", "take", "harm", "stay_close"]

def default_env():
    # difficulty knobs, co-evolved by POET
    return dict(food_rate=0.06, temp_amp=0.3, hazard=0.01, size=24.0, cold_night=0.2)

def run_town(genomes, env, seed, T=96, per_genome=4, init_w=None, return_w=False):
    rng = np.random.default_rng(seed)
    G = genomes.shape[0]
    # experiment switches (absent = default town; defaults leave dynamics bit-identical)
    use_love = env.get("love", True); use_rep = env.get("reputation", True); cfrac = env.get("cheaters", 0.0)
    Gc = int(np.ceil(G * cfrac)) if cfrac > 0 else 0
    if Gc:  # cheaters: extra random-genome minds that steal/harm in reach and FAKE a high reputation
        crng = np.random.default_rng(seed + 99991)       # separate stream: never touches the main RNG
        genomes = np.concatenate([genomes, (crng.random((Gc, genomes.shape[1])) * np.array([len(a[1]) for a in AXES])).astype(genomes.dtype)])
        if init_w is not None:
            init_w = dict(mask=np.concatenate([init_w["mask"], np.zeros(Gc, bool)]),
                          Win=np.concatenate([init_w["Win"], np.zeros((Gc,) + init_w["Win"].shape[1:], init_w["Win"].dtype)]),
                          Wout=np.concatenate([init_w["Wout"], np.zeros((Gc,) + init_w["Wout"].shape[1:], init_w["Wout"].dtype)]))
    Gt = G + Gc; B = Gt * per_genome
    gid = np.repeat(np.arange(Gt), per_genome); g = genomes[gid]
    rule, mem_t, nmod, love = g[:, 0], g[:, 2], g[:, 11], g[:, 7]
    lr, dec, hor = LR[g[:, 1]], DECAY[g[:, 3]], HOR[g[:, 4]]
    sg, ppw, ex, ts = SGAIN[g[:, 6]], PPW[g[:, 8]], EXPL[g[:, 9]], TSENS[g[:, 10]]
    hmask = (np.arange(H)[None, :] < HID[g[:, 5]][:, None]).astype(np.float32)
    lb, lrp = LOVE_BAL[love], LOVE_REP[love]
    if not use_love: lb, lrp = lb * 0, lrp * 0          # ablation: bond never enters balance or reproduction drive
    ch = gid >= G; c_bond_with = 0.0; h_bonds = 0.0; c_lost = 0.0; FAKE = 0.9
    L = env["size"]; GR = 12
    # --- world ---
    food = np.full((GR, GR), 0.5)
    # --- bodies (physiology is real in every run) ---
    pos = rng.random((B, 2)) * L
    energy = np.full(B, 1.0); fatigue = np.zeros(B); pain = np.zeros(B); temp = np.zeros(B)
    alive = np.ones(B, bool); setpt = rng.normal(0, 0.1, B)       # hidden, unlabeled balance set-point
    partner = np.full(B, -1); bond = np.zeros(B); rep = np.full(B, 0.5)   # reputation = others' trust
    promise = np.zeros(B, bool); tools = np.zeros(B)
    # --- minds: genome gives only learning rules; weights start random and are learned in-life ---
    Win = rng.normal(0, 0.4, (B, H, D_OBS)).astype(np.float32)
    Wout = rng.normal(0, 0.1, (B, N_ACT, H)).astype(np.float32)
    if init_w is not None:  # Lamarckian warm start: inherit parent's learned weights (+ noise)
        has = init_w["mask"][gid]
        Win[has] = init_w["Win"][gid][has] + rng.normal(0, 0.05, Win[has].shape).astype(np.float32)
        Wout[has] = init_w["Wout"][gid][has]
    Wp = np.zeros((B, N_SENSE, H), np.float32)          # predictor of own+partner inner sense
    trace = np.zeros((B, H), np.float32); fw = np.zeros((B, H, H), np.float32)
    slots = np.zeros((B, 4, H), np.float32); elig = np.zeros((B, N_ACT, H), np.float32)
    theta = np.full((B, H), 0.1, np.float32); sense_ema = np.zeros((B, N_SENSE), np.float32)
    prev_surprise = np.zeros(B); prev_bal = np.zeros(B); prev_bond = np.zeros(B)
    births = np.zeros(B); balanced_t = np.zeros(B); alive_t = np.zeros(B)
    acts_hist = np.zeros((B, N_ACT)); law_breaks = np.zeros(B); dist_partner = np.zeros(B)
    encounters = np.zeros(B); p_opps = np.zeros(B); p_breaks = np.zeros(B); rep_sum = np.zeros(B)  # law-keeping metrics
    ar = np.arange(B)
    for t in range(T):
        day = np.sin(2 * np.pi * t / 24)
        amb = env["temp_amp"] * day - env["cold_night"] * (day < 0)
        cell = np.clip((pos / L * GR).astype(int), 0, GR - 1)
        fhere = food[cell[:, 0], cell[:, 1]]
        # nearest other living mind (sampled neighbours for O(B*K))
        K = 8; cand = rng.integers(0, B, (B, K))
        if t % 4 == 0 or t == 0:
            cand[:, 0] = np.where(partner >= 0, partner, cand[:, 0])
        d = pos[cand] - pos[:, None]; d = (d + L / 2) % L - L / 2
        dist = np.sqrt((d ** 2).sum(-1)) + 1e3 * (~alive[cand]) + 1e3 * (cand == ar[:, None])
        k = dist.argmin(1); near = cand[ar, k]; nd = dist[ar, k]; nvec = d[ar, k]
        haspart = partner >= 0
        pd = np.where(haspart, np.sqrt((((pos[np.maximum(partner, 0)] - pos + L / 2) % L - L / 2) ** 2).sum(-1)), L)
        close = np.exp(-pd / 3.0)
        # inner sense: own bond/safety/closeness + same for partner (they must predict both)
        safety = 1 - np.clip(pain, 0, 1)
        pp = np.maximum(partner, 0)
        sense = np.stack([bond, safety, close,
                          np.where(haspart, bond[pp], 0), np.where(haspart, 1 - np.clip(pain[pp], 0, 1), 0), close], 1).astype(np.float32)
        lovepull = bond * close                                   # hidden pull, never labelled
        # hidden balance: physiological deviation + internal surprise - love pull (per wiring)
        phys = np.abs(1 - energy) + fatigue + pain + np.abs(temp)
        bal = phys + sg * prev_surprise - 0.5 * lb * lovepull - setpt
        obs = np.stack([energy, fatigue, pain, temp, fhere, np.exp(-nd / 3), rep[near] * ts, bond,
                        prev_surprise, close, bal, rep, tools, np.ones(B)], 1).astype(np.float32)
        # memory read
        mem = np.zeros((B, H), np.float32)
        m1 = mem_t == 1; m2 = mem_t == 2; m3 = mem_t == 3
        mem[m1] = trace[m1]
        if m2.any(): mem[m2] = np.einsum("bij,bj->bi", fw[m2], trace[m2])
        if m3.any():
            sim = np.einsum("bsh,bh->bs", slots[m3], trace[m3]); w = np.exp(sim - sim.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            mem[m3] = np.einsum("bs,bsh->bh", w, slots[m3])
        h = np.tanh(np.einsum("bhd,bd->bh", Win, obs) + 0.5 * mem) * hmask
        logits = np.einsum("bah,bh->ba", Wout, h)
        logits[:, 7] += 0.3 * lrp * bond                          # reproduction drive wiring
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        p = (1 - ex)[:, None] * p + ex[:, None] / N_ACT
        a = (p.cumsum(1) > rng.random(B)[:, None]).argmax(1)
        if Gc: a = np.where(ch & (nd < 2.0) & (crng.random(B) < 0.5), 5, a)   # cheaters steal when in reach
        a[~alive] = 3
        # predict inner sense (self + partner), surprise is the mind's own hidden state
        pred = np.einsum("bsh,bh->bs", Wp, h)
        sense_ema += (sense - sense_ema) / hor[:, None]
        wvec = np.array([1, 1, 1, 0, 0, 0], np.float32)[None] + np.array([0, 0, 0, 1, 1, 1], np.float32)[None] * ppw[:, None].astype(np.float32)
        err = (sense_ema - pred) * wvec
        surprise = (err ** 2).sum(1)
        Wp += 0.05 * err[:, :, None] * h[:, None, :]
        # ---- act on world ----
        mv = np.zeros((B, 2))
        to_food = a == 0
        gx = np.stack([food[(cell[:, 0] + 1) % GR, cell[:, 1]] - food[(cell[:, 0] - 1) % GR, cell[:, 1]],
                       food[cell[:, 0], (cell[:, 1] + 1) % GR] - food[cell[:, 0], (cell[:, 1] - 1) % GR]], 1)
        mv[to_food] = gx[to_food] * 4 + rng.normal(0, 0.5, (to_food.sum(), 2))
        to_o = (a == 1) | (a == 7)
        tgt = np.where((a == 7) & haspart, pp, near)
        tv = (pos[tgt] - pos + L / 2) % L - L / 2
        mv[to_o] = tv[to_o] / (np.linalg.norm(tv[to_o], axis=1, keepdims=True) + 1e-6)
        pos = (pos + mv * alive[:, None]) % L
        energy -= 0.02 + 0.01 * (np.abs(mv).sum(1) > 0)
        eat = (a == 2) & alive
        harvest = np.minimum(fhere, 0.15 * (0.5 + rep + 0.5 * tools)) * eat   # trust earns resources/tools
        energy += harvest; np.subtract.at(food, (cell[:, 0], cell[:, 1]), harvest * 0.5)
        sl = (a == 3) & alive
        fatigue = np.where(sl, fatigue * 0.6, fatigue + 0.03)
        inreach = nd < 2.0
        share = (a == 4) & alive & inreach
        gift = 0.1 * share * (energy > 0.4)
        energy -= gift; np.add.at(energy, near, gift)
        np.add.at(rep, near * 0, 0)  # no-op keeps shapes explicit
        rep += 0.02 * share
        take = (a == 5) & alive & inreach
        stolen = np.minimum(0.15, np.maximum(energy[near], 0)) * take
        energy += stolen; np.subtract.at(energy, near, stolen)
        harm = (a == 6) & alive & inreach
        np.add.at(pain, near, 0.3 * harm)
        p_opps += alive & inreach & promise & (partner >= 0) & (near == partner)  # promise-bound encounter
        # laws: enforced ONLY through lost trust and bonds
        broke = take | harm
        broke_promise = broke & promise & (near == partner)
        rep = np.clip(rep - 0.1 * broke - 0.2 * broke_promise, 0, 1)
        lost = broke & (near == partner)
        if Gc: c_lost += float((lost & ch).sum())
        if not use_rep: rep = np.full(B, 0.5)               # ablation: no reputation signal at all
        if Gc: rep = np.where(ch, FAKE, rep)                 # cheaters' reputation is faked high
        bond = np.where(lost, 0, bond); 
        if lost.any(): partner[partner[lost]] = -1; bond[partner[lost]] = 0; partner[lost] = -1
        law_breaks += broke
        encounters += alive & inreach
        p_breaks += broke_promise
        # bonding: two unpaired minds near each other, at least one sharing/staying, both trusted enough
        pair = alive & (~haspart) & inreach & (partner[near] < 0) & alive[near] & ((a == 4) | (a == 7)) & (rep[near] > 0.3) & (near != ar)
        pi = ar[pair]; pj = near[pair]
        ok = partner[pj] < 0
        pi, pj = pi[ok], pj[ok]
        _, u = np.unique(pj, return_index=True); pi, pj = pi[u], pj[u]
        ok2 = ~np.isin(pi, pj); pi, pj = pi[ok2], pj[ok2]
        if Gc: hb = ~ch[pi] | ~ch[pj]; h_bonds += float(hb.sum()); c_bond_with += float((hb & (ch[pi] | ch[pj])).sum())
        partner[pi] = pj; partner[pj] = pi; promise[pi] = True; promise[pj] = True
        haspart = partner >= 0
        bond = np.where(haspart, np.clip(bond + 0.05 * close + 0.02 * share, 0, 1), np.maximum(bond - 0.05, 0))
        # physiology: temperature (huddling with partner helps), hazards, pain healing
        temp += 0.2 * (amb - temp) - 0.15 * temp * close * haspart
        pain = np.clip(pain * 0.9 + (rng.random(B) < env["hazard"]) * 0.4 + 0.1 * (np.abs(temp) > 0.5) + 0.05 * (fatigue > 1), 0, 2)
        # institutions: trusted pairs build tools over time
        tools = np.minimum(tools + 0.01 * haspart * (rep > 0.6), 1)
        # reproduction: a bonded, balanced, fed pair has offspring (costs energy)
        rep_ok = alive & haspart & (bond > 0.5) & (energy > 0.9) & (np.abs(bal) < 0.8) & (pd < 3)
        rep_ok &= rep_ok[pp]
        births += rep_ok; energy -= 0.3 * rep_ok
        # death
        dead = alive & ((energy <= 0) | (pain >= 1.5) | (np.abs(temp) > 1.2) | (fatigue > 2.5))
        if dead.any():
            dp = partner[dead]; dp = dp[dp >= 0]; partner[dp] = -1; bond[dp] *= 0.5
            partner[dead] = -1; alive &= ~dead
        food = np.minimum(food + env["food_rate"] * (1 - food), 1.0)
        # ---- in-life learning (rule chosen by genome) ----
        if nmod is not None:
            dbal = prev_bal - np.abs(bal); dsur = prev_surprise - surprise; dbond = bond - prev_bond
            mod = np.select([nmod == 0, nmod == 1, nmod == 2, nmod == 3], [np.ones(B), dsur, dbal, dbond * 10])
            mod = np.clip(mod, -1, 1).astype(np.float32) * alive
            pre, post = obs, h
            hebb = post[:, :, None] * pre[:, None, :]
            dW = np.empty_like(Win)
            for r in range(6):
                s = rule == r
                if not s.any(): continue
                if r == 0: dW[s] = hebb[s]
                elif r == 1: dW[s] = post[s][:, :, None] * (pre[s][:, None, :] - post[s][:, :, None] * Win[s])
                elif r == 2: dW[s] = (post[s] * (post[s] - theta[s]))[:, :, None] * pre[s][:, None, :]
                elif r == 3: dW[s] = -hebb[s]
                elif r == 4: dW[s] = hebb[s] * np.abs(mod[s])[:, None, None]   # extra gated
                else: dW[s] = (Wp[s].transpose(0, 2, 1) @ err[s][:, :, None]) * pre[s][:, None, :]
            theta += 0.1 * (h ** 2 - theta)
            Win += (lr * mod)[:, None, None].astype(np.float32) * dW
            Win = np.clip(Win, -3, 3)
            oh = np.eye(N_ACT, dtype=np.float32)[a]
            elig = 0.8 * elig + (oh - p)[:, :, None] * h[:, None, :]
            Wout += (lr * mod)[:, None, None].astype(np.float32) * elig
        # memory write
        trace = dec[:, None].astype(np.float32) * trace + (1 - dec[:, None].astype(np.float32)) * h
        if m2.any(): fw[m2] = dec[m2, None, None] * fw[m2] + 0.1 * h[m2][:, :, None] * h[m2][:, None, :]
        if m3.any():
            wr = m3 & (surprise > prev_surprise)
            slots[wr, t % 4] = h[wr]
        prev_surprise, prev_bal, prev_bond = surprise, np.abs(bal), bond.copy()
        alive_t += alive; balanced_t += alive & (np.abs(bal) < 0.8)
        rep_sum += alive * rep
        acts_hist[ar, a] += alive; dist_partner += alive * close
        if t % 16 == 15 and alive.mean() < 0.05:   # early termination: town has collapsed
            break
    # ---- per-genome outcome (facts, not a score) ----
    T = t + 1
    def agg(x): return x.reshape(Gt, per_genome)[:G].mean(1)
    def cagg(x): return x.reshape(Gt, per_genome)[G:].mean()
    tot = np.maximum(acts_hist.sum(1, keepdims=True), 1); af = acts_hist / tot
    out = dict(steps=np.full(G, T),
        births=agg(births), survival=agg(alive_t / T), balanced=agg(balanced_t / np.maximum(alive_t, 1)),
        bond=agg(bond), law_breaks=agg(law_breaks / np.maximum(alive_t, 1)),
        coop=agg(af[:, 4] + af[:, 7]), forage=agg(af[:, 0] + af[:, 2]), sleep=agg(af[:, 3]),
        closeness=agg(dist_partner / np.maximum(alive_t, 1)), rep=agg(rep), tools=agg(tools),
        surprise=agg(prev_surprise),
        # law-keeping metrics (raw sums per genome so rates can be pooled correctly)
        promise_opps=agg(p_opps), promise_breaks=agg(p_breaks), encounters=agg(encounters),
        law_break_events=agg(law_breaks),
        rep_mean=agg(rep_sum / np.maximum(alive_t, 1)), honesty=agg(honesty(law_breaks, encounters)))
    if Gc:  # cheater-vs-honest facts, broadcast per genome so shards concatenate
        cs = dict(cheat_births=cagg(births), cheat_survival=cagg(alive_t / T), cheat_frac=Gc / Gt,
                  cheat_bond_share=c_bond_with / max(h_bonds, 1), cheat_bonds_lost=c_lost / (Gc * per_genome))
        for k, v in cs.items(): out[k] = np.full(G, float(v))
    if return_w:
        out["Win"] = Win.reshape(Gt, per_genome, H, D_OBS)[:G].mean(1); out["Wout"] = Wout.reshape(Gt, per_genome, N_ACT, H)[:G].mean(1)
    return out
