"""Array-module-generic copy of town.run_town (numpy or CuPy). With xp=numpy and default env it is
bit-identical to town.run_town (tested). Extra env knobs, absent by default: rep_noise (others perceive
reputation + N(0, rep_noise), separate RNG stream), rep_penalty (scales the trust lost per broken law)."""
import numpy as np
from . import town as _C
from .town import (AXES, NAMES, LR, DECAY, HOR, HID, SGAIN, PPW, EXPL, TSENS, LOVE_BAL, LOVE_REP,
                   D_OBS, H, N_ACT, N_SENSE, promise_keeping_rate, reputation_gap, ACTS, default_env)

class _CupyRng:
    def __init__(self, xp, seed): self.r = xp.random.RandomState(int(seed) % (2**32)); self.xp = xp
    def random(self, size=None): return self.r.random_sample(size)
    def normal(self, m, s, size=None): return self.r.normal(m, s, size)
    def integers(self, lo, hi, size=None): return self.r.randint(lo, hi, size)
def _rng(xp, seed): return np.random.default_rng(seed) if xp is np else _CupyRng(xp, seed)
def _np(xp, x): return x if xp is np else xp.asnumpy(x)
def _add_at(xp, a, idx, v):
    if xp is np: np.add.at(a, idx, v)
    else:
        import cupyx; cupyx.scatter_add(a, idx, xp.asarray(v, a.dtype))
def _sub_at(xp, a, idx, v):
    if xp is np: np.subtract.at(a, idx, v)
    else:
        import cupyx; cupyx.scatter_add(a, idx, -xp.asarray(v, a.dtype))
def honesty(law_breaks, encounters, xp=np):
    return 1.0 - xp.asarray(law_breaks, float) / xp.maximum(xp.asarray(encounters, float), 1.0)

def run_town(genomes, env, seed, T=96, per_genome=4, init_w=None, return_w=False, xp=None):
    xp = np if xp is None else xp
    rng = _rng(xp, seed)
    LR, DECAY, HOR, HID, SGAIN, PPW, EXPL, TSENS, LOVE_BAL, LOVE_REP = (xp.asarray(c) for c in (_C.LR, _C.DECAY, _C.HOR, _C.HID, _C.SGAIN, _C.PPW, _C.EXPL, _C.TSENS, _C.LOVE_BAL, _C.LOVE_REP))
    genomes = xp.asarray(genomes)
    if init_w is not None: init_w = {k: xp.asarray(v) for k, v in init_w.items()}
    G = genomes.shape[0]
    # experiment switches (absent = default town; defaults leave dynamics bit-identical)
    use_love = env.get("love", True); use_rep = env.get("reputation", True); cfrac = env.get("cheaters", 0.0)
    Gc = int(xp.ceil(G * cfrac)) if cfrac > 0 else 0
    if Gc:  # cheaters: extra random-genome minds that steal/harm in reach and FAKE a high reputation
        crng = _rng(xp, seed + 99991)       # separate stream: never touches the main RNG
        genomes = xp.concatenate([genomes, (crng.random((Gc, genomes.shape[1])) * xp.array([len(a[1]) for a in AXES])).astype(genomes.dtype)])
        if init_w is not None:
            init_w = dict(mask=xp.concatenate([init_w["mask"], xp.zeros(Gc, bool)]),
                          Win=xp.concatenate([init_w["Win"], xp.zeros((Gc,) + init_w["Win"].shape[1:], init_w["Win"].dtype)]),
                          Wout=xp.concatenate([init_w["Wout"], xp.zeros((Gc,) + init_w["Wout"].shape[1:], init_w["Wout"].dtype)]))
    Gt = G + Gc; B = Gt * per_genome
    gid = xp.repeat(xp.arange(Gt), per_genome); g = genomes[gid]
    rule, mem_t, nmod, love = g[:, 0], g[:, 2], g[:, 11], g[:, 7]
    lr, dec, hor = LR[g[:, 1]], DECAY[g[:, 3]], HOR[g[:, 4]]
    sg, ppw, ex, ts = SGAIN[g[:, 6]], PPW[g[:, 8]], EXPL[g[:, 9]], TSENS[g[:, 10]]
    hmask = (xp.arange(H)[None, :] < HID[g[:, 5]][:, None]).astype(xp.float32)
    lb, lrp = LOVE_BAL[love], LOVE_REP[love]
    if not use_love: lb, lrp = lb * 0, lrp * 0          # ablation: bond never enters balance or reproduction drive
    ch = gid >= G; c_bond_with = 0.0; h_bonds = 0.0; c_lost = 0.0; FAKE = 0.9
    L = env["size"]; GR = 12
    # --- world ---
    food = xp.full((GR, GR), 0.5)
    # --- bodies (physiology is real in every run) ---
    pos = rng.random((B, 2)) * L
    energy = xp.full(B, 1.0); fatigue = xp.zeros(B); pain = xp.zeros(B); temp = xp.zeros(B)
    alive = xp.ones(B, bool); setpt = rng.normal(0, 0.1, B)       # hidden, unlabeled balance set-point
    partner = xp.full(B, -1); bond = xp.zeros(B); rep = xp.full(B, 0.5)   # reputation = others' trust
    promise = xp.zeros(B, bool); tools = xp.zeros(B)
    # --- minds: genome gives only learning rules; weights start random and are learned in-life ---
    Win = rng.normal(0, 0.4, (B, H, D_OBS)).astype(xp.float32)
    Wout = rng.normal(0, 0.1, (B, N_ACT, H)).astype(xp.float32)
    if init_w is not None:  # Lamarckian warm start: inherit parent's learned weights (+ noise)
        has = init_w["mask"][gid]
        Win[has] = init_w["Win"][gid][has] + rng.normal(0, 0.05, tuple(Win[has].shape)).astype(xp.float32)
        Wout[has] = init_w["Wout"][gid][has]
    Wp = xp.zeros((B, N_SENSE, H), xp.float32)          # predictor of own+partner inner sense
    trace = xp.zeros((B, H), xp.float32); fw = xp.zeros((B, H, H), xp.float32)
    slots = xp.zeros((B, 4, H), xp.float32); elig = xp.zeros((B, N_ACT, H), xp.float32)
    theta = xp.full((B, H), 0.1, xp.float32); sense_ema = xp.zeros((B, N_SENSE), xp.float32)
    prev_surprise = xp.zeros(B); prev_bal = xp.zeros(B); prev_bond = xp.zeros(B)
    births = xp.zeros(B); balanced_t = xp.zeros(B); alive_t = xp.zeros(B)
    acts_hist = xp.zeros((B, N_ACT)); law_breaks = xp.zeros(B); dist_partner = xp.zeros(B)
    encounters = xp.zeros(B); p_opps = xp.zeros(B); p_breaks = xp.zeros(B); rep_sum = xp.zeros(B)  # law-keeping metrics
    ar = xp.arange(B)
    rnoise = float(env.get("rep_noise", 0.0)); nrng = _rng(xp, seed + 7777) if rnoise > 0 else None
    rpen = float(env.get("rep_penalty", 1.0))
    for t in range(T):
        day = xp.sin(2 * xp.pi * t / 24)
        amb = env["temp_amp"] * day - env["cold_night"] * (day < 0)
        cell = xp.clip((pos / L * GR).astype(int), 0, GR - 1)
        fhere = food[cell[:, 0], cell[:, 1]]
        # nearest other living mind (sampled neighbours for O(B*K))
        K = 8; cand = rng.integers(0, B, (B, K))
        if t % 4 == 0 or t == 0:
            cand[:, 0] = xp.where(partner >= 0, partner, cand[:, 0])
        d = pos[cand] - pos[:, None]; d = (d + L / 2) % L - L / 2
        dist = xp.sqrt((d ** 2).sum(-1)) + 1e3 * (~alive[cand]) + 1e3 * (cand == ar[:, None])
        k = dist.argmin(1); near = cand[ar, k]; nd = dist[ar, k]; nvec = d[ar, k]
        nz = None if nrng is None else nrng.normal(0, rnoise, (B,))
        seen = rep[near] if nz is None else xp.clip(rep[near] + nz, 0, 1)
        haspart = partner >= 0
        pd = xp.where(haspart, xp.sqrt((((pos[xp.maximum(partner, 0)] - pos + L / 2) % L - L / 2) ** 2).sum(-1)), L)
        close = xp.exp(-pd / 3.0)
        # inner sense: own bond/safety/closeness + same for partner (they must predict both)
        safety = 1 - xp.clip(pain, 0, 1)
        pp = xp.maximum(partner, 0)
        sense = xp.stack([bond, safety, close,
                          xp.where(haspart, bond[pp], 0), xp.where(haspart, 1 - xp.clip(pain[pp], 0, 1), 0), close], 1).astype(xp.float32)
        lovepull = bond * close                                   # hidden pull, never labelled
        # hidden balance: physiological deviation + internal surprise - love pull (per wiring)
        phys = xp.abs(1 - energy) + fatigue + pain + xp.abs(temp)
        bal = phys + sg * prev_surprise - 0.5 * lb * lovepull - setpt
        obs = xp.stack([energy, fatigue, pain, temp, fhere, xp.exp(-nd / 3), seen * ts, bond,
                        prev_surprise, close, bal, rep, tools, xp.ones(B)], 1).astype(xp.float32)
        # memory read
        mem = xp.zeros((B, H), xp.float32)
        m1 = mem_t == 1; m2 = mem_t == 2; m3 = mem_t == 3
        mem[m1] = trace[m1]
        if m2.any(): mem[m2] = xp.einsum("bij,bj->bi", fw[m2], trace[m2])
        if m3.any():
            sim = xp.einsum("bsh,bh->bs", slots[m3], trace[m3]); w = xp.exp(sim - sim.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            mem[m3] = xp.einsum("bs,bsh->bh", w, slots[m3])
        h = xp.tanh(xp.einsum("bhd,bd->bh", Win, obs) + 0.5 * mem) * hmask
        logits = xp.einsum("bah,bh->ba", Wout, h)
        logits[:, 7] += 0.3 * lrp * bond                          # reproduction drive wiring
        p = xp.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        p = (1 - ex)[:, None] * p + ex[:, None] / N_ACT
        a = (p.cumsum(1) > rng.random(B)[:, None]).argmax(1)
        if Gc: a = xp.where(ch & (nd < 2.0) & (crng.random(B) < 0.5), 5, a)   # cheaters steal when in reach
        a[~alive] = 3
        # predict inner sense (self + partner), surprise is the mind's own hidden state
        pred = xp.einsum("bsh,bh->bs", Wp, h)
        sense_ema += (sense - sense_ema) / hor[:, None]
        wvec = xp.array([1, 1, 1, 0, 0, 0], xp.float32)[None] + xp.array([0, 0, 0, 1, 1, 1], xp.float32)[None] * ppw[:, None].astype(xp.float32)
        err = (sense_ema - pred) * wvec
        surprise = (err ** 2).sum(1)
        Wp += 0.05 * err[:, :, None] * h[:, None, :]
        # ---- act on world ----
        mv = xp.zeros((B, 2))
        to_food = a == 0
        gx = xp.stack([food[(cell[:, 0] + 1) % GR, cell[:, 1]] - food[(cell[:, 0] - 1) % GR, cell[:, 1]],
                       food[cell[:, 0], (cell[:, 1] + 1) % GR] - food[cell[:, 0], (cell[:, 1] - 1) % GR]], 1)
        mv[to_food] = gx[to_food] * 4 + rng.normal(0, 0.5, (int(to_food.sum()), 2))
        to_o = (a == 1) | (a == 7)
        tgt = xp.where((a == 7) & haspart, pp, near)
        tv = (pos[tgt] - pos + L / 2) % L - L / 2
        mv[to_o] = tv[to_o] / (xp.linalg.norm(tv[to_o], axis=1, keepdims=True) + 1e-6)
        pos = (pos + mv * alive[:, None]) % L
        energy -= 0.02 + 0.01 * (xp.abs(mv).sum(1) > 0)
        eat = (a == 2) & alive
        harvest = xp.minimum(fhere, 0.15 * (0.5 + rep + 0.5 * tools)) * eat   # trust earns resources/tools
        energy += harvest; _sub_at(xp, food, (cell[:, 0], cell[:, 1]), harvest * 0.5)
        sl = (a == 3) & alive
        fatigue = xp.where(sl, fatigue * 0.6, fatigue + 0.03)
        inreach = nd < 2.0
        share = (a == 4) & alive & inreach
        gift = 0.1 * share * (energy > 0.4)
        energy -= gift; _add_at(xp, energy, near, gift)
        _add_at(xp, rep, near * 0, 0)  # no-op keeps shapes explicit
        rep += 0.02 * share
        take = (a == 5) & alive & inreach
        stolen = xp.minimum(0.15, xp.maximum(energy[near], 0)) * take
        energy += stolen; _sub_at(xp, energy, near, stolen)
        harm = (a == 6) & alive & inreach
        _add_at(xp, pain, near, 0.3 * harm)
        p_opps += alive & inreach & promise & (partner >= 0) & (near == partner)  # promise-bound encounter
        # laws: enforced ONLY through lost trust and bonds
        broke = take | harm
        broke_promise = broke & promise & (near == partner)
        rep = xp.clip(rep - 0.1 * broke - 0.2 * broke_promise, 0, 1) if rpen == 1.0 else xp.clip(rep - rpen * (0.1 * broke + 0.2 * broke_promise), 0, 1)
        lost = broke & (near == partner)
        if Gc: c_lost += float((lost & ch).sum())
        if not use_rep: rep = xp.full(B, 0.5)               # ablation: no reputation signal at all
        if Gc: rep = xp.where(ch, FAKE, rep)                 # cheaters' reputation is faked high
        bond = xp.where(lost, 0, bond); 
        if lost.any(): partner[partner[lost]] = -1; bond[partner[lost]] = 0; partner[lost] = -1
        law_breaks += broke
        encounters += alive & inreach
        p_breaks += broke_promise
        # bonding: two unpaired minds near each other, at least one sharing/staying, both trusted enough
        pair = alive & (~haspart) & inreach & (partner[near] < 0) & alive[near] & ((a == 4) | (a == 7)) & ((rep[near] if nz is None else xp.clip(rep[near] + nz, 0, 1)) > 0.3) & (near != ar)
        pi = ar[pair]; pj = near[pair]
        ok = partner[pj] < 0
        pi, pj = pi[ok], pj[ok]
        _, u = xp.unique(pj, return_index=True); pi, pj = pi[u], pj[u]
        ok2 = ~xp.isin(pi, pj); pi, pj = pi[ok2], pj[ok2]
        if Gc: hb = ~ch[pi] | ~ch[pj]; h_bonds += float(hb.sum()); c_bond_with += float((hb & (ch[pi] | ch[pj])).sum())
        partner[pi] = pj; partner[pj] = pi; promise[pi] = True; promise[pj] = True
        haspart = partner >= 0
        bond = xp.where(haspart, xp.clip(bond + 0.05 * close + 0.02 * share, 0, 1), xp.maximum(bond - 0.05, 0))
        # physiology: temperature (huddling with partner helps), hazards, pain healing
        temp += 0.2 * (amb - temp) - 0.15 * temp * close * haspart
        pain = xp.clip(pain * 0.9 + (rng.random(B) < env["hazard"]) * 0.4 + 0.1 * (xp.abs(temp) > 0.5) + 0.05 * (fatigue > 1), 0, 2)
        # institutions: trusted pairs build tools over time
        tools = xp.minimum(tools + 0.01 * haspart * (rep > 0.6), 1)
        # reproduction: a bonded, balanced, fed pair has offspring (costs energy)
        rep_ok = alive & haspart & (bond > 0.5) & (energy > 0.9) & (xp.abs(bal) < 0.8) & (pd < 3)
        rep_ok &= rep_ok[pp]
        births += rep_ok; energy -= 0.3 * rep_ok
        # death
        dead = alive & ((energy <= 0) | (pain >= 1.5) | (xp.abs(temp) > 1.2) | (fatigue > 2.5))
        if dead.any():
            dp = partner[dead]; dp = dp[dp >= 0]; partner[dp] = -1; bond[dp] *= 0.5
            partner[dead] = -1; alive &= ~dead
        food = xp.minimum(food + env["food_rate"] * (1 - food), 1.0)
        # ---- in-life learning (rule chosen by genome) ----
        if nmod is not None:
            dbal = prev_bal - xp.abs(bal); dsur = prev_surprise - surprise; dbond = bond - prev_bond
            mod = xp.select([nmod == 0, nmod == 1, nmod == 2, nmod == 3], [xp.ones(B), dsur, dbal, dbond * 10])
            mod = xp.clip(mod, -1, 1).astype(xp.float32) * alive
            pre, post = obs, h
            hebb = post[:, :, None] * pre[:, None, :]
            dW = xp.empty_like(Win)
            for r in range(6):
                s = rule == r
                if not s.any(): continue
                if r == 0: dW[s] = hebb[s]
                elif r == 1: dW[s] = post[s][:, :, None] * (pre[s][:, None, :] - post[s][:, :, None] * Win[s])
                elif r == 2: dW[s] = (post[s] * (post[s] - theta[s]))[:, :, None] * pre[s][:, None, :]
                elif r == 3: dW[s] = -hebb[s]
                elif r == 4: dW[s] = hebb[s] * xp.abs(mod[s])[:, None, None]   # extra gated
                else: dW[s] = (Wp[s].transpose(0, 2, 1) @ err[s][:, :, None]) * pre[s][:, None, :]
            theta += 0.1 * (h ** 2 - theta)
            Win += (lr * mod)[:, None, None].astype(xp.float32) * dW
            Win = xp.clip(Win, -3, 3)
            oh = xp.eye(N_ACT, dtype=xp.float32)[a]
            elig = 0.8 * elig + (oh - p)[:, :, None] * h[:, None, :]
            Wout += (lr * mod)[:, None, None].astype(xp.float32) * elig
        # memory write
        trace = dec[:, None].astype(xp.float32) * trace + (1 - dec[:, None].astype(xp.float32)) * h
        if m2.any(): fw[m2] = dec[m2, None, None] * fw[m2] + 0.1 * h[m2][:, :, None] * h[m2][:, None, :]
        if m3.any():
            wr = m3 & (surprise > prev_surprise)
            slots[wr, t % 4] = h[wr]
        prev_surprise, prev_bal, prev_bond = surprise, xp.abs(bal), bond.copy()
        alive_t += alive; balanced_t += alive & (xp.abs(bal) < 0.8)
        rep_sum += alive * rep
        acts_hist[ar, a] += alive; dist_partner += alive * close
        if t % 16 == 15 and alive.mean() < 0.05:   # early termination: town has collapsed
            break
    # ---- per-genome outcome (facts, not a score) ----
    T = t + 1
    def agg(x): x = _np(xp, x); return x.reshape(Gt, per_genome)[:G].mean(1)
    def cagg(x): x = _np(xp, x); return x.reshape(Gt, per_genome)[G:].mean()
    tot = xp.maximum(acts_hist.sum(1, keepdims=True), 1); af = acts_hist / tot
    out = dict(steps=xp.full(G, T),
        births=agg(births), survival=agg(alive_t / T), balanced=agg(balanced_t / xp.maximum(alive_t, 1)),
        bond=agg(bond), law_breaks=agg(law_breaks / xp.maximum(alive_t, 1)),
        coop=agg(af[:, 4] + af[:, 7]), forage=agg(af[:, 0] + af[:, 2]), sleep=agg(af[:, 3]),
        closeness=agg(dist_partner / xp.maximum(alive_t, 1)), rep=agg(rep), tools=agg(tools),
        surprise=agg(prev_surprise),
        # law-keeping metrics (raw sums per genome so rates can be pooled correctly)
        promise_opps=agg(p_opps), promise_breaks=agg(p_breaks), encounters=agg(encounters),
        law_break_events=agg(law_breaks),
        rep_mean=agg(rep_sum / xp.maximum(alive_t, 1)), honesty=agg(honesty(law_breaks, encounters, xp)))
    if Gc:  # cheater-vs-honest facts, broadcast per genome so shards concatenate
        cs = dict(cheat_births=cagg(births), cheat_survival=cagg(alive_t / T), cheat_frac=Gc / Gt,
                  cheat_bond_share=c_bond_with / max(h_bonds, 1), cheat_bonds_lost=c_lost / (Gc * per_genome))
        for k, v in cs.items(): out[k] = xp.full(G, float(v))
    if return_w:
        out["Win"] = _np(xp, Win).reshape(Gt, per_genome, H, D_OBS)[:G].mean(1); out["Wout"] = _np(xp, Wout).reshape(Gt, per_genome, N_ACT, H)[:G].mean(1)
    return {k: _np(xp, v) if not isinstance(v, np.ndarray) else v for k, v in out.items()}
