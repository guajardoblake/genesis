"""Pure per-step kernels extracted verbatim from town.py (numpy reference for the JAX port).
Inputs are explicit arrays; randomness is passed in, so numpy and JAX can be compared on identical draws."""
import numpy as np

def mind_forward(Win, Wout, obs, mem_t, trace, fw, slots, hmask, lrp, bond, ex, u):
    """Memory read -> hidden -> action logits -> exploration-mixed policy -> sampled action (u ~ U[0,1))."""
    B, H = trace.shape; N_ACT = Wout.shape[1]
    mem = np.zeros((B, H), np.float32)
    m1 = mem_t == 1; m2 = mem_t == 2; m3 = mem_t == 3
    mem[m1] = trace[m1]
    mem[m2] = np.einsum("bij,bj->bi", fw[m2], trace[m2])
    if m3.any():
        sim = np.einsum("bsh,bh->bs", slots[m3], trace[m3]); w = np.exp(sim - sim.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        mem[m3] = np.einsum("bs,bsh->bh", w, slots[m3])
    h = np.tanh(np.einsum("bhd,bd->bh", Win, obs) + 0.5 * mem) * hmask
    logits = np.einsum("bah,bh->ba", Wout, h)
    logits[:, 7] += 0.3 * lrp * bond
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    p = (1 - ex)[:, None] * p + ex[:, None] / N_ACT
    a = (p.cumsum(1) > u[:, None]).argmax(1)
    return h, p, a

def predict_sense(Wp, h, sense, sense_ema, hor, ppw):
    pred = np.einsum("bsh,bh->bs", Wp, h)
    sense_ema = sense_ema + (sense - sense_ema) / hor[:, None]
    wvec = np.array([1, 1, 1, 0, 0, 0], np.float32)[None] + np.array([0, 0, 0, 1, 1, 1], np.float32)[None] * ppw[:, None].astype(np.float32)
    err = (sense_ema - pred) * wvec
    surprise = (err ** 2).sum(1)
    Wp = Wp + 0.05 * err[:, :, None] * h[:, None, :]
    return Wp, sense_ema, err, surprise

def law_step(a, alive, inreach, near, partner, promise, rep):
    """Law enforcement through reputation only + law-keeping counters for this step."""
    take = (a == 5) & alive & inreach; harm = (a == 6) & alive & inreach
    p_opp = alive & inreach & promise & (partner >= 0) & (near == partner)
    broke = take | harm
    broke_promise = broke & promise & (near == partner)
    rep = np.clip(rep - 0.1 * broke - 0.2 * broke_promise, 0, 1)
    return rep, broke, broke_promise, p_opp
