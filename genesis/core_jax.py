"""JAX port (CPU-validated, GPU/TPU-ready) of the per-step kernels in core_np.py.
Status: first kernels only (mind forward, inner-sense prediction, law step). The full town loop is not ported yet."""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@jax.jit
def mind_forward(Win, Wout, obs, mem_t, trace, fw, slots, hmask, lrp, bond, ex, u):
    m1 = (mem_t == 1)[:, None]; m2 = (mem_t == 2)[:, None]; m3 = (mem_t == 3)[:, None]
    fwm = jnp.einsum("bij,bj->bi", fw, trace)
    sim = jnp.einsum("bsh,bh->bs", slots, trace); w = jax.nn.softmax(sim, axis=1)
    slm = jnp.einsum("bs,bsh->bh", w, slots)
    mem = jnp.where(m1, trace, 0) + jnp.where(m2, fwm, 0) + jnp.where(m3, slm, 0)
    h = jnp.tanh(jnp.einsum("bhd,bd->bh", Win, obs) + 0.5 * mem) * hmask
    logits = jnp.einsum("bah,bh->ba", Wout, h)
    logits = logits.at[:, 7].add(0.3 * lrp * bond)
    p = jax.nn.softmax(logits, axis=1)
    p = (1 - ex)[:, None] * p + ex[:, None] / Wout.shape[1]
    a = jnp.argmax(jnp.cumsum(p, 1) > u[:, None], axis=1)
    return h, p, a

@jax.jit
def predict_sense(Wp, h, sense, sense_ema, hor, ppw):
    pred = jnp.einsum("bsh,bh->bs", Wp, h)
    sense_ema = sense_ema + (sense - sense_ema) / hor[:, None]
    wvec = jnp.array([1, 1, 1, 0, 0, 0], jnp.float32)[None] + jnp.array([0, 0, 0, 1, 1, 1], jnp.float32)[None] * ppw[:, None].astype(jnp.float32)
    err = (sense_ema - pred) * wvec
    return Wp + 0.05 * err[:, :, None] * h[:, None, :], sense_ema, err, (err ** 2).sum(1)

@jax.jit
def law_step(a, alive, inreach, near, partner, promise, rep):
    take = (a == 5) & alive & inreach; harm = (a == 6) & alive & inreach
    p_opp = alive & inreach & promise & (partner >= 0) & (near == partner)
    broke = take | harm; broke_promise = broke & promise & (near == partner)
    rep = jnp.clip(rep - 0.1 * broke - 0.2 * broke_promise, 0, 1)
    return rep, broke, broke_promise, p_opp
