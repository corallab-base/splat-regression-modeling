"""SRM backend: a weighted mixture of wrapped Gaussians ("splats") on any Environment.

Delegates the manifold-specific pieces (log map, Jacobian correction) to the
Environment and the wrapped-Gaussian math to
``srms.lib.manifold_splat.eval_wrapped_gaussian`` — the same generic evaluator
already used for the sphere and SE(2) backends there. This replaces the
original ``torus.py``'s hand-rolled ``eval_splat_torus``, which duplicated
that math with the wrap baked in by name.

Parameters ``(V, A, B)``: ``V: [k, p]`` weights, ``A: [k, d, d]`` scale/rotation,
``B: [k, d]`` centres — same convention as ``srms/lib/splat.py``.

Every backend (this one, and eventually ``mlp``/``kan``) exposes the same three
entry points so a training strategy (``srms/methods/strategies``) never needs
to know which one it's using:

- ``init_params(key, env, cfg) -> params`` — reads whatever ``cfg`` fields it
  needs (here: ``cfg.num_splats``, ``cfg.init_scale``); other backends read
  their own fields off the same flat ``cfg``.
- ``eval_raw(params, X, env) -> [n, p]`` — the raw (unfactored) field value.
- ``post_step(params, cfg, env) -> params`` — called by every strategy's
  training loop after each optimizer step (see ``post_step`` below for why).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

SplatParams = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def init_params(key: jax.Array, env, cfg, p: int = 1) -> SplatParams:
    """Initialise (V, A, B) with small positive weights and centres drawn from env.sample_domain.

    Centres need environment-specific sampling (a box for the torus, a unit sphere for
    SphereEnvironment), not a generic uniform box, so this delegates to the same host-side sampler
    RRT*/collocation already use rather than assuming ``env.domain`` is a box.

    With ``cfg.densify`` the model starts at ``cfg.init_splats`` and grows via ``adapt``; otherwise it
    is a fixed mixture of ``cfg.num_splats``.

    ``V`` is drawn ``|N(0, cfg.init_v_scale)|`` rather than left at exactly zero — the pre-``srms/``
    exploration (``_archive/preexisting/eikonal_splat.py``'s ``init_splat_params``) found a plain
    zero/signed init gets stuck organizing which splats should be positive vs. negative contributions
    from a flat start, whereas starting every splat as a small positive bump was a 59.8% MSE
    improvement (``autoresearch_results/results.tsv`` iter 8) with further gains from tuning the scale
    up to 0.1 (iter 10). Sign is otherwise still free to flip during training — only the init is
    constrained. Set ``cfg.init_v_scale = 0`` to recover the old exact-zero init.
    """
    k = cfg.init_splats if getattr(cfg, "densify", False) else cfg.num_splats
    vkey, key = jax.random.split(key)
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    centres_np = env.sample_domain(np.random.default_rng(seed), k)
    centres = jnp.asarray(centres_np, dtype=jnp.float32)
    scales = jnp.repeat((cfg.init_scale * jnp.eye(env.tangent_dim))[None], k, axis=0)
    v_scale = getattr(cfg, "init_v_scale", 0.0)
    V = jnp.abs(jax.random.normal(vkey, (k, p))) * v_scale if v_scale > 0.0 else jnp.zeros((k, p))
    return V, scales, centres


def eval_raw(params: SplatParams, X: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the raw splat mixture g(x) = Σ_j V[j]·N_w(x | B[j], A[j]) at each row of X. Returns [n, p].

    Centres are read through ``env.wrap_point`` so the density is always evaluated at a centre that
    lies *on* the manifold — a wrapped Gaussian is undefined otherwise. This makes the loss exactly
    invariant to ‖B‖, which in turn makes ``dL/dB`` exactly tangent (measured radial component
    4e-10), so the optimizer only ever moves a centre along the manifold.

    That invariance is necessary but **not sufficient**: with nothing pinning ‖B‖, AdamW's weight
    decay shrinks it, and the angular step ≈‖ΔB‖/‖B‖ then grows without bound. ``post_step`` performs
    the actual retraction on the parameters; see its docstring for the measurements.

    A mathematical no-op on the chart manifolds: ``wrap(x − wrap(mu)) ≡ wrap(x − mu)`` on the torus,
    and hyperbolic's ``_clamp_ball`` is the identity for interior points.
    """
    V, A, B = params
    B = env.wrap_point(B)
    # Per-splat work hoisted out of the per-point loop (see docstring).
    A_inv = jnp.linalg.inv(A)
    det_A = jnp.abs(jnp.linalg.det(A))
    pre = jax.vmap(env.splat_precompute)(B)
    norm_const = (2.0 * jnp.pi) ** (env.tangent_dim / 2.0)

    def rho_at_x(x: jnp.ndarray) -> jnp.ndarray:
        def one(pre_j, a_inv, det):
            # One call for both: jac_factor used to recompute the distance log_map already had.
            v, jac = env.log_and_jac(pre_j, x)
            z = a_inv @ v
            return jnp.exp(-0.5 * jnp.dot(z, z)) / (norm_const * (det + 1e-12)) * jac

        return jax.vmap(one)(pre, A_inv, det_A)

    return jax.vmap(rho_at_x)(X) @ V


def post_step(params: SplatParams, cfg, env=None) -> SplatParams:
    """Floor each splat's covariance singular values, and retract the centres onto the manifold.

    **Retraction (``env`` given).** A wrapped Gaussian is only defined for a centre lying *on* the
    manifold, but ``B`` is an unconstrained optimizer variable. This matters only for **embedded**
    manifolds: on S² ``B`` must satisfy ‖B‖ = 1 and a plain ambient gradient step walks it off —
    measured 0.105 off the unit sphere by step 65, at which point ``_sphere_frame``'s Householder
    construction (which assumes a unit vector) is no longer orthonormal and the density goes to NaN.
    ``env.wrap_point`` is the projection retraction; it agrees with the exponential retraction
    ``Exp_B`` to O(‖step‖³) (measured 3.3e-4 at step 0.1, 4.3e-7 at 0.01), which is far below float32
    noise at this learning rate.

    Doing it here, on the *parameters*, rather than inside ``eval_raw`` matters. Normalizing at
    evaluation time makes the loss invariant to ‖B‖, so nothing opposes AdamW's weight decay: ‖B‖ then
    decays as (1−lr·wd)^step (measured 1.00 → 0.33 over 2400 steps) and, since the angular step on a
    centre is ≈‖ΔB‖/‖B‖, the effective step size *grows* as the norm shrinks — 3× by step 2400 — and
    training diverged at step 2481. Retracting the parameters pins ‖B‖ = 1 and removes that coupling.

    A no-op on the chart manifolds: ``wrap`` is idempotent on the torus and hyperbolic's
    ``_clamp_ball`` is the identity for interior points, so no established result moves.

    **Scale floor — this is the fix that makes adaptive densification stable.** A sum of Gaussians
    cannot represent the true Eikonal kink at obstacle boundaries and the cut locus, so unconstrained
    optimization chases it by driving a splat's covariance to zero — an effectively infinite ``‖∇T‖``
    spike (up to ~1e7 was measured) which then blows up any speed-match residual. Flooring the SVD
    singular values at ``cfg.scale_floor`` makes that collapse impossible. Applied after every
    optimizer step; jittable, so it lives inside the strategies' ``step``.

    The floor is a no-op when ``cfg.scale_floor <= 0``, and irrelevant to the fixed-count model
    (which never densifies and so never triggers the collapse), but harmless there.
    """
    V, A, B = params
    if env is not None:
        B = env.wrap_point(B)
    floor = getattr(cfg, "scale_floor", 0.0)
    if floor > 0.0:
        U, S, Vt = jnp.linalg.svd(A)
        A = jnp.einsum("kij,kj,kjl->kil", U, jnp.clip(S, floor, None), Vt)
    return V, A, B


def adapt(params: SplatParams, opt_state, residual_fn, env, cfg, rng: np.random.Generator):
    """3DGS-style prune + spawn, preserving Adam moments for the splats that survive.

    Prunes splats whose weight is below ``cfg.prune_thresh``, then spawns up to ``cfg.spawn_per`` new
    ones at the highest-``residual_fn`` free-space points, capped at ``cfg.max_splats``.

    The optimizer state is grown **surgically**: every leaf whose leading axis indexes splats is
    sliced to the survivors and zero-padded for the spawns. A plain ``optimizer.init()`` reset would
    re-kick every already-converged splat, which is what made earlier densification runs diverge.
    Scalar leaves (Adam's step count) pass through untouched, so this is robust to ``optax.chain``
    nesting without depending on the state's namedtuple layout.

    Returns ``(params, opt_state, num_splats)``.
    """
    V = np.asarray(params[0])
    old_k = len(V)
    keep = np.abs(V[:, 0]) > cfg.prune_thresh
    if keep.sum() < 16:  # never prune the model into nothing
        keep = np.ones(old_k, bool)
    idx = np.where(keep)[0]

    spawn_B = np.zeros((0, env.dim), np.float32)
    room = cfg.max_splats - len(idx)
    if room > 0:
        pool = env.sample_domain(rng, 3000)
        pool = pool[env.sdf_np(pool) > 0.1]  # spawn in free space, not inside obstacles
        if len(pool):
            residual = np.asarray(residual_fn(jnp.asarray(pool, jnp.float32)))
            k = int(min(cfg.spawn_per, room, len(pool)))
            spawn_B = pool[np.argsort(residual)[-k:]].astype(np.float32)
    ns = len(spawn_B)

    def regrow(leaf):
        a = np.asarray(leaf)
        if a.ndim >= 1 and a.shape[0] == old_k:
            return jnp.asarray(np.concatenate([a[idx], np.zeros((ns,) + a.shape[1:], a.dtype)], 0))
        return leaf

    new_v = np.concatenate([V[idx], np.full((ns, V.shape[1]), 1e-4, np.float32)])
    new_a = np.concatenate(
        [
            np.asarray(params[1])[idx],
            np.repeat((cfg.spawn_scale * np.eye(env.tangent_dim))[None], ns, 0).astype(np.float32),
        ]
    )
    new_b = np.concatenate([np.asarray(params[2])[idx], spawn_B])
    new_params = (jnp.asarray(new_v), jnp.asarray(new_a), jnp.asarray(new_b))
    return new_params, jax.tree_util.tree_map(regrow, opt_state), len(new_v)


def num_params(params: SplatParams) -> int:
    """Total trainable scalars — k·(p + d² + d). Reported so SRM/MLP can be compared at matched size."""
    return int(sum(np.prod(np.shape(x)) for x in jax.tree_util.tree_leaves(params)))


def decay_mask(params: SplatParams) -> SplatParams:
    """Weight-decay mask for ``optax.adamw``: decay ``(V, A)``, not ``B``.

    AdamW's decoupled weight decay is a magnitude/complexity prior — sensible on the mixture weights
    ``V`` and scale/rotation ``A``, but ``B`` is a *position*, and decaying it is a literal spatial
    force pulling every splat's centre toward the chart origin every step, unrelated to anything the
    loss is asking for. Harmless on the torus/sphere (origin is a geometrically arbitrary point there,
    same distance-to-everywhere by symmetry) but actively wrong on a truncated environment like
    ``poincare_hyperbolic``, where the origin is the one region that's already easiest to fit (least
    curvature, typically the source) — measured there to visibly weaken the learned correction (see
    ``ntfields.py``'s use of this mask).
    """
    V, A, B = params
    return True, True, False
