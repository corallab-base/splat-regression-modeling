"""NTFields training strategy: ``T(q_s, q_g) = ‖q_s − q_g‖ / τ(q_s, q_g)``.

Port of NTFields (Ni & Qureshi, ICLR 2023, arXiv 2210.00120) to this repo's *fixed-source* setting.
Arrival time is the geodesic distance divided by a network-predicted normalized speed ``τ ∈ (0, 1]``
(``τ=1`` ⇒ free space, straight-line unit-speed travel time; ``τ→0`` near obstacles inflates ``T``).
``τ`` is evaluated through whichever backend is passed in (``srms/methods/backends/{srm,mlp}``) via a
sigmoid, ``τ = τ_min + (1−τ_min)·σ(g(θ)+bias)`` — ``bias`` starts ``τ`` near 1 (free space) at init;
``tau_min=0`` (the default) recovers the *un-floored* paper formulation exactly, ``σ(·)`` alone
bounding ``τ`` to ``(0, 1)``. ``tau_min>0`` is an ablation knob (the original ``torus.py``'s
``solve_ntfields`` used ``tau_min=0.25`` to floor the ``τ→0 ⇒ T→∞`` runaway), off by default here.
``T(start)=0`` automatically since ``base(start)=0``.

**Loss — paper Eq. 4.** NTFields matches predicted speed ``S = 1/‖∇T‖`` against the ground-truth
speed model ``S⋆`` with the *isotropic* penalty

    |1 − √(S⋆/S)| + |1 − √(S/S⋆)|                                              (Eq. 4, one endpoint)

which is symmetric under ``S ↔ S⋆`` and uses a square root "to smooth the loss function's gradient".
Writing ``q = S⋆/S = ‖∇T‖_{g⁻¹} · S⋆``, this is ``|1 − √q| + |1 − 1/√q|`` — see ``isotropic_loss``.

.. note::
   This replaces an earlier ``mean((q + 1/q − 2)²)`` objective in this module. That form is also
   symmetric and zero iff ``q=1``, but it is quartic near the optimum, whereas Eq. 4 is L1-like —
   a different objective, not the paper's. The module documented itself as a straight port, so the
   loss was corrected to match; the old form lives on in ``weak_supervision.py``, which is this
   repo's own method and does not claim paper fidelity.

**Deviations from the paper** (see also ``srms/methods/ntfields_TODO.md``):

- *Fixed source, all goals.* This repo learns ``T(θ)``: the source is fixed and the goal ranges over
  the **entire manifold** — the field *is* the set of all goals, which is what makes it a value
  function. The paper instead learns a two-point ``T(q_s, q_g)`` with a symmetric ``⊗`` architecture
  and evaluates Eq. 4 at both endpoints, so only the ``q_g`` half of the loss exists here. This is a
  deliberate choice of problem setting, not a truncated version of one; a different *source* means a
  different field, which is the premise of the planned editability study. Everything else in the
  formulation carries over unchanged.
- *No causal weighting.* NTFields has no such term, so this strategy **ignores ``cfg.causal``**
  (it prints a note when the flag is set). H-NTFields' own ``L_C`` lives in ``hntfields.py``.
- *Speed model.* The paper's ``S⋆`` is a clipped obstacle-distance ramp (Eq. 3); this repo's
  environments supply their own smooth ``env.slowness`` (``S⋆ = 1/slowness``). That is a property of
  the *scene*, shared identically by every strategy and backend, so it does not bias the comparison.
- *Collocation.* The paper trains on a fixed dataset of sampled ``(q_s, q_g)`` pairs; this repo
  resamples collocation points every step (the house style of ``eikonal.py``/``weak_supervision.py``).
- *Backend.* The paper's ResNet + workspace-CNN encoder is replaced by this repo's ``srm``/``mlp``
  backends — that substitution is the point of the comparison, not an accident of it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.methods.strategies import training_aids
from srms.methods.strategies.eikonal import sample_collocation


def tau(backend, params, theta: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """Normalized speed ``τ(θ) = τ_min + (1−τ_min)·σ(g(θ)+bias) ∈ [τ_min, 1)``."""
    raw = backend.eval_raw(params, theta[None, :], env)[0, 0]
    return tau_min + (1.0 - tau_min) * jax.nn.sigmoid(raw + tau_bias)


def field_ntfields(backend, params, theta: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """``T(θ) = geodesic(θ, start) / τ(θ)``."""
    base = env.geodesic(theta, jnp.asarray(env.start, dtype=jnp.float32))
    return base / tau(backend, params, theta, env, tau_bias, tau_min)


def predict(backend, params, thetas: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """Evaluate the NTFields field over a batch of points, returned as [N]."""
    return jax.vmap(lambda t: field_ntfields(backend, params, t, env, tau_bias, tau_min))(thetas)


def time_grad(backend, params, thetas: jnp.ndarray, env, tau_bias: float, tau_min: float) -> jnp.ndarray:
    """``∇T`` at each θ, as [n, dim]. Shared by ``pntfields``/``hntfields``."""

    def t_single(x):
        return field_ntfields(backend, params, x, env, tau_bias, tau_min)

    return jax.vmap(jax.grad(t_single))(thetas)


def grad_norm(grad: jnp.ndarray, thetas: jnp.ndarray, env) -> jnp.ndarray:
    """``‖∇T‖_{g⁻¹}`` — the Riemannian gradient norm, i.e. the predicted ``1/S``."""
    metric = jax.vmap(env.metric_inv)(thetas)
    grad_sq = jnp.einsum("ni,nij,nj->n", grad, metric, grad)
    return jnp.sqrt(jnp.clip(grad_sq, 1e-12, None))


def speed_ratio(
    backend, params, thetas: jnp.ndarray, slow: jnp.ndarray, env, tau_bias: float, tau_min: float
) -> jnp.ndarray:
    """``q = S⋆/S = ‖∇T‖_{g⁻¹} / slowness`` (``S = 1/‖∇T‖``, ``S⋆ = 1/slowness``); ``q=1`` solves Eikonal."""
    grad = time_grad(backend, params, thetas, env, tau_bias, tau_min)
    return grad_norm(grad, thetas, env) / slow


def isotropic_loss(q: jnp.ndarray) -> jnp.ndarray:
    """NTFields Eq. 4 (single endpoint): ``|1 − √(S⋆/S)| + |1 − √(S/S⋆)| = |1 − √q| + |1 − 1/√q|``."""
    root = jnp.sqrt(jnp.clip(q, 1e-12, None))
    return jnp.abs(1.0 - root) + jnp.abs(1.0 - 1.0 / root)


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit the NTFields ``T=base/τ`` factorization with the paper's isotropic speed loss (Eq. 4).

    ``progress_fn(step, metrics)``, if given, is called every ``cfg.log_every`` steps with a dict of
    scalar training metrics (``loss``, ``eikonal``) — e.g. to log to mlflow (see ``run.py``).

    Two numerical notes. Eq. 4's ``1/√q`` term has a genuine singularity as ``q → 0``, and at
    ``tau_min=0`` an unfloored ``1/τ`` can blow up on its own, with no obstacle-band gating to damp
    collocation points that wander near it — hence the clip inside ``isotropic_loss``, matching
    ``weak_supervision.py`` and this repo's archived PINN scripts for the same loss family. The
    optimizer is AdamW, masked per ``backend.decay_mask`` so decay lands on magnitude/shape
    parameters and not on srm's splat centres — see ``srm.py``'s ``decay_mask`` docstring for the
    measured effect (on ``poincare_hyperbolic``, unmasked decay on centres measurably weakened the
    learned correction by pulling every splat toward the chart origin every step, an effect with no
    analogue in the released MLP-based implementation this strategy otherwise follows). Weight decay
    is 0.02, not the released implementation's 0.1: measured on ``lorentz_hyperbolic`` (deterministic
    XLA ops, see ``autoresearch_results/lorentz_hyperbolic_srm_results.tsv`` iter 14), 0.1 over-shrinks
    ``V``/``A`` for this scene (rel_RMS 0.1588 at 0.1, 0.1488 at 0.02, 0.1536 at 0.0 — 0.02 is a real
    optimum, not a fidelity concession).
    """
    if cfg.causal:
        print("[ntfields] paper-faithful: cfg.causal ignored (NTFields has no causal weighting)", flush=True)

    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)

    # AdamW; weight_decay=0.02 tuned for this repo's splat backend (see solve()'s docstring), not
    # the released implementation's 0.1 (NTFields models/model_3d.py).
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=0.02, mask=backend.decay_mask)
    )
    opt_state = optimizer.init(params)

    def loss_fn(p, colloc, slow):
        q = speed_ratio(backend, p, colloc, slow, env, cfg.tau_bias, cfg.tau_min)
        eikonal_loss = jnp.mean(isotropic_loss(q))
        return eikonal_loss, {"eikonal": eikonal_loss}

    @jax.jit
    def step(p, state, colloc, slow):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, colloc, slow)
        updates, state = optimizer.update(grads, state, p)
        p = backend.post_step(optax.apply_updates(p, updates), cfg, env)
        return p, state, loss, aux

    def spawn_residual(pts):
        """Where to put new capacity: the Eq. 4 speed mismatch at each candidate centre."""
        return isotropic_loss(speed_ratio(backend, params, pts, env.slowness(pts), env, cfg.tau_bias, cfg.tau_min))

    # The model, not the schedule, decides how many splats it needs (see training_aids).
    densifier = training_aids.DensifyController(cfg)
    progress = trange(cfg.steps, desc="ntfields")
    for i in progress:
        colloc = sample_collocation(env, rng, cfg.num_collocation, cfg.source_radius, cfg.colloc_exclude_obstacles)
        slow = env.slowness(colloc)
        params, opt_state, loss, aux = step(params, opt_state, colloc, slow)
        densifier.record(loss)
        if densifier.should_densify(i, len(params[0])):
            params, opt_state, k = backend.adapt(params, opt_state, spawn_residual, env, cfg, rng)
            if k is not None:
                progress.write(f"[ntfields] step {i}: densify → {k} splats ({backend.num_params(params)} params)")
        if i % cfg.log_every == 0:
            progress.set_description(f"ntfields — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
            if progress_fn is not None:
                progress_fn(
                    i,
                    {
                        "loss": float(loss),
                        "eikonal": float(aux["eikonal"]),
                        "num_params": backend.num_params(params),
                    },
                )
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    print(f"[ntfields] {densifier.summary(len(params[0]) if isinstance(params, tuple) else 0)}", flush=True)
    print(f"[ntfields] final model: {backend.num_params(params)} trainable parameters", flush=True)
    return params
