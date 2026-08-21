"""CLI entry point: solve the Eikonal time-to-go on an Environment with a chosen strategy.

Generalizes the original ``torus.py``'s ``main`` to dispatch through the
``STRATEGIES`` registry instead of an inline solver dict, and to build the
environment (obstacles, slowness field, ground truth) from the ``ENVIRONMENTS``
registry (``srms/environments``) instead of module-level torus functions.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Literal

import jax
import mlflow
import numpy as np
import tyro

from srms.environments import ENVIRONMENTS, sampling
from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, hntfields, ntfields, pntfields, weak_supervision
from srms.viz import render, render_prediction
from srms.viz_3d import render_3d

STRATEGIES = {
    "eikonal": eikonal,
    "weak_supervision": weak_supervision,
    "ntfields": ntfields,
    "pntfields": pntfields,
    "hntfields": hntfields,
}

# Strategies whose field is the NTFields T = base/τ factorization, so they share `predict`'s signature.
_NTFIELDS_FAMILY = ("ntfields", "pntfields", "hntfields")


@dataclasses.dataclass
class Config:
    """Configuration for a self-supervised Eikonal solve on a Riemannian manifold.

    Every field below is a scene, method or budget choice; the *why* for each method's own knobs
    lives in that method's module docstring (``srms/methods/strategies/*.py``) rather than being
    duplicated here.

    Groups:
        manifold — ``environment`` (torus Tⁿ, K=0 and an abelian Lie group; sphere Sⁿ, K=+1;
            hyperbolic Hⁿ in the Poincaré ball or Lorentz Model, K=−1; ``so3``, K=¼ and a
            non-abelian Lie group) and ``dim``, its intrinsic dimension. The SRM backend is
            identical across all of them: it reads only ``log_map`` / ``jac_factor`` /
            ``metric_inv`` off the environment, which is the entire per-manifold
            delta. ``environments/test_manifolds.py`` verifies each against an exact identity before
            any training runs.
        method / backend — ``eikonal`` (free field; see its docstring for a known init defect),
            ``weak_supervision`` (RRT* prior the PDE refines), or the published baselines
            ``ntfields`` / ``pntfields`` / ``hntfields``, which share the ``T = base/τ`` field so the
            comparison isolates the objective. ``backend`` is ``srm`` (splat mixture) or ``mlp``.
        scene — ``start``, ``num_obstacles``, ``obstacle_radius`` (geodesic, ball obstacles — the
            default shape on every environment), ``slowness_max``, ``slow_width``; ``trunc_radius``
            bounds the hyperbolic chart (H^d is unbounded), shared by both hyperbolic environments
            so their domains are comparable (Lorentz's ``domain_radius = 2·artanh(trunc_radius)``,
            matching Poincaré's own ``wall_distance`` exactly). ``num_ray_obstacles``/``ray_length``/
            ``ray_thickness`` add optional capsule-thickened geodesic-ray obstacles on top of the
            ball obstacles, on both hyperbolic environments identically (off — 0 rays — by default).
        budget — ``steps``, ``lr``, ``num_collocation``, ``colloc_exclude_obstacles`` (drop ``sdf<0``
            collocation points, ``eikonal``/``ntfields``/``pntfields`` only), ``seed``, ``resolution``
            (scoring grid only).
        adaptive capacity (``srm``) — the model picks its own size: it grows where the residual is
            and stops when a densify pass buys less than ``densify_min_gain`` fractional residual
            reduction *per splat added*. That test is scale-free in both the loss and the spawn
            schedule, so one threshold transfers across manifolds, seeds and obstacle sets.
            ``max_splats`` is a runaway backstop, not a target; ``densify_freeze_frac`` settles the
            basis for the final stretch; ``scale_floor`` is the stabiliser without which a splat
            collapses onto the Eikonal kink. This is the capability a fixed-width MLP lacks —
            ``mlp.adapt`` is necessarily a no-op.

    Ground truth is computed only *after* training returns (see ``main``); nothing in the training
    path can reach it, which ``environments/test_selfsupervised.py`` enforces.

    """

    environment: Literal["torus", "plane", "sphere", "poincare_hyperbolic", "lorentz_hyperbolic"] = "torus"
    dim: int = 2
    method: Literal["eikonal", "weak_supervision", "ntfields", "pntfields", "hntfields"] = "eikonal"
    backend: Literal["srm", "mlp", "mlp_raw", "mlp_resnet"] = "srm"
    # scene
    start: tuple[float, ...] | None = None
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    trunc_radius: float = 0.9  # hyperbolic only
    num_ray_obstacles: int = 0  # hyperbolic only (poincare_hyperbolic / lorentz_hyperbolic)
    ray_length: tuple[float, float] = (0.8, 1.8)  # hyperbolic only
    ray_thickness: tuple[float, float] = (0.08, 0.15)  # hyperbolic only
    # causal weighting (all strategies)
    causal: bool = True
    causal_strength: float = 5.0
    causal_anneal: bool = True
    # eikonal strategy
    source_radius: float = 0.25
    n_sphere: int = 32
    physics_weight: float = 1.0
    # weak_supervision strategy (RRT*)
    rrt_iters: int = 350
    rrt_step: float = 0.5
    rrt_radius: float = 0.9
    roadmap_nodes: int = 300
    roadmap_gamma: float = 0.05
    roadmap_hop: int = 5
    base_reg: float = 1.0
    # ntfields strategy
    tau_bias: float = 4.0
    tau_min: float = 0.0
    # pntfields strategy (P-NTFields: viscosity + progressive speed scheduling)
    viscosity_eps: float = 0.01
    alpha_init: float = 0.5
    alpha_hold_frac: float = 0.25
    alpha_final: float = 1.05
    # hntfields strategy (H-NTFields: roadmap bounds + TD-NTFields PDE losses)
    hnt_lambda_e: float = 1e-2
    hnt_lambda_td: float = 1e-3
    hnt_lambda_n: float = 1e-3
    hnt_lambda_r: float = 1e-2
    hnt_lambda_c: float = 0.5
    hnt_dt: float = 0.02
    hnt_nodes: int = 300
    hnt_pool: int = 40000
    hnt_connect_radius: float = 1.5
    hnt_max_radius: float = 0.3
    hnt_detach_causal: bool = True
    # mlp backend
    mlp_width: int = 128
    mlp_depth: int = 3
    mlp_omega0: float = 30.0
    # srm backend — adaptive densification (3DGS-style grow/prune)
    densify: bool = True
    init_splats: int = 128
    max_splats: int = 4096
    densify_every: int = 400
    densify_min_gain: float = 1e-5
    densify_freeze_frac: float = 0.1
    spawn_per: int = 48
    prune_thresh: float = 5e-4
    spawn_scale: float = 0.3
    scale_floor: float = 0.07
    # training / output
    num_splats: int = 384
    num_collocation: int = 2048
    colloc_exclude_obstacles: bool = False
    steps: int = 4000
    lr: float = 1e-2
    init_scale: float = 0.35
    init_v_scale: float = 0.3
    resolution: int = 120
    seed: int = 1
    error_clip: float = 0.2
    checkpoint_every: int = 1500
    log_every: int = 25
    out_dir: str = "figures"


def _build_env(cfg: Config):
    """Construct the configured Environment, filling in a dim-appropriate default start if unset."""
    if cfg.environment == "torus":
        start = cfg.start if cfg.start is not None else (-1.5,) * cfg.dim
        return ENVIRONMENTS["torus"](
            start=start,
            dim=cfg.dim,
            num_obstacles=cfg.num_obstacles,
            obstacle_radius=cfg.obstacle_radius,
            slowness_max=cfg.slowness_max,
            slow_width=cfg.slow_width,
            seed=cfg.seed,
        )
    if cfg.environment == "plane":
        start = cfg.start if cfg.start is not None else (-1.5,) * cfg.dim
        return ENVIRONMENTS["plane"](
            start=start,
            dim=cfg.dim,
            num_obstacles=cfg.num_obstacles,
            obstacle_radius=cfg.obstacle_radius,
            slowness_max=cfg.slowness_max,
            slow_width=cfg.slow_width,
            seed=cfg.seed,
        )
    if cfg.environment == "poincare_hyperbolic":
        start = cfg.start if cfg.start is not None else (0.0,) * cfg.dim
        return ENVIRONMENTS["poincare_hyperbolic"](
            start=start,
            dim=cfg.dim,
            num_obstacles=cfg.num_obstacles,
            obstacle_radius=cfg.obstacle_radius,
            num_ray_obstacles=cfg.num_ray_obstacles,
            ray_length=cfg.ray_length,
            ray_thickness=cfg.ray_thickness,
            slowness_max=cfg.slowness_max,
            slow_width=cfg.slow_width,
            trunc_radius=cfg.trunc_radius,
            seed=cfg.seed,
        )
    if cfg.environment == "lorentz_hyperbolic":
        start = cfg.start if cfg.start is not None else (0.0,) * cfg.dim + (1.0,)
        # domain_radius = 2*artanh(trunc_radius): the same conversion PoincareHyperbolicEnvironment
        # applies internally (its own wall_distance), so one --trunc_radius sizes both hyperbolic
        # environments' domains identically instead of trunc_radius=0.9 (a Poincaré ball-fraction)
        # being handed to Lorentz as a *geodesic radius* directly — a ~3x-smaller domain than intended.
        domain_radius = 2.0 * float(np.arctanh(min(cfg.trunc_radius, 1.0 - 1e-6)))
        return ENVIRONMENTS["lorentz_hyperbolic"](
            start=start,  # apex (0,...,0,1) also satisfies H^n's <x,x>_eta=-1, same default as sphere
            n=cfg.dim,
            domain_radius=domain_radius,
            num_obstacles=cfg.num_obstacles,
            obstacle_radius=cfg.obstacle_radius,
            num_ray_obstacles=cfg.num_ray_obstacles,
            ray_length=cfg.ray_length,
            ray_thickness=cfg.ray_thickness,
            slowness_max=cfg.slowness_max,
            slow_width=cfg.slow_width,
            seed=cfg.seed,
        )
    if cfg.environment == "sphere":
        start = cfg.start if cfg.start is not None else (0.0,) * cfg.dim + (1.0,)
        return ENVIRONMENTS["sphere"](
            start=start,
            n=cfg.dim,
            num_obstacles=cfg.num_obstacles,
            obstacle_radius=cfg.obstacle_radius,
            slowness_max=cfg.slowness_max,
            slow_width=cfg.slow_width,
            seed=cfg.seed,
        )


def main(cfg: Config) -> None:
    """Solve the Eikonal PDE with obstacles, then score against a dense fast-marching grid.

    **Ground truth is computed only after ``strategy.solve`` returns, and that ordering is the
    point.** Every method here is self-supervised: the solvers see only the scene (``slowness``,
    ``sdf``), the analytic base geodesic, and collocation samples. Computing the fast-marching field
    up front and closing over it — as this function used to, for checkpoint figures — made it
    *possible in principle* for training to touch it. Deferring it makes that structurally impossible
    rather than merely true, which is worth more than a comment. ``environments/test_selfsupervised.py``
    additionally asserts that no strategy module so much as references ``ground_truth``.
    """
    env = _build_env(cfg)
    backend = BACKENDS[cfg.backend]
    dense = getattr(env, "has_dense_gt", cfg.dim == 2)  # env decides; 3-D grids are tractable now

    thetas = shape = gt = inside = coords = edges = None
    if dense:
        thetas, shape = env.grid(cfg.resolution)
        gt = env.ground_truth(cfg.resolution)
        inside = np.asarray(env.sdf(thetas)) < 0.0
        # duck-typed: only environments with a curvilinear chart (e.g. PoincareHyperbolicEnvironment's
        # Poincaré disk) define render_grid_xy/render_grid_edges_xy; torus/sphere fall back to
        # viz.render's default rectangular imshow(extent=...) path (coords=edges=None)
        if hasattr(env, "render_grid_xy"):
            coords = env.render_grid_xy(cfg.resolution)
        if hasattr(env, "render_grid_edges_xy"):
            edges = env.render_grid_edges_xy(cfg.resolution)

    roadmap = None
    if cfg.method == "weak_supervision":
        roadmap = sampling.build_roadmap(
            env, env.start, cfg.rrt_iters, cfg.rrt_step, cfg.rrt_radius, cfg.roadmap_nodes, cfg.seed
        )

    def predict_current(current) -> np.ndarray:
        if cfg.method == "weak_supervision":
            nodes, costs = roadmap
            return np.asarray(
                weak_supervision.predict(
                    backend, current, thetas, nodes, costs, cfg.roadmap_gamma, env, cfg.roadmap_hop
                )
            )
        if cfg.method in _NTFIELDS_FAMILY:
            return np.asarray(ntfields.predict(backend, current, thetas, env, cfg.tau_bias, cfg.tau_min))
        return np.asarray(eikonal.predict(backend, current, thetas, env))

    checkpoint = None
    if dense:

        def checkpoint(current, stepnum: int) -> None:
            """Mid-training snapshot of the *prediction only*.

            Deliberately GT-free: this callback runs inside the training loop, so anything it can see
            is something training could in principle be tuned against. It renders the field and
            reports its range; scoring happens once, after ``solve`` returns.
            """
            out_name = f"{cfg.environment}_ckpt_{stepnum}.png"
            field = predict_current(current)
            render_prediction(env, cfg, field, inside, shape, out_name=out_name)
            print(
                f"  [ckpt {stepnum}] saved {out_name}  T range [{np.nanmin(field):.3f}, {np.nanmax(field):.3f}]",
                flush=True,
            )


    run_name = f"{cfg.environment}-{cfg.method}-{cfg.backend}-d{cfg.dim}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "environment": cfg.environment,
                "method": cfg.method,
                "backend": cfg.backend,
                "dim": str(cfg.dim),
                # same for the srm/mlp pair of a scene -> group by this tag to compare backends head-to-head
                "comparison_group": f"{cfg.environment}-{cfg.method}-d{cfg.dim}",
            }
        )
        mlflow.log_params(dataclasses.asdict(cfg))

        def progress_fn(step: int, metrics: dict) -> None:
            mlflow.log_metrics(metrics, step=step)

        strategy = STRATEGIES[cfg.method]
        splat = strategy.solve(env, cfg, backend, checkpoint, progress_fn=progress_fn)
        with open(f"{cfg.out_dir}/splat.pkl", "wb") as f:  # save params + scene for the near-obstacle diagnostic
            pickle.dump(
                {
                    "splat": jax.tree_util.tree_map(np.asarray, splat),
                    "obstacles": env.obstacles,
                    "cfg": dataclasses.asdict(cfg),
                },
                f,
            )

        if not dense:
            print(f"{cfg.environment} dim={cfg.dim}: no dense-grid ground truth available; training done.")
            print(f"saved {cfg.out_dir}/splat.pkl  ({len(env.obstacles)} obstacles)")
            return

        # ---- scoring: the FIRST point at which ground truth exists in this process -----------------
        gt = env.ground_truth(cfg.resolution)
        out_name = f"{cfg.environment}_obstacles.png"
        prediction = predict_current(splat)
        metrics = render(env, cfg, gt, prediction, inside, shape, out_name=out_name, coords=coords, edges=edges)
        metrics["num_params"] = backend.num_params(splat)  # so srm/mlp are comparable at equal accuracy
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(f"{cfg.out_dir}/{out_name}")
        render_3d(env, cfg, gt, prediction, inside, shape, thetas)  # no-op off torus/S² (see srms/viz_3d.py)
        print(f"saved {cfg.out_dir}/{out_name}  ({len(env.obstacles)} obstacles)")
        print(
            f"RMS={metrics['rms']:.4e}  max|err|={metrics['max_abs']:.4e}  "
            f"rel_RMS={metrics['rel_rms']:.4e}  num_params={metrics['num_params']}"
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
