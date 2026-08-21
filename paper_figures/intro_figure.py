"""Intro figure: SRM predicted time-to-go on the torus, sphere, and hyperbolic plane, as three
separate same-shape figures — ``intro_figure_torus.png``, ``intro_figure_sphere.png``,
``intro_figure_hyperbolic.png``.

``RUNS`` is the dict to edit: one ``srms.run.Config`` per figure, controlling which training run
backs it (method, backend, seed, steps, ``out_dir``, ...). If ``out_dir/splat.pkl`` already exists
(e.g. the torus/sphere defaults below, which point at ``run_baselines.sh``-produced checkpoints)
it's loaded as-is; otherwise ``main`` trains it fresh through the normal ``srms.run.main`` pipeline
before rendering — so pointing an entry at a new ``out_dir`` (or changing its method/backend/seed)
is enough to swap in a different run.

Torus/sphere render as 3D ambient surfaces via ``srms/visualization/utils_3d.py``'s matplotlib
``render_ambient_mpl`` (the Plotly-based ``render_surface`` needs ``plotly``/``kaleido``, not
installed here). Hyperbolic renders as the Poincaré disk instead (``srms.viz``'s curvilinear
``pcolormesh`` chart, the same one ``srms.run.main`` uses for the flat [GT | pred | error] figure) —
a 2D chart reads far more legibly for H^2 than the unbounded, self-occluding hyperboloid sheet.
No titles/axis labels (no captions) on any panel; only the hyperbolic panel keeps its colorbar,
since it's the one panel where the field's absolute scale isn't otherwise legible from the plot.
Every panel draws white iso-value contours of the field (8 evenly-spaced levels, ``_contour_levels``)
— baked into the surface's own facecolors for torus/sphere (``render_ambient_mpl``'s
``contour_levels``; mplot3d has no real depth buffer, so a separate line overlay on a self-occluding
surface renders behind it unpredictably) and as a normal ``ax.contour`` overlay on the flat
Poincaré disk, where that problem doesn't exist.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import savefig, use_latex_fonts
from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, ntfields
from srms.run import Config, _build_env
from srms.run import main as train_main
from srms.viz import _draw_field
from srms.visualization.utils_3d import close_periodic, render_ambient_mpl, torus_embedding

use_latex_fonts()

# ---- edit this to change which training run backs each figure --------------------------------
RUNS: dict[str, Config] = {
    "torus": Config(
        environment="torus", method="ntfields", backend="srm", dim=2,
        # out_dir renamed (posv = positive-V init, srms/methods/backends/srm.py's init_params):
        # the old "torus_ntfields_srm*" checkpoints predate that fix, so reusing that out_dir would
        # have _ensure_trained silently keep serving pre-fix runs instead of retraining.
        resolution=120, seed=1, out_dir="figures/compare/torus_ntfields_posv_srm",
    ),
    "sphere": Config(
        environment="sphere", method="ntfields", backend="srm", dim=2, tau_min=0.25,
        resolution=120, seed=1, out_dir="figures/compare/sphere_ntfields_tau025_posv_srm",
    ),
    "lorentz_hyperbolic": Config(
        environment="lorentz_hyperbolic", method="ntfields", backend="srm", dim=2,
        # steps/num_collocation left at Config's defaults (4000/2048) — matches torus/sphere above,
        # and results_figure.py's _MLP_OVERRIDES mirrors this budget so the aggregate table's
        # fairness check still passes.
        resolution=120, seed=1, out_dir="paper_figures/runs/lorentz_hyperbolic_ntfields_srm_4k",
    ),
}

# Output filename slug per RUNS key (the hyperbolic run is keyed by its environment name, but the
# figure is named after the chart it renders).
_FILE_SLUGS = {"torus": "torus", "sphere": "sphere", "lorentz_hyperbolic": "hyperbolic"}

FIGSIZE = (6.0, 6.0)  # shared by all three figures

# Grid resolution to *render* at, independent of each run's training/scoring resolution (evaluating
# a trained splat at a finer grid than it was scored on is just more points, not a retrain).
_RENDER_RESOLUTION = {"torus": 500, "sphere": 500}

# How far render_ambient_mpl's set_box_aspect zooms in to fill the (axis-off) panel — tuned per
# shape, the max that doesn't clip a corner off the silhouette. A flat, wide torus and a round
# sphere fill a square viewport differently, so one shared zoom leaves the sphere with a visible
# margin (round shape well inside its bounding square) while already being near-tight for the
# torus (wide, low-profile — closer to the square's own aspect).
_ZOOM = {"torus": 1.5, "sphere": 1.8}

_VIEW_ANGLE = {"torus": (20.0, 45.0), "sphere": (60.0, -50.0)}  # (elev, azim) degrees

# Methods sharing the NTFields T = base/tau factorization, whose predict() takes the extra
# tau_bias/tau_min args — mirrors srms.run.main's _NTFIELDS_FAMILY dispatch.
_NTFIELDS_FAMILY = ("ntfields", "pntfields", "hntfields")


def _contour_levels(field: np.ndarray) -> list[float]:
    """8 evenly-spaced iso-value levels spanning the field, excluding the (degenerate) 0 and max
    endpoints — matches ``srms/viz_3d.py``'s ``_render_manifold_3d`` convention."""
    vmax = float(np.nanmax(field))
    return np.linspace(0.0, vmax, 20)[1:-1].tolist()


def _ensure_trained(cfg: Config) -> None:
    """Train ``cfg``'s run if its checkpoint doesn't exist yet; otherwise a no-op (reused as-is)."""
    if not Path(f"{cfg.out_dir}/splat.pkl").exists():
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        train_main(cfg)


def _predict_field(cfg: Config, backend, splat, thetas, env) -> np.ndarray:
    """Evaluate the trained splat's field, dispatching on method the same way ``srms.run.main`` does."""
    if cfg.method in _NTFIELDS_FAMILY:
        return np.asarray(ntfields.predict(backend, splat, thetas, env, cfg.tau_bias, cfg.tau_min))
    return np.asarray(eikonal.predict(backend, splat, thetas, env))


def _load_splat(cfg: Config):
    """Load a trained (env, backend, splat) triple from ``cfg.out_dir``'s checkpoint."""
    with open(f"{cfg.out_dir}/splat.pkl", "rb") as f:
        ckpt = pickle.load(f)
    saved_cfg = Config(**ckpt["cfg"])  # the exact config training ran with, not RUNS[name]
    env = _build_env(saved_cfg)
    backend = BACKENDS[saved_cfg.backend]
    splat = jax.tree_util.tree_map(jnp.asarray, ckpt["splat"])
    if any(bool(jnp.isnan(leaf).any()) for leaf in jax.tree_util.tree_leaves(splat)):
        # A known SRM training-instability failure mode (documented for sphere/weak_supervision+srm
        # and hyperbolic elsewhere in this codebase): a blown-up run silently checkpoints NaN
        # params, which plot_surface then renders as a near-empty wireframe of stray facets rather
        # than erroring — misleadingly looking like a rendering bug instead of a bad checkpoint.
        raise ValueError(f"{cfg.out_dir}/splat.pkl has NaN parameters — pick a different out_dir or retrain")
    return saved_cfg, env, backend, splat


def _save_ambient_panel(name: str, cfg: Config) -> None:
    """Torus/sphere: render the predicted field as a 3D ambient surface, no title, no colorbar."""
    saved_cfg, env, backend, splat = _load_splat(cfg)
    resolution = _RENDER_RESOLUTION.get(name, saved_cfg.resolution)
    thetas, shape = env.grid(resolution)
    field = _predict_field(saved_cfg, backend, splat, thetas, env).reshape(shape)
    inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
    field = np.where(inside, np.nan, field)

    if name == "torus":
        angles = np.asarray(thetas).reshape(shape + (2,))
        x, y, z = torus_embedding(angles[..., 0], angles[..., 1])
        start_xyz = tuple(float(v) for v in torus_embedding(np.array(env.start[0]), np.array(env.start[1])))
        obstacle_xyz = [tuple(float(v) for v in torus_embedding(np.array(o[0]), np.array(o[1]))) for o in env.obstacles]
        periodic = (True, True)
    else:  # sphere: env.grid() already returns ambient unit-vector points
        points = np.asarray(thetas).reshape(shape + (-1,))
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        start_xyz = tuple(float(v) for v in env.start)
        obstacle_xyz = [tuple(float(v) for v in o[:-1]) for o in env.obstacles]
        periodic = (False, True)

    x, y, z, field = (close_periodic(a, periodic) for a in (x, y, z, field))

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    render_ambient_mpl(
        ax, x, y, z, field, start_xyz=start_xyz, obstacle_xyz=obstacle_xyz,
        contour_levels=_contour_levels(field), title="", zoom=_ZOOM[name],
        elev=_VIEW_ANGLE[name][0], azim=_VIEW_ANGLE[name][1],
    )
    # No axis/title/colorbar on this panel (render_ambient_mpl already turns the 3D axis off), so
    # there's nothing tight_layout needs room for — let the axes fill the whole figure and leave
    # cropping the remaining whitespace to savefig's bbox_inches="tight".
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path = f"paper_figures/intro_figure_{_FILE_SLUGS[name]}.png"
    savefig(fig, out_path)
    plt.close(fig)


def _save_poincare_panel(cfg: Config) -> None:
    """Hyperbolic: render the predicted field on the Poincaré disk chart, no title, with colorbar."""
    saved_cfg, env, backend, splat = _load_splat(cfg)
    thetas, shape = env.grid(saved_cfg.resolution)
    field = _predict_field(saved_cfg, backend, splat, thetas, env).reshape(shape)
    inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
    img = np.where(inside, np.nan, field)

    coords = env.render_grid_xy(saved_cfg.resolution)
    edges = env.render_grid_edges_xy(saved_cfg.resolution)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    handle, mesh1, mesh2 = _draw_field(ax, env, img, shape, coords, edges, "viridis", 0.0, float(np.nanmax(img)))
    ax.contour(mesh1, mesh2, img, levels=_contour_levels(img), colors="white", linewidths=0.7, alpha=0.75)
    ax.contour(mesh1, mesh2, inside.astype(float), levels=[0.5], colors="black", linewidths=1.2)
    ax.plot(*env.render_marker_deg(), "*", color="red", markersize=14, markeredgecolor="white")
    # No facecolor and no spines/ticks — just the disk itself and the unit-circle boundary
    # `_draw_field` already draws as a patch (kept: it's the disk's actual edge, not a plot frame).
    ax.set_axis_off()
    fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = "paper_figures/intro_figure_hyperbolic.png"
    savefig(fig, out_path)
    plt.close(fig)


def main() -> None:
    for cfg in RUNS.values():
        _ensure_trained(cfg)

    _save_ambient_panel("torus", RUNS["torus"])
    _save_ambient_panel("sphere", RUNS["sphere"])
    _save_poincare_panel(RUNS["lorentz_hyperbolic"])


if __name__ == "__main__":
    main()
