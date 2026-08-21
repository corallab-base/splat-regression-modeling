"""Results figure: SRM vs MLP predicted time-to-go against ground truth, as one 3x3 grid of flat 2D
chart panels — rows are manifolds (torus, sphere, hyperbolic H^2), columns are
(ground truth | SRM | MLP).

Reuses ``paper_figures/intro_figure.py``'s exact ``RUNS`` as the SRM column, and derives the MLP
column from it via ``dataclasses.replace(backend="mlp", ...)`` rather than a separately
hand-specified config. **That's the one place to change a row's obstacle set**: every scene field
(``num_obstacles``, ``obstacle_radius``, ``seed``, ...) comes from ``intro_figure.RUNS``'s Config for
that manifold, so editing it there changes the ground truth *and* both predictions for that row
together — obstacle sampling depends only on those shared scene fields, not on ``backend``/``method``
(verified: the existing ``*_srm``/``*_mlp`` checkpoint pairs this reuses already have identical
sampled obstacles). If a run's checkpoint doesn't exist yet it's trained fresh via
``intro_figure._ensure_trained`` (the normal ``srms.run.main`` pipeline), same as intro_figure.py.
The 3x3 figure itself is drawn from these ``RUNS`` alone (one representative, seed=1, run per
manifold/backend) — the aggregate table described below is a separate, wider sweep over 5 seeds.

Rendering goes through ``srms.viz``'s own ``_draw_field`` — the same chart ``srms.run.main`` draws
its [GT | prediction | error] figure with: rectangular ``imshow(extent=...)`` for torus/sphere,
curvilinear ``pcolormesh`` (Poincaré disk) for hyperbolic — the manifold's native flat chart, not a
3D embedding (contrast ``intro_figure.py``).

The hyperbolic row is drawn on a much finer grid than it was trained/scored on
(``_CHART_RENDER_RESOLUTION``): ``imshow`` (torus/sphere) embeds its array as a resampled, filtered
raster even in a vector PDF, so the training-resolution grid still looks smooth; ``pcolormesh``
(Poincaré disk) instead draws one flat-shaded polygon per grid cell as real PDF vector paths, so at
that same resolution the panel — and its obstacle silhouette, a marching-squares contour over the
same coarse grid — visibly pixelates once large on the page. Evaluating a trained splat (and
re-running fast marching for ground truth) at a finer grid than it was scored on is just more
points, not a retrain — same trick as ``intro_figure.py``'s own ``_RENDER_RESOLUTION``.

Also prints one LaTeX table (final loss, final error, parameter count; one row-group per metric,
one row per manifold) comparing RSRM against MLP as mean +/- std over ``SEEDS`` = 5 training seeds
per manifold/backend (``SEED_RUNS``, built from ``RUNS`` via ``_seed_variant``: same scene/method,
seed 1 reuses ``RUNS``'s own checkpoint, seeds 2-5 get their own ``out_dir`` and — since scene
sampling depends on ``cfg.seed`` too (see ``srms.run._build_env``) — their own independently sampled
obstacles, so each seed is a full independent trial, not just a different init). Error/parameters
are recomputed from each checkpoint itself (same philosophy as ``make_tables.py``: never scraped
from a log that could disagree with the model). Loss isn't recoverable from the checkpoint alone
(``srms.run.main`` never saves it — see its ``splat.pkl`` dump) so it comes from the mlflow run that
produced it, keyed by ``out_dir`` (``_query_training_run``). Also prints each manifold's iteration
count; if RSRM and MLP (or seeds within one backend) don't all match, that's an unfair comparison and
this raises rather than printing a table that hides it.
"""

from __future__ import annotations

import dataclasses
import math

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from paper_figures.intro_figure import RUNS as SRM_RUNS
from paper_figures.intro_figure import _contour_levels, _ensure_trained, _load_splat, _predict_field
from paper_figures.style import savefig, use_latex_fonts
from srms.run import Config
from srms.viz import _draw_field

use_latex_fonts()



# Per-manifold exceptions to the default backend="mlp" derivation below — a different MLP flavor,
# network size, and/or training budget than the SRM run it's compared against. lorentz_hyperbolic's
# baseline is mlp_raw (mlp.py without Fourier features — the plainer, more standard MLP) at a wider
# 256x4 network, trained at the same steps/num_collocation as RUNS["lorentz_hyperbolic"] itself
# (intro_figure.py's Config defaults, 4000/2048 — matching torus/sphere) so main()'s
# equal-iteration-count check still passes. out_dir is spelled out explicitly rather than derived
# (the generic "*_srm" -> "*_mlp" rename below would otherwise collide with an older checkpoint at
# a stale budget).
_MLP_OVERRIDES: dict[str, dict] = {
    "lorentz_hyperbolic": {
        "backend": "mlp_raw",
        "mlp_depth": 4,
        "mlp_width": 256,
        "out_dir": "paper_figures/runs/lorentz_hyperbolic_ntfields_mlp_raw_4k",
    },
}


def _mlp_variant(cfg: Config, name: str) -> Config:
    """``cfg`` with backend switched to the MLP baseline for manifold ``name`` — ``"mlp"`` by
    default, with scene/method/resolution/seed/steps/num_collocation untouched (so it's guaranteed
    to describe the same obstacle set and training budget as ``cfg`` itself), unless ``name`` has an
    entry in ``_MLP_OVERRIDES``. ``out_dir`` mirrors this repo's own ``*_srm`` -> ``*_mlp`` checkpoint
    naming convention when not overridden."""
    overrides = _MLP_OVERRIDES.get(name, {})
    out_dir = overrides.get(
        "out_dir",
        cfg.out_dir[: -len("srm")] + "mlp" if cfg.out_dir.endswith("srm") else f"{cfg.out_dir}_mlp",
    )
    extra = {k: v for k, v in overrides.items() if k not in ("backend", "out_dir")}
    return dataclasses.replace(cfg, backend=overrides.get("backend", "mlp"), out_dir=out_dir, **extra)


def _seed_variant(cfg: Config, seed: int) -> Config:
    """``cfg`` at a different seed — same scene spec, method, and backend, but (since scene sampling
    depends on ``cfg.seed`` too, see ``srms.run._build_env``) an independently sampled scene *and* a
    fresh training trial. ``seed == cfg.seed`` returns ``cfg`` itself unchanged (reuses its existing
    checkpoint rather than colliding with it); any other seed gets its own ``out_dir`` suffix so it
    trains into its own checkpoint instead of overwriting it."""
    if seed == cfg.seed:
        return cfg
    return dataclasses.replace(cfg, seed=seed, out_dir=f"{cfg.out_dir}_seed{seed}")


# {manifold: {"srm": Config, "mlp": Config}} — see module docstring: edit intro_figure.RUNS, not
# this. The single representative (seed=1) run per manifold/backend that the 3x3 figure plots.
RUNS: dict[str, dict[str, Config]] = {name: {"srm": cfg, "mlp": _mlp_variant(cfg, name)} for name, cfg in SRM_RUNS.items()}

SEEDS = (1, 2, 3, 4, 5)
# {manifold: {"srm": [Config x len(SEEDS)], "mlp": [...]}} — what the aggregate table is computed
# over. seed 1 in each list *is* RUNS[name][backend] (see _seed_variant), so no run is trained twice.
SEED_RUNS: dict[str, dict[str, list[Config]]] = {
    name: {backend: [_seed_variant(cfg, seed) for seed in SEEDS] for backend, cfg in pair.items()}
    for name, pair in RUNS.items()
}

_ROW_TITLES = {"torus": "Torus $T^2$", "sphere": "Sphere $S^2$", "lorentz_hyperbolic": "Hyperbolic $H^2$"}
_COLUMN_TITLES = ["Ground truth", "RSRM", "MLP"]
# Manifold display name for the aggregate table's rows (RUNS's own iteration order: torus, sphere,
# lorentz_hyperbolic -> "Hyperbolic", matching intro_figure._FILE_SLUGS's file-naming convention).
_MANIFOLD_LABELS = {"torus": "Torus", "sphere": "Sphere", "lorentz_hyperbolic": "Hyperbolic"}

PANEL_SIZE = (4.6, 4.6)  # inches, per grid cell

# Grid resolution to *render* the Poincaré-disk (curvilinear-chart) row at, decoupled from that
# run's own cfg.resolution (training/scoring) — see module docstring for why pcolormesh needs this
# and imshow doesn't.
_CHART_RENDER_RESOLUTION = 500


def _load(cfg: Config):
    """Load ``cfg``'s trained checkpoint. Returns ``(env, saved_cfg, backend, splat)``."""
    saved_cfg, env, backend, splat = _load_splat(cfg)
    return env, saved_cfg, backend, splat


def _eval(env, saved_cfg: Config, backend, splat, resolution: int):
    """Evaluate the trained splat's field on ``env``'s dense grid at ``resolution`` scalars/point —
    ``resolution`` need not equal ``saved_cfg.resolution``; more evaluation points isn't a retrain.
    Returns ``(thetas, shape, pred)``."""
    thetas, shape = env.grid(resolution)
    pred = _predict_field(saved_cfg, backend, splat, thetas, env).reshape(shape)
    return thetas, shape, pred


def _query_training_loss(out_dir: str) -> float:
    """Final training loss for ``out_dir``'s most recent finished mlflow run. ``splat.pkl`` (see
    ``srms/run.py``'s ``main``) never saves a loss curve — only the trained params — so this has to
    come from the run log that produced the checkpoint rather than the checkpoint itself."""
    runs = mlflow.search_runs(
        experiment_names=["srms"],
        filter_string=f"params.out_dir = '{out_dir}' and attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"no finished mlflow run logged for out_dir={out_dir!r} — the results table needs its "
            "final loss, which only the run log carries (retrain via srms.run if the checkpoint "
            "predates its mlflow run, e.g. after clearing mlruns/)"
        )
    row = runs.iloc[0]
    loss = row.get("metrics.loss")
    if loss is None or (isinstance(loss, float) and math.isnan(loss)):
        raise RuntimeError(f"mlflow run {row['run_id']} for out_dir={out_dir!r} logged no 'loss' metric")
    return float(loss)


def _stats_for(cfg: Config) -> dict:
    """One trained run's (loss, error, params, steps) — error/params recomputed from the checkpoint
    on its own scene (each seed samples an independent one, see ``_seed_variant``); loss comes from
    its mlflow run (``_query_training_loss``)."""
    env, saved_cfg, backend, splat = _load(cfg)
    thetas, shape, pred = _eval(env, saved_cfg, backend, splat, saved_cfg.resolution)
    inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
    gt = np.asarray(env.ground_truth(saved_cfg.resolution)).reshape(shape)
    err = np.where(inside, np.nan, pred - gt)
    return {
        "loss": _query_training_loss(saved_cfg.out_dir),
        "error": float(np.sqrt(np.nanmean(err**2))),
        "params": int(backend.num_params(splat)),
        "steps": saved_cfg.steps,
    }


def _aggregate(cfgs: list[Config]) -> dict:
    """Per-seed ``_stats_for`` over ``cfgs`` (same manifold/backend, one entry per ``SEEDS``),
    reduced to ``(mean, std)`` for loss/error/params. ``steps`` is carried through unreduced — it
    must be identical across every seed, or the caller's fairness check has nothing to compare."""
    runs = [_stats_for(cfg) for cfg in cfgs]
    steps = {r["steps"] for r in runs}
    if len(steps) != 1:
        raise ValueError(
            f"mismatched training budgets across seeds: steps={sorted(steps)} for {[c.out_dir for c in cfgs]} "
            "— comparison is unfair; retrain the stale ones to match"
        )
    agg = {"steps": steps.pop()}
    for key in ("loss", "error", "params"):
        values = np.array([r[key] for r in runs], dtype=float)
        agg[key] = (float(values.mean()), float(values.std()))
    return agg


def _fmt_num(v: float) -> str:
    """3 significant figures; scientific notation for very small magnitudes (losses/errors)."""
    if v == 0:
        return "0"
    if abs(v) < 1e-3:
        mantissa, exp = f"{v:.2e}".split("e")
        return rf"{float(mantissa):.2g} \times 10^{{{int(exp)}}}"
    return f"{float(f'{v:.3g}'):g}"


def _fmt_stat(mean: float, std: float) -> str:
    return rf"{_fmt_num(mean)} \pm {_fmt_num(std)}"


def _fmt_int_stat(mean: float, std: float) -> str:
    """Parameter counts: whole numbers, not 3-sig-fig-rounded."""
    return rf"{mean:.0f} \pm {std:.0f}"


# (row label, stats key, formatter for one (mean, std) pair)
_METRICS = [
    ("Final Loss", "loss", _fmt_stat),
    ("Final Error", "error", _fmt_stat),
    ("Parameters", "params", _fmt_int_stat),
]


def _agg_row(first_col: str, manifold_label: str, srm_stat: tuple[float, float], mlp_stat: tuple[float, float], fmt) -> str:
    """One table row, bolding whichever backend's mean is smaller (better)."""
    srm_wins = srm_stat[0] <= mlp_stat[0]
    srm_s, mlp_s = fmt(*srm_stat), fmt(*mlp_stat)
    srm_cell = f"$\\boldsymbol{{{srm_s}}}$" if srm_wins else f"${srm_s}$"
    mlp_cell = f"$\\boldsymbol{{{mlp_s}}}$" if not srm_wins else f"${mlp_s}$"
    return f"        {first_col} & {manifold_label} & {srm_cell} & {mlp_cell} \\\\"


def _print_results_table(stats: dict[str, tuple[dict, dict]]) -> None:
    """Print the single aggregate LaTeX table: one \\multirow group per metric, one row per
    manifold (in ``stats``'s own iteration order — ``RUNS``'s, i.e. torus/sphere/hyperbolic)."""
    manifolds = list(stats.keys())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"    \caption{Comparison of our approach with \ac{MLP} on three manifolds. Mean and "
        r"standard deviation are reported across five training seeds.}",
        r"    \label{tab:results}",
        r"    \begin{tabular}{llcc}",
        r"        \toprule",
        r"        & & \textbf{RSRM (Ours)} & \textbf{MLP} \\",
        r"        \midrule",
    ]
    for i, (label, key, fmt) in enumerate(_METRICS):
        if i > 0:
            lines.append(r"        \midrule")
        for j, name in enumerate(manifolds):
            srm_agg, mlp_agg = stats[name]
            first_col = rf"\multirow{{{len(manifolds)}}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(_agg_row(first_col, _MANIFOLD_LABELS[name], srm_agg[key], mlp_agg[key], fmt))
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    print("\n".join(lines))


def main() -> None:
    for pair in SEED_RUNS.values():
        for cfgs in pair.values():
            for cfg in cfgs:
                _ensure_trained(cfg)

    # ---- aggregate table, over all 5 seeds -----------------------------------------------------
    table_stats: dict[str, tuple[dict, dict]] = {}
    for name, pair in SEED_RUNS.items():
        srm_agg = _aggregate(pair["srm"])
        mlp_agg = _aggregate(pair["mlp"])
        print(
            f"% {_MANIFOLD_LABELS[name]}: RSRM trained for {srm_agg['steps']} iterations, MLP for "
            f"{mlp_agg['steps']} iterations (x{len(SEEDS)} seeds each)"
        )
        if srm_agg["steps"] != mlp_agg["steps"]:
            raise ValueError(
                f"{name}: RSRM ran {srm_agg['steps']} iterations but MLP ran {mlp_agg['steps']} — "
                "comparison is unfair (mismatched training budgets); fix out_dir/steps before comparing"
            )
        table_stats[name] = (srm_agg, mlp_agg)
    _print_results_table(table_stats)
    print()

    # ---- 3x3 figure, from the seed=1 representative runs only ----------------------------------
    # A 4th, slim column holds each row's colorbar in its own dedicated axes (`cax=`) rather than
    # `fig.colorbar(ax=...)` carving space out of the MLP column's own axes: on the curvilinear
    # (Poincaré-disk) hyperbolic row, `_draw_field`'s `ax.set_aspect("equal")` (default
    # `adjustable="box"`) shrinks whichever axes loses width to a colorbar, so the MLP panel's
    # square rendered visibly smaller than GT's/RSRM's (torus/sphere's `imshow(aspect="auto")`
    # masks the same effect by just stretching to fill whatever box it's given — that's why this
    # only showed up in the hyperbolic row). Reserving the colorbar its own column keeps all three
    # data panels in every row exactly the same size regardless of aspect mode.
    n_rows = len(RUNS)
    cbar_frac = 0.06
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(PANEL_SIZE[0] * (3 + cbar_frac), PANEL_SIZE[1] * n_rows),
        gridspec_kw={"width_ratios": [1, 1, 1, cbar_frac]},
    )

    for row, (name, pair) in enumerate(RUNS.items()):
        # srm/mlp share one scene (see module docstring), so obstacles/chart geometry are computed
        # once here rather than once per column.
        env, saved_cfg, srm_backend, srm_splat = _load(pair["srm"])
        _, mlp_cfg, mlp_backend, mlp_splat = _load(pair["mlp"])

        # Scoring grid — each run's own cfg.resolution, exactly what it was trained/scored at.
        thetas, shape, srm_pred = _eval(env, saved_cfg, srm_backend, srm_splat, saved_cfg.resolution)
        _, _, mlp_pred = _eval(env, mlp_cfg, mlp_backend, mlp_splat, mlp_cfg.resolution)
        inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
        gt = np.asarray(env.ground_truth(saved_cfg.resolution)).reshape(shape)

        # Render grid — curvilinear (Poincaré-disk) charts redraw at _CHART_RENDER_RESOLUTION so
        # their pcolormesh doesn't pixelate (see module docstring); torus/sphere's imshow already
        # looks smooth at the scoring resolution, so they render unchanged.
        render_resolution = _CHART_RENDER_RESOLUTION if hasattr(env, "render_grid_xy") else saved_cfg.resolution
        if render_resolution != saved_cfg.resolution:
            thetas, shape, srm_pred = _eval(env, saved_cfg, srm_backend, srm_splat, render_resolution)
            _, _, mlp_pred = _eval(env, mlp_cfg, mlp_backend, mlp_splat, render_resolution)
            inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
            gt = np.asarray(env.ground_truth(render_resolution)).reshape(shape)

        blocked = inside.astype(float)
        gt_img = np.where(inside, np.nan, gt)
        vmax = float(np.nanmax(gt_img))
        # One shared set of levels (from ground truth) for all three columns of the row, so SRM's
        # and MLP's contours are directly comparable to ground truth's rather than each auto-scaled
        # to its own range.
        levels = _contour_levels(gt_img)
        marker = env.render_marker_deg()
        coords = env.render_grid_xy(render_resolution) if hasattr(env, "render_grid_xy") else None
        edges = env.render_grid_edges_xy(render_resolution) if hasattr(env, "render_grid_edges_xy") else None

        for col, field in enumerate((gt, srm_pred, mlp_pred)):
            ax = axes[row, col]
            img = np.where(inside, np.nan, field)
            if coords is None:
                # Rectangular imshow (torus/sphere) fills the whole panel, so this never shows —
                # skip it for the curvilinear Poincaré-disk chart (hyperbolic), which only covers
                # the disk and would otherwise leave a gray square in its four corners.
                ax.set_facecolor("lightgray")
            handle, mesh1, mesh2 = _draw_field(ax, env, img, shape, coords, edges, "viridis", 0.0, vmax)
            ax.contour(mesh1, mesh2, img, levels=levels, colors="white", linewidths=0.7, alpha=0.75)
            ax.contour(mesh1, mesh2, blocked, levels=[0.5], colors="black", linewidths=1.0)
            ax.plot(*marker, "*", color="red", markersize=10, markeredgecolor="white")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            if row == 0:
                ax.set_title(_COLUMN_TITLES[col], fontsize=22, fontweight="bold")
            if col == 0:
                ax.set_ylabel(_ROW_TITLES[name], fontsize=22, fontweight="bold")
            if col == 2:
                fig.colorbar(handle, cax=axes[row, 3])

    fig.tight_layout()
    out_path = "paper_figures/results_figure.png"
    savefig(fig, out_path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
