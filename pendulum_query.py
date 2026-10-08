"""pendulum_query.py — post-training query + video for the 2-link pendulum C-space demo.

Loads a ``splat.pkl`` trained via ``python -m srms.run --environment pendulum --method ntfields
--backend srm``, evaluates ``T(θ)`` at an arbitrary ``--goal`` (the source/start is fixed at training
time — NTFields as ported in this repo bakes ``start`` into the field, see
``srms/methods/strategies/ntfields.py``'s own docstring — but any goal is free to query post-hoc from
one trained field), extracts a start→goal path via the corrected ``extract_path_grid`` (standard
Eikonal/fast-marching backtracking: descend from the goal to the source, then reverse for display),
and renders a side-by-side MP4: left panel is the flat (θ1,θ2) time-to-go heatmap with the path
animating in; right panel is the workspace with the two-link arm moving along that path past the two
fixed obstacle circles.
"""

from __future__ import annotations

import dataclasses
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.animation import FFMpegWriter, FuncAnimation

from srms.environments.pendulum import _fk_np
from srms.environments.torus import wrap
from srms.methods.backends import BACKENDS
from srms.methods.strategies import ntfields
from srms.run import Config, _build_env
from srms.visualization.utils_3d import extract_path_grid


@dataclasses.dataclass
class QueryConfig:
    """Post-training query + video configuration for a trained pendulum ``splat.pkl``."""

    pkl: str = "figures/splat.pkl"
    goal: tuple[float, float] = (1.0, 0.0)
    resolution: int = 240  # denser than training's grid is fine/free: T(theta) is a continuous fn
    num_frames: int = 150
    path_num_steps: int = 700  # the default scene's path bends the elbow ~135° and back — a long walk
    path_step_size: float = 0.5
    fps: int = 30
    out: str = "figures/pendulum.mp4"


def _load(pkl_path: str):
    """Reconstruct (splat params, cfg dict, env, backend module) from a saved ``srms.run`` pickle.

    Rebuilds ``env`` via ``srms.run._build_env`` (not by hand-picking fields) so any defaulting it
    does — e.g. ``cfg.start`` is ``None`` unless ``--start`` was passed explicitly, resolved to a
    per-environment default only inside ``_build_env`` — is reproduced exactly, rather than
    duplicated (and potentially drifting out of sync) here.
    """
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    splat = jax.tree_util.tree_map(jnp.asarray, blob["splat"])
    cfg_dict = blob["cfg"]
    if cfg_dict["environment"] != "pendulum":
        raise ValueError(f"{pkl_path} was trained on environment={cfg_dict['environment']!r}, not 'pendulum'")
    env = _build_env(Config(**cfg_dict))
    return splat, cfg_dict, env, BACKENDS[cfg_dict["backend"]]


def _theta_to_index(theta: tuple[float, float], resolution: int) -> tuple[int, int]:
    """Inverse of ``env.grid()``'s ``meshgrid(indexing="xy")``: ``img[i, j]`` <-> ``theta1=axis[j],
    theta2=axis[i]`` (verified against ``TorusEnvironment.grid``/``PendulumEnvironment.grid``)."""
    step = 2.0 * np.pi / resolution
    j = int(round((float(wrap(jnp.asarray(theta[0]))) + np.pi) / step)) % resolution  # theta1 -> column
    i = int(round((float(wrap(jnp.asarray(theta[1]))) + np.pi) / step)) % resolution  # theta2 -> row
    return i, j


def _resample_arclength(path_theta: np.ndarray, num_frames: int) -> np.ndarray:
    """Resample an unwrapped (θ1,θ2) polyline to ``num_frames`` points at equal cumulative arc
    length, so the animation's config-space speed is constant regardless of the raw path's spacing."""
    deltas = np.linalg.norm(np.diff(path_theta, axis=0), axis=-1)
    arclen = np.concatenate([[0.0], np.cumsum(deltas)])
    if arclen[-1] < 1e-9:  # degenerate: start == goal
        return np.repeat(path_theta[:1], num_frames, axis=0)
    targets = np.linspace(0.0, arclen[-1], num_frames)
    theta1 = np.interp(targets, arclen, path_theta[:, 0])
    theta2 = np.interp(targets, arclen, path_theta[:, 1])
    return np.stack([theta1, theta2], axis=-1)


def _render_video(
    env, img: np.ndarray, thetas: jnp.ndarray, shape: tuple[int, int], frames_theta: np.ndarray, qcfg: QueryConfig
) -> None:
    """Render the side-by-side [config-space heatmap+path | workspace arm] MP4."""
    l1, l2 = env.link_lengths
    reach = l1 + l2

    # Wrap each frame's theta into the [-180,180] chart for the heatmap panel; insert a NaN wherever
    # a consecutive-frame jump exceeds ~half the chart width, so a periodic wrap-around draws as the
    # path exiting one edge and re-entering the opposite one, instead of a spurious line across the image.
    wrapped_deg = np.degrees(np.asarray(wrap(jnp.asarray(frames_theta))))
    jump = np.any(np.abs(np.diff(wrapped_deg, axis=0)) > 100.0, axis=-1)
    plot_pts = wrapped_deg.copy()
    plot_pts[1:][jump] = np.nan

    inside = np.asarray(env.sdf(thetas)).reshape(shape) < 0.0
    img_masked = np.where(inside, np.nan, img)  # mask obstacle interior -- no travel time drawn there
    mesh1, mesh2 = np.meshgrid(np.linspace(-180, 180, shape[1]), np.linspace(-180, 180, shape[0]))

    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(13, 6))

    # -- config-space panel: static heatmap + iso-time contours + obstacle silhouette + markers --
    # Same style as srms/viz.py's render() (pendulum_obstacles.png): obstacle interior masked to NaN
    # (drawn as the lightgray facecolor, not a travel-time color), 14 white iso-time contours over the
    # masked field, then a black contour at the obstacle boundary (sdf=0 / inside=0.5).
    ax_c.set_facecolor("lightgray")
    ax_c.imshow(img_masked, origin="lower", extent=env.render_extent, cmap="viridis", aspect="auto")
    ax_c.contour(mesh1, mesh2, img_masked, levels=14, colors="white", linewidths=0.6, alpha=0.7)
    ax_c.contour(mesh1, mesh2, inside.astype(float), levels=[0.5], colors="black", linewidths=1.2)
    ax_c.plot(*env.render_marker_deg(), "*", color="red", markersize=16, markeredgecolor="white", zorder=5)
    goal_deg = np.degrees(np.asarray(wrap(jnp.asarray(qcfg.goal))))
    ax_c.plot(*goal_deg, "*", color="lime", markersize=16, markeredgecolor="black", zorder=5)
    (path_line,) = ax_c.plot([], [], "-", color="white", linewidth=2, zorder=4)
    (path_dot,) = ax_c.plot([], [], "o", color="red", markersize=8, zorder=6)
    ax_c.set_xlabel(env.axis_labels[0])
    ax_c.set_ylabel(env.axis_labels[1])
    ax_c.set_title(env.title)

    # -- workspace panel: static obstacle circles + dynamic link segments --
    for cx, cy, r in env.obstacles:
        ax_w.add_patch(plt.Circle((cx, cy), r, facecolor="lightgray", edgecolor="black", alpha=0.85, zorder=1))
    ax_w.set_xlim(-reach - 0.3, reach + 0.3)
    ax_w.set_ylim(-reach - 0.3, reach + 0.3)
    ax_w.set_aspect("equal")
    ax_w.set_title("workspace")
    (link1,) = ax_w.plot([], [], "-o", color="steelblue", linewidth=4, markersize=6, zorder=3)
    (link2,) = ax_w.plot([], [], "-o", color="darkorange", linewidth=4, markersize=6, zorder=3)

    def update(i: int):
        path_line.set_data(plot_pts[: i + 1, 0], plot_pts[: i + 1, 1])
        path_dot.set_data([plot_pts[i, 0]], [plot_pts[i, 1]])
        p0, p1, p2 = _fk_np(frames_theta[i], l1, l2)
        link1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        link2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        return path_line, path_dot, link1, link2

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(frames_theta), interval=1000.0 / qcfg.fps, blit=False)
    anim.save(qcfg.out, writer=FFMpegWriter(fps=qcfg.fps, bitrate=1800))
    plt.close(fig)
    print(f"saved {qcfg.out}  ({len(frames_theta)} frames @ {qcfg.fps} fps)")


def main(qcfg: QueryConfig) -> None:
    splat, cfg, env, backend = _load(qcfg.pkl)

    if float(env.sdf_np(np.asarray(qcfg.goal))) < 0:
        raise ValueError(f"--goal {qcfg.goal} is inside an obstacle (sdf<0); pick a free-space goal")

    thetas, shape = env.grid(qcfg.resolution)
    img = np.asarray(
        ntfields.predict(backend, splat, thetas, env, cfg["tau_bias"], cfg["tau_min"])
    ).reshape(shape)

    start_idx = _theta_to_index(env.start, qcfg.resolution)
    goal_idx = _theta_to_index(qcfg.goal, qcfg.resolution)
    path_idx = extract_path_grid(
        img, start_idx=start_idx, goal_idx=goal_idx, num_steps=qcfg.path_num_steps, step_size=qcfg.path_step_size
    )

    axis = np.linspace(-np.pi, np.pi, qcfg.resolution, endpoint=False)
    step = 2.0 * np.pi / qcfg.resolution
    path_theta1 = axis[0] + path_idx[:, 1] * step  # column -> theta1
    path_theta2 = axis[0] + path_idx[:, 0] * step  # row -> theta2
    # Remove the +-pi seam discontinuities the mod-wrapped grid descent introduces, giving a smooth
    # continuous angle trajectory safe to feed straight into cos/sin (FK) and to arc-length resample.
    path_theta1 = np.unwrap(path_theta1, period=2 * np.pi)
    path_theta2 = np.unwrap(path_theta2, period=2 * np.pi)

    frames = _resample_arclength(np.stack([path_theta1, path_theta2], axis=-1), qcfg.num_frames)
    frames = frames[::-1]  # play goal -> start: time-to-go counts down to 0 at the source
    _render_video(env, img, thetas, shape, frames, qcfg)


if __name__ == "__main__":
    main(tyro.cli(QueryConfig))
