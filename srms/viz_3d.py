"""3D surface visualization for torus/sphere environments in the ``srms/`` pipeline.

Renders the ``[GT | prediction | error]`` fields on the manifold's actual embedded 3D shape
(torus ring, sphere) instead of the flat ``(θ1, θ2)`` chart ``srms.viz.render`` uses — for
figures where the manifold's geometry itself is part of the point being made (paper figures,
presentations). Dispatches on the environment type; a no-op for environments without a bounded
ambient embedding a reader would recognize (hyperbolic charts, SO(3), the plane).

Uses each environment's own ``grid()`` output as the ambient embedding — for the sphere this is
already the ambient unit-vector grid ``env.grid()`` returns (no lon/lat re-derivation), and for
the torus it's ``torus_embedding`` applied to the same angle grid ``env.grid()`` rasterizes —
so obstacles/start/the extracted path line up exactly with what training saw.
"""

from __future__ import annotations

import numpy as np

from srms.environments.sphere import SphereEnvironment
from srms.environments.torus import TorusEnvironment
from srms.visualization.utils_3d import (
    close_periodic,
    compute_field_contours,
    extract_path_grid,
    grid_path_to_xyz,
    nearest_grid_index,
    plotly_surface,
    render_surface_multiangle,
    torus_embedding,
)


def _torus_obstacle_ring(obs: tuple[float, ...], n: int = 64) -> np.ndarray:
    """Boundary loop (SDF = 0) of a torus obstacle, mapped onto the ambient tube surface.

    ``TorusEnvironment.sdf`` measures the wrapped Euclidean norm in angle space, so the boundary
    of a ball of radius ``r`` about centre ``(c1, c2)`` is exactly the angle-space circle
    ``(c1, c2) + r*(cos t, sin t)`` — no geodesic solving needed, unlike the sphere case below.
    """
    c1, c2, r = obs
    t = np.linspace(0.0, 2 * np.pi, n)
    x, y, z = torus_embedding(c1 + r * np.cos(t), c2 + r * np.sin(t))
    return np.stack([x, y, z], axis=-1)


def _sphere_obstacle_ring(obs: tuple[float, ...], n: int = 64) -> np.ndarray:
    """Boundary loop (SDF = 0) of a sphere obstacle — the small circle at geodesic distance
    ``radius`` from the cap's centre, built from an arbitrary orthonormal tangent frame there."""
    *centre, radius = obs
    centre = np.asarray(centre, dtype=float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(centre[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, centre) * centre
    u /= np.linalg.norm(u)
    v = np.cross(centre, u)
    t = np.linspace(0.0, 2 * np.pi, n)
    return np.cos(radius) * centre[None, :] + np.sin(radius) * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def _torus_obstacle_patch(obs: tuple[float, ...], n_r: int = 14, n_t: int = 48, offset: float = 0.012) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filled interior (SDF < 0) of a torus obstacle as an ``(n_r, n_t)`` ambient grid.

    Radial fan from the centre out to the boundary circle used by ``_torus_obstacle_ring``, pushed
    outward along the tube's local surface normal by ``offset`` (world units) so this translucent
    patch renders in front of the coincident main field surface instead of z-fighting with it.
    """
    c1, c2, r = obs
    s = np.linspace(0.0, 1.0, n_r)
    t = np.linspace(0.0, 2 * np.pi, n_t)
    ss, tt = np.meshgrid(s, t, indexing="ij")
    theta1 = c1 + ss * r * np.cos(tt)
    theta2 = c2 + ss * r * np.sin(tt)
    x, y, z = torus_embedding(theta1, theta2)
    nx, ny, nz = np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), np.sin(theta2)
    return x + offset * nx, y + offset * ny, z + offset * nz


def _sphere_obstacle_patch(obs: tuple[float, ...], n_r: int = 14, n_t: int = 48, offset: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filled interior (SDF < 0) of a sphere obstacle as an ``(n_r, n_t)`` ambient grid.

    Same tangent-frame construction as ``_sphere_obstacle_ring`` but with the geodesic radius swept
    from 0 to the obstacle's radius; pushed outward from the origin by ``offset`` (fractional) to
    avoid z-fighting with the sphere surface it's coincident with.
    """
    *centre, radius = obs
    centre = np.asarray(centre, dtype=float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(centre[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, centre) * centre
    u /= np.linalg.norm(u)
    v = np.cross(centre, u)
    s = np.linspace(0.0, 1.0, n_r)
    t = np.linspace(0.0, 2 * np.pi, n_t)
    ss, tt = np.meshgrid(s, t, indexing="ij")
    rad = ss * radius
    pts = (
        np.cos(rad)[..., None] * centre[None, None, :]
        + np.sin(rad)[..., None] * (np.cos(tt)[..., None] * u[None, None, :] + np.sin(tt)[..., None] * v[None, None, :])
    )
    pts = pts * (1.0 + offset)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def render_3d(
    env, cfg, gt: np.ndarray, prediction: np.ndarray, inside: np.ndarray, shape: tuple[int, ...], thetas: np.ndarray
) -> None:
    """Generate multi-angle static + interactive 3D surface figures for a dense torus/sphere solve.

    ``thetas`` is the same raveled grid ``run.main`` already built with ``env.grid(cfg.resolution)``
    and evaluated ``gt``/``prediction`` on — reused here rather than rebuilt, so the 3D embedding is
    guaranteed pixel-for-pixel consistent with what was scored. Called unconditionally from
    ``run.main`` after ``srms.viz.render``; environments this module doesn't know how to embed
    (anything but the 2-torus / S²) are skipped silently, since the flat-chart figure already exists.
    """
    if isinstance(env, TorusEnvironment) and env.dim == 2:
        _render_torus_3d(env, cfg, gt, prediction, inside, shape, thetas)
    elif isinstance(env, SphereEnvironment) and env.n == 2:
        _render_sphere_3d(env, cfg, gt, prediction, inside, shape, thetas)


def _render_torus_3d(
    env: TorusEnvironment, cfg, gt: np.ndarray, prediction: np.ndarray, inside: np.ndarray, shape, thetas: np.ndarray
) -> None:
    angles = np.asarray(thetas).reshape(shape + (2,))
    x, y, z = torus_embedding(angles[..., 0], angles[..., 1])

    start_xyz = tuple(float(v) for v in torus_embedding(np.array(env.start[0]), np.array(env.start[1])))
    obstacle_xyz = [tuple(float(v) for v in torus_embedding(np.array(o[0]), np.array(o[1]))) for o in env.obstacles]
    obstacle_rings = [_torus_obstacle_ring(o) for o in env.obstacles]
    obstacle_patches = [_torus_obstacle_patch(o) for o in env.obstacles]

    # A ring torus self-occludes: any fixed small camera set can hide an arbitrary tube point.
    # Aim "front" along theta1 (which side of the ring) at an elevation matching theta2's sign
    # (which side of the tube's cross-section — top or bottom) so the source is never the one
    # marker that happens to fall on the far side of every panel.
    theta1_s, theta2_s = float(env.start[0]), float(env.start[1])
    front = (25.0 if np.sin(theta2_s) >= 0 else -25.0, np.degrees(theta1_s))

    _render_manifold_3d(
        env, cfg, gt, prediction, inside, shape, x, y, z, start_xyz, obstacle_xyz, obstacle_rings, obstacle_patches,
        periodic=(True, True), front=front,
    )


def _render_sphere_3d(
    env: SphereEnvironment, cfg, gt: np.ndarray, prediction: np.ndarray, inside: np.ndarray, shape, thetas: np.ndarray
) -> None:
    points = np.asarray(thetas).reshape(shape + (3,))  # already ambient unit vectors — no re-derivation
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    start_xyz = tuple(float(v) for v in env.start)
    obstacle_xyz = [tuple(float(v) for v in o[:-1]) for o in env.obstacles]
    obstacle_rings = [_sphere_obstacle_ring(o) for o in env.obstacles]
    obstacle_patches = [_sphere_obstacle_patch(o) for o in env.obstacles]

    # Aim "front" straight at the source point on the sphere (same reasoning as the torus case above,
    # though a sphere is convex so occlusion is milder — mainly matters for a source near either pole).
    sx, sy, sz = start_xyz
    elev = float(np.clip(np.degrees(np.arcsin(np.clip(sz, -1.0, 1.0))), -60.0, 60.0))
    front = (elev if abs(elev) > 5.0 else 25.0, float(np.degrees(np.arctan2(sy, sx))))

    # grid() is (colatitude, azimuth): colatitude is cell-centred/non-periodic, azimuth is periodic.
    _render_manifold_3d(
        env, cfg, gt, prediction, inside, shape, x, y, z, start_xyz, obstacle_xyz, obstacle_rings, obstacle_patches,
        periodic=(False, True), front=front,
    )


def _render_manifold_3d(
    env,
    cfg,
    gt: np.ndarray,
    prediction: np.ndarray,
    inside: np.ndarray,
    shape,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start_xyz: tuple[float, float, float],
    obstacle_xyz: list[tuple[float, float, float]],
    obstacle_rings: list[np.ndarray],
    obstacle_patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    periodic: tuple[bool, bool],
    front: tuple[float, float],
) -> None:
    """Shared render body once torus/sphere have supplied their ambient grid — see callers above."""
    gt_img = np.where(inside, np.nan, gt).reshape(shape)
    pred_img = np.where(inside, np.nan, prediction).reshape(shape)
    error_img = pred_img - gt_img

    out_dir = f"{cfg.out_dir}/3d_views"
    vmax = float(np.nanmax(gt_img))

    # A diverged training run (e.g. NTFields hitting NaN loss) leaves `prediction` entirely NaN —
    # nothing meaningful to plot for prediction/error, and the gradient-descent path below would
    # have no finite direction to follow anywhere. Render ground-truth only rather than 8 blank
    # panels, and say so plainly instead of the silent-garbage-or-crash default.
    if not np.isfinite(pred_img).any():
        print(
            f"[viz_3d] prediction field is entirely non-finite (training likely diverged to NaN) — "
            f"rendering ground-truth-only 3D views to {out_dir}/"
        )
        xc, yc, zc = close_periodic(x, periodic), close_periodic(y, periodic), close_periodic(z, periodic)
        gt_c = close_periodic(gt_img, periodic)
        contour_levels = np.linspace(0.0, vmax, 10)[1:-1].tolist()
        gt_contours = compute_field_contours(gt_c, xc, yc, zc, contour_levels)
        render_surface_multiangle(
            xc, yc, zc, gt_c, out_dir=out_dir, title="Ground truth", cmap="viridis", front=front,
            start_xyz=start_xyz, obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings,
            obstacle_patches=obstacle_patches, field_contours=gt_contours, vmin=0.0, vmax=vmax,
        )
        plotly_surface(
            xc, yc, zc, gt_c, cmap="viridis", vmin=0.0, vmax=vmax, start_xyz=start_xyz,
            obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings, obstacle_patches=obstacle_patches,
            field_contours=gt_contours, title=f"{env.title} — ground truth", out_html=f"{cfg.out_dir}/3d_interactive_gt.html",
        )
        return

    # Path to the farthest free point in the domain — a self-contained, geometry-agnostic
    # illustration of the routing (no separate "goal" concept exists for a pure time-to-go field).
    start_idx = nearest_grid_index(x, y, z, start_xyz)
    free_gt = np.where(inside.reshape(shape), -np.inf, gt_img)
    goal_idx = np.unravel_index(int(np.argmax(free_gt)), free_gt.shape)
    path_idx = extract_path_grid(pred_img, start_idx=start_idx, goal_idx=goal_idx, num_steps=200, periodic=periodic)
    path_xyz = grid_path_to_xyz(path_idx, x, y, z)
    goal_xyz = tuple(float(v) for v in (x[goal_idx], y[goal_idx], z[goal_idx]))

    # Close the periodic seam(s) for rendering only — env.grid()'s angle axis is [start, start+2*pi)
    # (endpoint=False), so the raw mesh never reaches back to its own start and a plotted surface
    # shows a visible slit there. Path extraction above stays on the unclosed grid (mod-based wrap).
    xc, yc, zc = close_periodic(x, periodic), close_periodic(y, periodic), close_periodic(z, periodic)
    gt_c = close_periodic(gt_img, periodic)
    pred_c = close_periodic(pred_img, periodic)
    error_c = close_periodic(error_img, periodic)

    # Iso-value contours (skipped on the error panel, matching srms.viz.render's flat-chart
    # convention) — computed on the closed grids so a level line traces cleanly through the seam.
    contour_levels = np.linspace(0.0, vmax, 10)[1:-1].tolist()
    gt_contours = compute_field_contours(gt_c, xc, yc, zc, contour_levels)
    pred_contours = compute_field_contours(pred_c, xc, yc, zc, contour_levels)

    render_surface_multiangle(
        xc, yc, zc, gt_c, out_dir=out_dir, title="Ground truth", cmap="viridis", front=front,
        start_xyz=start_xyz, obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings,
        obstacle_patches=obstacle_patches, field_contours=gt_contours, vmin=0.0, vmax=vmax,
    )
    render_surface_multiangle(
        xc, yc, zc, pred_c, out_dir=out_dir, title="Prediction", cmap="plasma", front=front,
        start_xyz=start_xyz, goal_xyz=goal_xyz, obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings,
        obstacle_patches=obstacle_patches, field_contours=pred_contours, path_xyz=path_xyz, vmin=0.0, vmax=vmax,
    )
    err_absmax = float(np.nanmax(np.abs(error_img))) or 1.0
    render_surface_multiangle(
        xc, yc, zc, error_c, out_dir=out_dir, title="Error", cmap="bwr", front=front,
        start_xyz=start_xyz, obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings,
        obstacle_patches=obstacle_patches, vmin=-err_absmax, vmax=err_absmax,
    )

    plotly_surface(
        xc, yc, zc, gt_c, cmap="viridis", vmin=0.0, vmax=vmax, start_xyz=start_xyz,
        obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings, obstacle_patches=obstacle_patches,
        field_contours=gt_contours, title=f"{env.title} — ground truth", out_html=f"{cfg.out_dir}/3d_interactive_gt.html",
    )
    plotly_surface(
        xc, yc, zc, pred_c, cmap="plasma", vmin=0.0, vmax=vmax,
        start_xyz=start_xyz, goal_xyz=goal_xyz, obstacle_xyz=obstacle_xyz, obstacle_rings=obstacle_rings,
        obstacle_patches=obstacle_patches, field_contours=pred_contours, path_xyz=path_xyz,
        title=f"{env.title} — prediction", out_html=f"{cfg.out_dir}/3d_interactive_pred.html",
    )

    print(f"saved 3D visualizations to {out_dir}/ and {cfg.out_dir}/3d_interactive_*.html")
