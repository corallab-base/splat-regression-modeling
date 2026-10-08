"""Two-link planar pendulum (revolute arm) environment: flat T² configuration space, with a
configuration-space obstacle derived from real forward kinematics instead of angle-space circles.

Unlike ``TorusEnvironment`` (whose obstacles are circles placed directly in the chart), this
environment's obstacles are fixed circles in the *workspace* (Cartesian plane). ``sdf``/``slowness``
run forward kinematics — ``theta1`` absolute, ``theta2`` relative to link 1 — and take the minimum
point-to-segment clearance between either link and either obstacle, mirroring the same
min-over-obstacles idealization ``TorusEnvironment.sdf`` already uses, just with a real link-collision
clearance in place of a raw angle-space distance.

A fresh dataclass, not a ``TorusEnvironment`` subclass: ``srms/viz_3d.py``'s generic 3-D torus
renderer dispatches on ``isinstance(env, TorusEnvironment)`` and unpacks ``env.obstacles`` as
``(theta1, theta2, radius)`` angle-space circles — this environment's obstacles are workspace
Cartesian circles, so subclassing would make that renderer silently misinterpret them. Manifold
geometry (``log_map``/``exp_map``/``metric_inv``/``grid``/ground truth) is the same flat-T² math as
``TorusEnvironment`` at ``dim=2``, reusing its free module-level helpers directly.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from srms.environments.torus import _fast_marching_torus, _wrap_np, wrap

Obstacle = tuple[float, float, float]  # (cx, cy, radius) — workspace Cartesian, not angle space

_OBSTACLE_SEED_OFFSET = 5


def _fk(theta: jnp.ndarray, l1: float, l2: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward kinematics: pivot p0, elbow p1, end-effector p2. ``theta1`` absolute, ``theta2``
    relative to link 1. ``theta`` may be ``[2]`` (single config) or ``[N, 2]`` (batch)."""
    theta1, theta2 = theta[..., 0], theta[..., 1]
    p0 = jnp.zeros(theta.shape[:-1] + (2,))
    p1 = l1 * jnp.stack([jnp.cos(theta1), jnp.sin(theta1)], axis=-1)
    p2 = p1 + l2 * jnp.stack([jnp.cos(theta1 + theta2), jnp.sin(theta1 + theta2)], axis=-1)
    return p0, p1, p2


def _seg_point_dist(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Distance from fixed point ``p`` (``[2]``) to segment ``(a, b)`` (each ``[..., 2]``); standard
    clamped-projection point-to-segment distance."""
    ab = b - a
    t = jnp.clip(jnp.sum((p - a) * ab, axis=-1) / jnp.maximum(jnp.sum(ab * ab, axis=-1), 1e-12), 0.0, 1.0)
    return jnp.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _fk_np(theta: np.ndarray, l1: float, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy port of ``_fk`` (host-side, for RRT*'s hot loop and the workspace-panel video renderer)."""
    theta1, theta2 = theta[..., 0], theta[..., 1]
    p0 = np.zeros(theta.shape[:-1] + (2,))
    p1 = l1 * np.stack([np.cos(theta1), np.sin(theta1)], axis=-1)
    p2 = p1 + l2 * np.stack([np.cos(theta1 + theta2), np.sin(theta1 + theta2)], axis=-1)
    return p0, p1, p2


def _seg_point_dist_np(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy port of ``_seg_point_dist``."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _pendulum_sdf(
    theta: jnp.ndarray, l1: float, l2: float, thickness: float, obstacles: tuple[Obstacle, ...]
) -> jnp.ndarray:
    """Min point-to-segment clearance (minus link thickness) over both links and all obstacles."""
    p0, p1, p2 = _fk(theta, l1, l2)
    per = []
    for cx, cy, r in obstacles:
        centre = jnp.array([cx, cy])
        per.append(_seg_point_dist(centre, p0, p1) - r - thickness)
        per.append(_seg_point_dist(centre, p1, p2) - r - thickness)
    return jnp.min(jnp.stack(per, axis=0), axis=0)


def _pendulum_sdf_np(
    theta: np.ndarray, l1: float, l2: float, thickness: float, obstacles: tuple[Obstacle, ...]
) -> np.ndarray:
    """NumPy port of ``_pendulum_sdf``."""
    p0, p1, p2 = _fk_np(theta, l1, l2)
    per = []
    for cx, cy, r in obstacles:
        centre = np.array([cx, cy])
        per.append(_seg_point_dist_np(centre, p0, p1) - r - thickness)
        per.append(_seg_point_dist_np(centre, p1, p2) - r - thickness)
    return np.min(per, axis=0)


@dataclasses.dataclass
class PendulumEnvironment:
    """2-link planar revolute arm: flat T² config space, forward-kinematics-derived obstacle.

    ``obstacles``, when given, are fixed workspace circles (the intended use — a reproducible demo
    scene); when ``None`` obstacles are randomly sampled (mirroring ``TorusEnvironment``'s own
    reject-near-source sampler, realized through FK/collision instead of raw angle-space distance) —
    kept for direct-Python/test use, not exposed on the CLI (the demo always supplies exactly 2 fixed
    obstacles).
    """

    start: tuple[float, float] = (2.9, 0.0)
    link_lengths: tuple[float, float] = (1.0, 1.0)
    link_thickness: float = 0.06
    obstacles: tuple[Obstacle, ...] | None = None
    num_obstacles: int = 2
    obstacle_radius: tuple[float, float] = (0.3, 0.6)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != 2:
            raise ValueError(f"start has {len(self.start)} coords, but a 2-link pendulum needs exactly 2")
        self.dim = 2
        self.tangent_dim = 2
        self.domain: tuple[float, float] = (-float(np.pi), float(np.pi))
        self.axis_labels: tuple[str, str] = (r"$\theta_1$ (deg)", r"$\theta_2$ (deg)")
        self.render_extent: tuple[float, float, float, float] = (-180.0, 180.0, -180.0, 180.0)
        self.has_dense_gt = True
        self.obstacles = self.obstacles if self.obstacles is not None else self._sample_obstacles()

    @property
    def title(self) -> str:
        return f"pendulum $T^2$ — time-to-go ({len(self.obstacles)} obstacles)"

    @property
    def gt_label(self) -> str:
        return "ground truth — periodic FMM (flat chart)"

    def _sample_obstacles(self) -> tuple[Obstacle, ...]:
        """Reproducible workspace-circle obstacles, clear of the source (checked via real FK collision,
        not raw angle-space distance — see module docstring)."""
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        reach = sum(self.link_lengths)
        start = np.asarray(self.start)
        obstacles: list[Obstacle] = []
        while len(obstacles) < self.num_obstacles:
            centre = rng.uniform(-reach, reach, size=2)
            if np.linalg.norm(centre) > reach:
                continue
            radius = float(rng.uniform(*self.obstacle_radius))
            candidate: Obstacle = (float(centre[0]), float(centre[1]), radius)
            if _pendulum_sdf_np(start, *self.link_lengths, self.link_thickness, (candidate,)) > 0.3:
                obstacles.append(candidate)
        return tuple(obstacles)

    # ---- manifold geometry (flat T² — identical math to TorusEnvironment at dim=2) --------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Wrapped displacement x -/ mu — the flat-torus log map."""
        return wrap(x - mu)

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Flat torus: identical to log_map (ambient chart coincides with the tangent frame)."""
        return wrap(x - mu)

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exp_mu(v) = wrap(mu + v) — inverse of log_map on the flat torus."""
        return wrap(mu + v)

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Flat torus: trivial volume element."""
        return jnp.asarray(1.0)

    def splat_precompute(self, mu: jnp.ndarray):
        """Per-splat geometry hoisted out of the per-point loop; nothing to precompute here."""
        return mu

    def log_and_jac(self, pre, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(log_map, jac_factor) in one call — see base.Environment for why they are fused."""
        return wrap(x - pre), jnp.asarray(1.0)

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        return wrap(x)

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        return _wrap_np(x)

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Wrapped tangent vector pointing from a to b."""
        return _wrap_np(b - a)

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points on a flat circle of radius eps around the source."""
        z = rng.standard_normal((n, 2)).astype(np.float64)
        z /= np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
        return _wrap_np(np.asarray(self.start) + eps * z)

    def metric_inv(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Inverse metric g^{ij}; identity for the flat torus."""
        return jnp.eye(2)

    def geodesic(self, theta: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic flat-torus geodesic distance ‖wrap(θ − start)‖ (the known base)."""
        return jnp.linalg.norm(wrap(theta - start), axis=-1)

    # ---- obstacle / slowness field (genuinely new: forward kinematics + link collision) ----------

    def sdf(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Signed clearance to the nearest link-obstacle collision, via forward kinematics."""
        return _pendulum_sdf(thetas, *self.link_lengths, self.link_thickness, self.obstacles)

    def slowness(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside a colliding configuration."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(thetas) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy signed clearance (host-side, for RRT*'s hot loop)."""
        return _pendulum_sdf_np(points, *self.link_lengths, self.link_thickness, self.obstacles)

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy smooth slowness (host-side, for RRT*'s hot loop)."""
        sdf = self.sdf_np(points)
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(sdf / self.slow_width))

    # ---- sampling / ground truth --------------------------------------------

    @property
    def volume(self) -> float:
        """Riemannian volume of the region ``sample_domain`` covers — flat, so just the chart box."""
        return float((2.0 * np.pi) ** 2)

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform samples in [-π, π)² — volume-uniform, since the torus is flat."""
        return rng.uniform(-np.pi, np.pi, size=(n, 2))

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int]]:
        """Periodic grid of θ over [-π, π)², raveled to [resolution², 2], plus its shape."""
        axis = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
        grid1, grid2 = np.meshgrid(axis, axis, indexing="xy")
        thetas_np = np.stack([grid1.ravel(), grid2.ravel()], axis=-1)
        return jnp.asarray(thetas_np, dtype=jnp.float32), (resolution, resolution)

    def ground_truth(self, resolution: int, start: tuple[float, float] | None = None) -> np.ndarray:
        """Periodic fast marching of ‖∇T‖ = slowness on the flat-torus grid; returns a raveled array."""
        start = self.start if start is None else start
        thetas, shape = self.grid(resolution)
        speed = 1.0 / np.asarray(self.slowness(thetas)).reshape(shape)
        return _fast_marching_torus(speed, start, resolution).ravel()

    def render_marker_deg(self) -> tuple[float, float]:
        """(θ1, θ2) position of the source in degrees."""
        deg = np.degrees(np.asarray(self.start))
        return float(deg[0]), float(deg[1])
