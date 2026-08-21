"""Flat n-torus Tⁿ environment with spherical obstacles and a smooth slowness field.

``T²`` (``dim=2``) is the configuration space of a 2-joint revolute arm; the same construction
generalizes to any ``dim`` joints. Splats are placed *intrinsically* (angle space), evaluated at the
wrapped log map ``wrap(θ − B)`` so periodicity is automatic and the model dimension equals the joint
count. The metric is kept separate (``metric_inv``: identity here, ``M(θ)⁻¹`` for the arm) so a
curved metric plugs in later without touching splat placement.

This is a straight port of the environment-specific half of the original
``torus.py`` (see git history) behind the ``Environment`` interface in
``srms/environments/base.py``, generalized from a fixed ``T²`` to ``Tⁿ``.
Grid/ground-truth (dense fast marching) and rendering only make sense for ``dim=2``
(a dense grid is intractable and unplottable beyond that) and raise ``NotImplementedError``
otherwise; training itself (mesh-free collocation) works at any ``dim``.
"""

from __future__ import annotations

import dataclasses
import heapq

import jax
import jax.numpy as jnp
import numpy as np

Obstacle = tuple[float, ...]  # (*centre[dim], radius)

_OBSTACLE_SEED_OFFSET = 5


def wrap(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to [-π, π) (jax)."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _wrap_np(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π) (NumPy)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclasses.dataclass
class TorusEnvironment:
    """Flat Tⁿ with a smooth slowness field rising around spherical obstacles."""

    start: tuple[float, ...] = (-1.5, -1.5)
    dim: int = 2
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != self.dim:
            raise ValueError(f"start has {len(self.start)} coords but dim={self.dim}")
        self.tangent_dim = self.dim
        self.domain: tuple[float, float] = (-float(np.pi), float(np.pi))
        # $\theta$, not a raw θ glyph: LaTeX (unlike matplotlib's own mathtext) has no legacy-font
        # slot for a bare Unicode Greek letter, so this crashes any checkpoint render that happens
        # while text.usetex=True is active (e.g. a paper_figures/*.py script that trains a fresh
        # checkpoint after calling style.use_latex_fonts()) with "Unicode character θ not set up
        # for use with LaTeX" — math mode works under both usetex and mathtext.
        self.axis_labels: tuple[str, str] = (r"$\theta_1$ (deg)", r"$\theta_2$ (deg)")
        self.render_extent: tuple[float, float, float, float] = (-180.0, 180.0, -180.0, 180.0)
        self.has_dense_gt = self.dim in (2, 3)  # dense fast marching tractable at 2-D and 3-D
        self.obstacles: tuple[Obstacle, ...] = self._sample_obstacles()

    @property
    def title(self) -> str:
        # $T^{...}$, not a raw "T^..." — a bare "^" outside math mode is LaTeX's superscript
        # operator and crashes ("Missing $ inserted") any render done while text.usetex=True
        # (see TorusEnvironment.axis_labels above for the same class of bug).
        return f"torus $T^{self.dim}$ — time-to-go ({self.num_obstacles} obstacles)"

    @property
    def gt_label(self) -> str:
        return "ground truth — periodic FMM (flat chart)"

    def _sample_obstacles(self) -> tuple[Obstacle, ...]:
        """Reproducible spherical obstacles in angle space, clear of the source."""
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        start = np.asarray(self.start)
        obstacles: list[Obstacle] = []
        while len(obstacles) < self.num_obstacles:
            centre = rng.uniform(-np.pi, np.pi, size=self.dim)
            radius = float(rng.uniform(*self.obstacle_radius))
            if np.linalg.norm(_wrap_np(centre - start)) > radius + 0.3:
                obstacles.append((*centre.tolist(), radius))
        return tuple(obstacles)

    # ---- manifold geometry -----------------------------------------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Wrapped displacement x -/ mu — the flat-torus log map."""
        return wrap(x - mu)

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Flat torus: identical to log_map (ambient chart coincides with the tangent frame)."""
        return wrap(x - mu)

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exp_mu(v) = wrap(mu + v) — inverse of log_map on the flat torus (see base.Environment)."""
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
        """``n`` points on a flat sphere of radius eps around the source."""
        z = rng.standard_normal((n, self.dim)).astype(np.float64)
        z /= np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
        return _wrap_np(np.asarray(self.start) + eps * z)

    def metric_inv(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Inverse metric g^{ij}; identity for the flat torus (the arm swaps in M(θ)⁻¹)."""
        return jnp.eye(self.dim)

    def geodesic(self, theta: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic flat-torus geodesic distance ‖wrap(θ − start)‖ (the known base)."""
        return jnp.linalg.norm(wrap(theta - start), axis=-1)

    # ---- obstacle / slowness field -----------------------------------------

    def sdf(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Signed distance (wrapped) to the union of obstacle balls."""
        per = [jnp.linalg.norm(wrap(thetas - jnp.array(obs[:-1])), axis=-1) - obs[-1] for obs in self.obstacles]
        return jnp.min(jnp.stack(per, axis=0), axis=0)

    def slowness(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside obstacles."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(thetas) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy signed distance (host-side, for RRT*'s hot loop)."""
        return np.min(
            [np.linalg.norm(_wrap_np(points - np.array(obs[:-1])), axis=-1) - obs[-1] for obs in self.obstacles],
            axis=0,
        )

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy smooth slowness (host-side, for RRT*'s hot loop)."""
        sdf = self.sdf_np(points)
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(sdf / self.slow_width))

    # ---- sampling / ground truth --------------------------------------------

    @property
    def volume(self) -> float:
        """Riemannian volume of the region ``sample_domain`` covers — flat, so just the chart box."""
        return float((2.0 * np.pi) ** self.dim)

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform samples in [-π, π)^dim — volume-uniform, since the torus is flat."""
        return rng.uniform(-np.pi, np.pi, size=(n, self.dim))

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, ...]]:
        """Periodic grid of θ over [-π, π)^dim, raveled to [resolution^dim, dim], plus its shape.

        Tractable at dim 2 and 3 (resolution³ cells); beyond that a dense grid is hopeless and
        training remains mesh-free.
        """
        if self.dim not in (2, 3):
            raise NotImplementedError("grid()/ground_truth() need a dense grid — only tractable at dim<=3")
        axis = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
        mesh = np.meshgrid(*([axis] * self.dim), indexing="ij" if self.dim == 3 else "xy")
        thetas_np = np.stack([m.ravel() for m in mesh], axis=-1)
        return jnp.asarray(thetas_np, dtype=jnp.float32), (resolution,) * self.dim

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Periodic fast marching of ‖∇T‖ = slowness on the flat-torus grid; returns a raveled array."""
        start = self.start if start is None else start
        thetas, shape = self.grid(resolution)
        if self.dim == 2:
            speed = 1.0 / np.asarray(self.slowness(thetas)).reshape(shape)
            return _fast_marching_torus(speed, start, resolution).ravel()
        # dim == 3: the shared anisotropic marcher, with uniform spacing and all axes periodic
        from srms.environments.marching import fast_march_3d

        step = 2 * np.pi / resolution
        pts = np.asarray(thetas, dtype=float)
        seed = np.linalg.norm(_wrap_np(pts - np.asarray(start)), axis=-1).reshape(shape)
        return fast_march_3d(
            spacing=np.full(shape + (3,), step),
            slowness=np.asarray(self.slowness(thetas)).reshape(shape),
            blocked=np.zeros(shape, bool),
            seed_time=np.where(seed <= 1.5 * step, seed, np.inf),
            periodic=(True, True, True),
        ).ravel()

    def render_marker_deg(self) -> tuple[float, float]:
        """(θ1, θ2) position of the source in degrees (only meaningful at dim=2)."""
        deg = np.degrees(np.asarray(self.start))
        return float(deg[0]), float(deg[1])


def _fast_marching_torus(speed: np.ndarray, start: tuple[float, float], n: int) -> np.ndarray:
    """Periodic fast marching of |∇T| = 1/speed on the flat-torus grid (wrap-around neighbours)."""
    step = 2 * np.pi / n
    axis = np.linspace(-np.pi, np.pi, n, endpoint=False)
    grid1, grid2 = np.meshgrid(axis, axis)
    slowness = 1.0 / speed
    time = np.full((n, n), np.inf)
    accepted = np.zeros((n, n), dtype=bool)
    heap: list[tuple[float, int, int]] = []
    seed = np.sqrt(_wrap_np(grid1 - start[0]) ** 2 + _wrap_np(grid2 - start[1]) ** 2)
    for j, i in zip(*np.where(seed <= step)):
        time[j, i] = float(seed[j, i])
        heapq.heappush(heap, (float(time[j, i]), int(j), int(i)))

    while heap:
        _, j, i = heapq.heappop(heap)
        if accepted[j, i]:
            continue
        accepted[j, i] = True
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nj, ni = (j + dj) % n, (i + di) % n
            if accepted[nj, ni]:
                continue
            left, right = time[nj, (ni - 1) % n], time[nj, (ni + 1) % n]
            down, up = time[(nj - 1) % n, ni], time[(nj + 1) % n, ni]
            along_x = min(
                left if accepted[nj, (ni - 1) % n] else np.inf, right if accepted[nj, (ni + 1) % n] else np.inf
            )
            along_y = min(down if accepted[(nj - 1) % n, ni] else np.inf, up if accepted[(nj + 1) % n, ni] else np.inf)
            cost = slowness[nj, ni] * step
            if np.isinf(along_x) and np.isinf(along_y):
                continue
            if np.isinf(along_x):
                candidate = along_y + cost
            elif np.isinf(along_y):
                candidate = along_x + cost
            else:
                lo, hi = min(along_x, along_y), max(along_x, along_y)
                candidate = (
                    lo + cost if hi - lo >= cost else 0.5 * (lo + hi + np.sqrt(max(0.0, 2 * cost**2 - (hi - lo) ** 2)))
                )
            if candidate < time[nj, ni]:
                time[nj, ni] = candidate
                heapq.heappush(heap, (float(candidate), int(nj), int(ni)))
    return time
