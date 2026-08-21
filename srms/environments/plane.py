"""Flat, bounded, non-periodic plane R^d — a control for isolating "non-periodic" from "curved".

Built specifically to separate two things the ``poincare_hyperbolic`` comparison conflated: negative
curvature (exponentially growing volume near a boundary, a metric that blows up there) and simple
non-periodicity (a bounded chart with no wraparound identification at its edges, unlike the torus).
This environment has *neither* curvature nor periodicity — flat like the torus (``jac_factor=1``,
``metric_inv=I``), but bounded with a genuine wall like ``poincare_hyperbolic``'s truncated ball
instead of a periodic identification at the edges. Same chart size as the torus's ``[-π, π]^dim``
domain, so obstacle radii, ``init_scale``, ``source_radius`` etc. transfer without retuning and any
gap that opens up when comparing to ``torus`` under identical hyperparameters is attributable to
periodicity, not scale.

If ``srm`` underperforms here the way it does on ``poincare_hyperbolic``, non-periodicity itself
(independent of curvature) is implicated. If ``srm`` matches ``mlp``/``torus``-level fits here,
the earlier hyperbolic gap is more likely curvature-specific (the metric blowup near the truncation
boundary, addressed only partially by the ``wrap_point``/weight-decay fixes in this codebase's
history — see ``poincare_hyperbolic.py``).

``wrap_point``/``wrap_point_np`` retract to the workspace box (not merely leave points as-is), matching
the same lesson learned there: an optimizer-owned position (e.g. srm's splat centres) needs a genuine
boundary retraction, or gradient descent can waste capacity drifting it outside the scored region.
"""

from __future__ import annotations

import dataclasses
import heapq

import jax
import jax.numpy as jnp
import numpy as np

Obstacle = tuple[float, ...]  # (*centre[dim], radius)

_OBSTACLE_SEED_OFFSET = 5


@dataclasses.dataclass
class PlaneEnvironment:
    """Flat, bounded R^d with a smooth slowness field rising around spherical obstacles; no wrap."""

    start: tuple[float, ...] = (-1.5, -1.5)
    dim: int = 2
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    half_width: float = float(np.pi)  # domain [-half_width, half_width]^dim -- matches torus exactly
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != self.dim:
            raise ValueError(f"start has {len(self.start)} coords but dim={self.dim}")
        if float(np.max(np.abs(self.start))) >= self.half_width:
            raise ValueError(f"start must lie inside the workspace ‖x‖_inf < {self.half_width}")
        self.tangent_dim = self.dim
        self.domain: tuple[float, float] = (-self.half_width, self.half_width)
        self.axis_labels: tuple[str, str] = ("x1", "x2")
        self.render_extent: tuple[float, float, float, float] = (
            -self.half_width,
            self.half_width,
            -self.half_width,
            self.half_width,
        )
        self.has_dense_gt = self.dim in (2, 3)
        self.obstacles: tuple[Obstacle, ...] = self._sample_obstacles()

    @property
    def title(self) -> str:
        # $R^{...}$, not a raw "R^..." — see torus.py's TorusEnvironment.title for why.
        return f"plane $R^{self.dim}$ (non-periodic control) — time-to-go ({self.num_obstacles} obstacles)"

    @property
    def gt_label(self) -> str:
        return "ground truth — bounded FMM (no wrap)"

    def _sample_obstacles(self) -> tuple[Obstacle, ...]:
        """Reproducible spherical obstacles, clear of the source -- same recipe as torus.py, no wrap."""
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        start = np.asarray(self.start)
        obstacles: list[Obstacle] = []
        while len(obstacles) < self.num_obstacles:
            centre = rng.uniform(-self.half_width, self.half_width, size=self.dim)
            radius = float(rng.uniform(*self.obstacle_radius))
            if np.linalg.norm(centre - start) > radius + 0.3:
                obstacles.append((*centre.tolist(), radius))
        return tuple(obstacles)

    # ---- manifold geometry -----------------------------------------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Plain Euclidean displacement x - mu -- the flat, non-periodic log map."""
        return x - mu

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return x - mu

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return mu + v

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Flat: trivial volume element, same as the torus."""
        return jnp.asarray(1.0)

    def splat_precompute(self, mu: jnp.ndarray):
        return mu

    def log_and_jac(self, pre, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return x - pre, jnp.asarray(1.0)

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Retract onto the workspace box -- the boundary is a genuine wall, not a periodic seam.

        Unlike the torus (``wrap`` = modular identification, no true edge) this domain has a real
        edge, so any optimizer-owned position (srm's splat centres, RRT* steering, ...) needs an
        explicit retraction here or it can drift out and waste capacity -- see
        ``poincare_hyperbolic.py``'s ``wrap_point`` for the measured cost of skipping this.
        """
        return jnp.clip(x, -self.half_width, self.half_width)

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), -self.half_width, self.half_width)

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Plain Euclidean tangent vector a->b; no wrap."""
        return np.asarray(b, dtype=float) - np.asarray(a, dtype=float)

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points on a flat sphere of radius eps around the source."""
        z = rng.standard_normal((n, self.dim)).astype(np.float64)
        z /= np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
        return np.asarray(self.start) + eps * z

    def metric_inv(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Identity -- flat, same as the torus."""
        return jnp.eye(self.dim)

    def geodesic(self, theta: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic flat-plane distance ‖θ − start‖ (the known base); no wrap, unlike the torus."""
        return jnp.linalg.norm(theta - start, axis=-1)

    # ---- obstacle / slowness field -----------------------------------------

    def sdf(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Signed distance to the union of obstacle balls; no wrap."""
        per = [jnp.linalg.norm(thetas - jnp.array(obs[:-1]), axis=-1) - obs[-1] for obs in self.obstacles]
        return jnp.min(jnp.stack(per, axis=0), axis=0)

    def slowness(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside obstacles."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(thetas) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        return np.min(
            [np.linalg.norm(points - np.array(obs[:-1]), axis=-1) - obs[-1] for obs in self.obstacles], axis=0
        )

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        sdf = self.sdf_np(points)
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(sdf / self.slow_width))

    # ---- sampling / ground truth --------------------------------------------

    @property
    def volume(self) -> float:
        return float((2.0 * self.half_width) ** self.dim)

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform samples in [-half_width, half_width)^dim -- volume-uniform, flat like the torus."""
        return rng.uniform(-self.half_width, self.half_width, size=(n, self.dim))

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, ...]]:
        if self.dim not in (2, 3):
            raise NotImplementedError("grid()/ground_truth() need a dense grid -- only tractable at dim<=3")
        axis = np.linspace(-self.half_width, self.half_width, resolution)
        mesh = np.meshgrid(*([axis] * self.dim), indexing="ij" if self.dim == 3 else "xy")
        thetas_np = np.stack([m.ravel() for m in mesh], axis=-1)
        return jnp.asarray(thetas_np, dtype=jnp.float32), (resolution,) * self.dim

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Bounded (non-periodic) fast marching of ‖∇T‖ = slowness on the plane grid."""
        start = self.start if start is None else start
        thetas, shape = self.grid(resolution)
        if self.dim == 2:
            speed = 1.0 / np.asarray(self.slowness(thetas)).reshape(shape)
            return _fast_marching_plane(speed, start, self.half_width, resolution).ravel()
        from srms.environments.marching import fast_march_3d

        step = 2.0 * self.half_width / (resolution - 1)
        pts = np.asarray(thetas, dtype=float)
        seed = np.linalg.norm(pts - np.asarray(start), axis=-1).reshape(shape)
        return fast_march_3d(
            spacing=np.full(shape + (3,), step),
            slowness=np.asarray(self.slowness(thetas)).reshape(shape),
            blocked=np.zeros(shape, bool),
            seed_time=np.where(seed <= 1.5 * step, seed, np.inf),
            periodic=(False, False, False),
        ).ravel()

    def render_marker_deg(self) -> tuple[float, float]:
        return float(self.start[0]), float(self.start[1])


def _fast_marching_plane(speed: np.ndarray, start: tuple[float, float], half_width: float, n: int) -> np.ndarray:
    """Bounded fast marching of |∇T| = 1/speed on the plane grid -- the torus marcher's twin, minus
    wrap-around neighbours (no ``% n`` indexing; a query past the grid edge simply doesn't exist)."""
    step = 2.0 * half_width / (n - 1)
    axis = np.linspace(-half_width, half_width, n)
    grid1, grid2 = np.meshgrid(axis, axis)
    slowness = 1.0 / speed
    time = np.full((n, n), np.inf)
    accepted = np.zeros((n, n), dtype=bool)
    heap: list[tuple[float, int, int]] = []
    seed = np.sqrt((grid1 - start[0]) ** 2 + (grid2 - start[1]) ** 2)
    for j, i in zip(*np.where(seed <= step)):
        time[j, i] = float(seed[j, i])
        heapq.heappush(heap, (float(time[j, i]), int(j), int(i)))

    while heap:
        _, j, i = heapq.heappop(heap)
        if accepted[j, i]:
            continue
        accepted[j, i] = True
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nj, ni = j + dj, i + di
            if not (0 <= nj < n and 0 <= ni < n) or accepted[nj, ni]:
                continue
            along_x = min(
                time[nj, ni - 1] if ni - 1 >= 0 and accepted[nj, ni - 1] else np.inf,
                time[nj, ni + 1] if ni + 1 < n and accepted[nj, ni + 1] else np.inf,
            )
            along_y = min(
                time[nj - 1, ni] if nj - 1 >= 0 and accepted[nj - 1, ni] else np.inf,
                time[nj + 1, ni] if nj + 1 < n and accepted[nj + 1, ni] else np.inf,
            )
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
