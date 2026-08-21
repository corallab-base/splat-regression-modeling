"""S^n environment (n-sphere embedded in R^{n+1}) with spherical-cap obstacles and a smooth
slowness field.

``S²`` (``n=2``, points in R³) is the canonical case; the same construction generalizes to any
``n``. Unlike the flat torus, S^n has no global non-singular *intrinsic* chart, so points are
represented in *ambient* coordinates (unit vectors in R^{n+1}: ``dim = n + 1``) while splats still
need an intrinsic tangent-space scale matrix (``tangent_dim = n``). The tangent frame at a centre
``mu`` is built from a Householder reflection mapping a reference axis to ``mu`` (JAX-safe: static
shapes, no dynamic indexing), generalizing the cross-product frame used for S² specifically in
``srms/lib/manifold_splat.py``.

The Eikonal PDE loss needs the *intrinsic* gradient magnitude of a field that is only ever
evaluated through ambient coordinates; ``metric_inv(x) = I - x xᵀ`` (the tangent-plane projector)
gives exactly that via ``grad @ metric_inv(x) @ grad`` (see ``srms/methods/strategies/eikonal.py``),
since the intrinsic gradient of a restriction to an embedded submanifold is the ambient gradient
projected onto the tangent space, regardless of how the field is extended off the manifold.

Grid/ground-truth (dense fast marching) and rendering only make sense for ``n=2`` (a dense grid is
intractable and unplottable beyond that) and raise ``NotImplementedError`` otherwise; training
itself (mesh-free collocation) works at any ``n``.
"""

from __future__ import annotations

import dataclasses
import heapq
import math

import jax
import jax.numpy as jnp
import numpy as np

from srms.environments import marching

Obstacle = tuple[float, ...]  # (*centre[dim] unit vector, angular radius)

_OBSTACLE_SEED_OFFSET = 5
_EPS = 1e-3


# ---- n-sphere geometry primitives (JAX) ------------------------------------------------------


def _sphere_frame(mu: jnp.ndarray, n: int) -> jnp.ndarray:
    """Orthonormal frame for T_mu S^n: [dim, n], dim = n + 1.

    Built as a Householder reflection mapping a reference axis (e0, or e1 if mu is nearly parallel
    to e0) to mu; the images of the remaining standard basis vectors are then an orthonormal frame
    of the tangent space. Both reference choices are computed and selected with ``jnp.where`` so the
    function has no dynamic-shape control flow (jit/grad safe).
    """
    dim = n + 1
    basis = jnp.eye(dim)
    e0, e1 = basis[:, 0], basis[:, 1]
    rest0 = basis[:, 1:]
    rest1 = jnp.concatenate([basis[:, :1], basis[:, 2:]], axis=1)
    near = jnp.abs(mu[0]) > 0.9
    ref = jnp.where(near, e1, e0)
    rest = jnp.where(near, rest1, rest0)
    w = mu - ref
    w_sq = jnp.dot(w, w)
    safe = w_sq > 1e-10
    w_safe = jnp.where(safe, w, e0)
    denom = jnp.where(safe, w_sq, 1.0)
    reflected = rest - 2.0 * jnp.outer(w_safe, w_safe @ rest) / denom
    return jnp.where(safe, reflected, rest)


def _theta_perp(mu: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Geodesic distance theta and unit tangent direction e_perp from mu toward x."""
    cos_theta = jnp.clip(jnp.dot(mu, x), -1.0 + _EPS, 1.0 - _EPS)
    theta = jnp.arccos(cos_theta)
    perp = x - cos_theta * mu
    e_perp = perp / jnp.maximum(jnp.linalg.norm(perp), 1e-8)
    return theta, e_perp


def wrap(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to [-π, π) (jax); kept for parity with torus.py, unused on S^n itself."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


@dataclasses.dataclass
class SphereEnvironment:
    """S^n (unit sphere in R^{n+1}) with a smooth slowness field rising around geodesic caps."""

    start: tuple[float, ...] = (0.0, 0.0, 1.0)
    n: int = 2
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.3, 0.6)
    slowness_max: float = 10.0
    slow_width: float = 0.1
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != self.n + 1:
            raise ValueError(f"start has {len(self.start)} coords but n={self.n} needs {self.n + 1}")
        if abs(float(np.linalg.norm(self.start)) - 1.0) > 1e-3:
            raise ValueError(f"start must be a unit vector on S^{self.n}, got norm {np.linalg.norm(self.start):.4f}")
        self.dim = self.n + 1
        self.tangent_dim = self.n
        self.domain: tuple[float, float] = (-1.0, 1.0)
        # Math mode, not raw ψ/θ glyphs — see torus.py's TorusEnvironment for why (LaTeX has no
        # legacy-font slot for a bare Unicode Greek letter, unlike matplotlib's own mathtext).
        self.axis_labels: tuple[str, str] = (r"$\psi$ longitude (deg)", r"$\theta$ colatitude (deg)")
        self.render_extent: tuple[float, float, float, float] = (-180.0, 180.0, 0.0, 180.0)
        self.has_dense_gt = self.n in (2, 3)
        self.obstacles: tuple[Obstacle, ...] = self._sample_obstacles()

    @property
    def title(self) -> str:
        # $S^{...}$, not a raw "S^..." — see torus.py's TorusEnvironment.title for why.
        return f"sphere $S^{self.n}$ — time-to-go ({self.num_obstacles} obstacles)"

    @property
    def gt_label(self) -> str:
        return "ground truth — anisotropic FMM (geodesic-polar grid)"

    def _sample_obstacles(self) -> tuple[Obstacle, ...]:
        """Reproducible geodesic-cap obstacles, clear of the source."""
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        start = np.asarray(self.start, dtype=float)
        obstacles: list[Obstacle] = []
        while len(obstacles) < self.num_obstacles:
            z = rng.standard_normal(self.dim)
            centre = z / np.linalg.norm(z)
            radius = float(rng.uniform(*self.obstacle_radius))
            cos_theta = np.clip(np.dot(centre, start), -1.0, 1.0)
            if np.arccos(cos_theta) > radius + 0.3:
                obstacles.append((*centre.tolist(), radius))
        return tuple(obstacles)

    # ---- manifold geometry -----------------------------------------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Intrinsic tangent-frame coordinates (size tangent_dim) of x at mu, for splat evaluation."""
        theta, e_perp = _theta_perp(mu, x)
        frame = _sphere_frame(mu, self.n)
        return theta * (e_perp @ frame)

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Ambient tangent vector (size dim) at mu pointing toward x; ‖·‖ = geodesic distance."""
        theta, e_perp = _theta_perp(mu, x)
        return theta * e_perp

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exp_mu(v) from tangent-frame coordinates v (size tangent_dim) — inverse of log_map.

        Lifts v through the same Householder frame log_map projects onto, then walks the great
        circle: cos‖v‖·mu + sin‖v‖·direction. See base.Environment for why this exists.
        """
        ambient = _sphere_frame(mu, self.n) @ v
        norm = jnp.linalg.norm(ambient)
        direction = ambient / jnp.maximum(norm, 1e-12)
        return jnp.cos(norm) * mu + jnp.sin(norm) * direction

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """|det ∂Log_mu/∂x| = (θ/sinθ)^(n-1), the inverse Riemannian volume element on S^n."""
        theta, _ = _theta_perp(mu, x)
        return (theta / jnp.maximum(jnp.sin(theta), 1e-8)) ** (self.n - 1)

    def splat_precompute(self, mu: jnp.ndarray):
        """Per-splat geometry: the tangent frame at mu, which the per-point loop must not rebuild.

        ``_sphere_frame`` constructs a [dim, n] Householder matrix and depends only on the splat
        centre, but evaluating k splats at n points used to build it n*k times instead of k. That was
        the bulk of the sphere's 8x per-step cost against the flat torus.
        """
        return mu, _sphere_frame(mu, self.n)

    def log_and_jac(self, pre, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(log_map, jac_factor) sharing one ``_theta_perp`` call instead of two."""
        mu, frame = pre
        theta, e_perp = _theta_perp(mu, x)
        jac = (theta / jnp.maximum(jnp.sin(theta), 1e-8)) ** (self.n - 1)
        return theta * (e_perp @ frame), jac

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project back onto the unit sphere (identity if already unit norm)."""
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ambient tangent vector at a pointing toward b; ‖·‖ = geodesic distance. Broadcasts."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        cos_theta = np.clip(np.sum(a * b, axis=-1), -1.0 + _EPS, 1.0 - _EPS)
        theta = np.arccos(cos_theta)
        perp = b - cos_theta[..., None] * a
        e_perp = perp / np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-8)
        return theta[..., None] * e_perp

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points at exact geodesic distance eps from the source (random tangent directions)."""
        start = np.asarray(self.start, dtype=float)
        tangent = rng.standard_normal((n, self.dim))
        tangent = tangent - (tangent @ start)[:, None] * start[None, :]
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-12
        return np.cos(eps) * start[None, :] + np.sin(eps) * tangent

    def metric_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Tangent-plane projector I - x xᵀ; recovers the intrinsic squared-gradient norm from an
        ambient gradient (the round metric induced by the isometric embedding)."""
        return jnp.eye(self.dim) - jnp.outer(x, x)

    def geodesic(self, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic great-circle distance arccos(<start, x>) (the known base)."""
        cos_theta = jnp.clip(jnp.sum(x * start, axis=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        return jnp.arccos(cos_theta)

    # ---- obstacle / slowness field -----------------------------------------

    def sdf(self, points: jnp.ndarray) -> jnp.ndarray:
        """Signed geodesic distance to the union of obstacle caps."""
        per = []
        for obs in self.obstacles:
            centre, radius = jnp.array(obs[:-1]), obs[-1]
            cos_theta = jnp.clip(points @ centre, -1.0 + 1e-7, 1.0 - 1e-7)
            per.append(jnp.arccos(cos_theta) - radius)
        return jnp.min(jnp.stack(per, axis=0), axis=0)

    def slowness(self, points: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside obstacles."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(points) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy signed distance (host-side, for RRT*'s hot loop)."""
        per = []
        for obs in self.obstacles:
            centre, radius = np.array(obs[:-1]), obs[-1]
            cos_theta = np.clip(points @ centre, -1.0 + 1e-7, 1.0 - 1e-7)
            per.append(np.arccos(cos_theta) - radius)
        return np.min(per, axis=0)

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy smooth slowness (host-side, for RRT*'s hot loop)."""
        sdf = self.sdf_np(points)
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(sdf / self.slow_width))

    # ---- sampling / ground truth --------------------------------------------

    @property
    def volume(self) -> float:
        """Surface area of S^n = 2π^((n+1)/2)/Γ((n+1)/2); the sampler is uniform against it."""
        return float(2.0 * np.pi ** (self.dim / 2.0) / math.gamma(self.dim / 2.0))

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform samples on S^n (Gaussian direction, normalized) — volume-uniform."""
        z = rng.standard_normal((n, self.dim))
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int]]:
        """Lat-long grid (cell-centred colatitude θ x periodic azimuth ψ), fixed to the (0,...,0,1)
        axis; raveled to [resolution², dim=3], plus its (resolution, resolution) shape."""
        if self.n == 3:
            polar, shape, _ = marching.polar_grid_3d(resolution, float(np.pi), np.sin)
            r, u = marching.polar_to_unit3(polar)
            pts = np.concatenate([np.sin(r)[:, None] * u, np.cos(r)[:, None]], axis=-1)
            return jnp.asarray(pts, dtype=jnp.float32), shape
        if self.n != 2:
            raise NotImplementedError("grid()/ground_truth() need a dense grid — only tractable at n=2,3")
        d_theta = np.pi / resolution
        d_psi = 2.0 * np.pi / resolution
        theta_axis = (np.arange(resolution) + 0.5) * d_theta
        psi_axis = -np.pi + np.arange(resolution) * d_psi
        grid_theta, grid_psi = np.meshgrid(theta_axis, psi_axis, indexing="ij")
        ambient = np.stack(
            [np.sin(grid_theta) * np.cos(grid_psi), np.sin(grid_theta) * np.sin(grid_psi), np.cos(grid_theta)],
            axis=-1,
        )
        return jnp.asarray(ambient.reshape(-1, 3), dtype=jnp.float32), (resolution, resolution)

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Fast marching of |∇T| = 1/speed on the S² lat-long grid; returns a raveled array."""
        start = self.start if start is None else start
        points, shape = self.grid(resolution)
        if self.n == 3:
            _, _, spacing = marching.polar_grid_3d(resolution, float(np.pi), np.sin)
            pts = np.asarray(points, dtype=float)
            seed = np.arccos(np.clip(pts @ np.asarray(start, dtype=float), -1.0, 1.0))
            return marching.fast_march_3d(
                spacing=spacing,
                slowness=np.asarray(self.slowness(points)).reshape(shape),
                blocked=np.zeros(shape, bool),
                seed_time=np.where(seed <= 2.0 * np.pi / resolution, seed, np.inf).reshape(shape),
                periodic=(False, False, True),
                wrap_fn=marching.polar_topology(shape),
            ).ravel()
        speed = 1.0 / np.asarray(self.slowness(points)).reshape(shape)
        return _fast_marching_sphere(speed, start, resolution).ravel()

    def render_marker_deg(self) -> tuple[float, float]:
        """(ψ, θ) position of the source in the render chart's degree units."""
        start = np.asarray(self.start, dtype=float)
        psi = np.degrees(np.arctan2(start[1], start[0]))
        theta = np.degrees(np.arccos(np.clip(start[2], -1.0, 1.0)))
        return float(psi), float(theta)


def _fast_marching_sphere(speed: np.ndarray, start: tuple[float, ...], resolution: int) -> np.ndarray:
    """Fast marching of |grad T| = 1/speed on S² in geodesic-polar (colatitude θ, azimuth ψ) coords.

    θ is cell-centred (avoids the pole singularity), ψ is periodic; the grid's physical (metric)
    spacing is (Δθ, sinθ·Δψ) since ds² = dθ² + sin²θ dψ² — an anisotropic orthogonal grid, unlike
    the flat torus's isotropic one, so each 2D upwind update solves the general anisotropic quadratic
    rather than `_fast_marching_torus`'s closed-form isotropic case. Grid is fixed to the
    (0,...,0,1) axis regardless of `start` (seeded by geodesic distance to `start`), matching
    `_fast_marching_torus`'s approach of seeding an axis-aligned grid from an arbitrary interior point.
    """
    n = resolution
    d_theta = np.pi / n
    d_psi = 2.0 * np.pi / n
    theta_axis = (np.arange(n) + 0.5) * d_theta
    psi_axis = -np.pi + np.arange(n) * d_psi
    grid_theta, grid_psi = np.meshgrid(theta_axis, psi_axis, indexing="ij")
    ambient = np.stack(
        [np.sin(grid_theta) * np.cos(grid_psi), np.sin(grid_theta) * np.sin(grid_psi), np.cos(grid_theta)], axis=-1
    )
    slowness = 1.0 / speed
    time = np.full((n, n), np.inf)
    accepted = np.zeros((n, n), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    start_vec = np.asarray(start, dtype=float)
    geo = np.arccos(np.clip(ambient @ start_vec, -1.0, 1.0))
    for i, j in zip(*np.where(geo <= d_theta)):
        time[i, j] = float(geo[i, j])
        heapq.heappush(heap, (float(time[i, j]), int(i), int(j)))

    def solve_quad(a: float, wa: float, b: float, wb: float, f: float) -> float:
        fallback = min(
            a + f / np.sqrt(wa) if not np.isinf(a) else np.inf,
            b + f / np.sqrt(wb) if not np.isinf(b) else np.inf,
        )
        if np.isinf(a) or np.isinf(b):
            return fallback
        A = wa + wb
        B = -2.0 * (a * wa + b * wb)
        C = a * a * wa + b * b * wb - f * f
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return fallback
        T = (-B + np.sqrt(disc)) / (2.0 * A)
        return T if T >= max(a, b) else fallback

    while heap:
        _, i, j = heapq.heappop(heap)
        if accepted[i, j]:
            continue
        accepted[i, j] = True
        for ni, nj in ((i - 1, j), (i + 1, j), (i, (j - 1) % n), (i, (j + 1) % n)):
            if not (0 <= ni < n) or accepted[ni, nj]:
                continue
            along_theta = min(
                time[ni - 1, nj] if 0 <= ni - 1 < n and accepted[ni - 1, nj] else np.inf,
                time[ni + 1, nj] if 0 <= ni + 1 < n and accepted[ni + 1, nj] else np.inf,
            )
            along_psi = min(
                time[ni, (nj - 1) % n] if accepted[ni, (nj - 1) % n] else np.inf,
                time[ni, (nj + 1) % n] if accepted[ni, (nj + 1) % n] else np.inf,
            )
            d_lat = max(np.sin(theta_axis[ni]), 1e-6) * d_psi
            candidate = solve_quad(along_theta, 1.0 / d_theta**2, along_psi, 1.0 / d_lat**2, slowness[ni, nj])
            if candidate < time[ni, nj]:
                time[ni, nj] = candidate
                heapq.heappush(heap, (float(candidate), int(ni), int(nj)))
    return time
