"""Hyperbolic H^d environment (Poincaré ball model) with geodesic-ball obstacles.

The third manifold in the torus / sphere / hyperbolic set, and the one that exercises *negative*
curvature. Points are chart coordinates in the open unit ball ``B^d = {x ∈ R^d : ‖x‖ < 1}``
(``dim = tangent_dim = d``, like the flat torus and unlike the embedded sphere), carrying the
conformal metric

    g_ij(x) = λ(x)² δ_ij,    λ(x) = 2 / (1 − ‖x‖²)

which is the constant-curvature −1 metric. Everything the ``Environment`` protocol needs follows in
closed form from Möbius addition ``⊕``:

    d(x, y)      = 2·artanh‖(−x) ⊕ y‖
    Log_μ(x)     = d(μ, x) · u/‖u‖,          u = (−μ) ⊕ x
    Exp_μ(v)     = μ ⊕ (tanh(‖v‖/2) · v/‖v‖)
    jac_factor   = (r / sinh r)^(d−1),        r = d(μ, x)
    metric_inv   = ((1 − ‖x‖²)/2)² · I

``Log`` is normalised so its *Euclidean* norm is the geodesic distance — the same convention the
sphere uses (``θ·e_perp``), which is what lets ``eval_wrapped_gaussian`` treat the tangent space as
plain R^d and keeps the splat's ``A`` matrix in Riemannian units on every manifold.

**The Jacobian factor is the whole curvature story.** Sphere (K=+1) uses ``(θ/sin θ)^(n−1)``,
hyperbolic (K=−1) uses ``(r/sinh r)^(d−1)``, torus (K=0) uses ``1``. These are the same Jacobi-field
expression at the three constant curvatures, and swapping ``sin → sinh`` is the *entire* per-manifold
delta in the density. ``environments/test_manifolds.py`` checks it against autodiff.

**Ground truth is nearly free, and that is a property of the metric, not a shortcut.** The metric is
conformal, so ‖∇T‖_g = λ(x)⁻¹‖∇T‖_euclid and the Riemannian Eikonal ‖∇T‖_g = s reduces to the
*Euclidean* Eikonal ‖∇T‖_euclid = s(x)·λ(x). The ordinary Cartesian fast marcher therefore computes
exact hyperbolic ground truth; ``test_manifolds.py`` checks it against the closed-form obstacle-free
distance ``arcosh(1 + 2‖x−y‖² / ((1−‖x‖²)(1−‖y‖²)))``.

**Two declared modelling choices** (H^d is unbounded and its volume grows like e^{(d−1)r}, so neither
is forced):

1. *Truncation.* The workspace is the ball ``‖x‖ ≤ trunc_radius``; its rim enters ``sdf`` as an
   ordinary obstacle, so it is a wall exactly like the geodesic balls are. This bounds the domain,
   supplies the render mask, and keeps T finite. At the default 0.9 the workspace has hyperbolic
   diameter 4·artanh(0.9) ≈ 5.9 — comparable to the torus's π√2 ≈ 4.4 and the sphere's π, so the
   scale-sensitive hyperparameters (``init_scale``, ``source_radius``, ``scale_floor``) transfer
   across all three without retuning.
2. *Sampling is chart-uniform*, not hyperbolic-volume-uniform. Volume-uniform sampling would pile
   essentially every collocation point into the thin annulus at the rim. Chart-uniform matches the
   Cartesian grid the RMS is scored on, so training and evaluation see the same measure.
"""

from __future__ import annotations

import dataclasses
import heapq
import math

import jax
import jax.numpy as jnp
import numpy as np

from srms.environments.lorentz_hyperbolic import dist_perp, ray_dist, ray_dist_np

BallObstacle = tuple[float, ...]  # (*centre[dim] chart coords, hyperbolic radius)
RayObstacle = tuple[float, ...]  # (*origin_ambient[dim+1], *direction_ambient[dim+1], length,
# thickness) — ambient Lorentz-hyperboloid coordinates, *not* chart coordinates: geodesics are
# circular arcs in the Poincaré ball but straight mink_dot algebra on the hyperboloid, so ray
# obstacles are sampled/evaluated via ``_to_hyperboloid`` and lorentz_hyperbolic's ``dist_perp``/
# ``ray_dist`` rather than re-deriving Möbius-model ray distances from scratch.
Obstacle = BallObstacle | RayObstacle

_OBSTACLE_SEED_OFFSET = 5
_MAX_NORM = 1.0 - 1e-6  # keep artanh finite for points pushed onto/past the ideal boundary


# ---- Poincaré-ball primitives (JAX) ----------------------------------------------------------


def _clamp_to_radius(x: jnp.ndarray, radius) -> jnp.ndarray:
    """Scale x down to norm <= radius, with a gradient that's well-defined (the identity) at x=0.

    ``jnp.linalg.norm``'s *gradient* is ``x/‖x‖`` — a genuine 0/0 exactly at the origin, even though
    the value is perfectly smooth there (only the derivative is not). Both callers below (``_clamp_ball``
    and ``PoincareHyperbolicEnvironment.wrap_point``) get evaluated *at* the origin unconditionally by
    other code: ``weak_supervision.py``'s roadmap always keeps a node at exactly ``env.start``
    (the origin by default), and ``roadmap_base`` evaluates ``env.slowness``/``env.wrap_point`` there
    as part of every query point's loss. Even though that sub-term doesn't depend on the query point
    and should contribute exactly 0 to its gradient, JAX still computes the *local* partial derivative
    there as NaN, and ``NaN * 0 == NaN`` in IEEE 754 — worse, that NaN silently poisons *every* other
    point's gradient too, not just this one, because ``roadmap_base`` batches all nodes' segment
    samples through one call to ``env.slowness``/``env.wrap_point``, and a NaN local gradient at one
    batch row survives being multiplied by a zero cotangent for unrelated rows during backprop
    (confirmed by tracing: fixing only ``_clamp_ball`` left ``wrap_point``'s own unguarded norm doing
    the same thing, and *every* one of 300 roadmap nodes' gradients came back NaN as a result, not just
    the one sitting at the origin).

    Substituting a nonzero dummy input for the norm computation only where ``x`` is exactly zero, then
    selecting the true value (``x`` itself — both callers are the identity near the origin, since
    ``radius`` is always > 0) via ``jnp.where`` for the forward pass, fixes this at the root: since
    ``jnp.where``'s gradient only flows through the branch it selects, the origin's gradient comes out
    to the identity — the actually-correct answer, since clamping never activates near the origin
    anyway — instead of NaN.
    """
    is_zero = jnp.all(x == 0.0, axis=-1, keepdims=True)
    safe_x = jnp.where(is_zero, jnp.ones_like(x), x)
    norm = jnp.linalg.norm(safe_x, axis=-1, keepdims=True)
    scaled = safe_x * jnp.minimum(1.0, radius / jnp.maximum(norm, 1e-12))
    return jnp.where(is_zero, x, scaled)


def _clamp_ball(x: jnp.ndarray) -> jnp.ndarray:
    """Pull a point strictly inside the unit ball (identity for ‖x‖ < 1 − 1e-6).

    Grid corners lie outside the ball (a Cartesian grid over [−R, R]^d reaches R√d), and those cells
    are masked for scoring but still *evaluated*. Clamping keeps artanh finite there instead of
    letting NaNs propagate into an otherwise-valid figure. See ``_clamp_to_radius`` for why this has a
    safe (non-NaN) gradient at the origin.
    """
    return _clamp_to_radius(x, _MAX_NORM)


def mobius_add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Möbius addition x ⊕ y on the curvature −1 Poincaré ball; broadcasts over leading axes.

    The denominator is (1 − ‖x‖‖y‖)² + 2(‖x‖‖y‖ + ⟨x,y⟩) ≥ (1 − ‖x‖‖y‖)² > 0 inside the ball, so no
    branch is needed — the ``maximum`` is belt-and-braces for clamped boundary points.
    """
    xy = jnp.sum(x * y, axis=-1, keepdims=True)
    x2 = jnp.sum(x * x, axis=-1, keepdims=True)
    y2 = jnp.sum(y * y, axis=-1, keepdims=True)
    num = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
    den = 1.0 + 2.0 * xy + x2 * y2
    return num / jnp.maximum(den, 1e-12)


def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Hyperbolic distance d(x, y) = 2·artanh‖(−x) ⊕ y‖; broadcasts, returns the batch shape."""
    u = mobius_add(-_clamp_ball(x), _clamp_ball(y))
    norm = jnp.linalg.norm(u, axis=-1)
    return 2.0 * jnp.arctanh(jnp.clip(norm, 0.0, _MAX_NORM))


def log_map(mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Log_μ(x), normalised so ‖Log_μ(x)‖ (Euclidean) = d(μ, x) — the sphere's convention."""
    u = mobius_add(-_clamp_ball(mu), _clamp_ball(x))
    norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    dist = 2.0 * jnp.arctanh(jnp.clip(norm, 0.0, _MAX_NORM))
    return dist * u / jnp.maximum(norm, 1e-12)


def exp_map(mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Exp_μ(v), inverse of ``log_map`` (v in the same Euclidean-norm-is-distance units)."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    direction = v / jnp.maximum(norm, 1e-12)
    return mobius_add(_clamp_ball(mu), jnp.tanh(0.5 * norm) * direction)


def _mobius_add_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy Möbius addition (host-side, for RRT*/roadmap hot loops)."""
    xy = np.sum(x * y, axis=-1, keepdims=True)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1, keepdims=True)
    num = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
    return num / np.maximum(1.0 + 2.0 * xy + x2 * y2, 1e-12)


def _clamp_ball_np(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, _MAX_NORM / np.maximum(norm, 1e-12))


def _to_hyperboloid(x: jnp.ndarray) -> jnp.ndarray:
    """Poincaré-ball chart point(s) -> Lorentz-hyperboloid ambient coordinates (size dim+1).

    The standard isometry between the two curvature -1 models: y = (2x, 1+‖x‖²) / (1-‖x‖²)
    satisfies the hyperboloid's ``<y,y>_eta = -1`` exactly for any x with ‖x‖ < 1 (a short algebra
    check: Σy_i² - y_last² = [4‖x‖² - (1+‖x‖²)²] / (1-‖x‖²)² = -1). Used only for ray-obstacle
    geometry (``_sample_ray_obstacles``/``sdf``'s ray terms), which reuses lorentz_hyperbolic's
    ``dist_perp``/``ray_dist`` on this ambient point rather than re-deriving Möbius-model ray
    distances.
    """
    x = _clamp_ball(x)
    norm_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    denom = jnp.maximum(1.0 - norm_sq, 1e-12)
    return jnp.concatenate([2.0 * x / denom, (1.0 + norm_sq) / denom], axis=-1)


def _to_hyperboloid_np(x: np.ndarray) -> np.ndarray:
    """NumPy counterpart of ``_to_hyperboloid``."""
    x = _clamp_ball_np(x)
    norm_sq = np.sum(x * x, axis=-1, keepdims=True)
    denom = np.maximum(1.0 - norm_sq, 1e-12)
    return np.concatenate([2.0 * x / denom, (1.0 + norm_sq) / denom], axis=-1)


def _log_map_np(mu: np.ndarray, x: np.ndarray) -> np.ndarray:
    u = _mobius_add_np(-_clamp_ball_np(mu), _clamp_ball_np(x))
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    dist = 2.0 * np.arctanh(np.clip(norm, 0.0, _MAX_NORM))
    return dist * u / np.maximum(norm, 1e-12)


def _distance_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u = _mobius_add_np(-_clamp_ball_np(x), _clamp_ball_np(y))
    return 2.0 * np.arctanh(np.clip(np.linalg.norm(u, axis=-1), 0.0, _MAX_NORM))


@dataclasses.dataclass
class PoincareHyperbolicEnvironment:
    """H^d in the Poincaré ball with a smooth slowness field rising around geodesic balls.

    Obstacles default to geodesic balls (``num_obstacles``, ``obstacle_radius``) — the same
    primitive torus/sphere/so3/LorentzHyperbolicEnvironment use, so the default scene is directly
    comparable across every manifold in this repo. Capsule-thickened geodesic *rays* are also
    available (``num_ray_obstacles``, off by default), matching LorentzHyperbolicEnvironment's ray
    obstacles exactly (same ``ray_length``/``ray_thickness`` distributions, same clearance rule) —
    see ``_sample_ray_obstacles``/``_to_hyperboloid`` for how ray geometry is evaluated in this
    chart despite geodesics here being circular arcs rather than straight lines.
    """

    start: tuple[float, ...] = (0.0, 0.0)
    dim: int = 2
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    num_ray_obstacles: int = 0
    ray_length: tuple[float, float] = (0.8, 1.8)
    ray_thickness: tuple[float, float] = (0.08, 0.15)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    trunc_radius: float = 0.9
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != self.dim:
            raise ValueError(f"start has {len(self.start)} coords but dim={self.dim}")
        if float(np.linalg.norm(self.start)) >= self.trunc_radius:
            raise ValueError(f"start must lie inside the truncation ball ‖x‖ < {self.trunc_radius}")
        self.tangent_dim = self.dim
        self.ambient_dim = self.dim + 1  # Lorentz-hyperboloid ambient size, for ray-obstacle geometry
        # Chart extent, not a period: only mlp.py reads it, and a ball has nothing periodic to encode.
        self.domain: tuple[float, float] = (-self.trunc_radius, self.trunc_radius)
        # Math-mode subscripts, not raw x₁/x₂ glyphs — see torus.py's TorusEnvironment for why
        # (LaTeX has no legacy-font slot for a bare Unicode subscript digit either).
        self.axis_labels: tuple[str, str] = (r"$x_1$ (Poincaré chart)", r"$x_2$ (Poincaré chart)")
        self.render_extent: tuple[float, float, float, float] = (
            -self.trunc_radius,
            self.trunc_radius,
            -self.trunc_radius,
            self.trunc_radius,
        )
        self.wall_distance = float(2.0 * np.arctanh(min(self.trunc_radius, _MAX_NORM)))
        self.has_dense_gt = self.dim in (2, 3)  # dense fast marching tractable at 2-D and 3-D
        # one shared rng stream so ball obstacles are sampled before rays, deterministic given seed
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        self.ball_obstacles: tuple[BallObstacle, ...] = self._sample_ball_obstacles(rng)
        self.ray_obstacles: tuple[RayObstacle, ...] = self._sample_ray_obstacles(rng)
        self.obstacles: tuple[Obstacle, ...] = self.ball_obstacles + self.ray_obstacles

    @property
    def title(self) -> str:
        # $H^{...}$, not a raw "H^..." — see torus.py's TorusEnvironment.title for why.
        return f"hyperbolic $H^{self.dim}$ (Poincaré ball) — time-to-go ({len(self.obstacles)} obstacles)"

    @property
    def gt_label(self) -> str:
        return "ground truth — Euclidean FMM, conformally rescaled"

    def _sample_ball_obstacles(self, rng: np.random.Generator) -> tuple[BallObstacle, ...]:
        """Reproducible geodesic-ball obstacles, clear of both the source and the rim wall — the
        same primitive torus/sphere/so3/LorentzHyperbolicEnvironment use, so a default (all-ball)
        scene here is directly comparable to those, and to LorentzHyperbolicEnvironment's own
        default scene in particular."""
        start = np.asarray(self.start, dtype=float)
        obstacles: list[BallObstacle] = []
        while len(obstacles) < self.num_obstacles:
            direction = rng.standard_normal(self.dim)
            direction /= np.linalg.norm(direction)
            centre = direction * (0.75 * self.trunc_radius) * rng.uniform(0.0, 1.0) ** (1.0 / self.dim)
            radius = float(rng.uniform(*self.obstacle_radius))
            clear_of_source = _distance_np(centre, start) > radius + 0.3
            clear_of_wall = self.wall_distance - _distance_np(centre, np.zeros(self.dim)) > radius + 0.3
            if clear_of_source and clear_of_wall:
                obstacles.append((*centre.tolist(), radius))
        return tuple(obstacles)

    def _sample_ray_obstacles(self, rng: np.random.Generator) -> tuple[RayObstacle, ...]:
        """Reproducible capsule-thickened geodesic-ray obstacles, mirroring
        LorentzHyperbolicEnvironment's ``_sample_ray_obstacles`` exactly (same "outward tangent"
        construction, same clearance rule) but sampled/stored in *ambient Lorentz-hyperboloid*
        coordinates via ``_to_hyperboloid`` — geodesics are circular arcs in the Poincaré ball, so
        the tangent-continuation algebra that construction relies on only closes in the hyperboloid
        chart, not here."""
        start_ambient = _to_hyperboloid(jnp.asarray(self.start, dtype=jnp.float32))
        obstacles: list[RayObstacle] = []
        while len(obstacles) < self.num_ray_obstacles:
            p_np = self.sample_domain(rng, 1)[0]
            length = float(rng.uniform(*self.ray_length))
            thickness = float(rng.uniform(*self.ray_thickness))
            p_ambient = _to_hyperboloid(jnp.asarray(p_np, dtype=jnp.float32))
            d, w = dist_perp(start_ambient, p_ambient)  # tangent at start pointing toward p
            if float(d) > thickness + 0.3:
                v = jnp.sinh(d) * start_ambient + jnp.cosh(d) * w  # tangent at p, continuing outward
                obstacles.append((*p_ambient.tolist(), *np.asarray(v).tolist(), length, thickness))
        return tuple(obstacles)

    # ---- manifold geometry -----------------------------------------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Intrinsic tangent coordinates of x at μ; ‖·‖ = geodesic distance."""
        return log_map(mu, x)

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Single global chart, so the ambient and tangent-frame log maps coincide."""
        return log_map(mu, x)

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exp_mu(v) = mu ⊕ tanh(‖v‖/2)·v̂ — inverse of log_map (see base.Environment)."""
        return exp_map(mu, v)

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """|det ∂Log_μ/∂x| = (r/sinh r)^(d−1) — the sphere's (θ/sin θ)^(n−1) at curvature −1.

        Inverse Riemannian volume element: det d Exp_μ|_v = (sinh‖v‖/‖v‖)^(d−1). Unlike the sphere's
        factor this is *bounded above by 1* and decays — negative curvature spreads geodesics apart,
        so a wrapped Gaussian's tail thins faster than its flat counterpart rather than piling up at
        a cut locus. There is no cut locus here at all (Exp is a global diffeomorphism), which is why
        the wrapped Gaussian is exact on H^d with a single term.
        """
        r = jnp.maximum(distance(mu, x), 1e-6)  # r ≥ 0, so this only regularises the 0/0 at r = 0
        return (r / jnp.sinh(r)) ** (self.dim - 1)

    def splat_precompute(self, mu: jnp.ndarray):
        """Per-splat geometry: the clamped centre, so the per-point loop does not re-clamp it."""
        return _clamp_ball(mu)

    def log_and_jac(self, pre, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(log_map, jac_factor) from one Mobius addition.

        ``jac_factor`` used to call ``distance``, which redid the whole Mobius/artanh computation that
        ``log_map`` had just done — roughly a 2x waste on every (point, splat) pair.
        """
        u = mobius_add(-pre, _clamp_ball(x))
        norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        dist = 2.0 * jnp.arctanh(jnp.clip(norm, 0.0, _MAX_NORM))
        r = jnp.maximum(jnp.squeeze(dist, -1), 1e-6)
        return dist * u / jnp.maximum(norm, 1e-12), (r / jnp.sinh(r)) ** (self.dim - 1)

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Retract onto the workspace: clamp to ``trunc_radius``, not just the open unit ball.

        Every optimizer-owned position that goes through this hook (srm's splat centres via
        ``post_step``, RRT*/roadmap steering, hntfields' TD-loss stepped query) should live in the
        *scored* workspace, not merely somewhere the hyperbolic distance formulas stay finite.
        ``_clamp_ball``'s ``1 - 1e-6`` margin is a numerical safety net for internal math (``sdf`` at
        grid corners, etc.); this is the actual geometric boundary — points past it are masked out of
        every score and treated as a wall by ``sdf`` (see ``_rim_sdf``), so nothing should be left free
        to drift out there. Measured effect on ``srm``: without this, an unmasked AdamW weight-decay
        pull toward the origin was the only thing discouraging centres from wandering past
        ``trunc_radius``, and removing that pull alone (see ``ntfields.py``'s ``decay_mask`` note) let
        45% of splats end up outside the workspace, wasting capacity that this retraction reclaims.
        See ``_clamp_to_radius`` for why this has a safe (non-NaN) gradient at the origin -- this
        function is exactly as exposed to that bug as ``_clamp_ball`` is, since ``roadmap_base``
        evaluates it directly at the roadmap's origin-sitting source node every step.
        """
        return _clamp_to_radius(x, self.trunc_radius)

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x * np.minimum(1.0, self.trunc_radius / np.maximum(norm, 1e-12))

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Tangent vector at a pointing toward b, ‖·‖ = geodesic distance; broadcasts."""
        return _log_map_np(np.asarray(a, dtype=float), np.asarray(b, dtype=float))

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points at exact hyperbolic distance eps from the source, via Exp_start."""
        start = np.asarray(self.start, dtype=float)
        tangent = rng.standard_normal((n, self.dim))
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-12
        return np.asarray(exp_map(jnp.asarray(start), jnp.asarray(eps * tangent)), dtype=float)

    def metric_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """g^{ij}(x) = ((1 − ‖x‖²)/2)² δ^{ij} — the conformal factor λ(x)⁻², squared out of g = λ²δ."""
        conformal = 0.5 * (1.0 - jnp.sum(_clamp_ball(x) ** 2, axis=-1))
        return conformal**2 * jnp.eye(self.dim)

    def geodesic(self, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic hyperbolic distance to the source (the known obstacle-free base)."""
        return distance(start, x)

    # ---- obstacle / slowness field -----------------------------------------

    def _rim_sdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Hard in/out indicator for the truncation ball — **not** a smooth wall.

        The truncation is where the *grid* stops, not a physical obstacle, so it must not contribute
        cost. An earlier version returned the smooth distance ``wall_distance − d(0,x)``, which fed
        the slowness ramp and turned the rim into the largest obstacle in the scene: because
        hyperbolic volume concentrates outward, that ramp covered 14.9% of the scored cells, inflated
        ground truth from a free-space max of ~2.94 to 9.33, and carried 47% of the total squared
        error. The torus and sphere are compact and need no truncation at all, so that handicap had no
        counterpart there and the three manifolds were not being asked the same question.

        Returning ±LARGE instead makes ``sdf`` reduce to the obstacle union inside the ball (so
        slowness is obstacle-driven, exactly as on the other two manifolds) while still marking the
        exterior for the render/score mask. Nothing is lost by dropping the wall: metric balls in a
        Hadamard manifold are geodesically convex, so no optimal path between interior points ever
        needs to leave, and ``λ → ∞`` already makes the ideal boundary unreachable.
        """
        large = jnp.asarray(1e3)
        return jnp.where(distance(jnp.zeros(self.dim), x) <= self.wall_distance, large, -large)

    def sdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Signed hyperbolic distance to the union of obstacle balls, obstacle rays (if any) *and
        the truncation wall*."""
        per = [distance(jnp.asarray(obs[:-1]), x) - obs[-1] for obs in self.ball_obstacles]
        if self.ray_obstacles:
            ambient_x = _to_hyperboloid(x)
            for obs in self.ray_obstacles:
                origin = jnp.array(obs[: self.ambient_dim])
                direction = jnp.array(obs[self.ambient_dim : 2 * self.ambient_dim])
                length, thickness = obs[-2], obs[-1]
                per.append(ray_dist(ambient_x, origin, direction, length) - thickness)
        per.append(self._rim_sdf(x))
        return jnp.min(jnp.stack(per, axis=0), axis=0)

    def slowness(self, x: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside obstacles / past the wall."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(x) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy signed distance (host-side, for RRT*'s hot loop)."""
        points = np.asarray(points, dtype=float)
        per = [_distance_np(np.array(obs[:-1]), points) - obs[-1] for obs in self.ball_obstacles]
        if self.ray_obstacles:
            ambient_points = _to_hyperboloid_np(points)
            for obs in self.ray_obstacles:
                origin = np.array(obs[: self.ambient_dim])
                direction = np.array(obs[self.ambient_dim : 2 * self.ambient_dim])
                length, thickness = obs[-2], obs[-1]
                per.append(ray_dist_np(ambient_points, origin, direction, length) - thickness)
        inside = _distance_np(np.zeros(self.dim), points) <= self.wall_distance
        per.append(np.where(inside, 1e3, -1e3))  # domain mask, not a wall — see _rim_sdf
        return np.min(per, axis=0)

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy smooth slowness (host-side, for RRT*'s hot loop)."""
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(self.sdf_np(points) / self.slow_width))

    # ---- sampling / ground truth --------------------------------------------

    @property
    def volume(self) -> float:
        """Riemannian volume of the truncated ball: Area(S^(d-1))·∫₀^R sinh^(d-1)(r) dr.

        Same quadrature the sampler inverts, so the two cannot disagree.
        """
        grid = np.linspace(0.0, self.wall_distance, 8192)
        radial = np.trapezoid(np.sinh(grid) ** (self.dim - 1), grid)
        sphere_area = 2.0 * np.pi ** (self.dim / 2.0) / math.gamma(self.dim / 2.0)
        return float(sphere_area * radial)

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform w.r.t. the **Riemannian volume** of the truncated ball — the same rule the other
        manifolds already follow.

        ``TorusEnvironment`` samples uniformly in its chart, which on a flat manifold *is* uniform by
        volume; ``SphereEnvironment`` samples uniformly on the sphere itself, likewise. Only here do
        the two differ, because ``dvol = λ(x)^d dx`` with λ = 2/(1−‖x‖²) reaching 10.5 at the default
        wall. Chart-uniform sampling therefore starves the region that holds most of the manifold: at
        d=2, the annulus ‖x‖ ∈ [0.8, 0.9] is 59% of the hyperbolic area but only 21% of the chart, and
        the resulting fit under-covered it badly — the band ‖x‖ ≥ 0.7 carried 92% of the squared error
        while being 46% of the scored cells, and the model's peak T reached only 4.73 against a true
        9.33. Sampling by volume removes that asymmetry rather than adding a hyperbolic-specific knob.

        In geodesic polar coordinates ``dvol ∝ sinh^(d−1)(r) dr dΩ``, so the radial CDF is inverted
        numerically (exact in the limit of the quadrature grid; the d=2 closed form is
        ``r = arcosh(1 + u·(cosh R − 1))``, which ``test_manifolds.py`` checks against).
        """
        direction = rng.standard_normal((n, self.dim))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-12
        grid = np.linspace(0.0, self.wall_distance, 8192)
        weight = np.sinh(grid) ** (self.dim - 1)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(grid))])
        radius = np.interp(rng.uniform(0.0, 1.0, size=n), cdf / cdf[-1], grid)
        return direction * np.tanh(0.5 * radius)[:, None]  # geodesic radius -> Poincaré chart radius

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int]]:
        """Cartesian chart grid over [−trunc_radius, trunc_radius]², raveled to [resolution², 2].

        The corners fall outside the ball; they are masked by ``sdf`` (the rim wall) exactly as
        obstacle interiors are, so they never enter the score.
        """
        if self.dim not in (2, 3):
            raise NotImplementedError("grid()/ground_truth() need a dense grid — only tractable at dim<=3")
        axis = np.linspace(-self.trunc_radius, self.trunc_radius, resolution)
        mesh = np.meshgrid(*([axis] * self.dim), indexing="ij" if self.dim == 3 else "xy")
        points = np.stack([m.ravel() for m in mesh], axis=-1)
        return jnp.asarray(points, dtype=jnp.float32), (resolution,) * self.dim

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Exact hyperbolic ground truth via a *Euclidean* fast marcher (see the module docstring).

        ‖∇T‖_g = s ⟺ ‖∇T‖_euclid = s·λ, so the Cartesian marcher run with the conformally rescaled
        slowness ``s(x)·2/(1−‖x‖²)`` returns hyperbolic travel time — no hyperbolic-specific solver.
        """
        start = self.start if start is None else start
        points, shape = self.grid(resolution)
        points_np = np.asarray(points, dtype=float)
        norm_sq = np.sum(points_np**2, axis=-1)
        conformal = 2.0 / np.maximum(1.0 - norm_sq, 1e-9)  # lambda(x); no dimension dependence
        slowness = np.asarray(self.slowness(points)) * conformal
        blocked = norm_sq >= self.trunc_radius**2
        step = 2.0 * self.trunc_radius / (resolution - 1)
        if self.dim == 2:
            return _fast_marching_ball(
                slowness.reshape(shape), blocked.reshape(shape), points_np.reshape(*shape, 2), start, step
            ).ravel()
        # dim == 3: isotropic metric, so every axis takes the same physical spacing lambda(x)*step.
        from srms.environments.marching import fast_march_3d

        seed = _distance_np(np.asarray(start, dtype=float), points_np).reshape(shape)
        return fast_march_3d(
            spacing=np.repeat((conformal * step).reshape(shape)[..., None], 3, axis=-1),
            slowness=np.asarray(self.slowness(points)).reshape(shape),
            blocked=blocked.reshape(shape),
            seed_time=np.where(seed <= 3.0 * float(np.max(conformal[~blocked] * step)), seed, np.inf),
            periodic=(False, False, False),
        ).ravel()

    def render_marker_deg(self) -> tuple[float, float]:
        """(x₁, x₂) source position in chart coordinates (the render chart is the ball itself)."""
        start = np.asarray(self.start, dtype=float)
        return float(start[0]), float(start[1])


def _fast_marching_ball(
    slowness: np.ndarray, blocked: np.ndarray, coords: np.ndarray, start: tuple[float, ...], step: float
) -> np.ndarray:
    """Non-periodic isotropic fast marching of ‖∇T‖ = slowness on a Cartesian grid with a mask.

    The torus marcher's twin minus the wrap-around neighbours, plus a ``blocked`` mask for cells
    outside the truncation ball. ``slowness`` here is already the conformally rescaled field, which
    is what makes the Euclidean update rule solve the hyperbolic Eikonal equation — see
    ``HyperbolicEnvironment.ground_truth``.
    """
    rows, cols = slowness.shape
    time = np.full((rows, cols), np.inf)
    accepted = blocked.copy()  # masked cells are never accepted and never relax a neighbour
    heap: list[tuple[float, int, int]] = []

    seed_dist = _distance_np(np.asarray(start, dtype=float), coords.reshape(-1, 2)).reshape(rows, cols)
    for j, i in zip(*np.where((seed_dist <= 2.0 * step) & ~blocked)):
        time[j, i] = float(seed_dist[j, i])
        heapq.heappush(heap, (float(time[j, i]), int(j), int(i)))

    while heap:
        _, j, i = heapq.heappop(heap)
        if accepted[j, i]:
            continue
        accepted[j, i] = True
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nj, ni = j + dj, i + di
            if not (0 <= nj < rows and 0 <= ni < cols) or accepted[nj, ni]:
                continue
            along_x = min(
                time[nj, ni - 1] if ni - 1 >= 0 and accepted[nj, ni - 1] else np.inf,
                time[nj, ni + 1] if ni + 1 < cols and accepted[nj, ni + 1] else np.inf,
            )
            along_y = min(
                time[nj - 1, ni] if nj - 1 >= 0 and accepted[nj - 1, ni] else np.inf,
                time[nj + 1, ni] if nj + 1 < rows and accepted[nj + 1, ni] else np.inf,
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
