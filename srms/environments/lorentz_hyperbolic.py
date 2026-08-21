"""H^n environment (n-dim hyperboloid, Lorentz/hyperboloid model, embedded in Minkowski R^{n,1}).

STATUS (see ``hyperbolic_TODO.md``): steps 1-5, i.e. all of it — geometry primitives
(``_lorentz_frame``, ``geodesic``, ``log_map``/``log_map_ambient``, ``jac_factor``,
``metric_inv``), obstacles/slowness, volume-correct sampling, manifold-projection utilities
(``sdf``, ``slowness``, ``sample_domain``, ``wrap_point``, ``displacement_np``,
``boundary_ring_np``), registration in ``ENVIRONMENTS``/``srms/run.py``, the geodesic-polar
grid/fast-marching ground truth (``grid``, ``ground_truth``, ``_fast_marching_hyperbolic``) at
n=2, and Poincaré-disk rendering (``render_grid_xy``, ``render_marker_deg``, wired into
``srms/viz.py``'s curvilinear ``pcolormesh`` path via ``srms/run.py``'s duck-typed dispatch) —
``--environment hyperbolic --method eikonal`` trains at any ``--dim`` and, at ``--dim 2``, scores
against ground truth and saves a [GT | prediction | error] figure on the Poincaré disk, same as
``--dim 2`` does for torus/sphere on their own charts. See hyperbolic_TODO.md's "Known issue" note
on a residual, occasional NaN risk in long training runs (same risk class already documented for
sphere+srm/weak_supervision+srm elsewhere in this codebase, not unique to this environment).

Points on ``H^n`` are unit-timelike vectors in R^{n,1}: ``x`` with Minkowski square
``<x,x>_eta = -1`` and ``x[-1] > 0`` (the upper sheet), signature ``eta = diag(1,...,1,-1)``
(last coordinate timelike). This is the direct analog of ``SphereEnvironment``'s ambient
unit-vector representation (``<x,x> = +1`` in R^{n+1}, Euclidean signature) with the Minkowski
form and hyperbolic trig (``cosh/sinh/arccosh``) standing in for the sphere's circular trig.

Unlike S^n, H^n has no cut locus (it's simply connected and non-positively curved — Cartan–
Hadamard), so the log map and its Jacobian are smooth everywhere: no antipodal clipping is
needed, and the tangent frame only needs one reference point (the apex), not two (contrast
``_sphere_frame``'s near-antipodal fallback).

The tangent-frame construction (``_lorentz_frame``) and the wrapped-density machinery this
feeds (``srms/lib/manifold_splat.py``'s ``eval_wrapped_gaussian``) follow Nagano, Yamaguchi,
Fujita & Koyama, *"A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based
Learning"* (ICML 2019) — the hyperbolic analog of the Said et al./Chevallier et al. sphere
construction already cited there.
"""

from __future__ import annotations

import dataclasses
import heapq
import math

import jax
import jax.numpy as jnp
import numpy as np

BallObstacle = tuple[float, ...]  # (*centre[dim], radius) — a geodesic ball, the same primitive
# torus/sphere/so3/PoincareHyperbolicEnvironment use.
RayObstacle = tuple[float, ...]  # (*origin[dim], *direction[dim], length, thickness) — a capsule-
# thickened geodesic ray: {cosh(t)*origin + sinh(t)*direction : t in [0, length]}, direction an
# eta-unit tangent vector at origin.
Obstacle = BallObstacle | RayObstacle

# Matches PoincareHyperbolicEnvironment's default wall_distance = 2*artanh(trunc_radius=0.9): the two
# hyperbolic environments' default scenes then have literally the same geodesic domain radius, and
# (bonus identity, not a coincidence) render_grid_xy's Poincaré-disk outer ring lands at exactly 0.9.
_DEFAULT_DOMAIN_RADIUS = 2.0 * math.atanh(0.9)

_EPS = 1e-3  # matches sphere.py's clip margin: bounds arccosh'/arccos' near the domain edge to a
# reasonable ~22 (vs. ~707 at 1e-6) rather than letting it approach a machine-precision-scale
# blowup — see hyperbolic_TODO.md's note on the NaN this margin being too tight caused in training
_SMALL_D = 1e-4
_OBSTACLE_SEED_OFFSET = 5


# ---- Minkowski bilinear form ------------------------------------------------------------------


def mink_dot(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Minkowski bilinear form <u,v>_eta = u[:-1]·v[:-1] - u[-1]*v[-1]; broadcasts over leading dims."""
    return jnp.sum(u[..., :-1] * v[..., :-1], axis=-1) - u[..., -1] * v[..., -1]


def _mink_dot_cols(w: jnp.ndarray, mat: jnp.ndarray) -> jnp.ndarray:
    """<w, mat[:, i]>_eta for each column i of mat [dim, n] -> [n]."""
    return w[:-1] @ mat[:-1, :] - w[-1] * mat[-1, :]


def _mink_dot_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy counterpart of mink_dot, for host-side sampling/RRT*."""
    return np.sum(u[..., :-1] * v[..., :-1], axis=-1) - u[..., -1] * v[..., -1]


def _radial_cdf(n: int, domain_radius: float, num: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Numerically inverted CDF of the hyperbolic-volume radial density sinh(r)^(n-1) on
    [0, domain_radius], used by sample_domain/_sample_ball_obstacles/_sample_ray_obstacles for
    volume-correct ball sampling.

    Hyperbolic volume grows like sinh(r)^(n-1) dr (the "sphere of radius r" has that much
    (n-1)-measure), so uniform-in-r sampling would badly undersample the outer region — unlike
    the torus/sphere, where uniform box/angle sampling already is volume-correct. Closed form
    only at n=2 (CDF ~ cosh(r)-1, invertible via arccosh); kept numeric (trapezoidal CDF +
    np.interp inversion) so this works at any n, matching every other Environment's
    ``sample_domain`` (training itself works at any dim, not just the n=2 rendering case).
    """
    r_grid = np.linspace(0.0, domain_radius, num)
    density = np.sinh(r_grid) ** (n - 1)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(r_grid))])
    cdf /= cdf[-1]
    return r_grid, cdf


# ---- H^n geometry primitives (JAX) -------------------------------------------------------------


def _lorentz_frame(mu: jnp.ndarray, n: int) -> jnp.ndarray:
    """Eta-orthonormal frame for T_mu H^n: [dim, n], dim = n + 1.

    Built as a Lorentz (eta-orthogonal) Householder-style reflection mapping the apex
    a = (0,...,0,1) to mu; the images of the first n standard basis vectors (an eta-orthonormal
    frame of T_a H^n, since eta restricted to span(e_1..e_n) is Euclidean) are then an
    eta-orthonormal frame of T_mu H^n. The reflecting vector w = a - mu is eta-null
    (eta(w,w) = 2(cosh(d)-1), d = dist(a,mu)) only at mu = a itself, not at some antipodal point
    as on the sphere — H^n has no cut locus — so only one reference point is needed here, unlike
    ``_sphere_frame``'s near-antipodal fallback. jnp.where keeps the mu=a case AD-safe (no 0/0).
    """
    dim = n + 1
    basis = jnp.eye(dim)
    apex = basis[:, -1]
    rest = basis[:, :n]
    w = apex - mu
    w_sq = mink_dot(w, w)
    safe = w_sq > 1e-10
    w_safe = jnp.where(safe, w, apex)
    denom = jnp.where(safe, w_sq, 1.0)
    coeffs = _mink_dot_cols(w_safe, rest)  # <w, e_i>_eta, [n]
    reflected = rest - 2.0 * jnp.outer(w_safe, coeffs) / denom
    return jnp.where(safe, reflected, rest)


def dist_perp(mu: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Geodesic distance d and eta-unit tangent direction e_perp from mu toward x.

    c = <mu,x>_eta <= -1 always (two points on the upper sheet); clipped away from -1 (x=mu) so
    arccosh's argument stays >= 1+eps. perp = x + c*mu is tangent at mu by construction
    (<mu,perp>_eta = c + c*<mu,mu>_eta = c - c = 0).

    eta(perp,perp) is *not* recomputed via mink_dot(perp,perp) — that mixes the (clipped) c with
    the (unclipped) mu,x, and for genuinely near-coincident mu,x this is a catastrophic-
    cancellation trap: eta(perp,perp) = -(c-c_true)^2 + (c_true^2-1), a *difference* of two small
    terms (unlike the sphere's Euclidean analog, which is a sum of squares and so can't cancel)
    that can land slightly negative or spuriously near zero even when d is safely clipped away
    from 0 — right in the unprotected zone just above a jnp.maximum floor where sqrt's gradient
    (and especially the *second* derivative the PDE loss's grad-of-grad exercises) blows up. This
    was reproducibly NaN-ing training within ~7 jit-compiled steps (hyperbolic_TODO.md) despite a
    sphere.py-matched clip margin, because the margin wasn't the problem. Fix: eta(perp,perp) =
    sinh(d)^2 = c^2-1, computed directly from the single already-clipped c — always >= ~2*eps,
    comfortably bounded away from zero by construction, no independent recomputation to cancel.
    """
    c = jnp.minimum(mink_dot(mu, x), -1.0 - _EPS)
    d = jnp.arccosh(-c)
    perp = x + c * mu
    perp_norm = jnp.sqrt(c * c - 1.0)  # = sinh(d), consistent with c by construction (see above)
    e_perp = perp / perp_norm
    return d, e_perp


def _d_over_sinh(d: jnp.ndarray) -> jnp.ndarray:
    """d / sinh(d), AD-safe at d=0 (Taylor: 1 - d^2/6 + 7d^4/360 - ...; limit is 1)."""
    small = jnp.abs(d) < _SMALL_D
    d_safe = jnp.where(small, jnp.ones_like(d), d)
    exact = d_safe / jnp.sinh(d_safe)
    taylor = 1.0 - d**2 / 6.0 + 7.0 * d**4 / 360.0
    return jnp.where(small, taylor, exact)


def ray_dist(x: jnp.ndarray, origin: jnp.ndarray, direction: jnp.ndarray, length: float) -> jnp.ndarray:
    """Geodesic distance from x to the ray {cosh(t)*origin + sinh(t)*direction : t in [0, length]}
    (origin on H^n, direction an eta-unit tangent vector at origin). Broadcasts over leading dims
    of x (origin/direction stay single).

    Closed form: minimizing eta(x, cosh(t)*origin+sinh(t)*direction) over the *full* line
    (t in R) by setting the t-derivative to zero gives
        t* = arctanh(-eta(x,direction) / eta(x,origin))
    (eta(x,origin) <= -1 always, so this is never a division by ~0); clip t* to [0, length] for
    the ray/segment, then reuse the point-to-point geodesic formula at that t (no need to build
    the ambient point cosh(t)*origin+sinh(t)*direction explicitly: eta(x, that point) is just
    cosh(t)*eta(x,origin) + sinh(t)*eta(x,direction), bilinearity). The arctanh argument is
    clipped defensively to (-1,1); verified empirically to already land there (uniqueness of the
    orthogonal projection onto a complete geodesic in CAT(0) space), same "verify before trusting"
    treatment as metric_inv got.
    """
    eta_xo = mink_dot(x, origin)
    eta_xd = mink_dot(x, direction)
    t_star = jnp.arctanh(jnp.clip(-eta_xd / eta_xo, -1.0 + 1e-6, 1.0 - 1e-6))
    t = jnp.clip(t_star, 0.0, length)
    c = jnp.minimum(jnp.cosh(t) * eta_xo + jnp.sinh(t) * eta_xd, -1.0 - _EPS)
    return jnp.arccosh(-c)


def ray_dist_np(x: np.ndarray, origin: np.ndarray, direction: np.ndarray, length: float) -> np.ndarray:
    """NumPy counterpart of ray_dist, for host-side sampling/RRT*."""
    eta_xo = _mink_dot_np(x, origin)
    eta_xd = _mink_dot_np(x, direction)
    t_star = np.arctanh(np.clip(-eta_xd / eta_xo, -1.0 + 1e-6, 1.0 - 1e-6))
    t = np.clip(t_star, 0.0, length)
    c = np.minimum(np.cosh(t) * eta_xo + np.sinh(t) * eta_xd, -1.0 - _EPS)
    return np.arccosh(-c)


def _point_dist_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy counterpart of ``geodesic``: point-to-point distance (not to a ray), for ball-obstacle
    clearance checks and ``sdf_np``'s ball terms."""
    c = np.minimum(_mink_dot_np(x, y), -1.0 - _EPS)
    return np.arccosh(-c)


# ---- H^n environment -----------------------------------------------------------------------


@dataclasses.dataclass
class LorentzHyperbolicEnvironment:
    """H^n (hyperboloid sheet in Minkowski R^{n,1}), Lorentz model, with a smooth slowness field
    rising around geodesic-ray obstacles, scoped to a geodesic ball of radius ``domain_radius``
    around ``start`` (H^n has infinite volume, unlike the periodic torus or compact sphere, so
    there's no "whole manifold" to place obstacles on or sample collocation points from —
    ``domain_radius`` plays the torus's ``[-π,π)^dim`` / the sphere's whole-S^n role).

    Obstacles default to geodesic balls — the same primitive torus/sphere/so3/
    PoincareHyperbolicEnvironment use (``num_obstacles``, ``obstacle_radius``), so the default scene
    is directly comparable across every manifold in this repo, hyperbolic included. Capsule-
    thickened geodesic *rays* are also available (``num_ray_obstacles``, off by default): each ray
    originates at a point sampled within ``domain_radius`` of ``start`` and continues *outward*
    along the ``start``-to-origin geodesic (i.e. its direction is the tangent at the origin point
    continuing away from ``start``, not toward it) for a random length. This casts a "shadow
    wedge" directly behind each ray from the source's point of view — a deliberately sharper
    occlusion test than a ball, given H^n's exponential volume growth makes that shadow region
    large. See hyperbolic_TODO.md's "Ray obstacles" section for the derivation.

    See module docstring STATUS.
    """

    start: tuple[float, ...] = (0.0, 0.0, 1.0)
    n: int = 2
    domain_radius: float = _DEFAULT_DOMAIN_RADIUS
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    num_ray_obstacles: int = 0
    ray_length: tuple[float, float] = (0.8, 1.8)
    ray_thickness: tuple[float, float] = (0.08, 0.15)
    slowness_max: float = 10.0
    slow_width: float = 0.1
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != self.n + 1:
            raise ValueError(f"start has {len(self.start)} coords but n={self.n} needs {self.n + 1}")
        start_arr = jnp.asarray(self.start, dtype=jnp.float32)
        c = float(mink_dot(start_arr, start_arr))
        if abs(c + 1.0) > 1e-3:
            raise ValueError(f"start must lie on H^{self.n} (<x,x>_eta=-1), got {c:.4f}")
        if self.start[-1] <= 0:
            raise ValueError("start must be on the upper sheet (last coordinate > 0)")
        self.dim = self.n + 1
        self.tangent_dim = self.n
        # placeholder box; unlike the torus this isn't periodic, so it only matters if the mlp
        # backend's periodic feature encoding (env.domain-keyed) is ever pointed at this environment
        self.domain: tuple[float, float] = (-self.domain_radius, self.domain_radius)
        self.axis_labels: tuple[str, str] = ("Poincaré disk x", "Poincaré disk y")
        # not used by viz.py's pcolormesh path (render_grid_xy supplies real coords instead) but
        # kept for Environment-protocol completeness; matches the disk's natural bounding box
        self.render_extent: tuple[float, float, float, float] = (-1.05, 1.05, -1.05, 1.05)
        self._start_frame_np = np.asarray(_lorentz_frame(start_arr, self.n))  # [dim, n], reused by
        # sample_domain/boundary_ring_np (both draw eta-unit tangent directions at start)
        self._radial_cdf_r, self._radial_cdf_p = _radial_cdf(self.n, self.domain_radius)
        # one shared rng stream so ball obstacles are sampled before rays, deterministic given seed
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        self.ball_obstacles: tuple[BallObstacle, ...] = self._sample_ball_obstacles(rng)
        self.ray_obstacles: tuple[RayObstacle, ...] = self._sample_ray_obstacles(rng)
        self.obstacles: tuple[Obstacle, ...] = self.ball_obstacles + self.ray_obstacles

    @property
    def title(self) -> str:
        return f"hyperbolic H^{self.n} — time-to-go ({len(self.obstacles)} obstacles)"

    def _sample_ball_obstacles(self, rng: np.random.Generator) -> tuple[BallObstacle, ...]:
        """Hardcoded geodesic-ball obstacles in the middle region of the domain, clear of the source.

        Positioned in the central region to create a navigational challenge between source and
        boundary, rather than periphery clutter.
        """
        start = np.asarray(self.start, dtype=np.float64)
        # For n=2: hardcoded spatial coordinates (x, y) in R^2, each paired with a radius
        # These are embedded to the hyperboloid: (x, y) → (x, y, sqrt(1 + x² + y²))
        # Coordinates chosen to be far enough from start (0,0,1) to pass distance check
        hardcoded_spatial = [
            (0.7, 0.5, 0.2),      # middle-right region
            (-1.0, 0.6, 0.20),     # middle-left region
            (0.5, -0.8, 0.20),     # middle-lower region
        ]
        obstacles: list[BallObstacle] = []
        for x, y, rad in hardcoded_spatial[:self.num_obstacles]:
            # Embed in R^{n+1}: for n=2, (x, y) → (x, y, t) where t = sqrt(1 + x² + y²)
            # This projects (x,y) to the upper sheet of the hyperboloid via wrap_point logic
            if self.n == 2:
                spatial = np.array([x, y], dtype=np.float64)
                t = np.sqrt(1.0 + np.sum(spatial * spatial))
                centre = np.concatenate([spatial, [t]])
            else:
                # For n != 2, sample instead (hardcoding only defined for n=2)
                centre = self.sample_domain(rng, 1)[0]
            # Use provided radius
            radius = rad
            # Safety check: ensure obstacle is clear of the source
            if float(_point_dist_np(centre, start)) > radius + 0.3:
                obstacles.append((*centre.tolist(), radius))
        # If we didn't get enough obstacles (e.g., some failed the distance check), pad with random ones
        while len(obstacles) < self.num_obstacles:
            centre = self.sample_domain(rng, 1)[0]
            radius = float(rng.uniform(*self.obstacle_radius))
            if float(_point_dist_np(centre, start)) > radius + 0.3:
                obstacles.append((*centre.tolist(), radius))
        return tuple(obstacles)

    def _sample_ray_obstacles(self, rng: np.random.Generator) -> tuple[RayObstacle, ...]:
        """Reproducible geodesic-ray obstacles within domain_radius of start, clear of the source.

        Each ray originates at a point p sampled via sample_domain, with direction = the tangent
        at p continuing the start->p geodesic *outward* (away from start). For a unit-speed
        geodesic gamma(t) = cosh(t)*start + sinh(t)*w (w = tangent at start toward p), the
        velocity at t=d (i.e. at p) is gamma'(d) = sinh(d)*start + cosh(d)*w — general facts for
        any hyperboloid geodesic: eta(gamma(t),gamma'(t))=0 (tangency) and eta(gamma'(t),gamma'(t))=1
        (unit speed) hold for all t, so this is automatically an eta-unit tangent vector at p
        without needing a separate frame construction. Since a geodesic ray extended past a point
        away from the basepoint moves monotonically farther from it, the ray's own closest point
        to start is always its origin p — so the clearance check only needs d(p, start).
        """
        start = jnp.asarray(self.start, dtype=jnp.float32)
        obstacles: list[RayObstacle] = []
        while len(obstacles) < self.num_ray_obstacles:
            p_np = self.sample_domain(rng, 1)[0]
            p = jnp.asarray(p_np, dtype=jnp.float32)
            length = float(rng.uniform(*self.ray_length))
            thickness = float(rng.uniform(*self.ray_thickness))
            d, w = dist_perp(start, p)  # tangent at start pointing toward p
            if float(d) > thickness + 0.3:
                v = jnp.sinh(d) * start + jnp.cosh(d) * w  # tangent at p, continuing outward
                obstacles.append((*p_np.tolist(), *np.asarray(v).tolist(), length, thickness))
        return tuple(obstacles)

    # ---- manifold geometry -----------------------------------------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Intrinsic tangent-frame coordinates (size tangent_dim) of x at mu, for splat evaluation."""
        d, e_perp = dist_perp(mu, x)
        frame = _lorentz_frame(mu, self.n)
        return d * _mink_dot_cols(e_perp, frame)

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Ambient tangent vector (size dim) at mu pointing toward x; eta-norm = geodesic distance."""
        d, e_perp = dist_perp(mu, x)
        return d * e_perp

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """|det d(log_map)/dx| = (d/sinh d)^(n-1), the inverse Riemannian volume element on H^n.

        Unlike the sphere's (theta/sin theta)^(n-1), this needs no clipping away from a
        cut-locus singularity — d/sinh(d) is smooth and bounded in (0,1] for all d >= 0.
        """
        d, _ = dist_perp(mu, x)
        return _d_over_sinh(d) ** (self.n - 1)

    def splat_precompute(self, mu: jnp.ndarray):
        """Per-splat geometry: the eta-orthonormal tangent frame at mu, which the per-point loop
        must not rebuild — mirrors ``sphere.py``'s ``splat_precompute`` (see its docstring for the
        n*k-vs-k cost this hoists out of ``srm.py``'s inner loop; ``_lorentz_frame`` depends only
        on the splat centre)."""
        return mu, _lorentz_frame(mu, self.n)

    def log_and_jac(self, pre, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(log_map, jac_factor) sharing one ``dist_perp`` call instead of two — mirrors
        ``sphere.py``'s ``log_and_jac``."""
        mu, frame = pre
        d, e_perp = dist_perp(mu, x)
        jac = _d_over_sinh(d) ** (self.n - 1)
        return d * _mink_dot_cols(e_perp, frame), jac

    def metric_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Quadratic form recovering the intrinsic squared-gradient norm from an ambient
        Euclidean-component gradient via grad @ metric_inv(x) @ grad: eta + x x^T.

        NOTE this is *not* eta + (eta x)(eta x)^T (the naive Minkowski-form analog of the
        sphere's I - xx^T projector) — that was tried first and caught wrong by
        hyperbolic_TODO.md's mandated finite-difference check (~5x relative error). The subtlety:
        jax.grad always returns *Euclidean* partials (the metric-free pairing between a
        differential and a tangent vector is always the plain dot product, regardless of which
        bilinear form defines the submanifold), so "raising the index" with eta and "projecting
        onto the tangent space" use *different* pairings once the ambient form isn't Euclidean.

        Derivation: solve eta(w,v) = grad_F·v (Euclidean dot) for tangent w (eta(x,w)=0). Ansatz
        w = eta⊙grad_F + lambda*x; tangency forces lambda = x·grad_F (Euclidean dot, using
        eta(x,x)=-1), giving w = eta⊙grad_F + (x·grad_F)*x and
            eta(w,w) = eta(grad_F,grad_F) + (x·grad_F)^2.
        (The same method with kappa = eta(x,x) = +1 and eta = Euclidean, i.e. the sphere, gives
        lambda = -(x·grad_F) and eta(w,w) = ||grad_F||^2 - (x·grad_F)^2 = grad_F^T(I-xx^T)grad_F —
        reproducing the sphere's known-correct projector, which calibrates that this derivation
        method itself is sound.)
        """
        eta_diag = jnp.concatenate([jnp.ones(self.dim - 1), -jnp.ones(1)])
        return jnp.diag(eta_diag) + jnp.outer(x, x)

    def geodesic(self, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic geodesic distance arccosh(-<x,start>_eta) (the known base)."""
        c = jnp.minimum(mink_dot(x, start), -1.0 - _EPS)
        return jnp.arccosh(-c)

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project onto H^n by fixing the spatial coordinates and solving for the timelike one:
        (s, t) -> (s, sqrt(1+|s|^2)). Exact identity for x already on H^n (t = sqrt(1+|s|^2)
        already holds), and — unlike rescaling x by 1/sqrt(-<x,x>_eta) along its own ray, which is
        only valid while x is genuinely timelike — well-defined for *any* ambient x, spacelike or
        null included, with no denominator to blow up.

        This replaced the eta-rescaling retraction after it caused real training failures: an
        unconstrained AdamW step on a splat centre ``B`` can — especially with weight_decay, which
        pulls B toward the ambient *origin*, exactly the rescaling retraction's singularity — drift
        B onto or past the light cone, where -<x,x>_eta <= 0 and the old formula's
        ``1/sqrt(max(..., 1e-12))`` doesn't correct that, it *amplifies* it (a huge scale on an
        already-drifted vector); measured ``|B|`` growing 9.5 -> 1.7e20 over 20 steps before the
        eventual overflow to inf/nan poisoned every parameter. This retraction cannot diverge that
        way: as the spatial part shrinks (e.g. under weight decay), sqrt(1+|s|^2) -> 1 and the point
        converges to the apex (0,...,0,1) — H^n's own origin — instead of a singularity.
        """
        spatial = x[..., :-1]
        t = jnp.sqrt(1.0 + jnp.sum(spatial * spatial, axis=-1, keepdims=True))
        return jnp.concatenate([spatial, t], axis=-1)

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        """NumPy counterpart of ``wrap_point``."""
        x = np.asarray(x, dtype=np.float64)
        spatial = x[..., :-1]
        t = np.sqrt(1.0 + np.sum(spatial * spatial, axis=-1, keepdims=True))
        return np.concatenate([spatial, t], axis=-1)

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ambient tangent vector at a pointing toward b; eta-norm = geodesic distance. Broadcasts.

        NumPy counterpart of log_map_ambient (same c/perp/e_perp construction, see dist_perp)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = np.minimum(_mink_dot_np(a, b), -1.0 - _EPS)
        d = np.arccosh(-c)
        perp = b + c[..., None] * a
        perp_sq = np.maximum(_mink_dot_np(perp, perp), 1e-12)
        e_perp = perp / np.sqrt(perp_sq)[..., None]
        return d[..., None] * e_perp

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points at exact geodesic distance eps from start (the Eikonal BC ring): random
        eta-unit tangent directions at start (Euclidean-unit combinations of the eta-orthonormal
        start frame — eta-unit since the frame is eta-orthonormal), exponentiated by eps."""
        z = rng.standard_normal((n, self.n))
        z /= np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
        tangent = z @ self._start_frame_np.T
        start = np.asarray(self.start, dtype=np.float64)
        return np.cosh(eps) * start[None, :] + np.sinh(eps) * tangent

    # ---- obstacle / slowness field -----------------------------------------

    def sdf(self, points: jnp.ndarray) -> jnp.ndarray:
        """Signed geodesic distance to the union of ball obstacles and capsule-thickened rays."""
        per = [self.geodesic(points, jnp.array(obs[:-1])) - obs[-1] for obs in self.ball_obstacles]
        for obs in self.ray_obstacles:
            origin = jnp.array(obs[: self.dim])
            direction = jnp.array(obs[self.dim : 2 * self.dim])
            length, thickness = obs[-2], obs[-1]
            per.append(ray_dist(points, origin, direction, length) - thickness)
        return jnp.min(jnp.stack(per, axis=0), axis=0)

    def slowness(self, points: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside obstacles."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(points) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy signed distance (host-side, for RRT*'s hot loop)."""
        per = [_point_dist_np(points, np.array(obs[:-1])) - obs[-1] for obs in self.ball_obstacles]
        for obs in self.ray_obstacles:
            origin = np.array(obs[: self.dim])
            direction = np.array(obs[self.dim : 2 * self.dim])
            length, thickness = obs[-2], obs[-1]
            per.append(ray_dist_np(points, origin, direction, length) - thickness)
        return np.min(per, axis=0)

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy smooth slowness (host-side, for RRT*'s hot loop)."""
        sdf = self.sdf_np(points)
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(sdf / self.slow_width))

    # ---- sampling --------------------------------------------------------

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Volume-correct samples in the geodesic ball of radius domain_radius around start (see
        _radial_cdf); direction is uniform (Euclidean-unit combination of the eta-orthonormal
        start frame), radius drawn from the sinh(r)^(n-1)-weighted CDF, embedded via the
        hyperboloid exponential map at start."""
        u = rng.uniform(0.0, 1.0, size=n)
        r = np.interp(u, self._radial_cdf_p, self._radial_cdf_r)
        z = rng.standard_normal((n, self.n))
        z /= np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
        tangent = z @ self._start_frame_np.T
        start = np.asarray(self.start, dtype=np.float64)
        return np.cosh(r)[:, None] * start[None, :] + np.sinh(r)[:, None] * tangent

    # ---- ground truth / rendering -----------------------------------------------------------

    def _r_psi_grid(self, resolution: int) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centred r (0, domain_radius], periodic ψ meshgrid (indexing="ij"), each shape
        (resolution, resolution). r is cell-centred (never exactly 0) to sidestep the r=0
        coordinate singularity where all ψ directions meet — same trick sphere.py's grid()/
        _fast_marching_sphere use to avoid their pole singularities. Shared by
        _geodesic_polar_ambient (training/ground-truth points) and render_grid_xy (Poincaré-disk
        plot coordinates) so both are guaranteed to describe the same grid."""
        d_r = self.domain_radius / resolution
        d_psi = 2.0 * np.pi / resolution
        r_axis = (np.arange(resolution) + 0.5) * d_r
        psi_axis = -np.pi + np.arange(resolution) * d_psi
        return np.meshgrid(r_axis, psi_axis, indexing="ij")

    def _geodesic_polar_ambient(self, resolution: int, centre: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Geodesic-polar (r, ψ) grid centered at ``centre``, mapped to ambient coordinates.

        ψ is periodic. r ranges over (0, domain_radius] — a *hard* outer boundary (no wraparound),
        since H^n's geodesic ball is bounded but not periodic in r, unlike the sphere's θ ∈ [0, π]
        (also non-periodic, but for a different reason: opposite poles, not a truncation). Returns
        [resolution², dim] ambient points (float64) and (resolution, resolution); shared by grid()
        (always centre=start) and ground_truth() (honors a start override, unlike grid() whose
        signature has no such parameter).
        """
        frame = np.asarray(_lorentz_frame(jnp.asarray(centre, dtype=jnp.float32), self.n))  # [dim, 2]
        grid_r, grid_psi = self._r_psi_grid(resolution)
        tangent = np.cos(grid_psi)[..., None] * frame[:, 0] + np.sin(grid_psi)[..., None] * frame[:, 1]
        ambient = np.cosh(grid_r)[..., None] * centre[None, None, :] + np.sinh(grid_r)[..., None] * tangent
        return ambient.reshape(-1, self.dim), (resolution, resolution)

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int]]:
        """Geodesic-polar grid centered at start; raveled to [resolution², dim], plus its
        (resolution, resolution) shape. Only meaningful at n=2 (a dense grid is intractable and
        unplottable beyond that); NotImplementedError otherwise (matches sphere.py/torus.py)."""
        if self.n != 2:
            raise NotImplementedError("grid()/ground_truth() need a dense grid — only tractable at n=2")
        ambient, shape = self._geodesic_polar_ambient(resolution, np.asarray(self.start, dtype=np.float64))
        return jnp.asarray(ambient, dtype=jnp.float32), shape

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Fast marching of |∇T| = 1/speed on the geodesic-polar grid; returns a raveled array."""
        if self.n != 2:
            raise NotImplementedError("grid()/ground_truth() need a dense grid — only tractable at n=2")
        centre = np.asarray(self.start if start is None else start, dtype=np.float64)
        ambient, shape = self._geodesic_polar_ambient(resolution, centre)
        speed = 1.0 / np.asarray(self.slowness(jnp.asarray(ambient, dtype=jnp.float32))).reshape(shape)
        return _fast_marching_hyperbolic(speed, resolution, self.domain_radius).ravel()

    def render_grid_xy(self, resolution: int) -> tuple[np.ndarray, np.ndarray]:
        """Poincaré-disk (X, Y) curvilinear coordinates for grid(resolution)'s (r, ψ) grid, for
        viz.py's pcolormesh rendering (duck-typed: torus/sphere don't define this, so run.py falls
        back to their rectangular imshow(extent=...) path).

        rho = tanh(r/2), angle = ψ — the closed-form map from geodesic-polar coordinates centered
        at a point to that *same point's* Poincaré-disk polar coordinates. This only needs the
        intrinsic (r, ψ) grid (via _r_psi_grid, identical to the one _geodesic_polar_ambient
        embeds into ambient space), not a literal ambient-Lorentz-to-disk projection — so it's
        correct for any `start`, not just the literal apex the standard x[:n]/(1+x[n]) formula is
        centered on. The default domain_radius (= 2*artanh(0.9), matching
        PoincareHyperbolicEnvironment's default trunc_radius) puts the outer ring at exactly
        tanh(domain_radius/2) = 0.9 — the same disk radius as that environment's truncation rim, so
        the two hyperbolic environments' default renders share an outer boundary.
        """
        if self.n != 2:
            raise NotImplementedError("render_grid_xy() needs a dense grid — only tractable at n=2")
        grid_r, grid_psi = self._r_psi_grid(resolution)
        rho = np.tanh(grid_r / 2.0)
        return rho * np.cos(grid_psi), rho * np.sin(grid_psi)

    def render_grid_edges_xy(self, resolution: int) -> tuple[np.ndarray, np.ndarray]:
        """Poincaré-disk cell-*edge* coordinates ((resolution+1, resolution+1) each), for
        viz.py's ``pcolormesh`` specifically.

        ``render_grid_xy``'s cell-*centre* coordinates are right for ``contour`` (which needs
        (X, Y) matching the field array's shape exactly) but wrong for ``pcolormesh``'s
        ``shading="auto"``: that infers edges by extrapolating halfway between adjacent centres,
        which silently breaks exactly at the ψ = -π/π seam (this is a periodic polar grid — the
        last column's centres are adjacent to the first column's in physical space but far apart
        in array order) — matplotlib warns "not monotonically increasing" and miscolors the seam.
        Supplying exact edges sidesteps the inference entirely: r edges span ``[0,
        domain_radius]`` (endpoints included, unlike the cell-*centred* r_axis), ψ edges span
        ``[-π, π]`` (closing the loop exactly).
        """
        if self.n != 2:
            raise NotImplementedError("render_grid_edges_xy() needs a dense grid — only tractable at n=2")
        r_edges = np.linspace(0.0, self.domain_radius, resolution + 1)
        psi_edges = np.linspace(-np.pi, np.pi, resolution + 1)
        grid_r, grid_psi = np.meshgrid(r_edges, psi_edges, indexing="ij")
        rho = np.tanh(grid_r / 2.0)
        return rho * np.cos(grid_psi), rho * np.sin(grid_psi)

    def render_marker_deg(self) -> tuple[float, float]:
        """(x, y) position of the source in the Poincaré disk — always (0,0), since the render
        grid is centered at start by construction (grid()/_geodesic_polar_ambient). Named _deg for
        Environment-protocol parity with torus/sphere's literal-degree markers; here it's
        dimensionless disk coordinates, not degrees (the protocol docstring already genericizes
        this as "the render chart's units")."""
        return 0.0, 0.0


def _fast_marching_hyperbolic(speed: np.ndarray, resolution: int, domain_radius: float) -> np.ndarray:
    """Fast marching of |grad T| = 1/speed on the geodesic-polar (r, ψ) grid centered at the source.

    Direct analog of ``_fast_marching_sphere``: the grid's physical (metric) spacing is
    (Δr, sinh(r)·Δψ) since ds² = dr² + sinh²(r) dψ² (vs. the sphere's ds² = dθ² + sin²θ dψ², i.e.
    (Δθ, sinθ·Δψ)) — the same anisotropic orthogonal grid, solved with the same general
    anisotropic-quadratic upwind update. Two differences from the sphere: (1) r has a hard outer
    boundary at domain_radius with no wraparound (H^n's geodesic ball is bounded but not periodic
    in r; ψ is still periodic), handled the same way the sphere already handles its own
    non-wrapping θ direction (``0 <= ni < n`` boundary check, no explicit domain_radius reference
    needed since it's baked into r_axis's range). (2) since the grid is source-*centered* by
    construction (``_geodesic_polar_ambient``'s ambient(r,ψ) is exactly geodesic distance r from
    the source), the innermost ring (row 0) is always the seed — no need for
    ``_fast_marching_sphere``'s explicit geo-distance search to find where the source falls on a
    fixed grid.
    """
    n = resolution
    d_r = domain_radius / n
    d_psi = 2.0 * np.pi / n
    r_axis = (np.arange(n) + 0.5) * d_r
    slowness = 1.0 / speed
    time = np.full((n, n), np.inf)
    accepted = np.zeros((n, n), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    for j in range(n):  # innermost ring: every cell here is at r_axis[0], equidistant from the source
        time[0, j] = float(r_axis[0])
        heapq.heappush(heap, (float(r_axis[0]), 0, j))

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
            along_r = min(
                time[ni - 1, nj] if 0 <= ni - 1 < n and accepted[ni - 1, nj] else np.inf,
                time[ni + 1, nj] if 0 <= ni + 1 < n and accepted[ni + 1, nj] else np.inf,
            )
            along_psi = min(
                time[ni, (nj - 1) % n] if accepted[ni, (nj - 1) % n] else np.inf,
                time[ni, (nj + 1) % n] if accepted[ni, (nj + 1) % n] else np.inf,
            )
            d_lat = max(np.sinh(r_axis[ni]), 1e-6) * d_psi
            candidate = solve_quad(along_r, 1.0 / d_r**2, along_psi, 1.0 / d_lat**2, slowness[ni, nj])
            if candidate < time[ni, nj]:
                time[ni, nj] = candidate
                heapq.heappush(heap, (float(candidate), int(ni), int(nj)))
    return time
