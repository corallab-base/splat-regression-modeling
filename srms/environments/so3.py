"""SO(3) environment — the rotation group, as a Lie group with its bi-invariant metric.

The first *Lie group* in this set, and the first intrinsically 3-dimensional manifold. It is also the
sharpest case for the representation argument: **the splat needs no embedding at all.** A splat lives
in the Lie algebra ``so(3) ≅ R³``, so ``tangent_dim = 3`` — the manifold's true dimension — and ``A``
is a 3×3 matrix. By contrast, a network that must *consume* a rotation as a vector of numbers has no
such option: Zhou et al. (CVPR 2019) prove no continuous representation of SO(3) exists in four or
fewer dimensions, so quaternion (4-D) and Euler (3-D) inputs are provably discontinuous and a ≥5-D
(in practice 6-D) encoding is mandatory. That is a theorem, not an observation, and it is exactly the
"per-manifold embedding" the splat formulation avoids.

**Storage vs. dimension.** Points are stored as unit quaternions (``dim = 4``), which makes SO(3)
structurally identical to ``SphereEnvironment``: an embedded manifold whose retraction is a
normalisation, so ``srm.post_step``'s existing machinery covers it with no new code. Storage is
bookkeeping; the splat only ever sees the 3-D tangent vector ``log_map`` returns.

**The double cover is the one thing that differs from S³.** ``q`` and ``−q`` are the *same* rotation
(SO(3) = S³/±1), which shows up in exactly two places: ``log_map`` negates the relative quaternion
when its scalar part is negative, and ``geodesic`` takes ``2·arccos|⟨q₁,q₂⟩|`` — the absolute value
*is* the quotient. Drop either and ``test_manifolds.py``'s isometry check fails loudly.

**Geometry.** SO(3) with the bi-invariant metric is locally isometric to a round S³ of radius 2
(``d_SO(3) = 2·d_S³``), hence constant curvature ¼:

- ``log_map(μ, x)``   axis-angle vector of ``μ⁻¹x``; ‖·‖ = rotation angle = geodesic distance
- ``jac_factor``      ``((θ/2) / sin(θ/2))²`` — the same Jacobi-field family as the sphere
  (``(θ/sinθ)^(n−1)``) and hyperbolic (``(r/sinh r)^(d−1)``), at K=¼ and d=3
- ``metric_inv``      ``¼(I − qqᵀ)`` — the sphere's tangent projector, scaled by the double cover
- ``sample_domain``   normalised Gaussian 4-vectors, which is **uniform w.r.t. Haar/Riemannian volume
  for free** — no sampling correction of the kind H² needed

Two conveniences worth noting against the other manifolds. ``jac_factor`` is **bounded on the whole
group** (it rises only to ``(π/2)² ≈ 2.47`` at the cut locus θ=π, where the sphere's ``θ/sinθ``
diverges), and SO(3) is **compact**, so unlike H² there is no truncation choice to declare.

**Frames: SO(3) has no seam.** A Lie group is parallelizable — left-translating a fixed basis of the
Lie algebra gives a globally smooth frame. So an anisotropic ``A`` means the same thing everywhere on
SO(3), unlike S², where the hairy-ball theorem forbids any global continuous frame and
``sphere.py``'s Householder branch jumps by 33.7° at ``|μ₀| = 0.9``. The Lie structure buys back
precisely what the topology of S² takes away.

Ground truth (a 3-D fast march) is not implemented yet; ``grid``/``ground_truth`` raise. Training is
mesh-free and works now.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from srms.environments import marching

Obstacle = tuple[float, ...]  # (*centre quaternion[4], angular radius in radians)

_OBSTACLE_SEED_OFFSET = 5


# ---- quaternion algebra (scalar-first: q = [w, x, y, z]) ---------------------------------------


def quat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product a ⊗ b."""
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return jnp.concatenate([jnp.array([aw * bw - jnp.dot(av, bv)]), aw * bv + bw * av + jnp.cross(av, bv)])


def quat_conj(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate (= inverse for unit quaternions)."""
    return jnp.concatenate([q[:1], -q[1:]])


def log_map(mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Axis-angle vector of ``μ⁻¹x``; Euclidean norm = rotation angle = geodesic distance.

    Uses ``arctan2`` rather than ``arccos`` deliberately: ``arccos``'s derivative diverges as θ→0,
    which is what forced ``sphere.py`` to clip ``cos θ`` at ``1−1e-3`` and thereby floor its geodesic
    distance at 0.045 rad. ``arctan2(‖v‖, w)`` has bounded derivatives there, so SO(3) does not
    inherit that pathology.
    """
    rel = quat_mul(quat_conj(mu), x)
    rel = jnp.where(rel[0] < 0, -rel, rel)  # q and -q are the same rotation: pick the short way round
    norm_v = jnp.linalg.norm(rel[1:])
    theta = 2.0 * jnp.arctan2(norm_v, jnp.abs(rel[0]))
    return theta * rel[1:] / jnp.maximum(norm_v, 1e-12)


def exp_map(mu: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """``μ ⊗ exp(ω/2)`` — inverse of ``log_map``. ``jnp.sinc`` keeps the θ→0 limit smooth."""
    theta = jnp.linalg.norm(omega)
    half = 0.5 * theta
    # sin(half)/theta = sinc(half/pi)/2, smooth at 0 (-> 1/2), so no branch is needed.
    delta = jnp.concatenate([jnp.array([jnp.cos(half)]), 0.5 * jnp.sinc(half / jnp.pi) * omega])
    return quat_mul(mu, delta)


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy Hamilton product, broadcasting over leading axes."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    v = aw * bv + bw * av + np.cross(av, bv)
    return np.concatenate([w, v], axis=-1)


def _log_map_np(mu: np.ndarray, x: np.ndarray) -> np.ndarray:
    """NumPy counterpart of ``log_map``; broadcasts."""
    mu = np.atleast_2d(mu)
    conj = np.concatenate([mu[..., :1], -mu[..., 1:]], axis=-1)
    rel = _quat_mul_np(conj, np.atleast_2d(x))
    rel = np.where(rel[..., :1] < 0, -rel, rel)
    norm_v = np.linalg.norm(rel[..., 1:], axis=-1, keepdims=True)
    theta = 2.0 * np.arctan2(norm_v, np.abs(rel[..., :1]))
    return np.squeeze(theta * rel[..., 1:] / np.maximum(norm_v, 1e-12))


def _distance_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between two unit quaternions; the ``abs`` is the SO(3) double cover."""
    dot = np.abs(np.sum(np.atleast_2d(a) * np.atleast_2d(b), axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


@dataclasses.dataclass
class SO3Environment:
    """SO(3) with a smooth slowness field rising around geodesic-ball obstacles."""

    start: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)  # identity rotation
    num_obstacles: int = 3
    obstacle_radius: tuple[float, float] = (0.35, 0.6)
    slowness_max: float = 10.0
    slow_width: float = 0.15
    seed: int = 1

    def __post_init__(self) -> None:
        if len(self.start) != 4:
            raise ValueError(f"start must be a unit quaternion (4 numbers), got {len(self.start)}")
        if abs(float(np.linalg.norm(self.start)) - 1.0) > 1e-3:
            raise ValueError(f"start must be a unit quaternion, got norm {np.linalg.norm(self.start):.4f}")
        self.dim = 4  # storage: unit quaternion
        self.tangent_dim = 3  # the manifold's true dimension — this is what the splat sees
        self.domain: tuple[float, float] = (-1.0, 1.0)  # storage range; SO(3) has no chart box
        # Math-mode subscripts, not raw g₁/g₂ glyphs — see torus.py's TorusEnvironment for why.
        self.axis_labels: tuple[str, str] = (r"Gibbs $g_1$", r"Gibbs $g_2$")
        self.render_extent: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
        self.has_dense_gt = True  # 3-D polar fast march is tractable (~res^3 cells)
        self.obstacles: tuple[Obstacle, ...] = self._sample_obstacles()

    @property
    def title(self) -> str:
        return f"SO(3) — time-to-go ({self.num_obstacles} obstacles)"

    @property
    def gt_label(self) -> str:
        return "ground truth — 3-D fast marching"

    def _sample_obstacles(self) -> tuple[Obstacle, ...]:
        """Reproducible geodesic-ball obstacles in SO(3), clear of the source."""
        rng = np.random.default_rng(self.seed + _OBSTACLE_SEED_OFFSET)
        start = np.asarray(self.start, dtype=float)
        obstacles: list[Obstacle] = []
        while len(obstacles) < self.num_obstacles:
            z = rng.standard_normal(4)
            centre = z / np.linalg.norm(z)
            radius = float(rng.uniform(*self.obstacle_radius))
            if float(np.ravel(_distance_np(centre, start))[0]) > radius + 0.3:
                obstacles.append((*centre.tolist(), radius))
        return tuple(obstacles)

    # ---- manifold geometry -----------------------------------------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Lie-algebra coordinates (size 3) of x at μ; ‖·‖ = rotation angle."""
        return log_map(mu, x)

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """SO(3)'s tangent vectors are Lie-algebra elements, already the intrinsic 3-vector."""
        return log_map(mu, x)

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exp_μ(v) — inverse of log_map (see base.Environment)."""
        return exp_map(mu, v)

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """|det ∂Log_μ/∂x| = ((θ/2)/sin(θ/2))², the K=¼ member of the Jacobi-field family.

        Bounded on the whole group: 1 at θ=0 rising only to (π/2)² ≈ 2.47 at the cut locus θ=π —
        unlike S², where ``θ/sinθ`` diverges there.
        """
        theta = self.geodesic(x[None, :], mu)[0]
        return (1.0 / jnp.sinc(0.5 * theta / jnp.pi)) ** 2

    def splat_precompute(self, mu: jnp.ndarray):
        """Per-splat geometry: the conjugate quaternion, hoisted out of the per-point loop."""
        return quat_conj(mu)

    def log_and_jac(self, pre, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(log_map, jac_factor) from one quaternion product.

        ``jac_factor`` used to call ``geodesic``, recomputing the angle ``log_map`` already had.
        Reading it from the same ``arctan2`` is also better conditioned near θ=0 than ``arccos``.
        """
        rel = quat_mul(pre, x)
        rel = jnp.where(rel[0] < 0, -rel, rel)
        norm_v = jnp.linalg.norm(rel[1:])
        theta = 2.0 * jnp.arctan2(norm_v, jnp.abs(rel[0]))
        jac = (1.0 / jnp.sinc(0.5 * theta / jnp.pi)) ** 2
        return theta * rel[1:] / jnp.maximum(norm_v, 1e-12), jac

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Retract onto the unit sphere in R⁴. Sign is deliberately *not* canonicalised: q and −q
        denote the same rotation and every consumer handles the double cover, whereas forcing w ≥ 0
        would introduce a discontinuity the optimizer would have to cross."""
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Lie-algebra vector from a to b, ‖·‖ = rotation angle; broadcasts."""
        return _log_map_np(np.asarray(a, dtype=float), np.asarray(b, dtype=float))

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` rotations at exact geodesic distance eps from the source."""
        omega = rng.standard_normal((n, 3))
        omega /= np.linalg.norm(omega, axis=-1, keepdims=True) + 1e-12
        start = jnp.asarray(self.start, dtype=jnp.float32)
        return np.asarray(jax.vmap(lambda w: exp_map(start, w))(jnp.asarray(eps * omega, jnp.float32)), dtype=float)

    def metric_inv(self, q: jnp.ndarray) -> jnp.ndarray:
        """g^{ij} = ¼(I − qqᵀ): the S³ tangent projector, scaled because d_SO(3) = 2·d_S³."""
        return 0.25 * (jnp.eye(4) - jnp.outer(q, q))

    def geodesic(self, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Rotation angle to the source. The ``abs`` implements the SO(3) = S³/±1 quotient."""
        dot = jnp.abs(jnp.sum(x * start, axis=-1))
        return 2.0 * jnp.arccos(jnp.clip(dot, 0.0, 1.0 - 1e-7))

    # ---- obstacle / slowness field -----------------------------------------

    def sdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Signed geodesic distance to the union of obstacle balls. SO(3) is compact — no truncation."""
        per = [self.geodesic(q, jnp.asarray(obs[:-1])) - obs[-1] for obs in self.obstacles]
        return jnp.min(jnp.stack(per, axis=0), axis=0)

    def slowness(self, q: jnp.ndarray) -> jnp.ndarray:
        """Smooth slowness: ~1 in free space, rising to slowness_max inside obstacles."""
        return 1.0 + (self.slowness_max - 1.0) * jax.nn.sigmoid(-self.sdf(q) / self.slow_width)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return np.min([_distance_np(np.array(o[:-1]), points) - o[-1] for o in self.obstacles], axis=0)

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        return 1.0 + (self.slowness_max - 1.0) / (1.0 + np.exp(self.sdf_np(points) / self.slow_width))

    # ---- sampling / ground truth --------------------------------------------

    @property
    def volume(self) -> float:
        """Riemannian volume of SO(3) with the bi-invariant metric = 8π².

        Not S³'s 2π²: distances here are *twice* the unit-S³ ones (d_SO(3) = 2·d_S³), so 3-volumes
        scale by 2³, and the double cover halves it — 2π²·8/2 = 8π². Directly: the geodesic-polar
        element 4sin²(r/2) integrated over r ∈ [0,π] gives 2π, times the S² area 4π.
        """
        return float(8.0 * np.pi**2)

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Haar-uniform rotations: normalised Gaussian 4-vectors.

        This is uniform w.r.t. the Riemannian volume with no correction — contrast H², where the
        chart and the volume measure disagree and sampling had to be reweighted explicitly.
        """
        z = rng.standard_normal((n, 4))
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int, int]]:
        """Geodesic-polar grid over the whole group, returned as unit quaternions.

        Coordinates are (rotation angle r about a direction u) — normal coordinates at the identity —
        so a grid cell is the rotation ``[cos(r/2), sin(r/2)·u]``. The chart is anchored at the
        identity regardless of ``start``, exactly as ``SphereEnvironment``'s lat-long grid is anchored
        at its pole; the source enters only through the seeding, which keeps the calibration honest.

        The metric ``ds² = dr² + 4sin²(r/2)·dΩ²`` makes this an orthogonal anisotropic grid, which is
        what ``marching.fast_march_3d`` consumes. Radial resolution is uniform over the full range
        r ∈ [0, π], so unlike a Gibbs chart nothing is discarded near the cut locus.
        """
        polar, shape, _ = marching.polar_grid_3d(resolution, float(np.pi), lambda r: 2.0 * np.sin(0.5 * r))
        r, u = marching.polar_to_unit3(polar)
        quats = np.concatenate([np.cos(0.5 * r)[:, None], np.sin(0.5 * r)[:, None] * u], axis=-1)
        return jnp.asarray(quats, dtype=jnp.float32), shape

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Fast marching of ‖∇T‖ = slowness over all of SO(3); returns a raveled array.

        Azimuth wraps; the radial axis and the direction-sphere colatitude do not (both are
        cell-centred, so no sample lands on a coordinate singularity). SO(3) is compact, so nothing is
        masked — every cell is in the domain.
        """
        start = self.start if start is None else start
        _, shape, spacing = marching.polar_grid_3d(resolution, float(np.pi), lambda r: 2.0 * np.sin(0.5 * r))
        quats, _ = self.grid(resolution)
        seed = np.asarray(_distance_np(np.asarray(start, dtype=float), np.asarray(quats, dtype=float)))
        return marching.fast_march_3d(
            spacing=spacing,
            slowness=np.asarray(self.slowness(quats)).reshape(shape),
            blocked=np.zeros(shape, bool),
            seed_time=np.where(seed <= 2.0 * np.pi / resolution, seed, np.inf).reshape(shape),
            periodic=(False, False, True),
            wrap_fn=marching.polar_topology(shape),
        ).ravel()

    def render_marker_deg(self) -> tuple[float, float]:
        """Gibbs-chart coordinates of the source (for eventual slice plots)."""
        q = np.asarray(self.start, dtype=float)
        q = q if q[0] >= 0 else -q
        g = q[1:] / max(q[0], 1e-9)
        return float(g[0]), float(g[1])
