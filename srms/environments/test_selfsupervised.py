"""Assert that no training path can see ground truth.

Every method in this repo is self-supervised: a solver may read the *scene* (``slowness``, ``sdf``,
obstacle geometry), the *analytic* base geodesic, and its own collocation samples — but never the
fast-marching field, which exists only to score the result afterwards.

That is easy to state and easy to violate by accident, so it is checked three ways rather than
asserted:

1. **Static.** No module under ``srms/methods`` may mention ``ground_truth`` at all.
2. **Interface.** Each strategy's ``solve`` is walked for every ``env.<attr>`` it touches, and the
   union must lie inside an allow-list of self-supervised quantities.
3. **Dynamic.** ``env.ground_truth`` is monkeypatched to raise, then a short solve is run on every
   strategy x manifold pair. If any of them reaches for it, the run dies.

Run: ``python -m srms.environments.test_selfsupervised`` (exit code 1 on any violation).

The third check is the one that would have caught the real incident this guards against: a solver
being handed ground truth through a callback rather than referencing it by name.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import jax

# Scene definition, closed-form geometry, or sampling — never anything a solver produced.
ALLOWED_ENV_ATTRS = {
    # scene / obstacles — known to the planner by construction
    "slowness",
    "slowness_np",
    "sdf",
    "sdf_np",
    "obstacles",
    "start",
    # analytic geometry — closed form, no fast marching involved
    "geodesic",
    "metric_inv",
    "log_map",
    "log_map_ambient",
    "exp_map",
    "jac_factor",
    "wrap_point",
    "wrap_point_np",
    "displacement_np",
    "boundary_ring_np",
    # sampling / bookkeeping
    "sample_domain",
    "dim",
    "tangent_dim",
    "domain",
}
FORBIDDEN = "ground_truth"


def check_static(root: pathlib.Path) -> list[str]:
    """1. No module under srms/methods may so much as name ``ground_truth``."""
    bad = []
    for path in sorted((root / "srms" / "methods").rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if FORBIDDEN in line and not line.lstrip().startswith("#"):
                bad.append(f"{path.relative_to(root)}:{lineno}: mentions {FORBIDDEN!r}")
    return bad


def check_interface(root: pathlib.Path) -> list[str]:
    """2. Every ``env.<attr>`` reached by a strategy module must be on the allow-list."""
    bad = []
    for path in sorted((root / "srms" / "methods" / "strategies").glob("*.py")):
        if path.name.startswith("_") or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text())
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "env"
        }
        for attr in sorted(used - ALLOWED_ENV_ATTRS):
            bad.append(f"{path.relative_to(root)}: reads env.{attr}, which is not allow-listed")
    return bad


def check_dynamic(strategies: dict, manifolds: list[tuple[str, dict]]) -> list[str]:
    """3. Run each strategy briefly with ``env.ground_truth`` booby-trapped to raise."""
    from srms.environments import ENVIRONMENTS
    from srms.methods.backends import BACKENDS
    from srms.run import Config

    bad = []
    for env_name, kwargs in manifolds:
        for method, strategy in strategies.items():
            cfg = Config(
                environment=env_name,
                method=method,
                backend="srm",
                steps=6,
                num_collocation=64,
                densify=False,
                num_splats=16,
                tau_min=0.01,
            )
            env = ENVIRONMENTS[env_name](**kwargs)

            def trap(*args, _m=method, _e=env_name, **kwargs):
                raise AssertionError(f"{_m} on {_e} called env.ground_truth during solve")

            env.ground_truth = trap  # type: ignore[method-assign]
            try:
                strategy.solve(env, cfg, BACKENDS["srm"])
            except AssertionError as exc:
                bad.append(str(exc))
            except (ValueError, TypeError, NotImplementedError, KeyError) as exc:
                # a strategy may legitimately not support a manifold; only a GT touch is a violation
                print(f"    ({method} on {env_name}: skipped — {type(exc).__name__}: {str(exc)[:70]})")
    return bad


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    from srms.methods.strategies import eikonal, ntfields, pntfields

    strategies = {"eikonal": eikonal, "ntfields": ntfields, "pntfields": pntfields}
    manifolds = [
        ("torus", {"dim": 2}),
        ("pendulum", {}),
        ("sphere", {"n": 2}),
        ("hyperbolic", {"dim": 2}),
        ("so3", {}),
    ]

    print("1. static: no 'ground_truth' anywhere under srms/methods")
    static = check_static(root)
    print("   " + ("OK" if not static else "\n   ".join(static)))

    print("2. interface: every env.<attr> a strategy reads is self-supervised")
    interface = check_interface(root)
    print("   " + ("OK" if not interface else "\n   ".join(interface)))

    print("3. dynamic: env.ground_truth booby-trapped during a short solve of each strategy x manifold")
    dynamic = check_dynamic(strategies, manifolds)
    print("   " + ("OK" if not dynamic else "\n   ".join(dynamic)))

    failures = static + interface + dynamic
    print(f"\n{len(failures)} violation(s). Training is {'self-supervised' if not failures else 'NOT clean'}.")
    return 1 if failures else 0


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    sys.exit(main())
