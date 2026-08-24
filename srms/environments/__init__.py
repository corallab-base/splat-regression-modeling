from srms.environments.lorentz_hyperbolic import LorentzHyperbolicEnvironment
from srms.environments.pendulum import PendulumEnvironment
from srms.environments.plane import PlaneEnvironment
from srms.environments.poincare_hyperbolic import PoincareHyperbolicEnvironment
from srms.environments.so3 import SO3Environment
from srms.environments.sphere import SphereEnvironment
from srms.environments.torus import TorusEnvironment

ENVIRONMENTS = {
    "torus": TorusEnvironment,
    "plane": PlaneEnvironment,
    "sphere": SphereEnvironment,
    "so3": SO3Environment,
    "lorentz_hyperbolic": LorentzHyperbolicEnvironment,
    "poincare_hyperbolic": PoincareHyperbolicEnvironment,
    "pendulum": PendulumEnvironment,
}

__all__ = [
    "ENVIRONMENTS",
    "LorentzHyperbolicEnvironment",
    "PendulumEnvironment",
    "PlaneEnvironment",
    "PoincareHyperbolicEnvironment",
    "SO3Environment",
    "SphereEnvironment",
    "TorusEnvironment",
]
