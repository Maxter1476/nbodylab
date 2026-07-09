"""nbodylab — symplectic N-body dynamics with analytic and literature anchors."""

from .setups import FIGURE_EIGHT_PERIOD, figure_eight, kepler_ellipse, two_body_circular
from .system import NBodySystem, accelerations, integrate, leapfrog, rk4

__all__ = [
    "FIGURE_EIGHT_PERIOD",
    "NBodySystem",
    "accelerations",
    "figure_eight",
    "integrate",
    "kepler_ellipse",
    "leapfrog",
    "rk4",
    "two_body_circular",
]

__version__ = "0.1.0"
