"""Reference systems with known behaviour."""

from __future__ import annotations

import numpy as np

from .system import NBodySystem

__all__ = ["two_body_circular", "figure_eight", "kepler_ellipse"]


def two_body_circular(m1: float = 1.0, m2: float = 1.0, separation: float = 1.0) -> NBodySystem:
    """Two bodies on a circular orbit about their barycenter (G = 1).

    Angular velocity omega^2 = G (m1 + m2) / a^3.
    """
    omega = np.sqrt((m1 + m2) / separation**3)
    r1 = separation * m2 / (m1 + m2)
    r2 = separation * m1 / (m1 + m2)
    return NBodySystem(
        masses=np.array([m1, m2]),
        positions=np.array([[-r1, 0, 0], [r2, 0, 0]]),
        velocities=np.array([[0, -r1 * omega, 0], [0, r2 * omega, 0]]),
    )


def kepler_ellipse(ecc: float = 0.6) -> NBodySystem:
    """Test particle (tiny mass) around a unit mass, starting at perihelion.

    Semi-major axis a = 1: perihelion r = 1 - e with speed sqrt((1+e)/(1-e)).
    """
    r_peri = 1.0 - ecc
    v_peri = np.sqrt((1.0 + ecc) / (1.0 - ecc))
    return NBodySystem(
        masses=np.array([1.0, 1e-12]),
        positions=np.array([[0, 0, 0], [r_peri, 0, 0]]),
        velocities=np.array([[0, 0, 0], [0, v_peri, 0]]),
    )


def figure_eight() -> NBodySystem:
    """The Chenciner-Montgomery figure-eight choreography (G = m = 1).

    Three equal masses chase each other around one figure-eight curve with
    period T ≈ 6.32591398. Initial conditions from Chenciner & Montgomery
    (2000) / Simó's numerics.
    """
    x1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    return NBodySystem(
        masses=np.ones(3),
        positions=np.array([x1, -x1, [0.0, 0.0, 0.0]]),
        velocities=np.array([-v3 / 2.0, -v3 / 2.0, v3]),
    )


FIGURE_EIGHT_PERIOD = 6.32591398
