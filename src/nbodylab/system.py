"""N-body gravitational systems and integrators.

Two integrators with deliberately different structure:

- **Leapfrog (kick-drift-kick)** — symplectic and time-reversible. Its energy
  error stays *bounded* forever (it oscillates), which is why it is the
  workhorse of long-term orbital dynamics.
- **Classic RK4** — higher order per step but not symplectic: its energy
  error *drifts secularly*. The test suite asserts this qualitative
  difference, not just raw accuracy.

Units are G = 1 unless a system says otherwise; every function is vectorized
over bodies (positions shape (n, 3)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NBodySystem", "accelerations", "leapfrog", "rk4", "integrate"]


@dataclass
class NBodySystem:
    """Point masses under Newtonian gravity."""

    masses: np.ndarray
    positions: np.ndarray  # (n, 3)
    velocities: np.ndarray  # (n, 3)
    g_constant: float = 1.0
    softening: float = 0.0

    def __post_init__(self) -> None:
        self.masses = np.asarray(self.masses, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float).copy()
        self.velocities = np.asarray(self.velocities, dtype=float).copy()
        n = len(self.masses)
        if self.positions.shape != (n, 3) or self.velocities.shape != (n, 3):
            raise ValueError("positions and velocities must have shape (n, 3)")
        if np.any(self.masses <= 0):
            raise ValueError("masses must be positive")

    def energy(self) -> float:
        """Total energy: kinetic + pairwise potential."""
        kinetic = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1))
        delta = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.sqrt(np.sum(delta**2, axis=-1) + self.softening**2)
        np.fill_diagonal(dist, np.inf)
        inv = 1.0 / dist
        potential = -0.5 * self.g_constant * np.sum(
            self.masses[:, None] * self.masses[None, :] * inv
        )
        return float(kinetic + potential)

    def angular_momentum(self) -> np.ndarray:
        """Total angular momentum vector."""
        return np.sum(self.masses[:, None] * np.cross(self.positions, self.velocities), axis=0)

    def center_of_mass_velocity(self) -> np.ndarray:
        return self.masses @ self.velocities / self.masses.sum()


def accelerations(
    positions: np.ndarray, masses: np.ndarray, g_constant: float = 1.0, softening: float = 0.0
) -> np.ndarray:
    """Pairwise gravitational accelerations, O(n^2) vectorized."""
    delta = positions[None, :, :] - positions[:, None, :]  # r_j - r_i
    dist2 = np.sum(delta**2, axis=-1) + softening**2
    np.fill_diagonal(dist2, np.inf)
    inv_d3 = dist2 ** (-1.5)
    return g_constant * np.sum(masses[None, :, None] * delta * inv_d3[:, :, None], axis=1)


def leapfrog(system: NBodySystem, dt: float) -> None:
    """One kick-drift-kick step, in place. Symplectic, second order."""
    acc = accelerations(system.positions, system.masses, system.g_constant, system.softening)
    system.velocities += 0.5 * dt * acc
    system.positions += dt * system.velocities
    acc = accelerations(system.positions, system.masses, system.g_constant, system.softening)
    system.velocities += 0.5 * dt * acc


def rk4(system: NBodySystem, dt: float) -> None:
    """One classic Runge-Kutta 4 step, in place. Fourth order, not symplectic."""

    def deriv(pos, vel):
        return vel, accelerations(pos, system.masses, system.g_constant, system.softening)

    p0, v0 = system.positions, system.velocities
    k1p, k1v = deriv(p0, v0)
    k2p, k2v = deriv(p0 + 0.5 * dt * k1p, v0 + 0.5 * dt * k1v)
    k3p, k3v = deriv(p0 + 0.5 * dt * k2p, v0 + 0.5 * dt * k2v)
    k4p, k4v = deriv(p0 + dt * k3p, v0 + dt * k3v)
    system.positions = p0 + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
    system.velocities = v0 + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)


def integrate(
    system: NBodySystem,
    dt: float,
    n_steps: int,
    *,
    method: str = "leapfrog",
    record_every: int = 0,
) -> dict[str, np.ndarray]:
    """Advance the system; optionally record trajectory and energy history."""
    steppers = {"leapfrog": leapfrog, "rk4": rk4}
    if method not in steppers:
        raise ValueError(f"unknown method {method!r}; have {sorted(steppers)}")
    step = steppers[method]
    times, positions, energies = [], [], []
    for i in range(n_steps):
        step(system, dt)
        if record_every and i % record_every == 0:
            times.append((i + 1) * dt)
            positions.append(system.positions.copy())
            energies.append(system.energy())
    return {
        "time": np.array(times),
        "positions": np.array(positions),
        "energy": np.array(energies),
    }
