import numpy as np
import pytest

from nbodylab import (
    FIGURE_EIGHT_PERIOD,
    NBodySystem,
    figure_eight,
    integrate,
    kepler_ellipse,
    leapfrog,
    two_body_circular,
)


def test_circular_orbit_returns_after_period():
    system = two_body_circular()
    omega = np.sqrt(2.0)  # (m1+m2)/a^3 = 2
    period = 2 * np.pi / omega
    dt = period / 2000
    start = system.positions.copy()
    integrate(system, dt, 2000)
    assert np.allclose(system.positions, start, atol=1e-4)


def test_kepler_orbit_conserves_all_invariants():
    system = kepler_ellipse(ecc=0.6)
    e0 = system.energy()
    l0 = system.angular_momentum()
    period = 2 * np.pi  # a = 1, G M = 1
    integrate(system, period / 5000, 5000)
    assert system.energy() == pytest.approx(e0, rel=1e-7)
    assert np.allclose(system.angular_momentum(), l0, rtol=1e-9)
    # back at perihelion after one period
    assert np.linalg.norm(system.positions[1] - [0.4, 0.0, 0.0]) < 2e-3


def test_kepler_aphelion_distance():
    """Half a period after perihelion the particle sits at r = a(1+e)."""
    ecc = 0.5
    system = kepler_ellipse(ecc=ecc)
    integrate(system, np.pi / 4000, 4000)  # half of T = 2 pi
    r_aph = np.linalg.norm(system.positions[1] - system.positions[0])
    assert r_aph == pytest.approx(1.0 + ecc, abs=2e-3)


def test_leapfrog_energy_bounded_rk4_drifts():
    """The defining symplectic property: over many periods leapfrog's energy
    error oscillates within a bound while RK4's drifts monotonically.

    A linear-in-time drift makes the max error over the last quarter of the
    run ~4x the max over the first quarter; a bounded oscillation makes that
    ratio ~1. Both signatures are asserted, plus that the drifter ends up
    worse despite RK4's higher order."""
    ecc = 0.6
    dt = 0.05
    n = 120_000  # ~950 orbits

    sys_lf = kepler_ellipse(ecc)
    e0 = sys_lf.energy()
    hist_lf = integrate(sys_lf, dt, n, record_every=500)["energy"]

    sys_rk = kepler_ellipse(ecc)
    hist_rk = integrate(sys_rk, dt, n, method="rk4", record_every=500)["energy"]

    err_lf = np.abs(hist_lf - e0) / abs(e0)
    err_rk = np.abs(hist_rk - e0) / abs(e0)
    q = len(err_lf) // 4
    assert err_lf[-q:].max() < 2.0 * err_lf[:q].max()  # bounded
    assert err_rk[-q:].max() > 3.0 * err_rk[:q].max()  # secular drift
    assert err_rk[-1] > 2.0 * err_lf[-1]  # and it loses the long game


def test_figure_eight_closes():
    """Three bodies must return to their initial state after one period of
    the Chenciner-Montgomery choreography."""
    system = figure_eight()
    start_pos = system.positions.copy()
    start_vel = system.velocities.copy()
    n = 20_000
    integrate(system, FIGURE_EIGHT_PERIOD / n, n)
    assert np.allclose(system.positions, start_pos, atol=5e-4)
    assert np.allclose(system.velocities, start_vel, atol=5e-4)


def test_figure_eight_is_a_choreography():
    """One third of a period later, body i occupies body (i+1)'s slot."""
    system = figure_eight()
    start = system.positions.copy()
    n = 9_000
    integrate(system, FIGURE_EIGHT_PERIOD / 3.0 / n, n)
    # bodies cycle 0 -> 2 -> 1 -> 0 along the eight
    permuted = start[[2, 0, 1]]
    assert np.allclose(system.positions, permuted, atol=5e-3)


def test_momentum_frame_preserved():
    rng = np.random.default_rng(0)
    system = NBodySystem(
        masses=rng.uniform(0.5, 2.0, 5),
        positions=rng.normal(0, 1, (5, 3)),
        velocities=rng.normal(0, 0.2, (5, 3)),
        softening=0.05,
    )
    # remove net momentum, then it must stay zero
    system.velocities -= system.center_of_mass_velocity()
    integrate(system, 0.002, 5000)
    assert np.allclose(system.center_of_mass_velocity(), 0.0, atol=1e-12)


def test_validation_errors():
    with pytest.raises(ValueError):
        NBodySystem(np.array([1.0]), np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        NBodySystem(np.array([-1.0]), np.zeros((1, 3)), np.zeros((1, 3)))
    system = two_body_circular()
    with pytest.raises(ValueError):
        integrate(system, 0.01, 10, method="euler")


def test_single_step_reversibility():
    """Leapfrog is time-reversible: step forward then backward returns
    exactly (to round-off) to the start."""
    system = figure_eight()
    p0, v0 = system.positions.copy(), system.velocities.copy()
    leapfrog(system, 0.01)
    leapfrog(system, -0.01)
    assert np.allclose(system.positions, p0, atol=1e-14)
    assert np.allclose(system.velocities, v0, atol=1e-14)
