"""README figure: figure-eight choreography + the symplectic energy story."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbodylab import FIGURE_EIGHT_PERIOD, figure_eight, integrate, kepler_ellipse


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Left: the figure-eight orbit
    system = figure_eight()
    n = 30_000
    result = integrate(system, FIGURE_EIGHT_PERIOD / n, n, record_every=10)
    trajectory = result["positions"]
    for body, color in enumerate(["#1f77b4", "#d62728", "#2ca02c"]):
        ax1.plot(trajectory[:, body, 0], trajectory[:, body, 1], color=color, lw=0.8)
        ax1.plot(*trajectory[-1, body, :2], "o", color=color, ms=9)
    ax1.set_aspect("equal")
    ax1.set_title("Chenciner–Montgomery figure-eight (3 equal masses, 1 period)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Right: energy error, leapfrog vs RK4, ~950 Kepler orbits
    dt, steps = 0.05, 120_000
    sys_lf = kepler_ellipse(0.6)
    e0 = sys_lf.energy()
    lf = integrate(sys_lf, dt, steps, record_every=200)
    sys_rk = kepler_ellipse(0.6)
    rk = integrate(sys_rk, dt, steps, method="rk4", record_every=200)

    ax2.semilogy(lf["time"], np.abs(lf["energy"] - e0) / abs(e0), lw=0.8,
                 label="leapfrog (symplectic): bounded")
    ax2.semilogy(rk["time"], np.abs(rk["energy"] - e0) / abs(e0), lw=0.8,
                 label="RK4: secular drift")
    ax2.set_xlabel("time (~950 orbits of e = 0.6 ellipse)")
    ax2.set_ylabel("|relative energy error|")
    ax2.set_title("Why symplectic integrators own long-term dynamics")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("docs/figures/showcase.png", dpi=150)
    print("wrote docs/figures/showcase.png")


if __name__ == "__main__":
    main()
