"""
2D Compressible CFD Solver — Mach 0.7 Flow Over a Circular Cylinder
===================================================================
RANS equations with SST k-ω turbulence model on a structured O-grid.

Usage:
    python main.py
"""

import time
import numpy as np
from mesh import Mesh
from solver import Solver
from turbulence import compute_blending_and_mut
from visualize import (
    plot_mach, plot_pressure_coeff, plot_tke,
    plot_eddy_viscosity_ratio, plot_residual_history,
    plot_surface_cp,
)


def main():
    print("=" * 60)
    print("   2D Compressible RANS Solver  -  SST k-omega")
    print("   Flow over a cylinder  |  Ma_inf = 0.7")
    print("=" * 60)

    # ---- Mesh ----
    mesh = Mesh(
        R=0.5,          # cylinder radius
        r_far=25.0,     # farfield radius (50 diameters away)
        ni=100,         # circumferential cells
        nj=60,          # radial cells
        stretch=1.10,   # geometric stretching near wall
    )
    print(f"\nMesh: {mesh.nj} × {mesh.ni} = {mesh.nj * mesh.ni} cells")
    print(f"  first cell height: {mesh.radii[1] - mesh.radii[0]:.4e}")
    print(f"  farfield radius:   {mesh.r_far}")

    # ---- Solver ----
    sol = Solver(
        mesh,
        gamma=1.4,
        CFL=0.4,
        Ma_inf=0.7,
        p_inf=101325.0,    # Pa
        T_inf=288.15,      # K
        Re=1.0e6,          # based on diameter
        Pr=0.72,
        Prt=0.9,
        max_iter=3000,
        tol=1e-8,
    )

    # ---- Run ----
    t0 = time.time()
    W = sol.iterate()
    elapsed = time.time() - t0
    print(f"\nElapsed time: {elapsed:.1f} s")

    # ---- Post-processing ----
    print("\nGenerating plots …")
    mu_lam = sol._laminar_viscosity(W)
    _, _, mu_t = compute_blending_and_mut(W, mu_lam, mesh)

    plot_mach(W, mesh, sol.gamma, 0.7)
    plot_pressure_coeff(W, mesh, sol.gamma, 0.7)
    plot_tke(W, mesh)
    plot_eddy_viscosity_ratio(W, mu_t, mu_lam, mesh)
    plot_residual_history(sol.residual_history)
    plot_surface_cp(W, mesh, sol.gamma, 0.7)

    print("\nDone.  Check the  results/  folder for output plots.")


if __name__ == "__main__":
    main()
