"""
Post-processing visualisation for the CFD solver.
Generates Mach number, pressure coefficient, turbulent kinetic energy,
and eddy-viscosity ratio contour plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_mach(W, mesh, gamma, Ma_inf):
    """Contour plot of local Mach number."""
    _ensure_dir()
    rho = W[..., 0]; u = W[..., 1]; v = W[..., 2]; p = W[..., 3]
    a = np.sqrt(np.maximum(gamma * p / (rho + 1e-30), 1e-30))
    V = np.sqrt(u ** 2 + v ** 2)
    Ma = V / a

    fig, ax = plt.subplots(figsize=(10, 10))
    levels = np.linspace(0.0, min(Ma.max(), 2.0), 60)
    cf = ax.contourf(mesh.xc, mesh.yc, Ma, levels=levels, cmap="jet", extend="both")
    plt.colorbar(cf, ax=ax, label="Mach number")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mesh.R * np.cos(theta), mesh.R * np.sin(theta), "k-", lw=2)
    ax.set_aspect("equal")
    ax.set_title(f"Mach Number Field  (Ma_inf = {Ma_inf})")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 3)
    fig.savefig(os.path.join(RESULTS_DIR, "mach_contour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved mach_contour.png")


def plot_pressure_coeff(W, mesh, gamma, Ma_inf):
    """Contour plot of pressure coefficient Cp."""
    _ensure_dir()
    p = W[..., 3]
    p_inf = 1.0 / gamma
    q_inf = 0.5 * gamma * p_inf * Ma_inf ** 2
    Cp = (p - p_inf) / (q_inf + 1e-30)

    fig, ax = plt.subplots(figsize=(10, 10))
    levels = np.linspace(-2.0, 2.0, 60)
    cf = ax.contourf(mesh.xc, mesh.yc, Cp, levels=levels, cmap="RdBu_r", extend="both")
    plt.colorbar(cf, ax=ax, label="Cp")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mesh.R * np.cos(theta), mesh.R * np.sin(theta), "k-", lw=2)
    ax.set_aspect("equal")
    ax.set_title(f"Pressure Coefficient  (Ma_inf = {Ma_inf})")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 3)
    fig.savefig(os.path.join(RESULTS_DIR, "cp_contour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved cp_contour.png")


def plot_tke(W, mesh):
    """Contour plot of turbulent kinetic energy k."""
    _ensure_dir()
    k = np.maximum(W[..., 4], 0.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    levels = np.linspace(0, k.max() * 0.8 + 1e-12, 60)
    cf = ax.contourf(mesh.xc, mesh.yc, k, levels=levels, cmap="hot", extend="max")
    plt.colorbar(cf, ax=ax, label="k (turbulent kinetic energy)")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mesh.R * np.cos(theta), mesh.R * np.sin(theta), "k-", lw=2)
    ax.set_aspect("equal")
    ax.set_title("Turbulent Kinetic Energy")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 3)
    fig.savefig(os.path.join(RESULTS_DIR, "tke_contour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved tke_contour.png")


def plot_eddy_viscosity_ratio(W, mu_t, mu_lam, mesh):
    """Contour plot of eddy-viscosity ratio mu_t / mu."""
    _ensure_dir()
    ratio = mu_t / (mu_lam + 1e-30)

    fig, ax = plt.subplots(figsize=(10, 10))
    vmax = min(ratio.max(), 5000.0)
    levels = np.linspace(0, vmax, 60)
    cf = ax.contourf(mesh.xc, mesh.yc, ratio, levels=levels, cmap="viridis", extend="max")
    plt.colorbar(cf, ax=ax, label="μ_t / μ")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mesh.R * np.cos(theta), mesh.R * np.sin(theta), "k-", lw=2)
    ax.set_aspect("equal")
    ax.set_title("Eddy Viscosity Ratio")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 3)
    fig.savefig(os.path.join(RESULTS_DIR, "mut_ratio_contour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved mut_ratio_contour.png")


def plot_residual_history(history):
    """Convergence history."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(history) + 1), history, "b-", lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 Density Residual")
    ax.set_title("Convergence History")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(os.path.join(RESULTS_DIR, "residual_history.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved residual_history.png")


def plot_surface_cp(W, mesh, gamma, Ma_inf):
    """Surface Cp distribution vs. angle."""
    _ensure_dir()
    p_wall = W[0, :, 3]
    p_inf = 1.0 / gamma
    q_inf = 0.5 * gamma * p_inf * Ma_inf ** 2
    Cp_wall = (p_wall - p_inf) / (q_inf + 1e-30)
    theta = mesh.theta * 180.0 / np.pi

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta, Cp_wall, "b-o", ms=2, lw=1.2)
    ax.set_xlabel("theta (degrees)")
    ax.set_ylabel("Cp")
    ax.set_title(f"Surface Pressure Coefficient  (Ma_inf = {Ma_inf})")
    ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.5)
    fig.savefig(os.path.join(RESULTS_DIR, "surface_cp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved surface_cp.png")
