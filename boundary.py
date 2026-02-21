"""
Boundary conditions for the 2D compressible CFD solver.
  - Farfield: characteristic-based (Riemann invariants)
  - Wall: no-slip, adiabatic, k=0, omega=wall-function
  - Periodic: circumferential wrap-around
"""

import numpy as np


def primitives(Q, gamma):
    """Convert conserved → primitive variables."""
    EPS = 1.0e-30
    rho = Q[..., 0]
    u   = Q[..., 1] / (rho + EPS)
    v   = Q[..., 2] / (rho + EPS)
    k   = Q[..., 4] / (rho + EPS)
    k   = np.maximum(k, 0.0)
    E   = Q[..., 3] / (rho + EPS)
    p   = (gamma - 1.0) * rho * (E - 0.5 * (u ** 2 + v ** 2) - k)
    p   = np.maximum(p, EPS)
    omega = Q[..., 5] / (rho + EPS)
    omega = np.maximum(omega, EPS)
    W = np.empty_like(Q)
    W[..., 0] = rho; W[..., 1] = u; W[..., 2] = v
    W[..., 3] = p;   W[..., 4] = k; W[..., 5] = omega
    return W


def conserved(W, gamma):
    """Convert primitive → conserved variables."""
    rho = W[..., 0]
    u   = W[..., 1]
    v   = W[..., 2]
    p   = W[..., 3]
    k   = W[..., 4]
    omega = W[..., 5]
    E   = p / ((gamma - 1.0) * rho + 1e-30) + 0.5 * (u ** 2 + v ** 2) + k
    Q = np.empty_like(W)
    Q[..., 0] = rho
    Q[..., 1] = rho * u
    Q[..., 2] = rho * v
    Q[..., 3] = rho * E
    Q[..., 4] = rho * k
    Q[..., 5] = rho * omega
    return Q


# ================================================================== #
#                      GHOST-CELL STATES                             #
# ================================================================== #
def wall_ghost(W_interior, mesh, mu_lam_wall, gamma):
    """
    Wall boundary (j = 0 face) — adiabatic no-slip.
    Returns ghost-cell primitive state (1, ni, 6).

    k = 0 at the wall.
    omega = 60 * nu / (beta1 * dy1^2)  (Menter's wall condition)
    """
    W1 = W_interior[0:1, :, :]          # first interior cell  (1, ni, :)
    Wg = W1.copy()

    # no-slip: reflect velocity
    Wg[..., 1] = -W1[..., 1]
    Wg[..., 2] = -W1[..., 2]
    # adiabatic: same pressure
    Wg[..., 3] = W1[..., 3]
    # k = 0
    Wg[..., 4] = -W1[..., 4]   # so face average = 0
    # omega: Menter wall BC
    dy1 = mesh.wall_dist[0:1, :]        # distance of first cell centre to wall
    nu  = mu_lam_wall[0:1, :] / (W1[..., 0] + 1e-30)
    beta1 = 0.075
    omega_wall = 60.0 * nu / (beta1 * dy1 ** 2 + 1e-30)
    Wg[..., 5] = 2.0 * omega_wall - W1[..., 5]
    return Wg


def farfield_ghost(W_interior, W_inf, mesh, gamma):
    """
    Farfield boundary (j = nj face) — characteristic-based.
    Returns ghost-cell primitive state (1, ni, 6).
    """
    nj = W_interior.shape[0]
    Wn = W_interior[-1:, :, :]          # last interior cell  (1, ni, :)

    # face outward normal direction (radially outward)
    xc = mesh.xc[-1:, :]
    yc = mesh.yc[-1:, :]
    rc = np.sqrt(xc ** 2 + yc ** 2) + 1e-30
    nx = xc / rc
    ny = yc / rc

    rho_i = Wn[..., 0];  u_i = Wn[..., 1];  v_i = Wn[..., 2]
    p_i   = Wn[..., 3]
    rho_inf = W_inf[0];  u_inf = W_inf[1];  v_inf = W_inf[2]
    p_inf   = W_inf[3];  k_inf = W_inf[4];  w_inf = W_inf[5]

    a_i   = np.sqrt(gamma * p_i / (rho_i + 1e-30))
    a_inf = np.sqrt(gamma * p_inf / (rho_inf + 1e-30))

    Vn_i   = u_i * nx + v_i * ny
    Vn_inf = u_inf * nx + v_inf * ny

    # Riemann invariants
    R_plus  = Vn_i   + 2.0 * a_i   / (gamma - 1.0)
    R_minus = Vn_inf - 2.0 * a_inf / (gamma - 1.0)

    Vn_b = 0.5 * (R_plus + R_minus)
    a_b  = 0.25 * (gamma - 1.0) * (R_plus - R_minus)
    a_b  = np.maximum(a_b, 1e-30)

    # check inflow / outflow
    outflow = (Vn_b >= 0.0).astype(float)
    inflow  = 1.0 - outflow

    # entropy: use interior for outflow, freestream for inflow
    s_i   = p_i   / (rho_i ** gamma + 1e-30)
    s_inf = p_inf / (rho_inf ** gamma + 1e-30)
    s_b   = outflow * s_i + inflow * s_inf

    rho_b = (a_b ** 2 / (gamma * s_b + 1e-30)) ** (1.0 / (gamma - 1.0))
    p_b   = rho_b * a_b ** 2 / gamma

    # tangential velocity: from interior for outflow, from freestream for inflow
    Vt_i   = (u_i - Vn_i * nx), (v_i - Vn_i * ny)
    Vt_inf = (u_inf - Vn_inf * nx), (v_inf - Vn_inf * ny)

    utx = outflow * Vt_i[0] + inflow * Vt_inf[0]
    uty = outflow * Vt_i[1] + inflow * Vt_inf[1]

    u_b = Vn_b * nx + utx
    v_b = Vn_b * ny + uty

    k_b = outflow * Wn[..., 4] + inflow * k_inf
    w_b = outflow * Wn[..., 5] + inflow * w_inf

    Wg = np.empty_like(Wn)
    # ghost = 2*boundary - interior  so face average = boundary value
    Wg[..., 0] = 2.0 * rho_b - rho_i
    Wg[..., 1] = 2.0 * u_b   - u_i
    Wg[..., 2] = 2.0 * v_b   - v_i
    Wg[..., 3] = 2.0 * p_b   - p_i
    Wg[..., 4] = 2.0 * k_b   - Wn[..., 4]
    Wg[..., 5] = 2.0 * w_b   - Wn[..., 5]

    # clamp to physical values
    Wg[..., 0] = np.maximum(Wg[..., 0], 1e-6)
    Wg[..., 3] = np.maximum(Wg[..., 3], 1e-6)
    Wg[..., 4] = np.maximum(Wg[..., 4], 0.0)
    Wg[..., 5] = np.maximum(Wg[..., 5], 1e-6)
    return Wg
