"""
AUSM+ inviscid flux splitting for 2D compressible flow.
Includes viscous (laminar + turbulent) flux computation.
"""

import numpy as np


# ================================================================== #
#                       AUSM+ split functions                        #
# ================================================================== #
def _Mplus(M):
    return np.where(np.abs(M) >= 1.0,
                    0.5 * (M + np.abs(M)),
                    0.25 * (M + 1.0) ** 2 + 0.125 * (M ** 2 - 1.0) ** 2)

def _Mminus(M):
    return np.where(np.abs(M) >= 1.0,
                    0.5 * (M - np.abs(M)),
                    -0.25 * (M - 1.0) ** 2 - 0.125 * (M ** 2 - 1.0) ** 2)

def _Pplus(M):
    a = 3.0 / 16.0
    return np.where(np.abs(M) >= 1.0,
                    0.5 * (1.0 + np.sign(M)),
                    0.25 * (M + 1.0) ** 2 * (2.0 - M) + a * M * (M ** 2 - 1.0) ** 2)

def _Pminus(M):
    a = 3.0 / 16.0
    return np.where(np.abs(M) >= 1.0,
                    0.5 * (1.0 - np.sign(M)),
                    0.25 * (M - 1.0) ** 2 * (2.0 + M) - a * M * (M ** 2 - 1.0) ** 2)


# ================================================================== #
def inviscid_flux(WL, WR, Sx, Sy, gamma):
    """
    AUSM+ numerical flux through a face.

    Parameters
    ----------
    WL, WR : ndarray (..., 6)   primitive [rho, u, v, p, k, omega]
    Sx, Sy : ndarray (...)      face area-vector components
    gamma  : float

    Returns
    -------
    flux : ndarray (..., 6)  flux of conserved variables
    """
    EPS = 1.0e-30
    rL, uL, vL, pL, kL, oL = (WL[..., i] for i in range(6))
    rR, uR, vR, pR, kR, oR = (WR[..., i] for i in range(6))

    dS = np.sqrt(Sx ** 2 + Sy ** 2) + EPS
    nx, ny = Sx / dS, Sy / dS

    VnL = uL * nx + vL * ny
    VnR = uR * nx + vR * ny

    aL = np.sqrt(np.maximum(gamma * pL / (rL + EPS), EPS))
    aR = np.sqrt(np.maximum(gamma * pR / (rR + EPS), EPS))
    a12 = 0.5 * (aL + aR)

    ML = VnL / (a12 + EPS)
    MR = VnR / (a12 + EPS)

    # mass flux per unit area  ×  dS  →  mass flow rate
    mdot = a12 * (_Mplus(ML) * rL + _Mminus(MR) * rR) * dS
    # interface pressure
    p12 = _Pplus(ML) * pL + _Pminus(MR) * pR

    # total enthalpy per unit mass  H = (rhoE + p) / rho
    EL = pL / ((gamma - 1.0) * rL + EPS) + 0.5 * (uL ** 2 + vL ** 2) + kL
    ER = pR / ((gamma - 1.0) * rR + EPS) + 0.5 * (uR ** 2 + vR ** 2) + kR
    HL = EL + pL / (rL + EPS)
    HR = ER + pR / (rR + EPS)

    # upwind selection
    pos = (mdot >= 0.0).astype(float)
    u_up = pos * uL + (1.0 - pos) * uR
    v_up = pos * vL + (1.0 - pos) * vR
    H_up = pos * HL + (1.0 - pos) * HR
    k_up = pos * kL + (1.0 - pos) * kR
    o_up = pos * oL + (1.0 - pos) * oR

    flux = np.empty_like(WL)
    flux[..., 0] = mdot
    flux[..., 1] = mdot * u_up + p12 * Sx
    flux[..., 2] = mdot * v_up + p12 * Sy
    flux[..., 3] = mdot * H_up
    flux[..., 4] = mdot * k_up
    flux[..., 5] = mdot * o_up
    return flux


# ================================================================== #
def viscous_flux_j(W, mu_lam, mu_t, mesh, gamma, Pr, Prt, sigma_k, sigma_w):
    """
    Viscous flux across interior j-faces (radial direction).
    Uses thin-layer approximation (face-normal gradients only).

    Parameters
    ----------
    W      : (nj, ni, 6)  primitive variables in interior cells
    mu_lam : (nj, ni)     laminar viscosity
    mu_t   : (nj, ni)     turbulent (eddy) viscosity
    mesh   : Mesh object
    Returns
    -------
    Fv     : (nj-1, ni, 6)  viscous flux at interior j-faces (j=1..nj-1)
    """
    nj, ni = W.shape[:2]
    Sx = mesh.Sx_j[1:-1, :]   # interior j-faces  (nj-1, ni)
    Sy = mesh.Sy_j[1:-1, :]
    dS = mesh.dS_j[1:-1, :]
    dn = mesh.dist_j + 1e-30  # (nj-1, ni)

    # left / right cell values
    WL = W[:-1, :, :]   # (nj-1, ni, 6)
    WR = W[1:, :, :]

    # face averages
    mu_f  = 0.5 * (mu_lam[:-1] + mu_lam[1:])
    mut_f = 0.5 * (mu_t[:-1]   + mu_t[1:])
    mu_eff = mu_f + mut_f

    rho_f = 0.5 * (WL[..., 0] + WR[..., 0])
    u_f   = 0.5 * (WL[..., 1] + WR[..., 1])
    v_f   = 0.5 * (WL[..., 2] + WR[..., 2])

    # face-normal unit vector
    nx = Sx / (dS + 1e-30)
    ny = Sy / (dS + 1e-30)

    # face-normal gradients
    du_dn = (WR[..., 1] - WL[..., 1]) / dn
    dv_dn = (WR[..., 2] - WL[..., 2]) / dn
    dk_dn = (WR[..., 4] - WL[..., 4]) / dn
    dw_dn = (WR[..., 5] - WL[..., 5]) / dn

    # approximate velocity gradients (thin-layer)
    dudx = du_dn * nx
    dudy = du_dn * ny
    dvdx = dv_dn * nx
    dvdy = dv_dn * ny
    div  = dudx + dvdy

    txx = mu_eff * (2.0 * dudx - (2.0 / 3.0) * div)
    tyy = mu_eff * (2.0 * dvdy - (2.0 / 3.0) * div)
    txy = mu_eff * (dudy + dvdx)

    # temperature gradient for heat flux
    pL = WL[..., 3]; pR = WR[..., 3]
    rL = WL[..., 0]; rR = WR[..., 0]
    cv = 1.0 / (gamma - 1.0)
    R_gas = 1.0  # non-dimensional; handled via gamma and p=rho*R*T
    TL = pL / (rL * R_gas + 1e-30)
    TR = pR / (rR * R_gas + 1e-30)
    dT_dn = (TR - TL) / dn
    cp = gamma * cv
    kappa = mu_f * cp / Pr + mut_f * cp / Prt

    Fv = np.zeros((nj - 1, ni, 6))
    Fv[..., 1] = (txx * Sx + txy * Sy)
    Fv[..., 2] = (txy * Sx + tyy * Sy)
    Fv[..., 3] = (u_f * (txx * Sx + txy * Sy) +
                   v_f * (txy * Sx + tyy * Sy) +
                   kappa * dT_dn * dS)
    Fv[..., 4] = (mu_f + sigma_k * mut_f) * dk_dn * dS
    Fv[..., 5] = (mu_f + sigma_w * mut_f) * dw_dn * dS
    return Fv
