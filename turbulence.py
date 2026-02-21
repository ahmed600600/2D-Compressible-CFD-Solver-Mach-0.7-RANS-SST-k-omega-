"""
Menter SST k-omega turbulence model.
Computes eddy viscosity, source terms, and blending functions.
"""

import numpy as np


# ================================================================== #
#                     SST  MODEL  CONSTANTS                          #
# ================================================================== #
# Inner layer (k-omega)
SIGMA_K1  = 0.85
SIGMA_W1  = 0.5
BETA_1    = 0.075
ALPHA_1   = 5.0 / 9.0

# Outer layer (k-epsilon transformed)
SIGMA_K2  = 1.0
SIGMA_W2  = 0.856
BETA_2    = 0.0828
ALPHA_2   = 0.44

# Shared
BETA_STAR = 0.09
A1        = 0.31
KAPPA     = 0.41


# ================================================================== #
def blend(phi1, phi2, F1):
    """Blending of inner/outer constants."""
    return F1 * phi1 + (1.0 - F1) * phi2


def compute_F1(k, omega, rho, mu, d, dkdx, dkdy, dwdx, dwdy):
    """First blending function F1."""
    EPS = 1.0e-30
    nu = mu / (rho + EPS)
    d2 = d ** 2 + EPS

    CDkw = np.maximum(
        2.0 * rho * SIGMA_W2 / (omega + EPS) *
        (dkdx * dwdx + dkdy * dwdy),
        1.0e-20
    )
    arg1_a = np.sqrt(np.maximum(k, 0.0)) / (BETA_STAR * omega * d + EPS)
    arg1_b = 500.0 * nu / (d2 * omega + EPS)
    arg1_c = 4.0 * rho * SIGMA_W2 * k / (CDkw * d2 + EPS)
    arg1 = np.minimum(np.maximum(arg1_a, arg1_b), arg1_c)
    return np.tanh(arg1 ** 4)


def compute_F2(k, omega, rho, mu, d):
    """Second blending function F2 (for eddy-viscosity limiter)."""
    EPS = 1.0e-30
    nu = mu / (rho + EPS)
    d2 = d ** 2 + EPS

    arg2_a = 2.0 * np.sqrt(np.maximum(k, 0.0)) / (BETA_STAR * omega * d + EPS)
    arg2_b = 500.0 * nu / (d2 * omega + EPS)
    arg2 = np.maximum(arg2_a, arg2_b)
    return np.tanh(arg2 ** 2)


def eddy_viscosity(rho, k, omega, F2, strain_mag):
    """
    Eddy viscosity with Bradshaw limiter.
    mu_t = rho * a1 * k / max(a1*omega, Omega*F2)
    """
    EPS = 1.0e-30
    mu_t = rho * A1 * np.maximum(k, 0.0) / (
        np.maximum(A1 * omega, strain_mag * F2) + EPS
    )
    return mu_t


def strain_magnitude(dudx, dudy, dvdx, dvdy):
    """Magnitude of the mean strain-rate tensor  sqrt(2*Sij*Sij)."""
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)
    return np.sqrt(2.0 * (S11 ** 2 + S22 ** 2 + 2.0 * S12 ** 2) + 1.0e-30)


# ================================================================== #
def source_terms(W, mu_lam, mu_t, F1, mesh, gamma):
    """
    Compute k and omega source terms at cell centres.

    Parameters
    ----------
    W      : (nj, ni, 6) primitive variables
    mu_lam : (nj, ni)
    mu_t   : (nj, ni)
    F1     : (nj, ni)
    mesh   : Mesh
    gamma  : float

    Returns
    -------
    Sk, Sw : (nj, ni) source terms for k and omega equations
    """
    EPS = 1.0e-30
    nj, ni = W.shape[:2]
    rho = W[..., 0]
    u   = W[..., 1]
    v   = W[..., 2]
    k   = np.maximum(W[..., 4], 0.0)
    omega = np.maximum(W[..., 5], EPS)

    # ---- velocity gradients (Green-Gauss at cell centres) ----
    # Approximate using central differences with periodic i
    # j-direction: forward/backward at boundaries, central interior
    dudx = np.zeros((nj, ni))
    dudy = np.zeros((nj, ni))
    dvdx = np.zeros((nj, ni))
    dvdy = np.zeros((nj, ni))

    # Use Green-Gauss:  grad(phi) = (1/A) * sum_faces phi_face * S_face
    area = mesh.area

    # j-face contributions
    Sx_j = mesh.Sx_j   # (nj+1, ni)
    Sy_j = mesh.Sy_j

    for jf in range(nj + 1):
        if jf == 0:
            u_f = u[0, :]     # wall: u=0 but we use cell value as approx
            v_f = v[0, :]
        elif jf == nj:
            u_f = u[-1, :]
            v_f = v[-1, :]
        else:
            u_f = 0.5 * (u[jf - 1, :] + u[jf, :])
            v_f = 0.5 * (v[jf - 1, :] + v[jf, :])

        if jf < nj:  # top face of cell jf
            dudx[jf, :] += u_f * Sx_j[jf + 1, :] if jf + 1 <= nj else 0
            dudy[jf, :] += u_f * Sy_j[jf + 1, :] if jf + 1 <= nj else 0

    # Simplified: use finite differences for velocity gradients
    # Circumferential (i) direction
    u_ip1 = np.roll(u, -1, axis=1)
    u_im1 = np.roll(u,  1, axis=1)
    v_ip1 = np.roll(v, -1, axis=1)
    v_im1 = np.roll(v,  1, axis=1)

    xc_ip1 = np.roll(mesh.xc, -1, axis=1)
    xc_im1 = np.roll(mesh.xc,  1, axis=1)
    yc_ip1 = np.roll(mesh.yc, -1, axis=1)
    yc_im1 = np.roll(mesh.yc,  1, axis=1)

    dx_i = xc_ip1 - xc_im1
    dy_i = yc_ip1 - yc_im1

    # Radial (j) direction
    dudx_r = np.zeros_like(u)
    dudy_r = np.zeros_like(u)
    dvdx_r = np.zeros_like(v)
    dvdy_r = np.zeros_like(v)

    # central differences in j
    dudx_r[1:-1] = (u[2:] - u[:-2]) / 2.0
    dudy_r[1:-1] = (u[2:] - u[:-2]) / 2.0
    dvdx_r[1:-1] = (v[2:] - v[:-2]) / 2.0
    dvdy_r[1:-1] = (v[2:] - v[:-2]) / 2.0

    # forward/backward at boundaries
    dudx_r[0]  = u[1] - u[0]
    dudx_r[-1] = u[-1] - u[-2]
    dvdx_r[0]  = v[1] - v[0]
    dvdx_r[-1] = v[-1] - v[-2]

    dx_j = np.diff(mesh.xc, axis=0)
    dy_j = np.diff(mesh.yc, axis=0)

    # Use chain rule on curvilinear coordinates
    # Approximate: dudx ≈ du/dxi * dxi/dx + du/deta * deta/dx
    # For simplicity, use magnitude-based gradients
    dist_i = mesh.dist_i + EPS
    dist_j_full = np.zeros((nj, ni))
    dist_j_full[1:-1] = 0.5 * (mesh.dist_j[:-1] + mesh.dist_j[1:])
    dist_j_full[0] = mesh.dist_j[0] if nj > 1 else 1.0
    dist_j_full[-1] = mesh.dist_j[-1] if nj > 1 else 1.0

    # Compute velocity gradients using local coordinate system
    # r-theta system is natural for O-grid
    rc = np.sqrt(mesh.xc ** 2 + mesh.yc ** 2) + EPS
    cos_t = mesh.xc / rc
    sin_t = mesh.yc / rc

    # radial velocity and tangential velocity
    ur = u * cos_t + v * sin_t
    ut = -u * sin_t + v * cos_t

    # radial gradients (j-direction ≈ r-direction)
    dur_dr = np.zeros_like(ur)
    dut_dr = np.zeros_like(ut)
    dur_dr[1:-1] = (ur[2:] - ur[:-2]) / (dist_j_full[1:-1] * 2.0 + EPS)
    dut_dr[1:-1] = (ut[2:] - ut[:-2]) / (dist_j_full[1:-1] * 2.0 + EPS)
    dur_dr[0] = (ur[1] - ur[0]) / (dist_j_full[0] + EPS)
    dur_dr[-1] = (ur[-1] - ur[-2]) / (dist_j_full[-1] + EPS)
    dut_dr[0] = (ut[1] - ut[0]) / (dist_j_full[0] + EPS)
    dut_dr[-1] = (ut[-1] - ut[-2]) / (dist_j_full[-1] + EPS)

    # azimuthal gradients (i-direction ≈ theta-direction)
    ur_ip1 = np.roll(ur, -1, axis=1)
    ur_im1 = np.roll(ur,  1, axis=1)
    ut_ip1 = np.roll(ut, -1, axis=1)
    ut_im1 = np.roll(ut,  1, axis=1)

    dur_dt = (ur_ip1 - ur_im1) / (2.0 * dist_i + EPS)
    dut_dt = (ut_ip1 - ut_im1) / (2.0 * dist_i + EPS)

    # Convert to Cartesian gradients
    # dudx = cos(t)*dur_dr - sin(t)*dut_dr  (approx, dominant terms)
    # dvdy = sin(t)*dur_dr + cos(t)*dut_dr
    dudx = cos_t * dur_dr - sin_t * dur_dt / (rc + EPS) - sin_t * dut_dr + cos_t * dut_dt / (rc + EPS)
    # Simplified: just use strain magnitude directly
    S = strain_magnitude_polar(dur_dr, dut_dr, ur, ut, rc, dur_dt, dut_dt, dist_i)

    # ---- Production of k ----
    Pk = mu_t * S ** 2
    # Limit production: Pk <= 20 * beta_star * rho * k * omega
    Pk = np.minimum(Pk, 20.0 * BETA_STAR * rho * k * omega)

    # ---- blended constants ----
    beta   = blend(BETA_1, BETA_2, F1)
    alpha  = blend(ALPHA_1, ALPHA_2, F1)
    sigma_w = blend(SIGMA_W1, SIGMA_W2, F1)

    # ---- k-equation source ----
    Sk = Pk - BETA_STAR * rho * k * omega

    # ---- omega-equation source ----
    # cross-diffusion term
    # gradients of k and omega in r and theta
    k_arr = np.maximum(W[..., 4], 0.0)
    w_arr = np.maximum(W[..., 5], EPS)

    dk_dr = np.zeros_like(k_arr)
    dw_dr = np.zeros_like(w_arr)
    dk_dr[1:-1] = (k_arr[2:] - k_arr[:-2]) / (dist_j_full[1:-1] * 2.0 + EPS)
    dw_dr[1:-1] = (w_arr[2:] - w_arr[:-2]) / (dist_j_full[1:-1] * 2.0 + EPS)
    dk_dr[0] = (k_arr[1] - k_arr[0]) / (dist_j_full[0] + EPS)
    dk_dr[-1] = (k_arr[-1] - k_arr[-2]) / (dist_j_full[-1] + EPS)
    dw_dr[0] = (w_arr[1] - w_arr[0]) / (dist_j_full[0] + EPS)
    dw_dr[-1] = (w_arr[-1] - w_arr[-2]) / (dist_j_full[-1] + EPS)

    k_ip1 = np.roll(k_arr, -1, axis=1)
    k_im1 = np.roll(k_arr,  1, axis=1)
    w_ip1 = np.roll(w_arr, -1, axis=1)
    w_im1 = np.roll(w_arr,  1, axis=1)
    dk_dt = (k_ip1 - k_im1) / (2.0 * dist_i + EPS)
    dw_dt = (w_ip1 - w_im1) / (2.0 * dist_i + EPS)

    cross_diff = 2.0 * (1.0 - F1) * rho * SIGMA_W2 / (omega + EPS) * (
        dk_dr * dw_dr + dk_dt * dw_dt
    )

    Sw = alpha * rho * S ** 2 / (omega + EPS) - beta * rho * omega ** 2 + cross_diff

    return Sk, Sw, dk_dr, dw_dr, dk_dt, dw_dt, S


def strain_magnitude_polar(dur_dr, dut_dr, ur, ut, rc, dur_dt, dut_dt, dist_i):
    """
    Strain-rate magnitude in approximate polar/curvilinear coordinates.
    S = sqrt(2 * Sij * Sij)
    """
    EPS = 1.0e-30
    # Approximate dominant components of strain rate
    Srr = dur_dr
    Stt = ur / (rc + EPS) + dut_dt / (rc + EPS)
    Srt = 0.5 * (dut_dr - ut / (rc + EPS) + dur_dt / (rc + EPS))
    return np.sqrt(2.0 * (Srr ** 2 + Stt ** 2 + 2.0 * Srt ** 2) + EPS)


def compute_blending_and_mut(W, mu_lam, mesh):
    """
    Compute F1, F2, and mu_t from the current state.

    Returns
    -------
    F1, F2, mu_t : (nj, ni) arrays
    """
    EPS = 1.0e-30
    rho   = W[..., 0]
    u     = W[..., 1]
    v     = W[..., 2]
    k     = np.maximum(W[..., 4], 0.0)
    omega = np.maximum(W[..., 5], EPS)
    d     = mesh.wall_dist + EPS
    nj, ni = rho.shape

    # Approximate gradients for cross-diffusion in F1
    dist_j_full = np.ones((nj, ni))
    if nj > 1:
        dist_j_full[1:-1] = 0.5 * (mesh.dist_j[:-1] + mesh.dist_j[1:])
        dist_j_full[0] = mesh.dist_j[0]
        dist_j_full[-1] = mesh.dist_j[-1]
    dist_i = mesh.dist_i + EPS

    dk_dr = np.zeros_like(k)
    dw_dr = np.zeros_like(omega)
    dk_dr[1:-1] = (k[2:] - k[:-2]) / (2.0 * dist_j_full[1:-1] + EPS)
    dw_dr[1:-1] = (omega[2:] - omega[:-2]) / (2.0 * dist_j_full[1:-1] + EPS)
    dk_dr[0]  = (k[1] - k[0]) / (dist_j_full[0] + EPS)
    dk_dr[-1] = (k[-1] - k[-2]) / (dist_j_full[-1] + EPS)
    dw_dr[0]  = (omega[1] - omega[0]) / (dist_j_full[0] + EPS)
    dw_dr[-1] = (omega[-1] - omega[-2]) / (dist_j_full[-1] + EPS)

    k_ip = np.roll(k, -1, axis=1)
    k_im = np.roll(k,  1, axis=1)
    w_ip = np.roll(omega, -1, axis=1)
    w_im = np.roll(omega,  1, axis=1)
    dk_dt = (k_ip - k_im) / (2.0 * dist_i + EPS)
    dw_dt = (w_ip - w_im) / (2.0 * dist_i + EPS)

    F1 = compute_F1(k, omega, rho, mu_lam, d, dk_dr, dk_dt, dw_dr, dw_dt)
    F2 = compute_F2(k, omega, rho, mu_lam, d)

    # Strain-rate magnitude for eddy-viscosity limiter
    rc = np.sqrt(mesh.xc ** 2 + mesh.yc ** 2) + EPS
    cos_t = mesh.xc / rc
    sin_t = mesh.yc / rc
    ur = u * cos_t + v * sin_t
    ut = -u * sin_t + v * cos_t

    dur_dr = np.zeros_like(ur)
    dut_dr = np.zeros_like(ut)
    dur_dr[1:-1] = (ur[2:] - ur[:-2]) / (2.0 * dist_j_full[1:-1] + EPS)
    dut_dr[1:-1] = (ut[2:] - ut[:-2]) / (2.0 * dist_j_full[1:-1] + EPS)
    dur_dr[0]  = (ur[1] - ur[0]) / (dist_j_full[0] + EPS)
    dur_dr[-1] = (ur[-1] - ur[-2]) / (dist_j_full[-1] + EPS)
    dut_dr[0]  = (ut[1] - ut[0]) / (dist_j_full[0] + EPS)
    dut_dr[-1] = (ut[-1] - ut[-2]) / (dist_j_full[-1] + EPS)

    ur_ip = np.roll(ur, -1, axis=1)
    ur_im = np.roll(ur,  1, axis=1)
    ut_ip = np.roll(ut, -1, axis=1)
    ut_im = np.roll(ut,  1, axis=1)
    dur_dth = (ur_ip - ur_im) / (2.0 * dist_i + EPS)
    dut_dth = (ut_ip - ut_im) / (2.0 * dist_i + EPS)

    S = strain_magnitude_polar(dur_dr, dut_dr, ur, ut, rc, dur_dth, dut_dth, dist_i)

    mu_t = eddy_viscosity(rho, k, omega, F2, S)

    return F1, F2, mu_t
