"""
Main solver loop for the 2D compressible RANS equations.
Explicit 3-stage Runge-Kutta with local time-stepping.
"""

import numpy as np
from mesh import Mesh
from flux import inviscid_flux, viscous_flux_j
from turbulence import (
    compute_blending_and_mut, source_terms,
    SIGMA_K1, SIGMA_K2, SIGMA_W1, SIGMA_W2, blend,
)
from boundary import primitives, conserved, wall_ghost, farfield_ghost


class Solver:
    """2D compressible RANS solver on a structured O-grid."""

    RK_ALPHA = [0.25, 0.5, 1.0]          # 3-stage RK coefficients

    def __init__(self, mesh, gamma=1.4, CFL=0.5, Ma_inf=0.7,
                 p_inf=101325.0, T_inf=288.15, Re=1.0e6,
                 Pr=0.72, Prt=0.9, max_iter=5000, tol=1e-6):
        self.mesh  = mesh
        self.gamma = gamma
        self.CFL   = CFL
        self.Pr    = Pr
        self.Prt   = Prt
        self.max_iter = max_iter
        self.tol   = tol

        # --- freestream conditions ---
        R_gas  = 287.058                  # J/(kg·K)
        self.R_gas = R_gas
        rho_inf = p_inf / (R_gas * T_inf)
        a_inf   = np.sqrt(gamma * p_inf / rho_inf)
        U_inf   = Ma_inf * a_inf
        mu_inf  = rho_inf * U_inf * mesh.R * 2.0 / Re   # based on diameter
        self.mu_ref = mu_inf

        # Non-dimensionalise
        L_ref = mesh.R * 2.0              # reference length = diameter
        self.rho_ref = rho_inf
        self.a_ref   = a_inf
        self.p_ref   = rho_inf * a_inf ** 2

        # Store non-dimensional freestream
        self.rho_inf = 1.0
        self.u_inf   = Ma_inf              # non-dim u = Ma
        self.v_inf   = 0.0
        self.p_inf   = 1.0 / gamma         # p / (rho a^2) = 1/gamma
        self.T_inf   = self.p_inf / (self.rho_inf * 1.0)
        self.mu_nd   = 1.0 / Re            # non-dim mu = 1/Re

        # turbulence freestream
        turb_intensity = 0.01
        turb_visc_ratio = 1.0
        self.k_inf = 1.5 * (turb_intensity * Ma_inf) ** 2
        self.k_inf = max(self.k_inf, 1e-8)
        self.w_inf = self.rho_inf * self.k_inf / (self.mu_nd * turb_visc_ratio + 1e-30)

        self.W_inf = np.array([
            self.rho_inf, self.u_inf, self.v_inf,
            self.p_inf, self.k_inf, self.w_inf
        ])

        # --- initialise flow field ---
        nj, ni = mesh.nj, mesh.ni
        self.W = np.empty((nj, ni, 6))
        for i in range(6):
            self.W[:, :, i] = self.W_inf[i]

        self.Q = conserved(self.W, gamma)
        self.residual_history = []

    # ------------------------------------------------------------------ #
    def _laminar_viscosity(self, W):
        """Non-dimensional Sutherland / constant viscosity."""
        return np.full(W.shape[:2], self.mu_nd)

    # ------------------------------------------------------------------ #
    def _local_timestep(self, W, mu_lam, mu_t):
        """CFL-based local time step."""
        EPS = 1.0e-30
        rho = W[..., 0]
        u   = W[..., 1]
        v   = W[..., 2]
        p   = W[..., 3]
        a   = np.sqrt(np.maximum(self.gamma * p / (rho + EPS), EPS))
        V   = np.sqrt(u ** 2 + v ** 2)

        # spectral radii in j and i directions (using face normals)
        mesh = self.mesh
        # average face normal magnitudes for each cell
        dS_j = 0.5 * (mesh.dS_j[:-1, :] + mesh.dS_j[1:, :])  # (nj, ni)
        dS_i = mesh.dS_i                                       # (nj, ni)

        lam_j = (V + a) * dS_j
        lam_i = (V + a) * dS_i

        # viscous contribution
        mu_eff = mu_lam + mu_t
        lam_v = 4.0 * mu_eff / (rho * mesh.area + EPS)

        dt = self.CFL * mesh.area / (lam_j + lam_i + lam_v + EPS)
        return dt

    # ------------------------------------------------------------------ #
    def _compute_residual(self, W, mu_lam, mu_t, F1):
        """Assemble the residual  R = -(1/A)*sum(fluxes) + sources."""
        mesh  = self.mesh
        gamma = self.gamma
        nj, ni = mesh.nj, mesh.ni

        # blended sigma_k, sigma_w for viscous flux
        sigma_k = blend(SIGMA_K1, SIGMA_K2, F1)
        sigma_w = blend(SIGMA_W1, SIGMA_W2, F1)
        # use average for viscous flux routine
        sigma_k_avg = np.mean(sigma_k)
        sigma_w_avg = np.mean(sigma_w)

        # ---- ghost cells -----
        Wg_wall = wall_ghost(W, mesh, mu_lam, gamma)       # (1, ni, 6)
        Wg_far  = farfield_ghost(W, self.W_inf, mesh, gamma)  # (1, ni, 6)

        # extended array (nj+2, ni, 6) with ghost layers
        We = np.concatenate([Wg_wall, W, Wg_far], axis=0)

        # ============================================================ #
        #                    j-direction fluxes                         #
        # ============================================================ #
        # j-face index f:  face f sits between cell f-1 and cell f
        # In the extended array, interior faces are f=1..nj  (nj faces)
        # plus boundary faces f=0 (below wall ghost) and f=nj+1 (above farfield ghost)
        # We need faces f=1..nj in the original indexing => f=1..nj in extended

        Fj = np.zeros((nj + 1, ni, 6))
        for f in range(nj + 1):
            WL_f = We[f, :, :]        # cell below face
            WR_f = We[f + 1, :, :]    # cell above face
            Sx = mesh.Sx_j[f, :]
            Sy = mesh.Sy_j[f, :]
            Fj[f] = inviscid_flux(WL_f, WR_f, Sx, Sy, gamma)

        # ---- viscous j-flux at interior faces (f=1..nj-1 in original) --
        # which are faces between interior cells only
        Fv_j = viscous_flux_j(W, mu_lam, mu_t, mesh, gamma,
                              self.Pr, self.Prt, sigma_k_avg, sigma_w_avg)
        # Fv_j shape = (nj-1, ni, 6)

        # ============================================================ #
        #                    i-direction fluxes                         #
        # ============================================================ #
        # periodic: face i sits between cell (j, i-1 mod ni) and cell (j, i)
        Fi = np.zeros((nj, ni, 6))
        for ic in range(ni):
            im = (ic - 1) % ni
            WL_f = W[:, im, :]    # (nj, 6)
            WR_f = W[:, ic, :]
            Sx = mesh.Sx_i[:, ic]
            Sy = mesh.Sy_i[:, ic]
            Fi[:, ic, :] = inviscid_flux(WL_f, WR_f, Sx, Sy, gamma)

        # ============================================================ #
        #                 Assemble residual                             #
        # ============================================================ #
        R = np.zeros_like(W)

        # j-direction: net outward inviscid flux for cell j
        #   top face = Fj[j+1]  (normal points outward = +j)
        #   bottom face = Fj[j] (normal points inward = -j)
        inviscid_net = Fj[1:, :, :] - Fj[:-1, :, :]   # (nj, ni, 6)

        # viscous j-flux at interior faces
        visc_j_net = np.zeros_like(W)
        visc_j_net[1:, :, :]  -= Fv_j          # bottom face of cell j (subtract = in)
        visc_j_net[:-1, :, :] += Fv_j          # top face of cell j-1 (add = out)

        # i-direction: net outward inviscid flux
        ip1 = np.roll(np.arange(ni), -1)
        inviscid_i_net = Fi[:, ip1, :] - Fi[:, :, :]  # (nj, ni, 6)

        flux_total = inviscid_net + inviscid_i_net - visc_j_net

        R = -flux_total / (mesh.area[:, :, None] + 1e-30)

        # ---- turbulence source terms ----
        Sk, Sw, *_ = source_terms(W, mu_lam, mu_t, F1, mesh, gamma)
        R[..., 4] += Sk
        R[..., 5] += Sw

        return R

    # ------------------------------------------------------------------ #
    def iterate(self, n_iter=None):
        """Run the solver for n_iter iterations (or self.max_iter)."""
        if n_iter is None:
            n_iter = self.max_iter

        gamma = self.gamma
        print(f"{'Iter':>6s}  {'L2(rho-res)':>12s}  {'L2(k-res)':>12s}")
        print("-" * 36)

        for n in range(1, n_iter + 1):
            Q0 = self.Q.copy()

            mu_lam = self._laminar_viscosity(self.W)
            F1, F2, mu_t = compute_blending_and_mut(self.W, mu_lam, self.mesh)
            dt = self._local_timestep(self.W, mu_lam, mu_t)

            # 3-stage Runge-Kutta
            for stage in range(3):
                alpha = self.RK_ALPHA[stage]
                self.W = primitives(self.Q, gamma)
                # clamp
                self.W[..., 0] = np.maximum(self.W[..., 0], 1e-8)
                self.W[..., 3] = np.maximum(self.W[..., 3], 1e-8)
                self.W[..., 4] = np.maximum(self.W[..., 4], 0.0)
                self.W[..., 5] = np.maximum(self.W[..., 5], 1e-8)

                R = self._compute_residual(self.W, mu_lam, mu_t, F1)
                self.Q = Q0 + alpha * dt[:, :, None] * R

            self.W = primitives(self.Q, gamma)
            self.W[..., 0] = np.maximum(self.W[..., 0], 1e-8)
            self.W[..., 3] = np.maximum(self.W[..., 3], 1e-8)
            self.W[..., 4] = np.maximum(self.W[..., 4], 0.0)
            self.W[..., 5] = np.maximum(self.W[..., 5], 1e-8)

            # convergence check
            dQ = self.Q - Q0
            res_rho = np.sqrt(np.mean(dQ[..., 0] ** 2))
            res_k   = np.sqrt(np.mean(dQ[..., 4] ** 2))
            self.residual_history.append(res_rho)

            if n % 100 == 0 or n == 1:
                print(f"{n:6d}  {res_rho:12.5e}  {res_k:12.5e}")

            if res_rho < self.tol:
                print(f"\n>>> Converged at iteration {n}  (res = {res_rho:.3e})")
                break

        else:
            print(f"\n>>> Reached max iterations ({n_iter}), "
                  f"final residual = {res_rho:.3e}")

        return self.W
