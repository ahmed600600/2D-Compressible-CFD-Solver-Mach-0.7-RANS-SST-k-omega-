"""
Structured O-grid mesh generation around a 2D circular cylinder.
Provides vertex coordinates, cell centres, face area-vectors, cell areas,
and wall-distance for the finite-volume solver.
"""

import numpy as np


class Mesh:
    """O-grid mesh around a circular cylinder centred at the origin."""

    def __init__(self, R=0.5, r_far=20.0, ni=200, nj=100, stretch=1.1):
        """
        Parameters
        ----------
        R       : cylinder radius  (m)
        r_far   : farfield radius  (m)
        ni      : cells in circumferential direction
        nj      : cells in radial direction
        stretch : geometric stretching ratio (>1 clusters near wall)
        """
        self.R = R
        self.r_far = r_far
        self.ni = ni
        self.nj = nj

        # --- circumferential angles (periodic, ni points) ---
        theta = np.linspace(0.0, 2.0 * np.pi, ni, endpoint=False)

        # --- radial levels (nj+1 vertices) with geometric stretching ---
        if abs(stretch - 1.0) < 1e-12:
            radii = np.linspace(R, r_far, nj + 1)
        else:
            dr0 = (r_far - R) * (stretch - 1.0) / (stretch ** nj - 1.0)
            radii = np.empty(nj + 1)
            radii[0] = R
            for k in range(nj):
                radii[k + 1] = radii[k] + dr0 * stretch ** k

        self.radii = radii
        self.theta = theta

        # --- vertex coordinates  shape = (nj+1, ni) ---
        TH, RR = np.meshgrid(theta, radii)
        self.xv = RR * np.cos(TH)
        self.yv = RR * np.sin(TH)

        self._compute_metrics()

    # ------------------------------------------------------------------ #
    def _compute_metrics(self):
        xv, yv = self.xv, self.yv
        ni, nj = self.ni, self.nj
        ip1 = np.roll(np.arange(ni), -1)  # periodic i+1

        # -- cell centres (average of 4 vertices) -------------------------
        self.xc = 0.25 * (xv[:-1, :] + xv[:-1, ip1] +
                          xv[1:, ip1] + xv[1:, :])
        self.yc = 0.25 * (yv[:-1, :] + yv[:-1, ip1] +
                          yv[1:, ip1] + yv[1:, :])

        # -- cell areas (cross product of diagonals) ----------------------
        d1x = xv[1:, ip1] - xv[:-1, :]
        d1y = yv[1:, ip1] - yv[:-1, :]
        d2x = xv[1:, :] - xv[:-1, ip1]
        d2y = yv[1:, :] - yv[:-1, ip1]
        self.area = 0.5 * np.abs(d1x * d2y - d1y * d2x)

        # -- j-face area vectors (radial boundary of cell) -----------------
        #    face(j,i) : vertex(j,i) -> vertex(j,ip1)
        #    normal points in +j direction (radially outward)
        dx_j = xv[:, ip1] - xv[:, :]
        dy_j = yv[:, ip1] - yv[:, :]
        self.Sx_j =  dy_j
        self.Sy_j = -dx_j
        self.dS_j = np.sqrt(self.Sx_j ** 2 + self.Sy_j ** 2)

        # -- i-face area vectors (circumferential boundary of cell) --------
        #    face(j,i) : vertex(j,i) -> vertex(j+1,i)
        #    normal points in +i direction
        dx_i = xv[1:, :] - xv[:-1, :]
        dy_i = yv[1:, :] - yv[:-1, :]
        self.Sx_i = -dy_i
        self.Sy_i =  dx_i
        self.dS_i = np.sqrt(self.Sx_i ** 2 + self.Sy_i ** 2)

        # -- wall distance from cell centre --------------------------------
        self.wall_dist = np.sqrt(self.xc ** 2 + self.yc ** 2) - self.R

        # -- distances between adjacent cell centres -----------------------
        # radial (j) neighbours  ->  (nj-1, ni)
        dxc = np.diff(self.xc, axis=0)
        dyc = np.diff(self.yc, axis=0)
        self.dist_j = np.sqrt(dxc ** 2 + dyc ** 2)

        # circumferential (i) neighbours  ->  (nj, ni)
        dxc_i = np.roll(self.xc, -1, axis=1) - self.xc
        dyc_i = np.roll(self.yc, -1, axis=1) - self.yc
        self.dist_i = np.sqrt(dxc_i ** 2 + dyc_i ** 2)
