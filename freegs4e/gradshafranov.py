"""
Contains various classes and functions related to the elliptic operator
of the Grad-Shafranov equation.

Copyright 2026 Ben Dudson, Tomas Rubio Cruz, Ubaid Qadri

This file is part of FreeGS4E.

FreeGS4E is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS4E is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS4E.  If not, see <http://www.gnu.org/licenses/>.

"""

from time import perf_counter

import numexpr as ne
import numpy as np
from numpy import pi
from scipy.fft import dst
from scipy.linalg import solve_banded
from scipy.sparse import csr_array, eye

# elliptic integrals of first and second kind (K and E)
from scipy.special import ellipe, ellipk

from .parallel_funcs import threaded_clip, threaded_elliptics_ek

# magnetic permeability of free space
mu0 = 4e-7 * pi


class GSElliptic:
    """
    Class representing the elliptc operator within the Grad-Shafranov
    equation:

        Δ^* = d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR

        where:
         -  R is the radial coordinate.
         -  Z is the vertical coordinate.

    """

    def __init__(self, Rmin):
        """
        Initializes the class.

        Parameters
        ----------
        Rmin : float
            Minimum major radius [m].

        """

        self.Rmin = Rmin

    def __call__(self, psi, dR, dZ):
        """
        Apply the elliptic operator to the flux function such that:

            (Δ^*) * ψ.

        Computes to second-order accuracy.

        Parameters
        ----------
        psi : np.array
            The total poloidal flux at each (R,Z) grid point [Webers/2pi].
        dR : float
            Radial grid size [m].
        dZ : float
            Vertical grid size [m].

        Returns
        -------
        np.array
            The operator applied to the total poloidal flux.
        """

        # number of radial and vertical grid points
        nx = psi.shape[0]
        ny = psi.shape[1]

        # to store output
        b = np.zeros([nx, ny])

        # pre-compute constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # compute the operator (can be vectorised)
        for x in range(1, nx - 1):
            R = self.Rmin + dR * x  # Major radius of this point
            for y in range(1, ny - 1):
                # Loop over points in the domain
                b[x, y] = (
                    psi[x, y - 1] * invdZ2
                    + (invdR2 + 1.0 / (2.0 * R * dR)) * psi[x - 1, y]
                    - 2.0 * (invdR2 + invdZ2) * psi[x, y]
                    + (invdR2 - 1.0 / (2.0 * R * dR)) * psi[x + 1, y]
                    + psi[x, y + 1] * invdZ2
                )
        return b

    def diag(self, dR, dZ):
        """
        Computes:

            -2 * ( (1/dR^2) + (1/dZ^2) ).

        Parameters
        ----------
        dR : float
            Radial grid size [m].
        dZ : float
            Vertical grid size [m].

        Returns
        -------
        float
            The value above.
        """
        return -2.0 / dR**2 - 2.0 / dZ**2


class GSsparse:
    """
    Class representing the elliptc operator within the Grad-Shafranov
    equation:

        Δ^* = d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR

        where:
         -  R is the radial coordinate.
         -  Z is the vertical coordinate.

    This class calculates the sparse version to second-order accuracy.

    """

    def __init__(self, Rmin, Rmax, Zmin, Zmax):
        """
        Initializes the class.

        Parameters
        ----------
        Rmin : float
            Minimum major radius [m].
        Rmax : float
            Maximum major radius [m].
        Zmin : float
            Minimum height [m].
        Zmax : float
            Maximum height [m].

        """

        # set parameters
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

    def __call__(self, nx, ny):
        """
        Generates the sparse elliptic operator Δ^* for a given number of
        grid points. Computes to second-order accuracy.

        Parameters
        ----------
        nx : int
            Number of radial grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        ny : int
            Number of vertical grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).

        Returns
        -------
        np.array
            The operator matrix.
        """

        # calculate grid spacing
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        # total number of points
        N = nx * ny

        # create a linked list sparse matrix
        A = eye(N, format="lil")

        # pre-compute constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # generate the operator values (can be optimsied)
        for x in range(1, nx - 1):
            R = self.Rmin + dR * x  # Major radius of this point
            for y in range(1, ny - 1):
                # Loop over points in the domain
                row = x * ny + y

                # y-1
                A[row, row - 1] = invdZ2

                # x-1
                A[row, row - ny] = invdR2 + 1.0 / (2.0 * R * dR)

                # diagonal
                A[row, row] = -2.0 * (invdR2 + invdZ2)

                # x+1
                A[row, row + ny] = invdR2 - 1.0 / (2.0 * R * dR)

                # y+1
                A[row, row + 1] = invdZ2

        # convert to Compressed Sparse Row (CSR) format
        return A.tocsr()


class GSsparse4thOrder:
    """
    Class representing the elliptc operator within the Grad-Shafranov
    equation:

        Δ^* = d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR

        where:
         -  R is the radial coordinate.
         -  Z is the vertical coordinate.

    This class calculates the sparse version to fourth-order accuracy.

    """

    # Coefficients for first derivatives
    # (index offset, weight)

    centred_1st = [
        (-2, 1.0 / 12),
        (-1, -8.0 / 12),
        (1, 8.0 / 12),
        (2, -1.0 / 12),
    ]

    offset_1st = [
        (-1, -3.0 / 12),
        (0, -10.0 / 12),
        (1, 18.0 / 12),
        (2, -6.0 / 12),
        (3, 1.0 / 12),
    ]

    # Coefficients for second derivatives
    # (index offset, weight)
    centred_2nd = [
        (-2, -1.0 / 12),
        (-1, 16.0 / 12),
        (0, -30.0 / 12),
        (1, 16.0 / 12),
        (2, -1.0 / 12),
    ]

    offset_2nd = [
        (-1, 10.0 / 12),
        (0, -15.0 / 12),
        (1, -4.0 / 12),
        (2, 14.0 / 12),
        (3, -6.0 / 12),
        (4, 1.0 / 12),
    ]

    def __init__(self, Rmin, Rmax, Zmin, Zmax):
        """
        Initializes the class.

        Parameters
        ----------
        Rmin : float
            Minimum major radius [m].
        Rmax : float
            Maximum major radius [m].
        Zmin : float
            Minimum height [m].
        Zmax : float
            Maximum height [m].

        """

        # set parameters
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

    def __call_old__(self, nx, ny):
        """
        Generates the sparse elliptic operator Δ^* for a given number of
        grid points. Computes to fourth-order accuracy.

        Uses an inefficient, but easy to check and debug entry-by-entry aproach to
        filling the matrix, unlike the faster but more complicated __call__ method.
        Kept as reference for debugging and and as a fallback.

        Parameters
        ----------
        nx : int
            Number of radial grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        ny : int
            Number of vertical grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).

        Returns
        -------
        np.array
            The operator matrix.
        """

        # calculate grid spacing
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        # total number of points, including boundaries
        N = nx * ny

        # create a linked list sparse matrix
        A = lil_matrix((N, N))

        # calculate constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # calculate entries (can be vectorised)
        for x in range(1, nx - 1):
            R = self.Rmin + dR * x  # Major radius of this point
            for y in range(1, ny - 1):
                row = x * ny + y

                # d^2 / dZ^2
                if y == 1:
                    # One-sided derivatives in Z
                    for offset, weight in self.offset_2nd:
                        A[row, row + offset] += weight * invdZ2
                elif y == ny - 2:
                    # One-sided, reversed direction.
                    # Note that for second derivatives the sign of the weights doesn't change
                    for offset, weight in self.offset_2nd:
                        A[row, row - offset] += weight * invdZ2
                else:
                    # Central differencing
                    for offset, weight in self.centred_2nd:
                        A[row, row + offset] += weight * invdZ2

                # d^2 / dR^2 - (1/R) d/dR

                if x == 1:
                    for offset, weight in self.offset_2nd:
                        A[row, row + offset * ny] += weight * invdR2

                    for offset, weight in self.offset_1st:
                        A[row, row + offset * ny] -= weight / (R * dR)

                elif x == nx - 2:
                    for offset, weight in self.offset_2nd:
                        A[row, row - offset * ny] += weight * invdR2

                    for offset, weight in self.offset_1st:
                        A[row, row - offset * ny] += weight / (R * dR)

                else:
                    for offset, weight in self.centred_2nd:
                        A[row, row + offset * ny] += weight * invdR2

                    for offset, weight in self.centred_1st:
                        A[row, row + offset * ny] -= weight / (R * dR)

        # set boundary rows
        for x in range(nx):
            for y in [0, ny - 1]:
                row = x * ny + y
                A[row, row] = 1.0
        for x in [0, nx - 1]:
            for y in range(ny):
                row = x * ny + y
                A[row, row] = 1.0

        # convert to Compressed Sparse Row (CSR) format
        return A.tocsr()

    def __call__(self, nx, ny):
        """
        Generates the sparse elliptic operator Δ^* for a given number of
        grid points. Computes to fourth-order accuracy.

        Parameters
        ----------
        nx : int
            Number of radial grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        ny : int
            Number of vertical grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).

        Returns
        -------
        np.array
            The operator matrix.
        """

        # calculate grid spacing
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        # total number of points, including boundaries
        N = nx * ny

        # calculate constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # The GS operator is constructed in COO format using lists of ndarrays. Each ndarray contains
        # the row/col/value of the entries for a given stencil.
        rows = []
        cols = []
        entries = []

        # set boundary entries

        for x in (0, nx - 1):
            y = np.arange(ny)

            row = x * ny + y
            entry = np.ones_like(row, dtype=np.float64)

            rows.append(row)
            cols.append(row)  # no offset for dirichlet bc
            entries.append(entry)

        for y in (0, ny - 1):
            x = np.arange(1, nx - 1)  # x=0, x=(nx-1) were already set above

            row = x * ny + y
            entry = np.ones_like(row, dtype=np.float64)

            rows.append(row)
            cols.append(row)  # no offset for dirichlet bc
            entries.append(entry)

        # set near-boundary entries (need offset stencils)

        # d^2 / dR^2 - (1/R) d/dR
        for x, offsign in zip(
            (1, nx - 2), (1, -1)
        ):  # offset on right (nx-2) has negative sign

            y = np.arange(1, ny - 1)
            R = self.Rmin + dR * x  # major radius of each point

            row = x * ny + y

            for offset, weight in self.offset_2nd:

                col = row + offsign * offset * ny
                entry = weight * invdR2
                entry = np.full_like(row, entry, dtype=np.float64)

                rows.append(row)
                cols.append(col)
                entries.append(entry)

            for offset, weight in self.offset_1st:

                col = row + offsign * offset * ny
                entry = (
                    -offsign * weight / (R * dR)
                )  # sign of entry depends on direction (offsign)
                entry = np.full_like(row, entry, dtype=np.float64)

                rows.append(row)
                cols.append(col)
                entries.append(entry)

        # d^2 / dZ^2
        for y, offsign in zip(
            (1, ny - 2), (1, -1)
        ):  # offset on top (ny-2) has negative sign

            x = np.arange(1, nx - 1)
            row = x * ny + y

            for offset, weight in self.offset_2nd:

                col = row + offsign * offset
                entry = weight * invdZ2
                entry = np.full_like(row, entry, dtype=np.float64)

                rows.append(row)
                cols.append(col)
                entries.append(entry)

        # set internal entries (use centred stencil)

        # build the largest rectangle in domain with only centred-scheme entries
        y_vals = np.arange(2, ny - 2)
        x_vals = np.arange(2, nx - 2)
        R = self.Rmin + dR * x_vals  # major radius of each point

        # 2d grid with row no. of each (x,y)
        row_2d = x_vals[:, np.newaxis] * ny + y_vals[np.newaxis, :]

        # iterate over differential operators:
        # longitudinal --> d^2/dZ^2; radial_1 --> d^2/dR^2; radial_2 --> -(1/R) d/dR

        operators = ["longitudinal", "radial_1", "radial_2"]
        stencils = {
            "longitudinal": self.centred_2nd,
            "radial_1": self.centred_2nd,
            "radial_2": self.centred_1st,
        }
        offscales = {"longitudinal": 1, "radial_1": ny, "radial_2": ny}

        for op in operators:

            stencil = stencils[op]
            offscale = offscales[op]

            size_stencil = len(stencil)
            stencil_offsets = offscale * np.array(
                [entry[0] for entry in stencil]
            )
            stencil_weights = np.array([entry[1] for entry in stencil])

            # 3d grids with the row/col/weight corresponding to each x,y,offset combo
            row = np.broadcast_to(
                row_2d[:, :, np.newaxis], (*row_2d.shape, size_stencil)
            )
            col = row + stencil_offsets[np.newaxis, np.newaxis, :]
            weights = np.broadcast_to(
                stencil_weights[np.newaxis, np.newaxis, :], row.shape
            )

            entry = None

            if op == "longitudinal":
                # d^2 / dZ^2
                entry = weights * invdZ2
            elif op == "radial_1":
                # d^2 / dR^2
                entry = weights * invdR2
            elif op == "radial_2":
                # -(1/R) d/dR
                entry = -weights / (R[:, np.newaxis, np.newaxis] * dR)
            else:
                raise KeyError(op)

            rows.append(row.flatten())
            cols.append(col.flatten())
            entries.append(entry.flatten())

            if op == "longitudinal":
                # set stencil for points where a centred scheme is used longitudinally
                # but not radially
                for x in (1, nx - 2):
                    y = np.arange(2, ny - 2)

                    row = x * ny + y
                    row = np.broadcast_to(
                        row[:, np.newaxis], (*row.shape, size_stencil)
                    )
                    col = row + stencil_offsets[np.newaxis, :]
                    weights = np.broadcast_to(
                        stencil_weights[np.newaxis, :], row.shape
                    )

                    entry = weights * invdZ2

                    rows.append(row.flatten())
                    cols.append(col.flatten())
                    entries.append(entry.flatten())

            else:
                # set stencil for points where a centred scheme is used radially
                # but not longitudinally
                for y in (1, ny - 2):
                    x = np.arange(2, nx - 2)

                    row = x * ny + y
                    row = np.broadcast_to(
                        row[:, np.newaxis], (*row.shape, size_stencil)
                    )
                    col = row + stencil_offsets[np.newaxis, :]
                    weights = np.broadcast_to(
                        stencil_weights[np.newaxis, :], row.shape
                    )

                    entry = None
                    if op == "radial_1":
                        entry = weights * invdR2
                    elif op == "radial_2":
                        entry = -weights / (R[:, np.newaxis] * dR)
                    else:
                        raise KeyError(op)

                    rows.append(row.flatten())
                    cols.append(col.flatten())
                    entries.append(entry.flatten())

        all_rows = np.concat(rows)
        all_cols = np.concat(cols)
        all_entries = np.concat(entries)

        A = csr_array((all_entries, (all_rows, all_cols)), dtype=np.float64)

        return A


class DSTsolver:
    """
    DST-based Poisson/Grad–Shafranov solver on a (R, Z) rectangular grid.

    Solves:  (d2/dR2 - (1/R)d/dR + d2/dZ2) psi = f  on the interior,
    with Dirichlet boundaries supplied on the outer edges via `rhs`.

    Input `rhs` format (shape (nr, nz)):
      - Z-boundaries: rhs[:, 0], rhs[:, -1]  -> Dirichlet psi
      - R-boundaries: rhs[0, 1:-1], rhs[-1, 1:-1] -> Dirichlet psi
      - Interior:     rhs[1:-1, 1:-1] -> source f
    """

    def __init__(self, R: np.ndarray, Z: np.ndarray, *, dtype=np.float64):

        if R.shape != Z.shape:
            raise ValueError(
                "shape mismatch: rhs shape {} does not match solver grid shape {}".format(
                    rhs.shape, self.R.shape
                )
            )

        self.Rmin = float(R[0, 0])
        self.Rmax = float(R[-1, 0])
        self.Zmin = float(Z[0, 0])
        self.Zmax = float(Z[0, -1])
        self.R = np.ascontiguousarray(R, dtype=dtype)
        self.Z = np.ascontiguousarray(Z, dtype=dtype)

        # Uniform spacings (assumed)
        self.dR = float(R[1, 0] - R[0, 0])
        self.dZ = float(Z[0, 1] - Z[0, 0])

        # Config
        self.dtype = dtype

        # Precompute batch tridiagonal diagonals and Z-eigenvalues
        self._init_matrix()

    # Orthnormal DST-I (self-inverse)
    def _dst1(self, x: np.ndarray) -> np.ndarray:
        """Apply DST-I (orthonormal) along the last axis; inverse is identical."""
        return dst(x, type=1, axis=-1, norm="ortho")

    def _init_matrix(self):
        R = self.R
        nr, nz = R.shape
        Nint = nz - 2
        if Nint < 1:
            raise ValueError(
                "Need at least 3 Z points to have an interior. Current number: {}".format(
                    nz
                )
            )

        # Z eigenvalues mu_m > 0 for Dirichlet interior FD Laplacian
        m = np.arange(1, Nint + 1, dtype=self.dtype)
        self.mu = (2.0 / self.dZ**2) * (
            1.0 - np.cos(m * np.pi / (Nint + 1))
        )  # length Nint

        # R-direction FD operator for (d2/dR2 - (1/R)d/dR)
        Rvec = R[:, 0]
        # Off-diagonals (centered derivatives)
        self.sub = -1.0 / (2.0 * -Rvec[1:] * self.dR) + np.full(
            nr - 1, 1.0 / self.dR**2, dtype=self.dtype
        )
        self.sup = +1.0 / (2.0 * -Rvec[:-1] * self.dR) + np.full(
            nr - 1, 1.0 / self.dR**2, dtype=self.dtype
        )
        self.main = np.full(nr, -2.0 / self.dR**2, dtype=self.dtype)

        # Impose Dirichlet at R boundaries by fixing rows:
        # main[0] = 1, main[-1] = 1, and ensure no coupling outside:
        self.main[0] = 1.0
        self.main[-1] = 1.0
        self.sub[-1] = 0.0  # nothing below last row
        self.sup[0] = 0.0  # nothing above first row

        # Correct SciPy banded matrix template
        ab = np.zeros((3, nr), dtype=self.dtype)
        ab[0, 1:] = self.sup[:]  # upper diag (length nr-1)
        ab[1, :] = self.main[:]  # main diag (length nr)
        ab[2, :-1] = self.sub[:]  # lower diag (length nr-1)

        self.ab_template = ab

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve for psi given rhs with mixed boundary/data layout as documented in the class docstring.
        """
        rhs = np.ascontiguousarray(rhs, dtype=self.dtype)

        if rhs.shape != self.R.shape:
            raise ValueError(
                "shape mismatch: rhs shape {} does not match solver grid shape {}".format(
                    rhs.shape, self.R.shape
                )
            )

        # Build "g" that matches Z-boundaries linearly in Z
        phi0 = rhs[:, 0]  # psi at Z=Zmin
        phiL = rhs[:, -1]  # psi at Z=Zmax
        zfrac = (self.Z - self.Zmin) / (self.Zmax - self.Zmin)
        g = phi0[:, None] + (phiL - phi0)[:, None] * zfrac  # shape (nr, nz)

        # Compute Δ*g (consistent finite differences)
        g_RR = np.zeros_like(g)
        g_RR[1:-1, :] = (g[:-2, :] - 2.0 * g[1:-1, :] + g[2:, :]) / self.dR**2

        g_ZZ = np.zeros_like(g)
        g_ZZ[:, 1:-1] = (g[:, :-2] - 2.0 * g[:, 1:-1] + g[:, 2:]) / self.dZ**2

        # (1/R)*∂g/∂R (centered)
        iR = 1.0 / (2.0 * self.R * self.dR)
        iRgR = np.zeros_like(g)
        iRgR[1:-1, :] = iR[1:-1, :] * (g[2:, :] - g[:-2, :])

        Delta_g = g_RR + g_ZZ - iRgR

        # Modified RHS for w: Δ* w = f - Δ* g, with w = 0 at Z boundaries
        F = rhs - Delta_g
        F_int = F[:, 1:-1]  # Z-interior

        # DST-I along Z (interior only). For type-1 + ortho, inverse == forward.
        F_hat = self._dst1(F_int)  # shape (nr, Nint)

        # Transform R-boundary values for w and inject into each mode's RHS
        w_b0_hat = self._dst1((rhs[0, 1:-1] - g[0, 1:-1]))  # (Nint,)
        w_bL_hat = self._dst1((rhs[-1, 1:-1] - g[-1, 1:-1]))  # (Nint,)

        # RHS per mode (transpose to (Nint, nr))
        rhs_batch = F_hat.T.copy()  # (Nint, nr)
        rhs_batch[:, 0] = w_b0_hat
        rhs_batch[:, -1] = w_bL_hat

        # Solve tridiagonal in R for each Z-mode: (D_RR - (1/R)D_R - mu_m * I) w_hat_m = rhs_m
        Nint = rhs_batch.shape[0]
        w_hat_modes = np.empty_like(rhs_batch)  # (Nint, nr)

        # Banded storage uses (l,u) = (1,1)
        for k in range(Nint):
            mu_k = self.mu[k]
            ab = self.ab_template.copy()
            # Shift interior diagonal by -mu_k (Z-separation term)
            ab[1, 1:-1] -= mu_k
            # Solve A_k * x_k = rhs_batch[k]
            w_hat_modes[k] = solve_banded((1, 1), ab, rhs_batch[k])

        # Inverse DST (same as forward) to reconstruct w on Z interior
        w_hat_int = w_hat_modes.T  # (nr, Nint)
        w_int = self._dst1(w_hat_int)

        # Assemble full w with zeros at Z boundaries
        w = np.zeros_like(rhs)
        w[:, 1:-1] = w_int

        psi = g + w
        return psi.reshape(-1)


def Greens(Rc, Zc, R, Z, limit_threading=False):
    """
    Calculate poloidal flux at (R,Z) due to a single unit of current at
    (Rc,Zc) using Greens function for the elliptic operator above. Greens
    function is given by:

        G(R, Z; Rc, Zc) = (μ0 / (2π)) * sqrt(R * Rc) * ((2 - k^2) * K(k^2) - 2 * E(k^2)) / k

    where:
     - k^2 = 4 R Rc / ((R + Rc)^2 + (Z - Zc)^2)
     - k = sqrt(k^2)

    and K(k^2) and E(k^2) are the complete elliptic integrals of the first
    and second kind.

    This function is multithreaded. Multithreading behavior can be controlled through the
    environment variable OMP_NUM_THREADS (preferred), NUMEXPR_NUM_THREADS or programatically
    through the use of the function set_num_threads() in freegs4e.parallel_funcs. Note that
    by default the number of threads is limited to 32 by numexpr: for use in HPC systems, it
    is recommended to set NUMEXPR_MAX_THREADS to the number of available cores.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    limit_threading: bool
        If True, forces SOME internal functions, with high threading overhead,
        to run single threaded. Multiple threads will still be used for low
        overhead functionalities.

    Returns
    -------
    float
        Value of the poloidal flux at (R,Z).
    """

    # TODO: the next version of numexpr should allow for cache_disabling, this could
    # help address memory issues.

    # calculate k^2
    k2 = ne.evaluate("4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)")

    # clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = threaded_clip(
        k2, 1e-10, 1.0 - 1e-10, out=k2, single_thread=limit_threading
    )

    # note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    eie, eik = threaded_elliptics_ek(k2, single_thread=limit_threading)

    res = ne.evaluate(
        "(mu0 / (2.0 * pi)) * sqrt(R * Rc) * ((2.0 - k2) * eik - 2.0 * eie) / sqrt(k2)",
    )

    return res


def GreensBz(Rc, Zc, R, Z, eps=1e-4):
    """
    Calculate vertical magnetic field at (R,Z) due to a single unit of current at
    (Rc,Zc) using Greens function for the elliptic operator above.

        Bz(R,Z) = (1/R) d psi/dR,

    where psi is found with the Greens function finite difference.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float
        Value of the vertical magnetic field at (R,Z) [T].
    """

    return (Greens(Rc, Zc, R + eps, Z) - Greens(Rc, Zc, R - eps, Z)) / (
        2.0 * eps * R
    )


def GreensBr(Rc, Zc, R, Z, eps=1e-4):
    """
    Calculate radial magnetic field at (R,Z) due to a single unit of current at
    (Rc,Zc) using Greens function for the elliptic operator above.

        Br(R,Z) = -(1/R) d psi/dZ,

    where psi is found with the Greens function finite difference.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float
        Value of the radial magnetic field at (R,Z) [T].
    """

    return (Greens(Rc, Zc, R, Z - eps) - Greens(Rc, Zc, R, Z + eps)) / (
        2.0 * eps * R
    )


def GreensdBzdr(Rc, Zc, R, Z, eps=2e-3):
    """
    Calculate radial derivative of vertical magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBz/dR (R,Z) = (Bz(R + eps, Z) - Bz(R - eps, Z))/ 2 * eps.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBz(Rc, Zc, R + eps, Z) - GreensBz(Rc, Zc, R - eps, Z)) / (
        2.0 * eps
    )


def GreensdBrdz(Rc, Zc, R, Z, eps=2e-3):
    """
    Calculate vertical derivative of radial magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBr/dZ (R,Z) = (Br(R, Z + eps) - Br(R, Z - eps))/ 2 * eps.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBr(Rc, Zc, R, Z + eps) - GreensBr(Rc, Zc, R, Z - eps)) / (
        2.0 * eps
    )
    # return GreensdBzdr(Rc, Zc, R, Z, eps)


def GreensdBzdz(Rc, Zc, R, Z, eps=2e-3):
    """
    Calculate vertical derivative of vertical magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBz/dZ (R,Z) = (Bz(R, Z + eps) - Bz(R, Z - eps))/ 2 * eps.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBz(Rc, Zc, R, Z + eps) - GreensBz(Rc, Zc, R, Z - eps)) / (
        2.0 * eps
    )


def GreensdBrdr(Rc, Zc, R, Z, eps=2e-3):
    """
    Calculate radial derivative of radial magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBr/dR (R,Z) = (Br(R + eps, Z) - Br(R + pes, Z))/ 2 * eps.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBr(Rc, Zc, R + eps, Z) - GreensBr(Rc, Zc, R - eps, Z)) / (
        2.0 * eps
    )


def GreensdBrdz(Rc, Zc, R, Z, eps=2e-3):
    """
    Calculate vertical derivative of radial magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBr/dZ (R,Z) = (Br(R, Z + eps) - Br(R, Z - eps))/ 2 * eps.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBr(Rc, Zc, R, Z + eps) - GreensBr(Rc, Zc, R, Z - eps)) / (
        2.0 * eps
    )


def GreensdBzdr(Rc, Zc, R, Z, eps=2e-3):
    """
    Calculate radial derivative of vertical magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBz/dR (R,Z) = (Bz(R + eps, Z) - Bz(R - eps, Z))/ 2 * eps.

    Parameters
    ----------
    Rc : float
        Radial position where current is located [m].
    Zc : float
        Vertical position where current is located [m].
    R : float
        Radial position where poloidal flux is to be calcualted [m].
    Z : float
        Vertical position where poloidal flux is to be calcualted [m].
    eps : float
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBz(Rc, Zc, R + eps, Z) - GreensBz(Rc, Zc, R - eps, Z)) / (
        2.0 * eps
    )
