"""
Defines class that represents the equilibrium state and its associated
quantities.

Modified substantially from the original FreeGS code.

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

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

import warnings

import matplotlib.pyplot as plt
import numpy as np
import shapely as sh
from numpy import array, exp, linspace, meshgrid, pi
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid, romb
from scipy.spatial.distance import pdist, squareform

from . import critical, machine, multigrid, polygons  # multigrid solver
from .boundary import fixedBoundary, freeBoundary  # finds free-boundary
from .gradshafranov import (  # operators which define the G-S equation
    GSsparse,
    GSsparse4thOrder,
    mu0,
)


class Equilibrium:
    """
    Represents the plasma equilibrium state and its associated properties.

    """

    def __init__(
        self,
        tokamak=machine.EmptyTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=freeBoundary,
        psi=None,
        current=0.0,
        order=4,
    ):
        """
        Initializes a plasma equilibrium.

        Parameters
        ----------
        tokamak : machine.Machine
            The set of active coils, passive structures, limiter, and magnetic probes in the tokamak.
        Rmin : float
            Minimum major radius [m].
        Rmax : float
            Maximum major radius [m].
        Zmin : float
            Minimum height [m].
        Zmax : float
            Maximum height [m].
        nx : int
            Number of radial grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        ny : int
            Number of vertical grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        boundary : callable
            The boundary condition function. Can be either `freeBoundary`
            or `fixedBoundary`.
        psi : np.array
            Initial guess for plasma flux [Webers/2pi]. If `None`, default initial guess used.
        current : float
            Plasma current [A].
        order : int
            Order of differential operators used in calculations.
            Must be either 2 or 4.
        """

        # assign tokamak object
        self.tokamak = tokamak

        # assign bounds of computational domain
        if Rmin > Rmax:
            raise ValueError("Rmin must be smaller than Rmax.")
        if Zmin > Zmax:
            raise ValueError("Zmin must be smaller than Zmax.")
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

        # assign number of grid points
        if nx < 0 or ny < 0:
            raise ValueError("nx and ny must be integers greater than zero.")
        self.nx = nx
        self.ny = ny

        # define 1D and 2D radial/vertical grids
        self.R_1D = linspace(Rmin, Rmax, nx)
        self.Z_1D = linspace(Zmin, Zmax, ny)
        self.R, self.Z = meshgrid(self.R_1D, self.Z_1D, indexing="ij")

        # assign grid sizes
        self.dR = self.R[1, 0] - self.R[0, 0]
        self.dZ = self.Z[0, 1] - self.Z[0, 0]

        # assign boundary function
        self._applyBoundary = boundary

        # assign initial guess for plasma flux (if None)
        if psi is None:
            # Starting guess for psi
            psi = self.create_psi_plasma_default()
            self.gpars = np.array([0.5, 0.5, 0, 2])

        if psi.shape != (nx, ny):
            raise ValueError("Shape of psi must match grid size (nx, ny).")
        self.plasma_psi = psi

        # generate Greens function mappings (used
        # in self.psi() to speed up calculations)
        self._pgreen = tokamak.createPsiGreens(self.R, self.Z)
        # self._updatePlasmaPsi(psi)  # Needs to be after _pgreen

        # assign plasma current
        self._current = current

        # deinfe the GS solver
        if order == 2:
            generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        elif order == 4:
            generator = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
        else:
            raise ValueError(
                "Invalid choice of order ({}). Valid values are 2 or 4.".format(
                    order
                )
            )
        self.order = order

        self._solver = multigrid.createVcycle(
            nx, ny, generator, nlevels=1, ncycle=1, niter=2, direct=True
        )

        # separatrix data not yet calculated
        self._separatrix_data_flag = False

    def create_psi_plasma_default(
        self, adaptive_centre=False, gpars=(0.5, 0.5, 0, 2)
    ):
        """
        Generate a Gaussian-shaped initial guess for the plasma flux.

        The flux function ψ(R, Z) is given by:

            ψ(R, Z) = exp(C - (|x - xc|^p + |y - yc|^p))

        where (xc, yc) is the center, C is a shift, and p is a power term.
        The function ensures the flux is zero on the boundaries.

        Parameters
        ----------
        adaptive_centre : bool, optional
            If True, the plasma core position (xc, yc) is adjusted dynamically.
            If False (default), it remains fixed.
        gpars : tuple (xc, yc, C, p)
            Parameters defining the flux shape:
            - xc, yc : float
                Coordinates of the flux center.
            - C : float
                Shift parameter in the exponent.
            - p : float
                Power of the distance term.

        Returns
        -------
        np.array
            The computed flux values on the given (R, Z) grid.
        """

        # define unit meshgrid
        xx, yy = meshgrid(
            linspace(0, 1, self.nx), linspace(0, 1, self.ny), indexing="ij"
        )

        # finds approximate (weighted) centre of the core plasma
        if adaptive_centre == True:
            ntot = np.sum(self.mask_inside_limiter)
            xc = (
                np.sum(
                    self.mask_inside_limiter
                    * linspace(0, 1, self.nx)[:, np.newaxis]
                )
                / ntot
            )
            yc = (
                np.sum(
                    self.mask_inside_limiter
                    * linspace(0, 1, self.ny)[np.newaxis, :]
                )
                / ntot
            )
        # else sets centre according to gpars
        else:
            xc, yc = gpars[:2]

        # generate the plasma flux
        psi = exp(
            gpars[2]
            - ((np.abs(xx - xc)) ** gpars[3] + (np.abs(yy - yc)) ** gpars[3])
        )

        # set zero flux on boundary
        psi[0, :] = 0.0
        psi[:, 0] = 0.0
        psi[-1, :] = 0.0
        psi[:, -1] = 0.0

        return psi

    def setSolverVcycle(self, nlevels=1, ncycle=1, niter=1, direct=True):
        """
        Sets a new linear solver based on the multigrid scheme.

        This method configures a multigrid V-cycle solver and assigns it to
        `self._solver`.

        Parameters
        ----------
        nlevels : int
            Number of resolution levels, including the original.
        ncycle : int
            Number of V-cycles to use.
        niter : int
            Number of linear solver (Jacobi) iterations per level.
        direct : bool
            If True, uses a direct solver at the coarsest level.

        Returns
        -------
        None
            This function modifies `self._solver` but does not return a value.
        """

        # set the solver
        self._solver = multigrid.createVcycle(
            nx=self.nx,
            ny=self.ny,
            generator=GSsparse(self.Rmin, self.Rmax, self.Zmin, self.Zmax),
            nlevels=nlevels,
            ncycle=ncycle,
            niter=niter,
            direct=direct,
        )

    def setSolver(self, solver):
        """
        Sets the linear solver to use. The given object/function must have a __call__ method
        which takes two inputs:

        solver(x, b)

        where x is the initial guess and b is the right hand side (this should solve Ax = b,
        returning the result).


        Parameters
        ----------
        solver : object
            The solver object.

        Returns
        -------
        None
            This function modifies `self._solver` but does not return a value.
        """

        self._solver = solver

    def callSolver(self, psi, rhs):
        """
        Calls the solver.

        Parameters
        ----------
        psi : np.array
            The initial guess [Webers/2pi].
        rhs : np.array
            The right hand side of the GS equation.

        Returns
        -------
        object
            Returns modified `self._solver` object.
        """

        return self._solver(psi, rhs)

    def getMachine(self):
        """
        Machine object.

        Parameters
        ----------

        Returns
        -------
        object
            Returns the tokamak object.

        """

        return self.tokamak

    def plasmaCurrent(self):
        """
        Returns the plasma current [Amps].

        Parameters
        ----------

        Returns
        -------
        float
            Returns the total plasma current [Amps].

        """

        return self._current

    def plasmaVolume(self):
        """
        Plasma volume [m^3] calculated using:

            V = 2 * pi * integral(R) dR dZ,

        where the integral is evaluated over the plasma core domain.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the total plasma volume [m^3].
        """

        # volume element
        dV = 2.0 * pi * self.R * self.dR * self.dZ

        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate volume in 2D
        return romb(romb(dV))

    def plasmaBr(self, R, Z):
        """
        Radial magnetic field due to plasma:

            Br_p(R, Z) = -(1 / R) * dψ_p/dZ (R, Z),

        where ψ_p is the plasma flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The radial magnetic field due to the plasma at the given (R, Z) positions [T].
        """

        return -self.psi_func(R, Z, dy=1, grid=False) / R

    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field Bz due to plasma::

            Bz_p(R, Z) = (1 / R) * dψ/dR (R, Z),

        where ψ_p is the plasma flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The vertical magnetic field due to the plasma at the given (R, Z) positions [T].
        """

        return self.psi_func(R, Z, dx=1, grid=False) / R

    def Br(self, R, Z):
        """
        Total radial magnetic field due to plasma and coils:

            Br(R, Z) = -(1 / R) * dψ/dZ (R, Z),

        where ψ is the total flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total radial magnetic field at the given (R, Z) positions [T].
        """

        return self.plasmaBr(R, Z) + self.tokamak.Br(R, Z)

    def Bz(self, R, Z):
        """
        Total vertical magnetic field due to plasma and coils:

            Bz(R, Z) = (1 / R) * dψ/dR (R, Z),

        where ψ is the total flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total vertical magnetic field at the given (R, Z) positions [T].
        """

        return self.plasmaBz(R, Z) + self.tokamak.Bz(R, Z)

    def Btor(self, R, Z):
        """
        Total toroidal magnetic field:

            Btor(R, Z) = I(R, Z) * fpol(ψ_n) + (1 - I(R, Z)) * fvac,

        where:
            - I(R, Z) = 1 inside core plasma, 0 outside.
            - fpol = R*Bt at specified values of normalised psi ψ_n (i.e.
        inside the plasma core).
            - fvac = = R*Bt in vacuum (i.e. outside plasma).

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The toroidal magnetic field at the given (R, Z) positions [T].
        """

        # find f = R * Btor in the core
        fpol = self.fpol(self.psiNRZ(self.R, self.Z))

        try:
            # combine values in core plasma with vacuum field outside
            fpol = (
                fpol * self._profiles.limiter_core_mask
                + (1.0 - self._profiles.limiter_core_mask) * self.fvac()
            )
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # interpolate over the field
        func = interpolate.RectBivariateSpline(
            self.R_1D, self.Z_1D, fpol / self.R
        )

        # if inputs are scalars, return scalar
        if np.isscalar(R) and np.isscalar(Z):
            return func(R, Z)[0, 0]
        elif R.ndim == 1 and Z.ndim == 1:
            return func(R.flatten(), Z.flatten(), grid=False)
        elif R.ndim == 2 and Z.ndim == 2:
            return func(R, Z, grid=False)
        else:
            raise ValueError("Unexpected input shape for R and Z.")

    def Bpol(self, R, Z):
        """
        Total poloidal magnetic field due to plasma and coils:

            Bpol(R, Z) = ( Br(R, Z)^2 + Bz(R, Z)^2 )^(1/2).

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The poloidal magnetic field at the given (R, Z) positions [T].
        """

        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)

        return np.sqrt((Br * Br) + (Bz * Bz))

    def psi(self):
        """
        Total poloidal flux due to plasma and coils on the
        entire (R,Z) grid:

            ψ(R, Z) = ψ_p (R, Z) + ψ_c (R, Z),

        where:
         -  ψ_p (R, Z) is the plasma flux.
         -  ψ_c (R, Z) is the coil flux.

        Parameters
        ----------

        Returns
        -------
        np.array
            The total poloidal flux due to the plasma and the coils [Webers/2pi].
        """

        return self.plasma_psi + self.tokamak.calcPsiFromGreens(self._pgreen)

    def psiRZ(self, R, Z):
        """
        Total poloidal flux due to plasma and coils (at chosen R and Z):

            ψ(R, Z) = ψ_p (R, Z) + ψ_c (R, Z),

        where:
         -  ψ_p (R, Z) is the plasma flux.
         -  ψ_c (R, Z) is the coil flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total poloidal flux due to the plasma and the coils at chosen (R,Z) locations [Webers/2pi].
        """

        return self.psi_func(R, Z, grid=False) + self.tokamak.psi(R, Z)

    def psiNRZ(self, R, Z):
        """
        Total poloidal flux due to plasma and coils (at chosen R and Z), normalised
        with respect to the flux on the magnetic axis and last closed flux surface:

            ψ_n(R, Z) = (ψ(R, Z) - ψ_a)/(ψ_b - ψ_a),

        where:
         -  ψ_a is the total flux on the magnetic axis.
         -  ψ_b is the total flux on the last closed flux surface.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total normalised poloidal flux due to the plasma and the coils at chosen (R,Z) locations.
        """

        return (self.psiRZ(R, Z) - self.psi_axis) / (
            self.psi_bndry - self.psi_axis
        )

    def psi_1D(self, N):
        """
        Total poloidal flux between magnetic axis and last closed flux surface (at
        equally spaced locations).

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The total poloidal flux due to the plasma and the coils between the magnetic axis
            and last closed flux surface [Webers/2pi].
        """

        return np.linspace(self.psi_axis, self.psi_bndry, N)

    def psiN_1D(self, N):
        """
        Total normalised poloidal flux between magnetic axis and last closed flux surface.

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The total normalised poloidal flux due to the plasma and the coils between the magnetic axis
            and last closed flux surface.
        """

        return np.linspace(0, 1, N)

    def rho_1D(
        self,
        N,
    ):
        """
        Toroidal flux ρ (at locations of normalised psi) such that:

            ρ = Φ(ψ) and

            Φ = - (1/(2 * pi)) integral(q(ψ)) dψ,

        where:
         - q is the safety factor profile as a function of ψ.

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The toroidal flux (at locations of normalised psi).
        """

        # calculate safety factor of the (normalised) flux surfaces
        qvals = self.q(self.psiN_1D(N))

        # compute the integral (cumulatively)
        integral = (-1.0 / (2.0 * np.pi)) * cumulative_trapezoid(
            qvals, self.psi_1D(N), initial=0.0
        )
        integral[0] = 0  # hard-code for floating point issues

        # convert to a scalar if only single output
        if len(integral) == 1:
            return integral[0]
        return integral

    def rhoN_1D(self, N):
        """
        Normalised toroidal flux ρ_n (at locations of normailsed psi) such that:

        ρ_n = sqrt( ρ / ρ_b ),

        where ρ_b is the toridal flux at the last closed flux surface.

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The normalised toroidal flux (at locations of normalised psi).
        """

        # calculate rho
        rho = self.rho_1D(N)

        return np.sqrt(rho / rho[-1])

    def fpol(self, psinorm):
        """
        Return f = R*Bt at specified values of normalised psi (i.e.
        inside the plasma core).

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            fpol at values of normalised psi [Tm].
        """

        return self._profiles.fpol(psinorm)

    def fvac(self):
        """
        Return vacuum f = R*Bt (i.e. outside the plasma core).

        Parameters
        ----------

        Returns
        -------
        float
            fvac value in vacuum [Tm].
        """

        return self._profiles.fvac()

    def q(self, psinorm):
        """
        Safety factor q at normalised flux values.

        Note that at both psinorm = 0 (magnetic axis) and psinorm = 1 (separatrix),
        calculating q is problematic (results may be inaccurate).

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            Safety factor q at values of normalised psi.
        """

        # print warning if needed
        if 0 in psinorm or 1 in psinorm:
            print(
                "psinorm contains 0 or 1, which may cause numerical issues druing safety factor calculation."
            )

        # find safety factor profile
        q = critical.find_safety(self, psinorm=psinorm)

        # convert to a scalar if only single output
        if len(q) == 1:
            return np.asscalar(q)

        return q

    def pprime(self, psinorm):
        """
        Return pprime (dp/dψ_n) at normalised flux values.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            pprime (dp/dψ_n) at normalised flux values [Pa/(Webers/2pi)].
        """

        return self._profiles.pprime(psinorm)

    def ffprime(self, psinorm):
        """
        Return ffprime (F * dF/dψ_n) at normalised flux values.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            ffprime (F * dF/dψ_n) at normalised flux values [T^2 m^2/(Wb/2pi)].
        """

        return self._profiles.ffprime(psinorm)

    def pressure(self, psinorm):
        """
        Return pressure (p(ψ_n)) at normalised flux values.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            pressure (p(ψ_n)) at normalised flux values [Pa].
        """

        return self._profiles.pressure(psinorm)

    def separatrix(self, ntheta=360, stdev=4):
        """
        Return (ntheta x 2) array of (R,Z) points on the last closed
        flux surface (plasma boundary), equally spaced in the geometric
        poloidal angle.

        Sometimes there may be spurious points far from the core plasma (due to
        open field lines), we exclude these by setting an average threshold distance
        between all the points (excluding those outside the threshold).

        Parameters
        ----------
        ntheta : int
            Number of points on the boundary to return.
        stdev : float
            Number of standard deviations after which outliers are excluded.

        Returns
        -------
        np.array
            (R,Z) points on the last closed flux surface (plasma boundary).
        """

        # initial points on core boundary
        points = np.array(critical.find_separatrix(self, ntheta=ntheta))[
            :, 0:2
        ]

        # compute pairwise distances using pdist
        dist_matrix = squareform(pdist(points))  # convert to square form
        mean_distances = np.mean(
            dist_matrix, axis=1
        )  # mean distance for each point

        # define a threshold (median + n * standard deviations)
        threshold = np.median(mean_distances) + stdev * np.quantile(
            dist_matrix.reshape(-1), q=0.8
        )

        # filter points
        filtered_points = points[mean_distances < threshold]

        return filtered_points

    def separatrix_area(self):
        """
        The area of the last closed flux surface [m^2].

        Parameters
        ----------

        Returns
        -------
        float
            Area of the last closed flux surface (plasma boundary) [m^2].
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return self._sep_area

    def separatrix_length(self):
        """
        The circumference of the last closed flux surface [m].

        Parameters
        ----------

        Returns
        -------
        float
            Circumference of the last closed flux surface (plasma boundary) [m].
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return self._sep_length

    def getForces(self):
        """
        Calculates the forces on the coils.

        Parameters
        ----------

        Returns
        -------
        dict
            Coil lables with the associated force [N].
        """

        return self.tokamak.getForces(self)

    def printForces(self):
        """
        Prints a table of forces on coils.

        Parameters
        ----------

        Returns
        -------
        None
            Table of forces on coils.
        """

        print("Forces on coils")

        def print_forces(forces, prefix=""):
            for label, force in forces.items():
                if isinstance(force, dict):
                    print(prefix + label + " (circuit)")
                    print_forces(force, prefix=prefix + "  ")
                else:
                    print(
                        prefix
                        + label
                        + " : R = {0:.2f} kN , Z = {1:.2f} kN".format(
                            force[0] * 1e-3, force[1] * 1e-3
                        )
                    )

        print_forces(self.getForces())

    def innerOuterSeparatrix(
        self,
        Z: float = 0.0,
    ):
        """
        Locate inboard and outboard R co-ordinates of last closed flux surface
        at a given Z position.

        Parameters
        ----------
        Z : float, optional
            The Z value at which to find the radii. Defaults to
            0.0.

        Returns
        -------
        float
            The inner separatrix major radius (at chosen Z) [m].
        float
            The outer separatrix major radius (at chosen Z) [m].
        """

        # Find the closest index to requested Z
        Zindex = np.argmin(abs(self.Z[0, :] - Z))

        # Normalise psi at this Z index
        psinorm = (self.psi()[:, Zindex] - self.psi_axis) / (
            self.psi_bndry - self.psi_axis
        )

        # Start from the magnetic axis
        Rindex_axis = np.argmin(abs(self.R[:, 0] - self.Rmagnetic()))

        # Inner separatrix
        # Get the maximum index where psi > 1 in the R index range from 0 to Rindex_axis
        outside_inds = np.argwhere(psinorm[:Rindex_axis] > 1.0)

        if outside_inds.size == 0:
            R_in = self.Rmin
        else:
            Rindex_inner = np.amax(outside_inds)

            # Separatrix should now be between Rindex_inner and Rindex_inner+1
            # Linear interpolation
            R_in = (
                self.R[Rindex_inner, Zindex]
                * (1.0 - psinorm[Rindex_inner + 1])
                + self.R[Rindex_inner + 1, Zindex]
                * (psinorm[Rindex_inner] - 1.0)
            ) / (psinorm[Rindex_inner] - psinorm[Rindex_inner + 1])

        # Outer separatrix
        # Find the minimum index where psi > 1
        outside_inds = np.argwhere(psinorm[Rindex_axis:] > 1.0)

        if outside_inds.size == 0:
            R_out = self.Rmax
        else:
            Rindex_outer = np.amin(outside_inds) + Rindex_axis

            # Separatrix should now be between Rindex_outer-1 and Rindex_outer
            R_out = (
                self.R[Rindex_outer, Zindex]
                * (1.0 - psinorm[Rindex_outer - 1])
                + self.R[Rindex_outer - 1, Zindex]
                * (psinorm[Rindex_outer] - 1.0)
            ) / (psinorm[Rindex_outer] - psinorm[Rindex_outer - 1])

        return R_in, R_out

    def innerOuterSeparatrix2(
        self,
        Z=0.0,
    ):
        """
        Alternative (simpler) function to find inboard and outboard R co-ordinates of
        last closed flux surface at any given Z position (using shapely).

        Parameters
        ----------
        Z : float, optional
            The Z value at which to find the radii. Defaults to
            0.0.

        Returns
        -------
        list
            List with the two (R,Z) coordinates: (R_in, Z) and (R_out, Z) [m].
        """

        # use shapely to define polygon of core plasma
        plasma_polygon = sh.LinearRing(self.separatrix(ntheta=360))

        # use shapely to define horizonal line at Z
        Z_line = sh.LineString([[0, Z], [100, Z]])

        # find intersection
        intersection = plasma_polygon.intersection(Z_line)

        # extract coordinates
        if intersection.is_empty:
            print("No intersection found.")
            intersection_coords = []
        elif intersection.geom_type == "Point":
            intersection_coords = [intersection.coords[0]]  # Single point
        elif intersection.geom_type == "MultiPoint":
            intersection_coords = [
                pt.coords[0] for pt in intersection.geoms
            ]  # Multiple points
        else:
            raise ValueError(
                f"Unexpected intersection type: {intersection.geom_type}"
            )

        return intersection_coords

    def intersectsWall(self):
        """
        Assess whether or not the core plasma touches the vessel
        walls.

        Parameters
        ----------

        Returns
        -------
        bool
            True if they interect, False if not.
        """

        # generate list of points for each
        separatrix = self.separatrix()
        wall = self.tokamak.wall

        return polygons.intersect(
            separatrix[:, 0], separatrix[:, 1], wall.R, wall.Z
        )

    def magneticAxis(self):
        """
        Returns the location of the magnetic axis and its total poloidal flux value.

        Parameters
        ----------

        Returns
        -------
        list
            Returns [R, Z, ψ(R, Z)] at magnetic axis location  [m, m, Webers/2pi].
        """

        # find critical points
        opts, xpts = critical.find_critical(self.R, self.Z, self.psi())

        # find closest O-point to the geometric axis
        geom_axis = self.geometricAxis()[0:2]
        o_point_ind = np.argmin(
            np.sum((opts[:, 0:2] - geom_axis) ** 2, axis=1)
        )

        return opts[o_point_ind, :]

    def Rcurrent(self):
        """
        The average major radius of the toroidal current distribution

        Parameters
        ----------

        Returns
        -------
        float
            average major radius [m].
        """

        if hasattr(self, "_profiles"):
            meanR = np.sum(self.R * self._profiles.jtor) / np.sum(
                self._profiles.jtor
            )
            return meanR
        else:
            print(
                "The equilibrium is not a GS solution. Please solve the eq. first."
            )

    def Zcurrent(self):
        """
        The average height of the toroidal current distribution

        Parameters
        ----------

        Returns
        -------
        float
            average height [m].
        """

        if hasattr(self, "_profiles"):
            meanZ = np.sum(self.Z * self._profiles.jtor) / np.sum(
                self._profiles.jtor
            )
            return meanZ
        else:
            print(
                "The equilibrium is not a GS solution. Please solve the eq. first."
            )

    def Rmagnetic(self):
        """
        The major radius of the magnetic axis.

        Parameters
        ----------

        Returns
        -------
        float
            Major radius of the magnetic axis [m].
        """

        return self.magneticAxis()[0]

    def Zmagnetic(self):
        """
        The height of the magnetic axis.

        Parameters
        ----------

        Returns
        -------
        float
            Vertical height of the magnetic axis [m].
        """

        return self.magneticAxis()[1]

    def geometricAxis(self):
        """
        The geometric axis (R_geom, Z_geom) of the plasma [m]. Calculated as
        the centre of a large number of points on the separatrix.

        Parameters
        ----------

        Returns
        -------
        np.array
            Geometric axis (R,Z) position [m].
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return np.array(
            [
                (self._sep_Rmax + self._sep_Rmin) / 2,
                (self._sep_Zmax + self._sep_Zmin) / 2,
            ]
        )

    def Rgeometric(self):
        """
        The major radius of the geometric axis.

        Parameters
        ----------

        Returns
        -------
        float
            Major radius of the geometric axis [m].
        """

        return self.geometricAxis()[0]

    def Zgeometric(self):
        """
        The height of the geometric axis.

        Parameters
        ----------

        Returns
        -------
        float
            Vertical height of the geometric axis [m].
        """

        return self.geometricAxis()[1]

    def minorRadius(self):
        """
        Calculate the minor radius of the plasma [m].

        Parameters
        ----------

        Returns
        -------
        float
            Minor radius of the plasma [m].
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return (self._sep_Rmax - self._sep_Rmin) / 2

    def aspectRatio(self):
        """
        Calculate the aspect ratio of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Aspect ratio of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return (self._sep_Rmax + self._sep_Rmin) / (
            self._sep_Rmax - self._sep_Rmin
        )

    def geometricElongation(self):
        """
        Calculate the geometric elongation of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Geometric elongation of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return (self._sep_Zmax - self._sep_Zmin) / (
            self._sep_Rmax - self._sep_Rmin
        )

    def geometricElongation_upper(self):
        """
        Calculate the geometric elongation of the upper part of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Geometric elongation of the upper part of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return (
            2
            * (self._sep_Zmax - self._sep_ZRmax)
            / (self._sep_Rmax - self._sep_Rmin)
        )

    def geometricElongation_lower(self):
        """
        Calculate the geometric elongation of the lower part of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Geometric elongation of the lower part of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        return (
            2
            * (self._sep_ZRmax - self._sep_Zmin)
            / (self._sep_Rmax - self._sep_Rmin)
        )

    def effectiveElongation(self):
        """
        Calculate the effective elongation of the plasma, using plasma volume.

        Parameters
        ----------

        Returns
        -------
        float
            Effective elongation of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return self.plasmaVolume() / (2.0 * R_geom * (np.pi**2) * (R_minor**2))

    def shafranov_shift(self):
        """
        Calculate the Shafranov shift of the plasma (i.e. the shift between the magnetic
        axis and the geometrix axis).

        Parameters
        ----------

        Returns
        -------
        float
            Shafranov shift of the plasma [del_R, del_Z] [m].
        """

        mag_axis = self.magneticAxis()
        geom_axis = self.geometricAxis()

        return np.array(
            [mag_axis[0] - geom_axis[0], mag_axis[1] - geom_axis[1]]
        )

    def triangularity_upper(self):
        """
        Calculate the upper triangularity of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Upper triangularity of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return (R_geom - self._sep_RZmin) / R_minor

    def triangularity_lower(self):
        """
        Calculate the lower triangularity of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Lower triangularity of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return (R_geom - self._sep_RZmax) / R_minor

    def triangularity(self):
        """
        Calculate the triangularity of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Triangularity of the plasma.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return (R_geom - (self._sep_RZmax + self._sep_RZmin) / 2) / R_minor

    def squareness(self):
        """
        This function calculates the squareness of the plasma core for each of the four quadrants:
        upper outer, upper lower, lower outer, and lower inner. Method outlined in:

        An analytic functional form for characterization and generation of axisymmetric plasma
        boundaries. Plasma Physics and Controlled Fusion. 2013. T C Luce.

        Parameters
        ----------

        Returns
        -------
        float
            Upper outer squareness.
        float
            Upper inner squareness.
        float
            Lower outer squareness.
        float
            Lower inner squareness.
        """

        # check if metrics are already calculated
        if self._separatrix_data_flag is False:
            self._separatrix_metrics()  # call function
            self._separatrix_data_flag = True  # update flag

        # create shapely object for plasma core
        plasma_boundary = sh.Polygon(self.separatrix())

        # same for "ideal" ellipses in each quadrant
        uo_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmax,
                Z0=self._sep_ZRmax,
                A=self._sep_Rmax - self._sep_RZmax,
                B=self._sep_Zmax - self._sep_ZRmax,
            )
        )
        ui_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmax,
                Z0=self._sep_ZRmax,
                A=self._sep_RZmax - self._sep_Rmin,
                B=self._sep_Zmax - self._sep_ZRmax,
            )
        )
        lo_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmin,
                Z0=self._sep_ZRmax,
                A=self._sep_Rmax - self._sep_RZmin,
                B=self._sep_ZRmax - self._sep_Zmin,
            )
        )
        li_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmin,
                Z0=self._sep_ZRmax,
                A=self._sep_RZmin - self._sep_Rmin,
                B=self._sep_ZRmax - self._sep_Zmin,
            )
        )

        # bounding box diagonals
        uo_diag = sh.LineString(
            [
                [self._sep_RZmax, self._sep_ZRmax],
                [self._sep_Rmax, self._sep_Zmax],
            ]
        )
        ui_diag = sh.LineString(
            [
                [self._sep_RZmax, self._sep_ZRmax],
                [self._sep_Rmin, self._sep_Zmax],
            ]
        )
        lo_diag = sh.LineString(
            [
                [self._sep_RZmin, self._sep_ZRmax],
                [self._sep_Rmax, self._sep_Zmin],
            ]
        )
        li_diag = sh.LineString(
            [
                [self._sep_RZmin, self._sep_ZRmax],
                [self._sep_Rmin, self._sep_Zmin],
            ]
        )

        # find intersecting line lengths
        uo_diag_core = uo_diag.intersection(plasma_boundary).length
        ui_diag_core = uo_diag.intersection(plasma_boundary).length
        lo_diag_core = uo_diag.intersection(plasma_boundary).length
        li_diag_core = uo_diag.intersection(plasma_boundary).length

        uo_diag_ellipse = uo_diag.intersection(uo_ellipse).length
        ui_diag_ellipse = uo_diag.intersection(ui_ellipse).length
        lo_diag_ellipse = uo_diag.intersection(lo_ellipse).length
        li_diag_ellipse = uo_diag.intersection(li_ellipse).length

        # calculate squarenesses
        s_uo = (uo_diag_core - uo_diag_ellipse) / (
            uo_diag.length - uo_diag_ellipse
        )
        s_ui = (ui_diag_core - ui_diag_ellipse) / (
            ui_diag.length - ui_diag_ellipse
        )
        s_lo = (lo_diag_core - lo_diag_ellipse) / (
            lo_diag.length - lo_diag_ellipse
        )
        s_li = (li_diag_core - li_diag_ellipse) / (
            li_diag.length - li_diag_ellipse
        )

        return s_uo, s_ui, s_lo, s_li

    def closest_wall_point(self):
        """
        This function calculates the (R,Z) point on the wall that is closest
        to the last closed flux surface boundary (and the corresponding distance).

        Parameters
        ----------

        Returns
        -------
        list
            Contains coordinates array and minimum distance to wall: [ (R,Z), min_distance ] [m,m,m].
        """

        # create shapely linestring objects
        plasma_boundary = sh.LineString(self.separatrix())
        wall = sh.LineString(
            np.array([self.tokamak.wall.R, self.tokamak.wall.Z]).T
        )

        # initialize variables
        closest_point = None
        min_distance = float("inf")

        # iterate over each point on the wall contour
        for point_coords in wall.coords:
            wall_point = sh.Point(point_coords)

            # calculate distance from wall point to plasma boundary
            initial_distance = plasma_boundary.distance(wall_point)

            if initial_distance < min_distance:
                # find the closest point on the plasma boundary
                candidate_closest_point = plasma_boundary.interpolate(
                    plasma_boundary.project(wall_point)
                )

                # recalculate the distance between the wall point and the closest point (due to interpolation)
                corrected_distance = wall_point.distance(
                    candidate_closest_point
                )

                if corrected_distance < min_distance:
                    min_distance = corrected_distance
                    closest_point = candidate_closest_point

        return [np.array(closest_point.coords[0]), min_distance]

    def internalInductance1(self):
        """
        Calculates li1 plasma internal inductance according to:

            li_1 = [(2 * integral(Bpol^2) dV) / (mu0 * Ip^2 * R_geom)] * [(1 + geom_elon^2)/(2 * eff_elon)],

        where:
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R_geom = radial coords of geometric axis.
         - geom_elon = geometric elongation.
         - eff_elon = effective elongation.

        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance (li1).
        """

        # extract Bpol squared
        B_pol_2 = self.Bpol(self.R, self.Z) ** 2

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral = romb(romb(B_pol_2 * dV))

        return (
            (2 * integral)
            / ((mu0 * self.plasmaCurrent()) ** 2 * self.Rgeometric())
        ) * (
            (1 + self.geometricElongation() ** 2)
            / (2.0 * self.effectiveElongation())
        )

    def internalInductance2(self):
        """
        Calculates li2 plasma internal inductance according to:

            li_2 = (2 * integral(Bpol^2) dV) / (mu0 * Ip^2 * R_mag),

        where:
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R_mag = radial coords of magnetic axis.


        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance (li2).
        """

        # extract Bpol squared
        B_pol_2 = self.Bpol(self.R, self.Z) ** 2

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral = romb(romb(B_pol_2 * dV))

        return (
            2
            * integral
            / ((mu0 * self.plasmaCurrent()) ** 2 * self.Rmagnetic())
        )

    def internalInductance3(self):
        """
        Calculates li3 plasma internal inductance according to:

            li_3 = (2 * integral(Bpol^2) dV) / (mu0 * Ip^2 * R_geom),


        where:
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R_geom = radial coords of geometric axis.

        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance (li3).
        """

        # extract Bpol squared
        B_pol_2 = self.Bpol(self.R, self.Z) ** 2

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral = romb(romb(B_pol_2 * dV))

        return (
            2
            * integral
            / ((mu0 * self.plasmaCurrent()) ** 2 * self.Rgeometric())
        )

    def poloidalBeta(self):
        """
        Calculates the poloidal beta from the following definition:

            betap = (8 * pi / (mu0 * Ip^2)) * integral(p) dR dZ,

        where:
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - p = plasma pressure (at each (R,Z)).

        Parameters
        ----------

        Returns
        -------
        float
            Returns the poloidal beta value.
        """

        # plasma pressure
        pressure = self.pressure(self.psiNRZ(self.R, self.Z))

        # mask with the core plasma
        try:
            pressure *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # calculate the poloidal beta by integrating pressure in 2D
        return (
            ((8.0 * pi) / mu0)
            * romb(romb(pressure))
            * self.dR
            * self.dZ
            / (self.plasmaCurrent() ** 2)
        )

    def poloidalBeta2(self):
        """
        Calculates an (alternative) poloidal beta from the following definition:

            betap = 2 * mu0 * integral(p) dV / integral(Bpol^2) dV,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the poloidal beta value.
        """

        # extract Bpol squared
        B_pol_2 = self.Bpol(self.R, self.Z) ** 2

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # plasma pressure
        pressure = self.pressure(self.psiNRZ(self.R, self.Z))

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = romb(romb(pressure * dV))
        field_integral_pol = romb(romb(B_pol_2 * dV))

        return 2 * mu0 * pressure_integral / field_integral_pol

    def toroidalBeta(self):
        """
        Calculates a toroidal beta from the following definition:

            betat = 2 * mu0 * integral(p) dV / integral(Btor^2) dV,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - Btor(R,Z) = toroidal magnetic field (at each (R,Z)).
         - dV = volume element.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the toroidal beta value.
        """

        # extract Btor squared
        B_tor_2 = self.Btor(self.R, self.Z) ** 2

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # plasma pressure
        pressure = self.pressure(self.psiNRZ(self.R, self.Z))

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = romb(romb(pressure * dV))

        # correct for errors in Btor and core masking
        np.nan_to_num(B_tor_2, copy=False)

        field_integral_tor = romb(romb(B_tor_2 * dV))

        return 2 * mu0 * pressure_integral / field_integral_tor

    def normalised_total_Beta(self):
        """
        Calculates the total beta from the following definition:

            normalised_total_Beta = ( (1 / poloidalBeta2) + (1/toroidalBeta) )^(-1).

        Parameters
        ----------

        Returns
        -------
        float
            Returns the normalised total beta value.

        """

        return 1.0 / (
            (1.0 / self.poloidalBeta2()) + (1.0 / self.toroidalBeta())
        )

    def strikepoints(
        self,
        quadrant="all",
        loc=None,
    ):
        """
        This function can be used to find the strikepoints of an equilibrium (i.e
        the points at which the psi_boundary contour intersect the wall.

        Parameters
        ----------
        quadrant: str
            Which strikepoints to return to the user, options are "all" (returns all points) or
            one of "lower left", "lower right", "upper left", or "upper right" quadrants (which
            returns point(s) in quadrant of poloidal plane with centre given by 'loc' (R,Z) pair).
        loc: tuple
            (R,Z) point at which to centre the quadrants choice [m].

        Returns
        -------
        np.array
            Returns an array of the (R,Z) strikepoints [m].

        """

        # find contour object for psi_boundary
        if self._profiles.flag_limiter:
            cs = plt.contour(
                self.R, self.Z, self.psi(), levels=[self._profiles.psi_bndry]
            )
        else:
            cs = plt.contour(
                self.R, self.Z, self.psi(), levels=[self._profiles.xpt[0][2]]
            )
        plt.close()  # this isn't the most elegant but we don't need the plot itself

        # for each item in the contour object there's a list of points in (r,z) (i.e. a line)
        psi_boundary_lines = []
        for i, item in enumerate(cs.allsegs[0]):
            psi_boundary_lines.append(item)

        # use the shapely package to find where each psi_boundary_line intersects the wall (or not)
        strikes = []
        wall = np.array([self.tokamak.wall.R, self.tokamak.wall.Z]).T
        curve1 = sh.LineString(wall)
        for j, line in enumerate(psi_boundary_lines):
            curve2 = sh.LineString(line)

            # find the intersection points
            intersection = curve2.intersection(curve1)

            # extract intersection points
            if intersection.geom_type == "Point":
                strikes.append(np.squeeze(np.array(intersection.xy).T))
            elif intersection.geom_type == "MultiPoint":
                strikes.append(
                    np.squeeze(
                        np.array([geom.xy for geom in intersection.geoms])
                    )
                )

        # check how many strikepoints
        if len(strikes) == 0:
            return None
        else:
            all_strikes = np.concatenate(strikes, axis=0)

        # which strikepoint(s) to return
        if quadrant == "all":
            return all_strikes
        else:
            if loc == None:
                raise ValueError(
                    f"Need to define quadrant centre point in 'loc' (R,Z)."
                )
            else:
                if quadrant == "lower left":
                    ind = np.where(
                        (all_strikes[:, 0] < loc[0])
                        & (all_strikes[:, 1] < loc[1])
                    )[0]
                elif quadrant == "lower right":
                    ind = np.where(
                        (all_strikes[:, 0] > loc[0])
                        & (all_strikes[:, 1] < loc[1])
                    )[0]
                elif quadrant == "upper left":
                    ind = np.where(
                        (all_strikes[:, 0] < loc[0])
                        & (all_strikes[:, 1] > loc[1])
                    )[0]
                elif quadrant == "upper right":
                    ind = np.where(
                        (all_strikes[:, 0] > loc[0])
                        & (all_strikes[:, 1] > loc[1])
                    )[0]
                else:
                    raise ValueError(
                        f"Unexpected quadrant: {quadrant}. Choose from 'all', "
                        f"'lower left', 'lower right', 'upper left', or 'upper right'."
                    )

        # return if quadrant value required
        if len(ind) == 0:
            return np.full(2, None)
        else:
            return np.array(
                [all_strikes[:, 0][ind], all_strikes[:, 1][ind]]
            ).T.squeeze()

    def solve(self, profiles, Jtor=None, psi=None, psi_bndry=None):
        """
        This is a legacy function that can be used to solve for the plasma equilibrium. It
        performs a linear Grad-Shafranov solve.

        Parameters
        ----------
        profiles: class
            Class containing the toroidal current density profile information.
        Jtor: np.array
            Toroidal current density field over (R,Z) [A/m^2].
        psi: np.array
            Total poloidal magnetic flux over (R,Z) [Webers/2pi].
        psi_bndry: float
            Total poloidal magnetic flux on the plasma boundary [Webers/2pi].

        Returns
        -------
        None
            Modifies the equilbirium class object in place.

        """

        # set profiles
        self._profiles = profiles

        # calculate toroidal current density if not given (with psi)
        if Jtor is None:
            if psi is None:
                psi = self.psi()
            Jtor = profiles.Jtor(self.R, self.Z, psi, psi_bndry=psi_bndry)

        # set plasma boundary
        # note that the Equilibrium is passed to the boundary function
        # since the boundary may need to run the GS solver (von Hagenow's method)
        self._applyBoundary(self, Jtor, self.plasma_psi)

        # right hand side of GS equation
        rhs = -mu0 * self.R * Jtor

        # copy boundary conditions
        rhs[0, :] = self.plasma_psi[0, :]
        rhs[:, 0] = self.plasma_psi[:, 0]
        rhs[-1, :] = self.plasma_psi[-1, :]
        rhs[:, -1] = self.plasma_psi[:, -1]

        # call elliptic solver
        plasma_psi = self._solver(self.plasma_psi, rhs)

        self._updatePlasmaPsi(plasma_psi)

        # update plasma current
        self._current = romb(romb(Jtor)) * self.dR * self.dZ

    def _updatePlasmaPsi(self, plasma_psi):
        """
        Sets the plasma psi data using spline interpoation coefficients.

        Parameters
        ----------
        plasma_psi: np.array
            Plasma poloidal magnetic flux over (R,Z) [Webers/2pi].

        Returns
        -------
        None
            Modifies plasma_psi, mask, psi_axis, and psi_bndry in the class
            object in place.

        """

        # set plasma psi
        self.plasma_psi = plasma_psi

        # update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], plasma_psi
        )

        # update locations of X-points, core mask, psi ranges.
        # Note that this may fail if there are no X-points, so it should not raise an error
        # Analyse the equilibrium, finding O- and X-points
        psi = self.psi()
        opt, xpt = critical.find_critical(self.R, self.Z, psi)
        self.psi_axis = opt[0][2]

        if len(xpt) > 0:
            self.psi_bndry = xpt[0][2]
            self.mask = critical.inside_mask(
                self.R, self.Z, psi, opt, xpt, self.mask_outside_limiter
            )

            # Use interpolation to find if a point is in the core.
            self.mask_func = interpolate.RectBivariateSpline(
                self.R[:, 0], self.Z[0, :], self.mask
            )
        elif self._applyBoundary is fixedBoundary:
            # No X-points, but using fixed boundary
            self.psi_bndry = psi[0, 0]  # Value of psi on the boundary
            self.mask = None  # All points are in the core region
        else:
            self.psi_bndry = None
            self.mask = None

    def plot(self, axis=None, show=True, oxpoints=True):
        """
        Plot the equilibrium.

        Parameters
        ----------
        axis: object
            Matplotlib axis object.
        show: bool
            Calls matplotlib.pyplot.show() before returning.
        oxpoints: bool
            Plot X points and O points.

        Returns
        -------
        axis
            Matplotlib axis object.

        """

        from .plotting import plotEquilibrium

        return plotEquilibrium(self, axis=axis, show=show, oxpoints=oxpoints)

    def _separatrix_metrics(self):
        """
        Function that returns min/max (R,Z) points on the last closed flux surface,
        its area, and its circumference.

        Depiction here: https://imas-data-dictionary.readthedocs.io/en/latest/_downloads/23716946c6f02da1817f55b2f453444d/DefinitionEqBoundary.svg

        Parameters
        ----------

        Returns
        -------
        None
            Modifies eq object in place.
        """

        # calculate core separatrix
        core_boundary = self.separatrix()

        # min/max values on core boundary
        Rmin = np.min(core_boundary[:, 0])
        Rmax = np.max(core_boundary[:, 0])
        Zmin = np.min(core_boundary[:, 1])
        Zmax = np.max(core_boundary[:, 1])

        # find corresponding indices for these
        ZRmin_arg = np.argmin(core_boundary[:, 0])
        ZRmax_arg = np.argmax(core_boundary[:, 0])
        RZmin_arg = np.argmin(core_boundary[:, 1])
        RZmax_arg = np.argmax(core_boundary[:, 1])

        # use indices to find corresponding coords
        ZRmin = core_boundary[ZRmin_arg, 1]
        ZRmax = core_boundary[ZRmax_arg, 1]
        RZmin = core_boundary[RZmin_arg, 0]
        RZmax = core_boundary[RZmax_arg, 0]

        # use shapely to define polygon of core plasma
        plasma_polygon = sh.Polygon(core_boundary)
        area = plasma_polygon.area
        length = plasma_polygon.length

        self._sep_Rmin = Rmin
        self._sep_Rmax = Rmax
        self._sep_Zmin = Zmin
        self._sep_Zmax = Zmax
        self._sep_ZRmin = ZRmin
        self._sep_ZRmax = ZRmax
        self._sep_RZmin = RZmin
        self._sep_RZmax = RZmax
        self._sep_area = area
        self._sep_length = length


def ellipse_points(R0, Z0, A, B, N=360):
    """
    This function generates the (R,Z) points of an ellipse:
    (x/A)**2 + (y/B)**2 = 1.

    Parameters
    ----------
    R0, Z0, A, B: floats
        Centre of ellipse (R0,Z0) and the ellipticity parameters A and B.
    N: int
        Number of points to generate on the ellipse.

    Returns
    -------
    np.ndarray
        (N x 2) numpy array of points on the ellipse.
    """

    theta = np.linspace(0, 2 * np.pi, N)
    R = R0 + A * np.cos(theta)
    Z = Z0 + B * np.sin(theta)

    return np.column_stack((R, Z))


if __name__ == "__main__":

    # Test the different spline interpolation routines

    import machine
    import matplotlib.pyplot as plt
    from numpy import ravel

    tokamak = machine.TestTokamak()

    Rmin = 0.1
    Rmax = 2.0

    eq = Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax)

    import constraints

    xpoints = [(1.2, -0.8), (1.2, 0.8)]
    constraints.xpointConstrain(eq, xpoints)

    psi = eq.psi()

    tck = interpolate.bisplrep(ravel(eq.R), ravel(eq.Z), ravel(psi))
    spline = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psi)
    f = interpolate.interp2d(eq.R[:, 0], eq.Z[0, :], psi, kind="cubic")

    plt.plot(eq.R[:, 10], psi[:, 10], "o")

    r = linspace(Rmin, Rmax, 1000)
    z = eq.Z[0, 10]
    plt.plot(r, f(r, z), label="f")

    plt.plot(r, spline(r, z), label="spline")

    plt.plot(r, interpolate.bisplev(r, z, tck), label="bisplev")

    plt.legend()
    plt.show()
