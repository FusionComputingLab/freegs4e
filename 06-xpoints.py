#!/usr/bin/env python
#
# Example demonstrating functions for creating and finding X-points

import matplotlib.pyplot as plt

import freegs4e

# Plotting routines
from freegs4e.plotting import plotCoils, plotConstraints, plotEquilibrium

tokamak = freegs4e.machine.TestTokamak()
eq = freegs4e.Equilibrium(tokamak=tokamak, nx=256, ny=256)

##########################################################
# Calculate currents in coils to create X-points
# in specified locations
#

xpoints = [(1.1, -0.8), (1.1, 0.8)]  # (R,Z) locations of X-points

control = freegs4e.control.constrain(xpoints=xpoints)
control(eq)  # Apply control to Equilibrium eq

psi = eq.psi()

print("=> Solved coil currents, created X-points")

ax = plotEquilibrium(eq, show=False)
plotCoils(tokamak.coils, axis=ax)
plotConstraints(control, axis=ax)
plt.show()

##########################################################
# Find critical points (O- and X-points)
#
#

import freegs4e.critical as critical

opt, xpt = critical.find_critical(eq.R, eq.Z, psi)

print("=> Found O- and X-points")

ax = plotEquilibrium(eq, show=False, oxpoints=False)
for r, z, _ in xpt:
    ax.plot(r, z, "ro")
for r, z, _ in opt:
    ax.plot(r, z, "go")
psi_bndry = xpt[0][2]
sep_contour = ax.contour(eq.R, eq.Z, psi, levels=[psi_bndry], colors="r")
plt.show()

##########################################################
# Create a mask array, 1 in the core and 0 outside
#
#

mask = critical.core_mask(eq.R, eq.Z, psi, opt, xpt)

print("=> Created X-point mask")

plt.contourf(eq.R, eq.Z, mask)
plt.show()
