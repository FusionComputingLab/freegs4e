# FreeGS4E support scope

## Purpose

FreeGS4E is the LGPL-licensed equilibrium backend used by FreeGSNKE. It is a
separate distribution for licensing and release purposes, but its supported
behaviour is defined primarily by the needs of FreeGSNKE.

FreeGS4E was forked from FreeGS 0.6.1 and still contains inherited modules that
are not part of the supported FreeGSNKE workflow. Their presence must not be
interpreted as a claim that FreeGS4E is a drop-in replacement for FreeGS.

## Backend functionality used by FreeGSNKE

FreeGSNKE currently relies on the following FreeGS4E functionality, directly or
through inheritance:

- Grad-Shafranov operators and Green's functions in `gradshafranov`.
- The direct elliptic solver exposed through `multigrid.createVcycle`.
- Critical-point, separatrix, plasma-mask, and bilinear-interpolation helpers.
- `Machine`, `Circuit`, `Wall`, `Coil`, and `MultiCoil` machine primitives.
- The `Equilibrium` grid, field, and equilibrium-diagnostic implementation.
- The profile bases `ConstrainBetapIp`, `ConstrainPaxisIp`, `Fiesta_Topeol`,
  `Lao85`, `TensionSpline`, and `GeneralPprimeFFprime`.
- Plotting helpers used by FreeGSNKE for equilibria, constraints, and probes.

Changes to this functionality must be tested against both the FreeGS4E and
FreeGSNKE test suites.

## Known unsupported behaviour

The following inherited FreeGS workflows are known not to satisfy their
original tests on the current codebase:

- The standalone `freegs4e.solve` Picard inverse workflow using
  `freegs4e.control.constrain`. FreeGSNKE provides its own static and evolutive
  solvers and does not use this route.
- Calling the equilibrium-oriented `critical.find_critical` wrapper on a field
  containing an X-point but no O-point. The lower-level `scan_for_crit` routine
  can identify such an isolated X-point, but the wrapper requires an O-point to
  order and filter X-points relative to the magnetic axis. FreeGSNKE plasma
  equilibria require that magnetic axis.

These behaviours are asserted explicitly in the test suite rather than hidden
through CI deselection. This keeps the current contract visible and makes any
change subject to review.

## Retained but not validated for FreeGSNKE

The following inherited functionality is not used by FreeGSNKE and is not part
of its supported backend contract:

- Standalone Picard and original FreeGS control interfaces.
- Coil-position optimisation in `optimise` and `optimiser`.
- Field-line tracing in `fieldtracer`.
- HDF5 `OutputFile` persistence in `dump`.
- A-EQDSK and DivGeo file utilities.
- Hard-coded legacy machine constructors such as `DIIID`, `MAST`, `TCV`, and
  the old built-in MAST-U descriptions.

Some of this code has isolated inherited tests. Passing such a test demonstrates
that specific behaviour only; it does not make the module a supported
FreeGSNKE interface. G-EQDSK interoperability is treated separately because it
is used alongside FreeGSNKE and has dedicated round-trip tests.

## User guidance

- Use FreeGSNKE APIs and examples for equilibrium solves.
- Do not use the original FreeGS documentation as an API specification for
  FreeGS4E.
- Report backend issues in the FreeGSNKE issue tracker unless working directly
  on the FreeGS4E backend.
