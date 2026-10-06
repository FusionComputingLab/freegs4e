import io

import pytest
from numpy import allclose

import freegs4e


def test_readwrite():
    """Test HDF5 reading/writing independently of the legacy Picard solver."""

    for tokamak in [
        freegs4e.machine.TestTokamak(),
        freegs4e.machine.MAST_sym(),
    ]:

        eq = freegs4e.Equilibrium(
            tokamak=tokamak,
            Rmin=0.1,
            Rmax=2.0,
            Zmin=-1.0,
            Zmax=1.0,
            nx=17,
            ny=17,
            boundary=freegs4e.boundary.freeBoundaryHagenow,
        )
        memory_file = io.BytesIO()

        with freegs4e.OutputFile(memory_file, "w") as f:
            f.write_equilibrium(eq)

        with freegs4e.OutputFile(memory_file, "r") as f:
            read_eq = f.read_equilibrium()

        assert tokamak == read_eq.tokamak
        assert allclose(eq.psi(), read_eq.psi())


def test_original_readwrite_solve_setup_is_unsupported():
    """Record the unsupported standalone solve formerly hidden by CI."""
    eq = freegs4e.Equilibrium(
        tokamak=freegs4e.machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=17,
        ny=17,
        boundary=freegs4e.boundary.freeBoundaryHagenow,
    )
    profiles = freegs4e.jtor.ConstrainPaxisIp(1e4, 1e6, 2.0)
    constrain = freegs4e.control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)],
        isoflux=[(1.1, -0.6, 1.1, 0.6)],
    )

    with pytest.raises(ValueError, match="No opoints found"):
        freegs4e.solve(
            eq, profiles, constrain, maxits=25, atol=1e-3, rtol=1e-1
        )
