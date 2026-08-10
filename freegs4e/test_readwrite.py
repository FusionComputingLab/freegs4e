import io

import pytest
from numpy import allclose

import freegs4e


@pytest.mark.xfail  # TODO: test has been failing for some time now
def test_readwrite():
    """Test reading/writing to a file round-trip"""

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
        profiles = freegs4e.jtor.ConstrainPaxisIp(1e4, 1e6, 2.0)

        # Note here the X-point locations and isoflux locations are not the same.
        # The result will be an unbalanced double null configuration, where the
        # X-points are on different flux surfaces.
        xpoints = [(1.1, -0.6), (1.1, 0.8)]
        isoflux = [(1.1, -0.6, 1.1, 0.6)]
        constrain = freegs4e.control.constrain(
            xpoints=xpoints, isoflux=isoflux
        )

        freegs4e.solve(
            eq, profiles, constrain, maxits=25, atol=1e-3, rtol=1e-1
        )

        memory_file = io.BytesIO()

        with freegs4e.OutputFile(memory_file, "w") as f:
            f.write_equilibrium(eq)

        with freegs4e.OutputFile(memory_file, "r") as f:
            read_eq = f.read_equilibrium()

        assert tokamak == read_eq.tokamak
        assert allclose(eq.psi(), read_eq.psi())
