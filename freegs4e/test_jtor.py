import numpy as np

from . import jtor


def test_psinorm_range():
    """Test that the profiles produce finite values outside core"""

    for profiles in [
        jtor.ConstrainPaxisIp(1e3, 2e5, 2.0),
        jtor.ConstrainBetapIp(1.0, 2e5, 2.0),
    ]:

        # Need to give a plasma psi
        R, Z = np.meshgrid(
            np.linspace(0.5, 1.5, 33), np.linspace(-1, 1, 33), indexing="ij"
        )
        psi = np.exp((-((R - 1.0) ** 2) - Z**2) * 3) + np.exp(
            (-((R - 1.0) ** 2) - (Z + 1) ** 2) * 3
        )

        current_density = profiles.Jtor(R, Z, psi)
        assert np.all(np.isfinite(current_density))

        assert profiles.pprime(1.0) == 0.0
        assert profiles.pprime(1.1) == 0.0
        assert np.isfinite(profiles.pprime(-0.32))

        assert profiles.ffprime(1.0) == 0.0
        assert profiles.ffprime(1.1) == 0.0
        assert np.isfinite(profiles.ffprime(-0.32))


def test_profile_jtor_stores_plotting_metadata():
    """Test that the generic Jtor wrapper records downstream metadata."""

    profiles = jtor.ConstrainBetapIp(1.0, 2e5, 2.0)
    R, Z = np.meshgrid(
        np.linspace(0.5, 1.5, 33), np.linspace(-1, 1, 33), indexing="ij"
    )
    psi = np.exp((-((R - 1.0) ** 2) - Z**2) * 3) + np.exp(
        (-((R - 1.0) ** 2) - (Z + 1) ** 2) * 3
    )

    profiles.Jtor(R, Z, psi)

    assert hasattr(profiles, "opt")
    assert hasattr(profiles, "xpt")
    assert hasattr(profiles, "flag_limiter")
    assert hasattr(profiles, "psi_axis")
    assert hasattr(profiles, "psi_bndry")
    assert hasattr(profiles, "diverted_core_mask")
    assert hasattr(profiles, "limiter_core_mask")
