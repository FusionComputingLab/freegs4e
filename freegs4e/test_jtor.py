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


def _continuity_test_profile(shape):
    profiles = jtor.ConstrainPaxisIp(1e3, 2e5, 2.0)
    profiles.mask_inside_limiter = np.ones(shape, dtype=bool)
    profiles.edge_mask = np.zeros(shape, dtype=bool)
    profiles.edge_mask[[0, -1], :] = True
    profiles.edge_mask[:, [0, -1]] = True
    profiles.diverted_core_mask = np.zeros(shape, dtype=bool)
    profiles.diverted_core_mask.flat[:16] = True
    return profiles


def test_alternative_xpoint_sets_matching_boundary(monkeypatch):
    """An accepted alternative X-point also supplies the returned boundary flux."""
    shape = (5, 5)
    R, Z = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    psi = np.ones(shape)
    opt = np.array([[2.0, 2.0, 1.0]])
    xpt = np.array([[2.0, 1.0, 0.9], [2.0, 3.0, 0.8]])
    primary_mask = np.zeros(shape, dtype=bool)
    primary_mask[2:4, 2:4] = True
    alternative_mask = np.zeros(shape, dtype=bool)
    alternative_mask[1:4, 1:4] = True

    monkeypatch.setattr(
        jtor.critical, "find_critical", lambda *args: (opt, xpt)
    )
    monkeypatch.setattr(
        jtor.critical,
        "inside_mask",
        lambda *args: (
            primary_mask if np.isclose(args[-1], 0.9) else alternative_mask
        ),
    )

    profiles = _continuity_test_profile(shape)
    _, selected_xpt, mask, psi_bndry = profiles.Jtor_part1(R, Z, psi)

    assert np.array_equal(selected_xpt, xpt[1:])
    assert np.isclose(psi_bndry, selected_xpt[0, 2])
    assert np.array_equal(mask, alternative_mask)


def test_alternative_xpoint_search_advances_after_edge_mask(monkeypatch):
    """An edge-touching alternative is skipped without stalling the search."""
    shape = (5, 5)
    R, Z = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    psi = np.ones(shape)
    opt = np.array([[2.0, 2.0, 1.0]])
    xpt = np.array([[2.0, 1.0, 0.9], [2.0, 3.0, 0.8], [3.0, 2.0, 0.7]])
    primary_mask = np.zeros(shape, dtype=bool)
    primary_mask[2:4, 2:4] = True
    edge_mask = np.ones(shape, dtype=bool)
    final_mask = np.zeros(shape, dtype=bool)
    final_mask[1:4, 1:4] = True
    boundaries = []

    def inside_mask(*args):
        psi_bndry = args[-1]
        boundaries.append(psi_bndry)
        if np.isclose(psi_bndry, 0.9):
            return primary_mask
        if np.isclose(psi_bndry, 0.8):
            return edge_mask
        return final_mask

    monkeypatch.setattr(
        jtor.critical, "find_critical", lambda *args: (opt, xpt)
    )
    monkeypatch.setattr(jtor.critical, "inside_mask", inside_mask)

    profiles = _continuity_test_profile(shape)
    _, selected_xpt, mask, psi_bndry = profiles.Jtor_part1(R, Z, psi)

    assert np.allclose(boundaries, [0.9, 0.8, 0.7])
    assert np.array_equal(selected_xpt, xpt[2:])
    assert np.isclose(psi_bndry, selected_xpt[0, 2])
    assert np.array_equal(mask, final_mask)
