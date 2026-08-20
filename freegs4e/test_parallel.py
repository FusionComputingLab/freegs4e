import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from . import parallel

# All tests are written under the assumption that the parallel functions are implemented as wrappers, and should give
# EXACTLY equivalent results to the serial versions


@pytest.fixture
def make_test_array():
    return np.random.rand(100, 100, 100)


def test_set_threads():

    num_threads = parallel.get_num_threads()
    parallel.set_num_threads(num_threads + 2)

    assert parallel.get_num_threads() == (num_threads + 2)

    parallel.set_num_threads(num_threads)


def test_take(make_test_array):

    idcs = np.random.randint(0, 100, size=(20, 3))
    reference = np.take(make_test_array, idcs)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs)
    ), "Failed on 2 threads"

    parallel.set_num_threads(61)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs)
    ), "Failed n_threads>idcs.shape[0]"

    parallel.set_num_threads(orig_num_threads)


def test_take_axis(make_test_array):

    idcs = np.random.randint(0, 100, size=(20, 3))
    reference = np.take(make_test_array, idcs, axis=1)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs, axis=1)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs, axis=1)
    ), "Failed on 2 threads"

    parallel.set_num_threads(21)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs, axis=1)
    ), "Failed n_threads>idcs.shape[0]"

    parallel.set_num_threads(orig_num_threads)


def test_clip(make_test_array):

    amin, amax = 0.25, 0.75
    reference = np.clip(make_test_array, amin, amax)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    assert np.all(
        reference == parallel.threaded_clip(make_test_array, amin, amax)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    assert np.all(
        reference == parallel.threaded_clip(make_test_array, amin, amax)
    ), "Failed on 2 threads"

    parallel.set_num_threads(101)
    assert np.all(
        reference == parallel.threaded_clip(make_test_array, amin, amax)
    ), "Failed n_threads>idcs.shape[0]"

    parallel.set_num_threads(orig_num_threads)


def test_elliptics(make_test_array):

    amin, amax = 0.25, 0.75
    refe = ellipe(make_test_array)
    refk = ellipk(make_test_array)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    rese, resk = parallel.threaded_elliptics_ek(make_test_array)
    assert np.all(
        (refe == rese) & (refk == resk)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    rese, resk = parallel.threaded_elliptics_ek(make_test_array)
    assert np.all((refe == rese) & (refk == resk)), "Failed on 2 threads"

    parallel.set_num_threads(101)
    rese, resk = parallel.threaded_elliptics_ek(make_test_array)
    assert np.all(
        (refe == rese) & (refk == resk)
    ), "Failed n_threads>idcs.shape[0]"

    parallel.set_num_threads(orig_num_threads)
