"""
Custom parallel library for use in FreeGS4E, based on Python threading for GIL
releasing operations. Wraps expensive numpy and scipy functions with parallel
implementations. Relies on numexpr for threadpool size control.

Copyright 2026 Tomas Rubio Cruz, STFC - Hartree Centre

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

import concurrent.futures
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numexpr as ne
import numpy as np
from numpy import clip, take
from scipy.special import ellipe, ellipk
from threadpoolctl import ThreadpoolController

try:
    # preferred path, preserves future numpy compatibility
    from numpy.lib.array_utils import normalize_axis_index
except ImportError:
    # kept for numpy 1.26 support
    from numpy.core.multiarray import normalize_axis_index

thread_controller = ThreadpoolController()


def get_num_threads():
    """
    Utility function to inquire the default number of threads used by functions in this
    parallel library.

    Returns
    -------
    int
        Number of threads
    """

    # for consistency in performance, always match the no. of threads used by numexpr
    return ne.get_num_threads()


def set_num_threads(num_threads):
    """
    Utility function to programatically set the default number of threads used by functions in
    this parallel library.

    Parameters
    ----------

    num_threads: int
        Number of threads
    """

    # for consistency in performance, always match the no. of threads used by numexpr
    ne.set_num_threads(num_threads)


@thread_controller.wrap(limits={"blas": 1, "openmp": 1})
def threaded_take(a, indices, axis=None, out=None, mode="raise"):

    num_threads = get_num_threads()

    if num_threads == 1:
        return np.take(a, indices, axis=axis, out=out, mode=mode)

    # perform checks on indices array

    if not isinstance(indices, np.ndarray):
        # we rely on numpy behavior for the parallelization
        raise TypeError("Only numpy ndarray indices are supported")

    elif not indices.shape or indices.shape[0] < num_threads:
        # don't parallelize if there are less indices than threads
        return np.take(a, indices, axis=axis, out=out, mode=mode)

    # determine the correct output shape

    outshape = None

    if axis is not None:
        inshape = a.shape
        idxshape = indices.shape

        if not axis < len(inshape):
            raise np.exceptions.AxisError(axis, len(inshape))

        outshape = list(inshape[:axis]) + list(idxshape)

        if axis not in (-1, len(inshape) - 1):
            outshape += list(inshape[axis + 1 :])

        outshape = tuple(outshape)

    else:
        outshape = indices.shape

    # perform checks on output array

    if out is None:
        # output array necessary for parallel implementation
        out = np.empty(outshape)
    elif not isinstance(out, np.ndarray):
        # we rely on numpy behavior for the parallelization
        raise TypeError("return arrays must be of ArrayType")
    elif out.shape != outshape:
        raise ValueError(
            f"Incorrect shape of output buffer: received {out.shape} but expected {outshape}"
        )

    # try to flatten arrays for better load balancing

    if axis is None:
        # no reshaping if axis (because shape would matter)

        if indices.flags.forc:
            # only reshape if indices and out are contiguous, otherwise no time savings
            try:
                out.resize(out.size)
                indices = indices.reshape(-1)
            except:
                warnings.warn(
                    "Output array has an abnormal data layout. This may affect performance"
                )
        else:
            warnings.warn(
                "Input array has an abnormal data layout. This may affect performance"
            )

    # launch threads

    with ThreadPoolExecutor(max_workers=num_threads) as executor:

        futures = []

        # length of dimension that will be decomposed
        main_len = indices.shape[0]

        step, rem = divmod(main_len, num_threads)
        end = 0

        for i in range(num_threads):

            start = end
            end = (
                start + step + (i + 1) * (i < rem)
            )  # first few slices get one more element to deal with remainder

            idcs_slice = indices[start:end]
            out_slice = out[start:end]

            # if non-zero axis is provided, output array needs to be sliced along said axis
            if axis:
                normalized_axis = normalize_axis_index(axis, ndim=a.ndim)
                idxr = (slice(None),) * normalized_axis + (slice(start, end),)
                out_slice = out[idxr]

            futures.append(
                executor.submit(
                    take, a, idcs_slice, axis=axis, out=out_slice, mode=mode
                )
            )

        # Threads don't raise exceptions unless joined explicitly. This is a low-overhead way of doing that
        tuple(
            f.result()
            for f in concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            ).done
        )

    # reshape if flattened
    if out.shape != outshape:
        # there should only be a shape mismatch if we resized alread, so it should be safe to do again
        out.resize(outshape)

    return out


@thread_controller.wrap(limits={"blas": 1, "openmp": 1})
def threaded_elliptics_ek(k2, out=None, single_thread=False):
    """
    Parallel wrapper for both scipy.special.ellipe() and scipy.special.ellipk(). Behavior of
    these functions can be consulted on scipy docs. `out` parameter is not supported and is
    ignored.

    On 2 threads, both integrals are simply calculated simultaneously. For larger threadpools,
    k2 is divided into slices and each thread calculates one of the integrals on its
    corresponding slice. If an odd number of threads was set, the next lower even number is
    used.

    Parameters
    ----------
    k2 : ndarray
        The parameter of the elliptic integral
    out: Any, optional
        Unused. Only for compatibility (matching signatures) with wrapped functions.
    single_thread: bool
        If True, the function is run in serial regarding of the pre-set number of threads

    Returns
    -------
    ndarray
        Value of the elliptic integral
    """

    # The wrapper enssures that BLAS/OpenMP threads will not be spawned by scipy as this
    # could cause oversubscription issues.

    if out:
        warnings.warn("out argument in threaded_elliptics_ek is ignored")

    num_threads_total = get_num_threads()

    if single_thread or num_threads_total == 1:
        return ellipe(k2), ellipk(k2)

    if not isinstance(k2, np.ndarray):
        # we rely on numpy behavior for the parallelization
        raise TypeError("Only numpy ndarrays are supported")

    # operating on a flattened view is slightly better for load balancing

    inshape = k2.shape

    if k2.flags.forc:
        # only reshape if k2 is contiguous, otherwise no time savings
        k2 = k2.reshape(-1)
    else:
        warnings.warn(
            "Input array has an abnormal data layout. This may affect performance"
        )

    num_threads = num_threads_total // 2

    # If there aren't enough elements to parallelize, don't
    if not k2.shape or k2.shape[0] < num_threads:
        k2 = k2.reshape(inshape)
        return ellipe(k2), ellipk(k2)

    # output arrays
    eie = np.empty(k2.shape)
    eik = np.empty(k2.shape)

    with ThreadPoolExecutor(max_workers=num_threads_total) as executor:

        futures = []

        main_len = k2.shape[0]  # length of dimension that will be decomposed
        step, rem = divmod(main_len, num_threads)
        end = 0

        for i in range(num_threads):

            start = end
            end = (
                start + step + (i + 1) * (i < rem)
            )  # first few slices get one more element to deal with remainder

            k2_slice = k2[start:end]
            futures.append(
                executor.submit(ellipe, k2_slice, out=eie[start:end])
            )
            futures.append(
                executor.submit(ellipk, k2_slice, out=eik[start:end])
            )

        # Threads don't raise exceptions unless joined explicitly. This is a low-overhead way of doing that
        tuple(
            f.result()
            for f in concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            ).done
        )

    eie.resize(inshape)
    eik.resize(inshape)

    return eie, eik


@thread_controller.wrap(limits={"blas": 1, "openmp": 1})
def threaded_clip(
    k2, /, amin, amax, *, out=None, single_thread=False, **kwargs
):
    """
    Parallel wrapper for numpy clip. Detailed behavior of the function can be consulted on
    numpy docs. This function only adds the argument `single_thread` (read below).

    k2 is divided into slices and each thread clips its corresponding slice.

    Parameters
    ----------
    k2 : ndarray
        Array containing the elements to clip
    a_min, a_max : array_like or None
        Minimum and maximum value. If ``None``, clipping is not performed on
        the corresponding edge. If both ``a_min`` and ``a_max`` are ``None``,
        the elements of the returned array stay the same. Both are broadcasted
        against ``a``.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.
    single_thread: bool
        If True, the function is run in serial regarding of the pre-set number of threads
    **kwargs:
        As per numpy docs. Note that the ufunc argument `where` is not currently supported
        and is ignored.
    Returns
    -------
    ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.
    """

    # The wrapper enssures that BLAS/OpenMP threads will not be spawned by scipy as this
    # could cause oversubscription issues.

    num_threads = get_num_threads()

    if single_thread or num_threads == 1:
        return clip(k2, amin, amax, out=out, **kwargs)

    # Manually handle edge cases introduced by parallel implementation

    if out is None:
        # output array necessary for parallel implementation
        out = np.empty(k2.shape)
    elif not isinstance(out, np.ndarray):
        # we rely on numpy behavior for the parallelization
        raise TypeError("return arrays must be of ArrayType")
    elif not isinstance(k2, np.ndarray):
        # we rely on numpy behavior for the parallelization
        raise TypeError("Only numpy ndarrays are supported")
    else:
        # need to check shape compatibility BEFORE slicing the arrays
        np.broadcast(k2, out)  # raises ValueError if not broadcastable

    if not (kwargs.pop("where", None) is None):
        # no support for `where` because: a) it is not used in freegs4e,
        # b) I haven't been able to identify how exceptions would be managed
        warnings.warn(
            "Argument `where` of numpy ufuncs not supported by threaded_clip. Ignored."
        )

    # Perform resizing when needed and for efficiency

    inshape = k2.shape
    outshape = out.shape

    # operating on a flattened view is slightly better for load balancing
    if k2.flags.forc:
        # only reshape if k2 is contiguous, otherwise no time savings
        k2 = k2.reshape(-1)
    else:
        warnings.warn(
            "Input array has an abnormal data layout. This may affect performance"
        )

    # If there aren't enough elements to parallelize, don't
    # It is important for this to happen BEFORE resizing out
    if not k2.shape or k2.shape[0] < num_threads:
        k2 = k2.reshape(inshape)
        return clip(k2, amin, amax, out=out, **kwargs)

    # prepare output array
    try:
        # parallel implementation relies on being able to get a reshaped VIEW of out
        out.resize(k2.shape)
    except:
        warnings.warn(
            "clip could not be performed in parallel due to abnormal data layout of output array"
        )
        return clip(k2, amin, amax, out=out, **kwargs)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:

        futures = []

        main_len = k2.shape[0]  # length of dimension that will be decomposed
        step, rem = divmod(main_len, num_threads)
        end = 0

        for i in range(num_threads):

            start = end
            end = (
                start + step + (i + 1) * (i < rem)
            )  # first few slices get one more element to deal with remainder

            k2_slice = k2[start:end]
            futures.append(
                executor.submit(clip, k2_slice, amin, amax, out=out[start:end])
            )

        # Threads don't raise exceptions unless joined explicitly. This is a low-overhead way of doing that
        tuple(
            f.result()
            for f in concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            ).done
        )

    out.resize(outshape)

    return out


class ThreadManagedRegion:
    """
    EXPERIMENTAL. Defines a context manager to set a specific number of threads for a region
    of code. Carries large overheads.
    """

    context_depth = 0  # helps keep track of nested managed regions

    def __init__(self, num_threads):

        self.preset_threads = get_num_threads()

        if isinstance(num_threads, int) and num_threads > 0:
            self.context_threads = num_threads
        elif num_threads == "default":
            self.context_threads = self.preset_threads
        elif num_threads == "max":
            num_avail = len(
                os.sched_getaffinity(0)
            )  # TODO: from Python 3.13, process_cpu_count() preferred
            self.context_threads = num_avail
        else:
            raise TypeError(
                "Invalid number of threads '{}'. Should be an integer >1, 'default' or 'max'".format(
                    num_threads
                )
            )

    def __enter__(self):
        ThreadManagedRegion.context_depth += 1
        if context_depth == 1:
            set_num_threads(self.context_threads)

    def __exit__(self, *_):
        if context_depth == 1:
            set_num_threads(self.preset_threads)
        ThreadManagedRegion.context_depth -= 1


class SingleThreadedRegion(ThreadManagedRegion):
    """
    EXPERIMENTAL. Defines a context manager that enforces single threaded execution in a region
    of code.
    """

    def __init__(self):
        super().__init__(1)
