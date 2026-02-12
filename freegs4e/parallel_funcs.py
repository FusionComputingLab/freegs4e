import concurrent.futures
import os
import threading

import numexpr as ne
import numpy as np
from numpy import clip
from scipy.special import ellipe, ellipk
from threadpoolctl import ThreadpoolController

# TODO: unify/centralize thread control
thread_controller = ThreadpoolController()


def get_num_threads():
    # for consistency in performance, always match the no. of threads used by numexpr
    return ne.get_num_threads()


def set_num_threads(num_threads):
    # for consistency in performance, always match the no. of threads used by numexpr
    ne.set_num_threads(num_threads)


class ReturnThread(threading.Thread):
    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs={},
        verbose=None,
    ):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    # Overriding the Thread.run function
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


@thread_controller.wrap(limits={"blas": 1, "openmp": 1})
def threaded_elliptics_ek(k2):

    # The wrapper prevents BLAS/OpenMP threads from being spawned by scipy (which is not expected to happen anyway), as this could cause oversubscription issues.

    # TODO: try to simplify slicing logic, see if using threadpool from concurrent.futures if feasible

    num_threads_total = get_num_threads()

    if num_threads_total == 1:
        return ellipe(k2), ellipk(k2)

    num_threads = num_threads_total // 2
    main_len = k2.shape[0]
    step = main_len // num_threads

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads_total
    ) as executor:

        eie = np.empty(k2.shape)
        eik = np.empty(k2.shape)

        for i in range(num_threads):

            start = i * step
            end = start + step
            end = (
                end if i != num_threads - 1 else main_len
            )  # last slice gets the remainder

            k2_slice = k2[start:end]
            executor.submit(ellipe, k2_slice, out=eie[start:end])
            executor.submit(ellipk, k2_slice, out=eik[start:end])

    return eie, eik


@thread_controller.wrap(limits={"blas": 1, "openmp": 1})
def threaded_clip(k2, amin, amax, out=None, **kwargs):

    # The wrapper prevents BLAS/OpenMP threads from being spawned by scipy (which is not expected to happen anyway), as this could cause oversubscription issues.

    # TODO: try to simplify slicing logic, see if using threadpool from concurrent.futures if feasible

    num_threads = get_num_threads()

    # Have to manually check for some np.clip exceptions that would affect / be affected by the slicing procedure in the parallel case
    if num_threads == 1:
        return clip(k2, amin, amax, out=out, **kwargs)
    elif not isinstance(out, np.ndarray):
        raise TypeError("return arrays must be of ArrayType")
    else:
        np.broadcast(k2, out)  # raises ValueError if not broadcastable

    main_len = k2.shape[0]
    step = main_len // num_threads

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:

        eie = np.empty(k2.shape)
        eik = np.empty(k2.shape)

        for i in range(num_threads):

            start = i * step
            end = start + step
            end = (
                end if i != num_threads - 1 else main_len
            )  # last slice gets the remainder

            k2_slice = k2[start:end]
            out_slice = out[start:end]
            executor.submit(clip, k2_slice, amin, amax, out=out_slice)

    return k2
