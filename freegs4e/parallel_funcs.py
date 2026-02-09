import threading

import numpy as np
from scipy.special import ellipe, ellipk


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


def threaded_elliptics_ek(k2, num_threads_total):

    if num_threads_total == 1:
        return ellipe(k2), ellipk(k2)

    threads_e = []
    threads_k = []

    num_threads = (
        num_threads_total // 2
    )  # no. of threads assigned to each function

    main_len = k2.shape[0]
    step = main_len // num_threads

    for i in range(num_threads):

        start = i * step
        end = start + step
        end = (
            end if i != num_threads - 1 else main_len
        )  # last slice gets the remainder

        threads_e.append(ReturnThread(target=ellipe, args=(k2[start:end],)))
        threads_k.append(ReturnThread(target=ellipk, args=(k2[start:end],)))

    for te, tk in zip(threads_e, threads_k):
        te.start()
        tk.start()

    eie = np.zeros_like(k2)
    eik = np.zeros_like(k2)

    for i in range(num_threads):

        start = i * step
        end = start + step
        end = (
            end if i != num_threads - 1 else main_len
        )  # last slice gets the remainder

        eie[start:end] = threads_e[i].join()
        eik[start:end] = threads_k[i].join()

    return eie, eik
