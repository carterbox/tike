"""Defines a worker Pool for multi-device managerment."""

__author__ = "Daniel Ching, Xiaodong Yu"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

from concurrent.futures import ThreadPoolExecutor
import typing
import warnings

import cupy as cp
import numpy as np


class _Device():

    def __init__(self, x) -> None:
        pass

    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        pass


np.asnumpy = np.asarray


class ThreadPool(ThreadPoolExecutor):
    """Python thread pool plus scatter gather methods.

    A Pool is a context manager which provides access to and communications
    amongst workers.

    Attributes
    ----------
    workers : int, tuple(int)
        The number of GPUs to use or a tuple of the device numbers of the GPUs
        to use. If the number of GPUs is less than the requested number, only
        workers for the available GPUs are allocated.
    device_count : int
        The total number of devices on the host as reported by CUDA runtime.
    num_workers : int
        Returns len(self.workers). For convenience.

    Raises
    ------
    ValueError
        When invalid GPU device ids are provided.
        When the current CUDA device does not match the first GPU id in the
        list of workers.
    """

    Device = cp.cuda.Device

    def __init__(
        self,
        workers,
        xp=cp,
        device_count=None,
    ):
        self.device_count = cp.cuda.runtime.getDeviceCount(
        ) if device_count is None else device_count
        if type(workers) is int:
            if workers < 1:
                raise ValueError(f"Provide workers > 0, not {workers}.")
            if workers > self.device_count:
                warnings.warn(
                    "Not enough CUDA devices for workers!"
                    f" Requested {workers} of {self.device_count} devices.")
                workers = min(workers, self.device_count)
            if workers == 1:
                # Respect "with cp.cuda.Device()" blocks for single thread
                workers = (cp.cuda.Device().id,)
            else:
                workers = tuple(range(workers))
        for w in workers:
            if w < 0 or w >= self.device_count:
                raise ValueError(f'{w} is not a valid GPU device number.')
        self.workers = workers
        self.xp = xp
        super().__init__(self.num_workers)

    def __enter__(self):
        if self.workers[0] != cp.cuda.Device().id:
            raise ValueError(
                "The primary worker must be the current device. "
                f"Use `with cupy.cuda.Device({self.workers[0]}):` to set the "
                "current device.")
        return self

    @property
    def num_workers(self):
        return len(self.workers)

    def _copy_to(
        self,
        x: typing.Union[cp.array, np.array],
        worker: int,
    ) -> cp.array:
        with self.Device(worker):
            return self.xp.asarray(x)

    def _copy_host(
        self,
        x: cp.array,
        worker: int,
    ) -> np.array:
        with self.Device(worker):
            return self.xp.asnumpy(x)

    def bcast(
        self,
        x: typing.Union[cp.array, np.array],
        stride: int = 1,
    ) -> typing.List[cp.array]:
        """Send each x to all device groups.

        Parameters
        ----------
        x : list
            A list of data to be broadcast.
        stride : int > 0
            The stride of the broadcast. e.g. stride=2 and num_gpu=8, then x[0]
            will be broadcast to workers[::2] while x[1] will go to
            workers[1::2].

        """

        def f(worker):
            idx = self.workers.index(worker) % stride
            return self._copy_to(x[idx], worker)

        return list(self.map(f, self.workers))

    def gather(
        self,
        x: typing.List[cp.array],
        worker: int = None,
        axis: int = 0,
    ) -> cp.array:
        """Concatenate x on a single worker along the given axis."""
        if self.num_workers == 1:
            return x[0]
        worker = self.workers[0] if worker is None else worker
        with self.Device(worker):
            return self.xp.concatenate(
                [self._copy_to(part, worker) for part in x],
                axis,
            )

    def gather_host(
        self,
        x: typing.List[cp.array],
        axis: int = 0,
    ) -> np.array:
        """Concatenate x on host along the given axis."""
        if self.num_workers == 1:
            return cp.asnumpy(x[0])

        def f(x, worker):
            return self._copy_host(x, worker)

        return np.concatenate(
            self.map(f, x, self.workers),
            axis,
        )

    def all_gather(
        self,
        x: typing.List[cp.array],
        axis: int = 0,
    ) -> typing.List[cp.array]:
        """Concatenate x on all workers along the given axis."""

        def f(worker):
            return self.gather(x, worker, axis)

        return list(self.map(f, self.workers))

    def scatter(
        self,
        x: typing.List[cp.array],
        stride: int = 1,
    ) -> typing.List[cp.array]:
        """Scatter each x with given stride.

        scatter_bcast(x=[0, 1], stride=3) -> [0, 0, 0, 1, 1, 1]

        Same as scatter_bcast, but with a different communication pattern. In
        this function, array are copied from their initial devices.

        Parameters
        ----------
        x : list
            Chunks to be sent to other devices.
        stride : int
            The size of a device group. e.g. stride=4 and num_gpu=8, then
            x[0] will be broadcast to workers[:4] while x[1] will go to
            workers[4:].

        """

        def f(worker):
            idx = self.workers.index(worker) // stride
            return self._copy_to(x[idx], worker)

        return list(self.map(f, self.workers))

    def scatter_bcast(
        self,
        x: typing.List[cp.array],
        stride: int = 1,
    ) -> typing.List[cp.array]:
        """Scatter each x with given stride and then broadcast nearby.

        scatter_bcast(x=[0, 1], stride=3) -> [0, 0, 0, 1, 1, 1]

        Same as scatter, but with a different communication pattern. In this
        function, arrays are first copied to a device in each group, then
        copied from that device locally.

        Parameters
        ----------
        x : list
            Chunks to be sent and copied.
        stride : int
            The stride length of the scatter. e.g. stride=4 and num_gpu=8, then
            x[0] will be broadcast to workers[:4] while x[1] will go to
            workers[4:].

        """

        def s(bworkers, chunk):

            def b(worker):
                return self._copy_to(chunk, worker)

            return list(self.map(b, bworkers, workers=bworkers))

        bworkers = []
        if stride == 1:
            sworkers = self.workers[:len(x)]
            for i in range(len(x)):
                bworkers.append(self.workers[i::len(x)])
        else:
            sworkers = self.workers[::stride]
            for i in sworkers:
                bworkers.append(self.workers[i:(i + stride)])

        a = self.map(s, bworkers, x, workers=sworkers)
        output = [None] * self.num_workers
        i, j = 0, 0
        for si in bworkers:
            for bi in si:
                output[bi] = a[i][j]
                j += 1
            i += 1
            j = 0

        return output

    def reduce_gpu(
        self,
        x: typing.List[cp.array],
        stride: int = 1,
        workers: typing.List[int] | None = None,
    ) -> typing.List[cp.array]:
        """Reduce x by addition to a device group from all other devices.

        reduce_gpu([0, 1, 2, 3, 4], stride=2) -> [6, 4]

        Parameters
        ----------
        x : list
            Chunks to be reduced to a device group.
        stride : int
            The stride of the reduction. e.g. stride=2 and num_gpu=8, then
            x[0::2] will be reduced to workers[0] while x[1::2] will be reduced
            to workers[1].

        """
        if self.num_workers == 1:
            return x

        def f(worker):
            i = self.workers.index(worker)
            x1 = 0
            for part in x[i::stride]:
                x1 += self._copy_to(part, worker)
            return x1

        workers = self.workers[:stride] if workers is None else workers
        return self.map(f, workers, workers=workers)

    def reduce_cpu(self, x: typing.List[cp.array]) -> np.array:
        """Reduce x by addition from all GPUs to a CPU buffer."""
        assert len(x) <= self.num_workers, (
            f"{len(x)} work is more than {self.num_workers} workers")
        return np.sum(self.map(self._copy_host, x, self.workers), axis=0)

    def reduce_mean(
        self,
        x: typing.List[cp.array],
        axis: int | typing.List[int],
        worker: int | None = None,
    ) -> cp.array:
        """Reduce x by addition to one GPU from all other GPUs."""
        if self.num_workers == 1:
            return x[0]
        worker = self.workers[0] if worker is None else worker
        return cp.mean(
            self.gather(x, worker=worker, axis=axis),
            keepdims=True,
            axis=axis,
        )

    def allreduce(
        self,
        x: typing.List[cp.array],
        stride: int | None = None,
    ) -> typing.List[cp.array]:
        """All-reduce x by addition within device groups.

        allreduce([0, 1, 2, 3, 4, 5, 6], stride=2) -> [1, 1, 5, 5, 9, 9, 6]

        Parameters
        ----------
        x : list
            Chunks to be all-reduced in grouped devices context.
        stride : int
            The size of a device group. e.g. s=4 and num_gpu=8, then x[:4] will
            perform all-reduce within workers[:4] while x[4:] will perform
            all-reduce within workers[4:].

        """
        if self.num_workers == 1:
            return x

        stride = len(x) if stride is None else stride
        assert stride >= 1, "Stride cannot be less than 1; it is {stride}."

        def f(worker):
            group_start = stride * (self.workers.index(worker) // stride)
            buff = 0
            for i in range(group_start, min(group_start + stride, len(x))):
                buff += self._copy_to(x[i], worker)
            return buff

        return list(self.map(
            f,
            self.workers,
        ))

    def map(
        self,
        func,
        *iterables,
        workers: typing.List[int] | None = None,
        **kwargs,
    ) -> list:
        """ThreadPoolExecutor.map, but wraps call in a cuda.Device context."""

        def f(worker, *args):
            with self.Device(worker):
                return func(*args, **kwargs)

        workers = self.workers if workers is None else workers

        # return list(super().map(f, workers, *iterables))
        return list(map(f, workers, *iterables))
