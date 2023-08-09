__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import torch
import numpy as np

from tike.operators.torch import OpticallyThinDiffraction
import tike.random
import tike.precision
import tike.ptycho


def test_thin_diffraction(size=256):
    """Load a dataset for reconstruction."""

    nscan = 27
    probe_shape = (32, 32)
    nprobe = 2

    scan = np.random.rand(nscan, 2) * (size - probe_shape[0] - 1)

    unique_kernels = tike.random.numpy_complex(
        nscan,
        nprobe,
        *probe_shape,
    )

    shared_kernels = np.ones(
        (1, 1, *probe_shape),
        dtype=tike.precision.cfloating,
    )
    shared_kernels = tike.ptycho.probe.add_modes_cartesian_hermite(
        shared_kernels, 5)
    shared_kernels.imag = 0

    try:
        import libimage
        fov = (libimage.load('coins', size) +
               1j * libimage.load('earring', size))
    except ModuleNotFoundError:
        fov = tike.random.numpy_complex(
            size,
            size,
        )

    operator = OpticallyThinDiffraction()

    patches = operator(
        torch.tensor(fov),
        torch.tensor(scan),
        torch.tensor(shared_kernels),
    )
    print(patches.shape)

    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        sample = torch.cat([p for p in patches[0]], dim=1)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(sample.real)
        plt.subplot(2, 1, 2)
        plt.imshow(sample.imag)
        plt.savefig('optically-thin-shared.png')
    except ModuleNotFoundError:
        pass

    patches = operator(
        torch.tensor(fov),
        torch.tensor(scan),
        torch.tensor(unique_kernels),
    )
    print(patches.shape)
