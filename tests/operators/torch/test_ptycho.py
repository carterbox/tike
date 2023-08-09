import numpy as np
import torch

import tike.ptycho
import tike.operators.torch
import tike.ptycho.probe


def test_consistent_simulate_torch(size=256):
    """Check ptycho.simulate for consistency."""

    try:
        import libimage
        import tike.precision
        fov = (libimage.load('coins', size) +
               1j * libimage.load('earring', size))
    except ModuleNotFoundError:
        import tike.random
        fov = tike.random.numpy_complex(
            size,
            size,
        )
    fov = fov.astype(tike.precision.cfloating)

    width = 64
    nscan = 5
    original = fov
    probe = 0.1 * tike.random.numpy_complex(1, 1, 1, width, width) + 1
    probe *= tike.ptycho.probe.gaussian(width, rin=0.45, rout=0.7)
    scan = (np.random.rand(nscan, 2) * (256 - 2 - width) + 1).astype('float32')
    # scan[:, 1] = 64
    # scan[:, 0] = 128.123456

    # noninteger = np.remainder(scan, 1)
    # integer = (scan - noninteger)
    # scan = integer

    data = tike.ptycho.simulate(
        detector_shape=probe.shape[-1],
        probe=probe,
        scan=scan,
        psi=original,
    )
    assert data.dtype == 'float32', data.dtype
    assert data.shape == (nscan, width, width)

    operator = tike.operators.torch.Ptycho()
    data1 = operator(
        psi=torch.tensor(original),
        scan=torch.tensor(scan),
        probe=torch.tensor(probe),
    ).detach().cpu().numpy()
    assert data1.dtype == 'float32', data1.dtype
    assert data1.shape == (nscan, width, width)

    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure(dpi=600)
        plt.subplot(3, 1, 1)
        plt.imshow(
            np.log(np.concatenate(data, axis=1)),
                        vmin=0,
            # vmax=1,
        )
        plt.title('linear')
        plt.colorbar()
        plt.subplot(3, 1, 2)
        plt.imshow(
            np.log(np.concatenate(data1, axis=1)),
            vmin=0,
            # vmax=1,
        )
        plt.title('Fourier')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        plt.imshow(
            np.concatenate(data1 - data, axis=1),
        )
        plt.colorbar()
        plt.savefig('simulate-torch.png')
    except ModuleNotFoundError:
        pass

    np.testing.assert_array_equal(data1.shape, data.shape)
    np.testing.assert_allclose(
        np.sqrt(data1),
        np.sqrt(data),
        atol=1e-6,
    )
