__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

from tike.operators.torch import Shift
import tike.random
import torch


def test_correct_shift(size=256):

    try:
        import libimage
        fov = (libimage.load('coins', size) +
               1j * libimage.load('earring', size))
    except ModuleNotFoundError:
        fov = tike.random.numpy_complex(
            size,
            size,
        )

    op = Shift()
    shifted = op(
        torch.tensor(fov),
        torch.tensor([4, 90]),
    )

    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(fov.real)
        plt.subplot(2, 1, 2)
        plt.imshow(fov.imag)
        plt.savefig('fov.png')
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(shifted.real)
        plt.subplot(2, 1, 2)
        plt.imshow(shifted.imag)
        plt.savefig('shifted.png')
    except ModuleNotFoundError:
        pass


def test_shift_simple():

    fov = torch.zeros((3, 9, 9))
    fov[:, 4, 4] = 1

    op = Shift()
    shifted = op(
        torch.tensor(fov),
        torch.tensor([
            [2, 5],
            [0, -3],
            [-1, 4],
        ]),
    )
    truth = torch.zeros((3, 9, 9))
    truth[0, (4 + 2) % 9, (4 + 5) % 9] = 1

    print(fov.dtype)
    print(fov)
    print()
    # print(truth.dtype)
    # print(truth)
    print()
    print(shifted.dtype)
    print(torch.round(shifted.real))

    # torch.testing.assert_close(shifted.real, truth)
