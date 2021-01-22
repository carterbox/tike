import tike.random
from tike.operators import Ptycho
import tike.ptycho
import cupy as cp
import libimage
import cupy as xp

import logging

import matplotlib.pyplot as plt
import tike.view

logger = logging.getLogger(__name__)


def update_farplane(op, farplane, data_):

    intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
    # rescale = cp.sum(cp.sqrt(data_ * intensity)) / cp.sum(intensity)
    # probe *= rescale
    # common_probe *= rescale
    # unique_probe *= rescale
    # intensity *= rescale * rescale
    # logger.info("object and probe rescaled by %f", rescale)

    cost = op.propagation.cost(data_, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane -= op.propagation.grad(data_, farplane, intensity)

    if __debug__:
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        cost = op.propagation.cost(data_, intensity)
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        # TODO: Only compute cost every 20 iterations or on a log sampling?

    return farplane


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    op = Ptycho(512, 512, 1, 1)

    data = libimage.load('coins', 512) + 1j * libimage.load('satyre', 512)
    data = cp.array(data, dtype='complex64')[None, ...]
    data = xp.square(xp.abs(data))
    plt.figure()
    tike.view.plot_complex(data[0].get())
    plt.show()

    farplane = tike.random.cupy_complex(*data.shape).astype('complex64')
    farplane = farplane[None, None, None]

    print(data.shape, farplane.shape)

    for i in range(10):
        farplane = update_farplane(op, farplane, data)
        plt.figure()
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        tike.view.plot_complex(intensity[0, 0].get())
        plt.show()
