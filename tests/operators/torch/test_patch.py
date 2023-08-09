#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import torch
import numpy as np

import tike.linalg
from tike.operators.torch import Patch
import tike.precision
import tike.random

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def test_patch_correctness(size=256, win=8):

    try:
        import libimage
        fov = (libimage.load('coins', size) +
               1j * libimage.load('earring', size))
    except ModuleNotFoundError:
        fov = tike.random.numpy_complex(
            size,
            size,
        )

    subpixel = 0.0
    positions = np.array(
        [
            [0, 0],
            [0, size - win],
            [size - win, 0],
            [size - win, size - win],
            [size // 2 - win // 2, size // 2 - win // 2],
            [subpixel, 3],
        ],
        dtype=np.intc,
    )
    truth = np.stack(
        (
            fov[:win, :win],
            fov[:win, -win:],
            fov[-win:, :win],
            fov[-win:, -win:],
            fov[size // 2 - win // 2:size // 2 - win // 2 + win,
                size // 2 - win // 2:size // 2 - win // 2 + win],
            (1.0 - subpixel) * fov[0:win, 3:3 + win] +
            subpixel * fov[1:1 + win, 3:3 + win],
        ),
        axis=0,
    )
    op = Patch()
    patches = op(
        images=torch.tensor(fov),
        positions=torch.tensor(positions),
        widths=(win, win),
    )

    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(fov.real)
        plt.savefig('fov.png')
        plt.figure()
        for i in range(len(positions)):
            plt.subplot(len(positions), 3, 3 * i + 1)
            plt.imshow(truth[i].real)
            plt.subplot(len(positions), 3, 3 * i + 2)
            plt.imshow(patches[i].real)
            plt.subplot(len(positions), 3, 3 * i + 3)
            plt.imshow(patches[i].real - truth[i].real, cmap=plt.cm.inferno)
            plt.colorbar()
        plt.savefig('patches.png')
    except ModuleNotFoundError:
        pass

    np.testing.assert_allclose(
        patches,
        truth,
        atol=1e-6,
    )


if __name__ == '__main__':
    unittest.main()
