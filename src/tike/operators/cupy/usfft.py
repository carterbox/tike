"""Provides unequally-spaced fast fourier transforms (USFFT).

The USFFT, NUFFT, or NFFT is a fast-fourier transform from an uniform domain to
a non-uniform domain or vice-versa. This module provides forward Fourier
transforms for those two cased. The inverse Fourier transforms may be created
by negating the frequencies on the non-uniform grid.
"""
from importlib_resources import files

import cupy as cp

_cu_source = files('tike.operators.cupy').joinpath('usfft.cu').read_text()
_scatter_kernel = cp.RawKernel(_cu_source, "scatter")
_gather_kernel = cp.RawKernel(_cu_source, "gather")


def _scatter(f, x, n, m, mu):
    G = cp.zeros([2 * n] * 3, dtype="complex64")
    const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu], dtype='float32')
    block = (min(_scatter_kernel.max_threads_per_block, (2 * m)**3),)
    grid = (1, 0, min(f.shape[0], 65535))
    _scatter_kernel(grid, block, (
        G,
        f.astype('complex64'),
        f.shape[0],
        x.astype('float32'),
        n,
        m,
        const.astype('float32'),
    ))
    return G


def _gather(Fe, x, n, m, mu):
    F = cp.zeros(x.shape[0], dtype="complex64")
    const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu], dtype='float32')
    block = (min(_gather_kernel.max_threads_per_block, (2 * m)**3),)
    grid = (1, 0, min(x.shape[0], 65535))
    _gather_kernel(grid, block, (
        F,
        Fe.astype('complex64'),
        x.shape[0],
        x.astype('float32'),
        n,
        m,
        const.astype('float32'),
    ))
    return F


def _get_kernel(xp, pad, mu):
    """Return the interpolation kernel for the USFFT."""
    xeq = xp.mgrid[-pad:pad, -pad:pad, -pad:pad]
    return xp.exp(-mu * xp.sum(xeq**2, axis=0)).astype('float32')


def eq2us(f, x, n, eps, xp, gather=_gather, fftn=None):
    """USFFT from equally-spaced grid to unequally-spaced grid.

    Parameters
    ----------
    f : (n, n, n) complex64
        The function to be transformed on a regular-grid of size n.
    x : (N, 3)
        The sampled frequencies on unequally-spaced grid.
    eps : float
        The desired relative accuracy of the USFFT.
    """
    fftn = xp.fft.fftn if fftn is None else fftn
    ndim = f.ndim
    pad = n // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (kernel)
    kernel = _get_kernel(xp, pad, mu)

    # FFT and compesantion for smearing
    fe = xp.zeros([2 * n] * ndim, dtype="complex64")
    fe[pad:end, pad:end, pad:end] = f / ((2 * n)**ndim * kernel)
    Fe = checkerboard(xp, fftn(checkerboard(xp, fe)), inverse=True)
    F = _gather(Fe, x, n, m, mu)

    return F


def us2eq(f, x, n, eps, xp, scatter=_scatter, fftn=None):
    """USFFT from unequally-spaced grid to equally-spaced grid.

    Parameters
    ----------
    f : (n**3) complex64
        Values of unequally-spaced function on the grid x
    x : (n**3) float
        The frequencies on the unequally-spaced grid
    n : int
        The size of the equall spaced grid.
    eps : float
        The accuracy of computing USFFT
    scatter : function
        The scatter function to use.
    """
    fftn = xp.fft.fftn if fftn is None else fftn
    pad = n // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (ker)
    kernel = _get_kernel(xp, pad, mu)

    G = _scatter(f, x, n, m, mu)

    # FFT and compesantion for smearing
    F = checkerboard(xp, fftn(checkerboard(xp, G)), inverse=True)
    F = F[pad:end, pad:end, pad:end] / ((2 * n)**3 * kernel)

    return F


def _unpad(array, width, mode='wrap'):
    """Remove padding from an array in-place.

    Parameters
    ----------
    array : array
        The array to strip.
    width : int
        The number of indices to remove from both sides along each dimension.
    mode : string
        'wrap' - Add the discarded regions to the array by wrapping them. The
        end regions are added to the beginning and the beginning regions are
        added the end of the new array.

    Returns
    -------
    array : array
        A view of the original array.
    """
    twice = 2 * width
    for _ in range(array.ndim):
        array[+width:+twice] += array[-width:]
        array[-twice:-width] += array[:width]
        array = array[width:-width]
        array = cp.moveaxis(array, 0, -1)
    return array


def _g(x):
    """Return -1 for odd x and 1 for even x."""
    return 1 - 2 * (x % 2)


def checkerboard(xp, array, axes=None, inverse=False):
    """In-place FFTshift for even sized grids only.

    If and only if the dimensions of `array` are even numbers, flipping the
    signs of input signal in an alternating pattern before an FFT is equivalent
    to shifting the zero-frequency component to the center of the spectrum
    before the FFT.
    """
    axes = range(array.ndim) if axes is None else axes
    for i in axes:
        if array.shape[i] % 2 != 0:
            raise ValueError(
                "Can only use checkerboard algorithm for even dimensions. "
                f"This dimension is {array.shape[i]}.")
        array = xp.moveaxis(array, i, -1)
        array *= _g(xp.arange(array.shape[-1]) + 1)
        if inverse:
            array *= _g(array.shape[-1] // 2)
        array = xp.moveaxis(array, -1, i)
    return array
