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
    ndim = x.shape[1]
    const = cp.array([cp.sqrt(cp.pi / mu)**ndim, -cp.pi**2 / mu],
                     dtype='float32')

    kernel_size = _next_power_two(2 * m)
    block = (kernel_size, kernel_size)
    grid = (x.shape[0], 2 * m)

    G = cp.zeros([2 * n] * ndim, dtype="complex64")
    _scatter_kernel(grid, block, (
        ndim,
        G,
        f.astype('complex64'),
        x.shape[0],
        x.astype('float32'),
        n,
        m,
        const.astype('float32'),
    ))
    return G


def _gather(Fe, x, n, m, mu):
    ndim = x.shape[1]
    const = cp.array([cp.sqrt(cp.pi / mu)**ndim, -cp.pi**2 / mu],
                     dtype='float32')

    kernel_size = _next_power_two(2 * m)
    block = (kernel_size, kernel_size)
    grid = (x.shape[0], 2 * m)

    F = cp.zeros(x.shape[0], dtype="complex64")
    _gather_kernel(grid, block, (
        ndim,
        F,
        Fe.astype('complex64'),
        x.shape[0],
        x.astype('float32'),
        n,
        m,
        const.astype('float32'),
    ))
    return F


def _get_kernel(pad, mu, ndim=3):
    """Return the interpolation kernel for the USFFT."""
    xeq = cp.array(
        cp.meshgrid(
            *[cp.arange(-pad, pad)] * ndim,
            indexing='ij',
        ),
        dtype='float32',
    )
    return cp.exp(-mu * cp.sum(xeq**2, axis=0))


def eq2us(f, x, n, eps, xp, gather=None, fftn=None):
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
    ndim = x.shape[1]
    pad = n // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (kernel)
    kernel = _get_kernel(pad, mu, ndim=ndim)

    # FFT and compesantion for smearing
    fe = xp.zeros([2 * n] * ndim, dtype="complex64")
    fe[pad:end, pad:end, pad:end] = f / ((2 * n)**ndim * kernel)
    Fe = checkerboard(xp, fftn(checkerboard(xp, fe)), inverse=True)
    F = _gather(Fe, x, n, m, mu)

    return F


def us2eq(f, x, n, eps, xp, scatter=None, fftn=None):
    """USFFT from unequally-spaced grid to equally-spaced grid.

    Parameters
    ----------
    f : (N) complex64
        Values of unequally-spaced function on the grid x
    x : (N, 3) float
        The frequencies on the unequally-spaced grid
    n : int
        The size of the equall spaced grid.
    eps : float
        The accuracy of computing USFFT
    """
    fftn = xp.fft.fftn if fftn is None else fftn
    ndim = x.shape[1]
    pad = n // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (ker)
    kernel = _get_kernel(pad, mu, ndim=ndim)

    G = _scatter(f, x, n, m, mu)

    # FFT and compesantion for smearing
    F = checkerboard(xp, fftn(checkerboard(xp, G)), inverse=True)
    F = F[pad:end, pad:end, pad:end] / ((2 * n)**ndim * kernel)

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


def _next_power_two(v):
    """Return the next highest power of 2 of 32-bit v.
    https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    """
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1
