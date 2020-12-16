"""Functions related to creating and manipulating probe arrays.

Ptychographic probes are represented as two separate components: a common probe
whose values are the same for all positions and the varying component. The
former is required as it provides the common probe constraint for ptychography
and the later relaxes the former constraint to accomodate real-world
illuminations which may vary with time.

The common components consist of a single array representing multiple
incoherent probes each of which may have an accompanying varying component.

The varying components are stored sparsely as two arrays, so full varying
probes are only reconstructed as needed. The first array is an array of
principal components (coherent modes) that are shared for all positions and the
second is an array of weights for each position.

Each incoherent probe may have its own set of coherent modes. The full varying
probe at a given position is reconstructed by adding the common probe to the
weighted sum of the coherent modes.

```
unique_probe = common_probe + np.sum(weights * coherent_modes)
```

Design comments
---------------
In theory, the probe representation could be implemented in as little as two
arrays: one with all of the shared components and one with the varying
components where the common_probe becomes the first coherent mode. Choosing to
keep the shared coherent_modes separate from the common_probe as a third array
provides backwards compatability and allows for storing fewer coherent_modes in
the case when only some incoherent probes are allowed to vary.

"""

import cupy as cp
import numpy as np


def get_unique(common_probe, m=None, weights=None):
    """Construct the m-th unique probe from a common_probe and weights.

    Parameters
    ----------
    common_probe : (..., 1, COHER, INCOH, WIDE, HIGH) complex64
        The common probe amongst all positions.
    m : int or list(int)
        The index of the requested probe
    weights : (..., POSI, COHER, INCOH) float32
        The relative intensity of the coherent probes at each position

    Returns
    -------
    unique_probes : (..., POSI, 1, 1, WIDE, HIGH)
    """
    if m is None:
        m = list(range(common_probe.shape[-3]))
    if type(m) is not list:
        m = [m]
    if weights is None:
        # The probe does not vary with position.
        return common_probe[..., 0:1, m, :, :].copy()
    else:
        return np.sum(
            common_probe[..., :, :, m, :, :] *
            weights[..., :, :, m, None, None],
            axis=-4,
            keepdims=True,
        )


def add_modes_random_phase(probe, nmodes):
    """Initialize additional probe modes by phase shifting the first mode.

    Parameters
    ----------
    probe : (:, :, :, M, :, :) array
        A probe with M > 0 incoherent modes.
    nmodes : int
        The number of desired modes.

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    all_modes = np.empty((*probe.shape[:-3], nmodes, *probe.shape[-2:]),
                         dtype='complex64')
    pw = probe.shape[-1]
    for m in range(nmodes):
        if m < probe.shape[-3]:
            # copy existing mode
            all_modes[..., m, :, :] = probe[..., m, :, :]
        else:
            # randomly shift the first mode
            shift = np.exp(-2j * np.pi * (np.random.rand(2, 1) - 0.5) *
                           ((np.arange(0, pw) + 0.5) / pw - 0.5))
            all_modes[..., m, :, :] = (probe[..., 0, :, :] * shift[0][None] *
                                       shift[1][:, None])
    return all_modes


# TODO: Possibly a faster implementation would use QR decomposition, but numpy
# only support 2D inputs for QR as of 2020.04.
def orthogonalize_gs(x, ndim=1):
    """Gram-schmidt orthogonalization for complex arrays.

    x : (..., nmodes, :, :) array_like
        The array with modes in the -3 dimension.

    ndim : int > 0
        The number of trailing dimensions to orthogonalize.

    """
    if ndim < 1:
        raise ValueError("Must orthogonalize at least one dimension!")

    def inner(x, y, axis=None):
        """Return the complex inner product of x and y along axis."""
        return np.sum(np.conj(x) * y, axis=axis, keepdims=True)

    unflat_shape = x.shape
    nmodes = unflat_shape[-ndim - 1]
    x_ortho = x.reshape(*unflat_shape[:-ndim], -1)

    for i in range(1, nmodes):
        u = x_ortho[..., 0:i, :]
        v = x_ortho[..., i:i + 1, :]
        projections = u * inner(u, v, axis=-1) / inner(u, u, axis=-1)
        x_ortho[..., i:i + 1, :] -= np.sum(projections, axis=-2, keepdims=True)

    if __debug__:
        # Test each pair of vectors for orthogonality
        for i in range(nmodes):
            for j in range(i):
                error = abs(
                    inner(x_ortho[..., i:i + 1, :],
                          x_ortho[..., j:j + 1, :],
                          axis=-1))
                assert np.all(error < 1e-5), (
                    f"Some vectors are not orthogonal!, {error}, {error.shape}")

    return x_ortho.reshape(unflat_shape)


def orthogonalize_eig(x):
    """Orthogonalize modes of x using eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (..., nmodes, :, :) array_like complex64
        An array of the probe modes vectorized

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    nmodes = x.shape[-3]
    # 'A' holds the dot product of all possible mode pairs. We only fill the
    # lower half of `A` because it is conjugate-symmetric
    A = cp.empty((*x.shape[:-3], nmodes, nmodes), dtype='complex64')
    for i in range(nmodes):
        for j in range(i + 1):
            A[..., i, j] = cp.sum(cp.conj(x[..., i, :, :]) * x[..., j, :, :],
                                  axis=(-1, -2))

    _, vectors = cp.linalg.eigh(A, UPLO='L')
    # np.linalg.eigh guarantees that the eigen values are returned in ascending
    # order, so we just reverse the order of modes to have them sorted in
    # descending order.

    # TODO: Optimize this double-loop
    x_new = cp.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            # Sort new modes by eigen value in decending order.
            x_new[..., nmodes - 1 -
                  j, :, :] += vectors[..., i, j, None, None] * x[..., i, :, :]
    assert x_new.shape == x.shape, [x_new.shape, x.shape]

    return x_new


if __name__ == "__main__":
    cp.random.seed(0)
    x = (cp.random.rand(7, 1, 9, 3, 3) +
         1j * cp.random.rand(7, 1, 9, 3, 3)).astype('complex64')
    x1 = orthogonalize_eig(x)
    assert x1.shape == x.shape, x1.shape
